"""TimeChat adapter (LLaMA-2-7B backbone) — video-path intervention.

TimeChat (CVPR'24, arxiv:2312.02051) = EVA-ViT-g + InstructBLIP Q-Former +
timestamp-aware Video Q-Former (window 32) + LLaMA-2-7B-chat (LoRA). It samples
a FIXED budget of 96 frames and binds each frame to its absolute timestamp, so
it grounds and answers in real seconds.

Integration (same adapter contract as the other adapters):
- FIXED frame budget (96) -> keep/remove interventions are done at the
  VIDEO-PATH level (crop the evidence window with ffmpeg, then re-sample 96
  frames from the crop), NOT by subsetting the 96 full-video frames. Same
  rationale as VideoMind / GVLLM / VTimeLLM: always 96 frames, in-distribution.
- QA prompt: SHARED (supportgain.config.prompts.build_qa_prompt), parsed by the
  shared parser.
- Grounding: SHARED GROUNDING_CORE (question-conditioned, NO answer leak) with
  TimeChat's native "start/end timestamps in seconds" output trigger. Because
  TimeChat is timestamp-aware, the two numbers ARE absolute seconds (no /100
  de-normalisation as with VTimeLLM's 0..99 indices).
- Blind = a synthetic all-black clip (ffmpeg lavfi color=black), so K_blind is a
  genuine no-visual-evidence control (TimeChat needs a real file, not zeros).
"""
import os
import re
import sys
import shutil
import tempfile
import subprocess
from typing import List, Optional, Tuple

import numpy as np
import torch

from supportgain.models.base import ModelAdapter
from supportgain.config.prompts import GROUNDING_CORE, build_qa_prompt
from supportgain.core.parser import parse_answer

from supportgain.paths import TIMECHAT_ROOT as TIMECHAT_REPO
CFG_PATH = os.path.join(TIMECHAT_REPO, 'eval_configs', 'timechat.yaml')
CKPT = os.path.join(TIMECHAT_REPO, 'ckpt', 'timechat', 'timechat_7b.pth')

N_FRAMES = 96  # TimeChat's max_frame_pos / training frame count


def _ensure_on_path():
    if TIMECHAT_REPO not in sys.path:
        sys.path.insert(0, TIMECHAT_REPO)


def _build_grounding_prompt(question: str) -> str:
    # SHARED GROUNDING_CORE (question-conditioned, no answer leak) + TimeChat's
    # native seconds-localization trigger so it emits "start - end seconds".
    core = GROUNDING_CORE.format(question=question.strip())
    return (f"{core}\nReturn the start and end timestamps in seconds of that "
            "segment, in the format 'start - end seconds'.")


class Adapter(ModelAdapter):
    name = 'timechat'

    SYSTEM = ("You are able to understand the visual content that the user "
              "provides. Follow the instructions carefully and explain your "
              "answers in detail.")

    # ---- lifecycle --------------------------------------------------------
    def load(self, device: str = 'cuda:0') -> None:
        _ensure_on_path()
        import argparse
        cwd = os.getcwd()
        os.chdir(TIMECHAT_REPO)   # yaml + from_config read ckpt/* relative to cwd
        try:
            from timechat.common.config import Config
            from timechat.common.registry import registry
            from timechat.conversation.conversation_video import (
                Chat, conv_llava_llama_2, StoppingCriteriaSub)
            from transformers import StoppingCriteriaList
            # trigger registration
            import timechat.models  # noqa: F401
            import timechat.processors  # noqa: F401
            import timechat.runners  # noqa: F401
            import timechat.tasks  # noqa: F401
            import timechat.datasets.builders  # noqa: F401

            args = argparse.Namespace(cfg_path=CFG_PATH, options=None)
            cfg = Config(args)
            mc = cfg.model_cfg
            mc.ckpt = CKPT
            gpu = int(device.split(':')[-1]) if ':' in device else 0
            mc.device_8bit = gpu
            model_cls = registry.get_model_class(mc.arch)
            model = model_cls.from_config(mc).to(device)
            model.eval()
            vp_cfg = cfg.datasets_cfg.webvid.vis_processor.train
            vis_processor = registry.get_processor_class(
                vp_cfg.name).from_config(vp_cfg)
        finally:
            os.chdir(cwd)

        self.Chat = Chat
        self.conv_proto = conv_llava_llama_2
        self._StopList = StoppingCriteriaList
        self._StopSub = StoppingCriteriaSub
        self.model = model
        self.chat = Chat(model, vis_processor, device=device)
        self.device = device

    def supports_video_path_intervention(self) -> bool:
        return True

    # ---- decode (for duration only; TimeChat re-decodes internally) -------
    def decode_video(self, video_path: str
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
        import decord
        vr = decord.VideoReader(video_path, num_threads=1)
        total = len(vr)
        fps = float(vr.get_avg_fps())
        dur = total / max(fps, 1e-6)
        placeholder = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        return placeholder, np.zeros(1), float(dur)

    # ---- ffmpeg helpers (video-path intervention) -------------------------
    def _even(self):
        return ['-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2']

    def _crop(self, path, s, e):
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(s),
                            '-to', str(e), '-i', path, *self._even(),
                            '-c:v', 'libx264', '-an', tmp], capture_output=True)
        if r.returncode != 0 or not os.path.exists(tmp):
            return None
        return tmp

    def _complement(self, path, s, e, duration):
        parts = []
        if s > 0.05:
            parts.append((0.0, s))
        if e < duration - 0.05:
            parts.append((e, duration))
        if not parts:
            return None
        segs = [self._crop(path, a, b) for a, b in parts]
        if any(x is None for x in segs):
            for x in segs:
                if x:
                    os.unlink(x)
            return None
        if len(segs) == 1:
            return segs[0]
        lst = tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False)
        for sg in segs:
            lst.write(f"file '{sg}'\n")
        lst.close()
        out = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat',
                            '-safe', '0', '-i', lst.name, '-c', 'copy', out],
                           capture_output=True)
        for sg in segs:
            os.unlink(sg)
        os.unlink(lst.name)
        if r.returncode != 0 or not os.path.exists(out):
            return None
        return out

    def _black(self, duration):
        d = max(1.0, min(float(duration), 8.0))
        out = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        r = subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
             '-i', f'color=black:s=224x224:r=8:d={d:.1f}',
             '-c:v', 'libx264', out], capture_output=True)
        if r.returncode != 0 or not os.path.exists(out):
            return None
        return out

    def _video_for(self, video_path, spec):
        """Return (path, is_temp) for a given intervention spec, or (None, _)."""
        if spec is None:
            return video_path, False
        kind = spec.get('kind')
        iv = spec.get('interval')
        dur = self.decode_video(video_path)[2]
        if kind == 'blind':
            return self._black(dur), True
        if kind in (None, 'full'):
            return video_path, False
        if iv is None:
            return None, False
        s, e = float(iv[0]), float(iv[1])
        if 'keep' in kind:
            return self._crop(video_path, s, e), True
        if 'remove' in kind:
            return self._complement(video_path, s, e, dur), True
        return None, False

    # ---- core Chat call ---------------------------------------------------
    def _embed_ids(self, ids):
        m = self.model.llama_model
        emb = (m.get_base_model().model.embed_tokens if self.model.lora
               else m.model.embed_tokens)
        return emb(ids)

    @torch.inference_mode()
    def _answer_greedy(self, conv, img_list, max_new_tokens=64, max_length=2000,
                       prime=None):
        """Deterministic (do_sample=False) reimplementation of Chat.answer so
        TEF metrics are reproducible. Upstream Chat.answer hardcodes
        do_sample=True/temperature=1.0. conv_llava_llama_2 uses sep2='</s>'.

        `prime` (e.g. 'The answer is (') is appended as decoder-side tokens so
        the model must commit to a letter -- the same trick VideoMind uses
        ('Best Option: (') to stop LLaMA-2-chat from hedging ('None of the
        above'). Grounding leaves prime=None."""
        conv.append_message(conv.roles[1], None)
        embs = self.chat.get_context_emb(conv, img_list)
        if prime:
            pid = self.model.llama_tokenizer(
                prime, return_tensors='pt',
                add_special_tokens=False).input_ids.to(self.device)
            embs = torch.cat([embs, self._embed_ids(pid)], dim=1)
        cur = embs.shape[1] + max_new_tokens
        begin = max(0, cur - max_length)
        embs = embs[:, begin:]
        stop = self._StopList([self._StopSub(
            stops=[torch.tensor([2]).to(self.device)])])
        out = self.model.llama_model.generate(
            inputs_embeds=embs, max_new_tokens=max_new_tokens,
            stopping_criteria=stop, num_beams=1, do_sample=False,
            min_length=1, repetition_penalty=1.0, length_penalty=1.0)
        tok = out[0]
        if tok[0] == 0:
            tok = tok[1:]
        if len(tok) and tok[0] == 1:
            tok = tok[1:]
        text = self.model.llama_tokenizer.decode(tok, add_special_tokens=False)
        text = text.split(conv.sep2)[0].split(conv.roles[1] + ':')[-1].strip()
        return (prime + text) if prime else text

    @torch.inference_mode()
    def _chat_once(self, video_path: str, prompt_text: str,
                   max_new_tokens: int = 64, prime=None) -> str:
        conv = self.conv_proto.copy()
        conv.system = self.SYSTEM
        img_list = []
        self.chat.upload_video_without_audio(
            video_path=video_path, conv=conv, img_list=img_list, n_frms=N_FRAMES)
        self.chat.ask(prompt_text, conv)
        return self._answer_greedy(conv, img_list, max_new_tokens=max_new_tokens,
                                   prime=prime)

    # ---- public inference -------------------------------------------------
    def predict_qa(self, frames: np.ndarray, question: str,
                   choices: List[str]) -> Tuple[int, str]:
        raise NotImplementedError('timechat uses predict_qa_video')

    def predict_qa_video(self, video_path: str, intervention_spec: Optional[dict],
                         question: str, choices: List[str]) -> Tuple[int, str]:
        path, is_temp = self._video_for(video_path, intervention_spec)
        if path is None:
            return -1, ''
        try:
            raw = self._chat_once(path, build_qa_prompt(question, choices),
                                  max_new_tokens=32, prime='The answer is (')
        finally:
            if is_temp and path and os.path.exists(path):
                os.unlink(path)
        return parse_answer(raw, choices), raw

    def predict_grounding(self, frames: np.ndarray, question: str,
                          predicted_answer: str, duration: float,
                          video_path: Optional[str] = None,
                          ) -> Tuple[Optional[Tuple[float, float]], str, Optional[str]]:
        # SHARED CORE, question-conditioned (predicted_answer intentionally unused).
        # Grounding always runs on the FULL video (timestamp-aware seconds).
        raw = self._chat_once(video_path, _build_grounding_prompt(question),
                              max_new_tokens=64)
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to|and|~)\s*(\d+(?:\.\d+)?)', raw)
        if not m:
            return None, raw, 'parse_fail'
        s, e = float(m.group(1)), float(m.group(2))
        if e < s:
            s, e = e, s
        s = max(0.0, min(duration, s))
        e = max(0.0, min(duration, e))
        return (s, e), raw, None
