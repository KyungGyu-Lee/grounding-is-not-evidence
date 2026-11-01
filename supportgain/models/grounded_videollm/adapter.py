"""TEF adapter for Grounded-VideoLLM (Phi-3.5 backbone).

Design (per TEF MODEL_SETUP_GUIDE + project decisions):
  * QA (1st)        : SHARED MCQ prompt (build_qa_prompt), letter output,
                      common parser — identical across all models.
  * Grounding (2nd) : SHARED instruction (GROUNDING_CORE, question-conditioned)
                      + the model's native <timestamp_grounding> trigger token.
                      Output is the model's NATIVE discrete time tokens
                      `<0>..<299>` (idx → duration*idx/300 s); only the PARSER
                      is model-specific. finalize_interval() unifies to one
                      absolute-second interval (longest).

Grounded-VideoLLM ingests a *video file* internally (read_frames_decord →
fixed 96 temporal frames @224 + 12 spatial segs @336), so interventions are
applied at the video-file level (opt-in supports_video_path_intervention):
  * phase2 keep   → read_frames_decord(clip=(start,end))   (evidence only)
  * phase2 remove → ffmpeg trim+concat of the complement, then decode
  * blind         → zero the pixel tensors

The repo ships its own top-level `models`, `mm_utils`, `datasets` packages
that collide with TEF's — we swap them in/out of sys.modules around import
(HawkEye pattern).
"""
import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import torch

from supportgain.models.base import ModelAdapter
from supportgain.config.prompts import build_qa_prompt, GROUNDING_CORE
from supportgain.config.settings import GROUNDING_REDUCE
from supportgain.core.parser import parse_answer
from supportgain.core.intervals import finalize_interval

from supportgain.paths import GVLLM_ROOT, ROOT as TEF_ROOT
WEIGHT_PATH = os.path.join(GVLLM_ROOT, 'weight_path')

# Top-level package names that must resolve to GVLLM (not TEF) during import.
_CONFLICTING_TOPLEVEL = ('models', 'mm_utils', 'datasets', 'overwatch', 'training')


@contextlib.contextmanager
def _gvllm_scope():
    """Temporarily make GVLLM's top-level packages importable instead of TEF's."""
    saved = {}
    for k in list(sys.modules.keys()):
        for prefix in _CONFLICTING_TOPLEVEL:
            if k == prefix or k.startswith(prefix + '.'):
                saved[k] = sys.modules.pop(k)
                break
    saved_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if p != TEF_ROOT]
    sys.path.insert(0, GVLLM_ROOT)
    cwd = os.getcwd()
    os.chdir(GVLLM_ROOT)
    try:
        yield
    finally:
        os.chdir(cwd)
        sys.path[:] = saved_path
        for k in list(sys.modules.keys()):
            for prefix in _CONFLICTING_TOPLEVEL:
                if k == prefix or k.startswith(prefix + '.'):
                    sys.modules.pop(k, None)
                    break
        sys.modules.update(saved)


class Adapter(ModelAdapter):
    name = 'grounded_videollm'

    def __init__(self):
        self.model = None
        self.device = 'cuda:0'
        self.dtype = torch.bfloat16
        self.num_frames = 96
        self.num_segs = 12
        self.num_temporal_tokens = 300
        self.max_txt_len = 2048
        # GVLLM_NATIVE=1 -> use the repo demo's native grounding template
        # ("Give you a textual query: '...'. When does the described content
        # occur ...") instead of TEF's GROUNDING_CORE. Still question-conditioned
        # (query = the question; no answer leakage), matching how GroundVTS-native
        # was run.
        self.native = os.environ.get('GVLLM_NATIVE', '0') == '1'
        # GVLLM_ANSCOND=1 -> answer-conditioned grounding: query = predicted
        # answer text (declarative), matching how Charades/ActivityNet feed a
        # declarative sentence. Breaks question-conditioned invariant (uses the
        # answer) -> paper-reproduction only, NOT the fair cross-model setup.
        self.anscond = os.environ.get('GVLLM_ANSCOND', '0') == '1'

    # this model intervenes at the video-file level
    def supports_video_path_intervention(self) -> bool:
        return True

    # ---- load -------------------------------------------------------------
    def load(self, device: str = 'cuda:0'):
        self.device = device
        with _gvllm_scope():
            from models.llava_next_video import LLAVA_NEXT_VIDEO
            from datasets.chat.base_template import (
                Phi_3_5_Template, DEFAULT_IMAGE_TOKEN, GROUNDING_TOKEN)
            from mm_utils.video_utils import read_frames_decord
            from mm_utils.utils import (
                frame_transform, INTERNVIDEO_MEAN, INTERNVIDEO_STD,
                OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)

            model = LLAVA_NEXT_VIDEO(
                dtype=self.dtype, stage='sft', max_txt_len=self.max_txt_len,
                num_frames=self.num_frames, num_segs=self.num_segs,
                num_temporal_tokens=self.num_temporal_tokens, lora=True,
                llm='phi3.5', attn_implementation='eager',
                config_path=os.path.join(WEIGHT_PATH, 'Phi-3.5-vision-instruct'),
                tokenizer_path=os.path.join(WEIGHT_PATH, 'Phi-3.5-mini-instruct'),
                pretrained_video_path=os.path.join(
                    WEIGHT_PATH,
                    'internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt'),
                pretrained_vision_proj_llm_path=os.path.join(
                    WEIGHT_PATH, 'Phi-3.5-vision-instruct-seperated/'),
            )
            ckpt_path = os.path.join(
                WEIGHT_PATH,
                'ckpt/sft_llava_next_video_phi3.5_mix_sft_'
                'multi_modal_projector_video_projecter_language_model.pth')
            ckpt = torch.load(ckpt_path, map_location='cpu')['model']
            for key in ('multi_modal_projector', 'video_projecter', 'language_model'):
                if key in ckpt:
                    getattr(model, key).load_state_dict(ckpt[key])
            model.eval()
            model.to(device)

        self.model = model
        self._read_frames = read_frames_decord
        self._tpl = Phi_3_5_Template()
        self._IMG = DEFAULT_IMAGE_TOKEN
        self._GND = GROUNDING_TOKEN
        self._vproc = frame_transform(image_size=224, mean=INTERNVIDEO_MEAN, std=INTERNVIDEO_STD)
        self._iproc = frame_transform(image_size=336, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        print(f'[grounded_videollm] loaded on {device}', flush=True)

    # ---- pixel building ---------------------------------------------------
    def _build_pixels(self, pixel_values):
        """pixel_values: (T, C, H, W) uint8 tensor from read_frames_decord."""
        tp = torch.tensor(np.array(
            [self._vproc(pixel_values[i]) for i in range(pixel_values.shape[0])]
        )).unsqueeze(0)
        npf = self.num_frames // self.num_segs
        idx = [(i * npf) + npf // 2 for i in range(self.num_segs)]
        sp = torch.tensor(np.array([self._iproc(pixel_values[i]) for i in idx])).unsqueeze(0)
        return tp.to(self.device), sp.to(self.device)

    def _gen(self, prompt, tp, sp, max_new_tokens):
        samples = {
            'video_ids': ['x'], 'question_ids': ['x'], 'prompts': [prompt],
            'temporal_pixel_values': tp, 'spatial_pixel_values': sp,
        }
        with torch.cuda.amp.autocast(enabled=True, dtype=self.dtype):
            with torch.inference_mode():
                out = self.model.generate(
                    samples, do_sample=False, num_beams=1,
                    max_new_tokens=max_new_tokens)[0]
        return out

    def _encode(self, human_value):
        conv = [{'from': 'human', 'value': human_value}, {'from': 'gpt', 'value': ''}]
        _, eos = self._tpl.separator.apply()
        return self._tpl.encode(conv).replace(eos, '')

    # ---- intervention source pixels --------------------------------------
    def _pixels_for(self, video_path, spec):
        """(temporal, spatial) pixel tensors for the intervention, or None."""
        blind = False
        clip = None
        tmp = None
        if spec is not None and spec.get('kind') not in (None, 'full'):
            kind = spec['kind']
            iv = spec.get('interval')
            if kind == 'blind':
                blind = True
            elif kind == 'keep':
                if iv is None:
                    return None
                clip = (float(iv[0]), float(iv[1]))
            elif kind == 'remove':
                if iv is None:
                    return None
                tmp = self._ffmpeg_complement(video_path, iv, spec['total_duration'])
                if tmp is None:
                    return None
                video_path = tmp
        try:
            pv, _, _, _, _ = self._read_frames(video_path, self.num_frames, 'middle', clip=clip)
        finally:
            if tmp:
                shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)
        tp, sp = self._build_pixels(pv)
        if blind:
            tp = torch.zeros_like(tp)
            sp = torch.zeros_like(sp)
        return tp, sp

    def _ffmpeg_complement(self, path, iv, duration):
        """Temp mp4 = video with [iv] removed (trim complement + concat)."""
        s, e, dur = float(iv[0]), float(iv[1]), float(duration)
        segs = []
        if s > 0.1:
            segs.append((0.0, s))
        if e < dur - 0.1:
            segs.append((e, dur))
        if not segs:
            return None
        tmpdir = tempfile.mkdtemp(prefix='gvllm_rm_')
        files = []
        for i, (a, b) in enumerate(segs):
            out = os.path.join(tmpdir, f'p{i}.mp4')
            r = subprocess.run(
                ['ffmpeg', '-y', '-loglevel', 'error', '-i', path,
                 '-ss', f'{a:.3f}', '-to', f'{b:.3f}',
                 '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                 '-c:v', 'libx264', '-an', out], capture_output=True)
            if r.returncode != 0 or not os.path.exists(out):
                shutil.rmtree(tmpdir, ignore_errors=True)
                return None
            files.append(out)
        if len(files) == 1:
            return files[0]
        listf = os.path.join(tmpdir, 'list.txt')
        with open(listf, 'w') as f:
            for fp in files:
                f.write(f"file '{fp}'\n")
        out = os.path.join(tmpdir, 'concat.mp4')
        r = subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
             '-i', listf, '-c', 'copy', out], capture_output=True)
        if r.returncode != 0 or not os.path.exists(out):
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None
        return out

    # ---- TEF contract -----------------------------------------------------
    def decode_video(self, video_path: str):
        pv, idx, fps, vlen, dur = self._read_frames(video_path, self.num_frames, 'middle')
        frames = pv.permute(0, 2, 3, 1).numpy()
        timestamps = np.asarray(idx, dtype=np.float64) / float(fps)
        return frames, timestamps, float(dur)

    def predict_qa(self, frames, question, choices):
        raise NotImplementedError(
            'grounded_videollm uses predict_qa_video (video-path intervention)')

    def predict_qa_video(self, video_path, intervention_spec, question, choices):
        px = self._pixels_for(video_path, intervention_spec)
        if px is None:
            return -1, ''
        tp, sp = px
        # SHARED MCQ prompt (identical across all models)
        prompt = self._encode(self._IMG + '\n' + build_qa_prompt(question, choices))
        text = self._gen(prompt, tp, sp, max_new_tokens=16)
        return parse_answer(text, choices), text

    def predict_grounding(self, frames, question, predicted_answer, duration,
                          video_path: Optional[str] = None):
        # SHARED instruction (GROUNDING_CORE, question-conditioned) + native
        # <timestamp_grounding> trigger. predicted_answer intentionally unused.
        pv, _, _, _, dur = self._read_frames(video_path, self.num_frames, 'middle')
        tp, sp = self._build_pixels(pv)
        # SHARED GROUNDING_CORE (question-conditioned) + native output directive
        # that triggers GVLLM's time-token grounding. The CORE wording is
        # identical across models; only this output directive + the parser are
        # native (mirrors Qwen's "Format only: start=X.X, end=Y.Y").
        if self.anscond or self.native:
            # Native demo template (inference.py prompt_grounding). The query is
            # the predicted answer (declarative, answer-conditioned) when ANSCOND,
            # else the question.
            q = (predicted_answer if (self.anscond and predicted_answer) else question)
            human = (self._IMG + ' ' + self._GND + '\n'
                     + f"Give you a textual query: '{str(q).strip()}'. "
                     "When does the described content occur in the video? "
                     "Please return the start and end timestamps.")
        else:
            human = (self._IMG + ' ' + self._GND + '\n'
                     + GROUNDING_CORE.format(question=question.strip())
                     + '\nPlease return the start and end timestamps.')
        prompt = self._encode(human)
        text = self._gen(prompt, tp, sp, max_new_tokens=64)
        # native parse: <idx> time tokens → seconds
        toks = [int(t) for t in re.findall(r'<(\d+)>', text)
                if 0 <= int(t) <= self.num_temporal_tokens]
        secs = sorted(dur * t / self.num_temporal_tokens for t in toks)
        spans = [(secs[0], secs[-1])] if len(secs) >= 2 else []
        s, e, flag = finalize_interval(spans, duration, reduce=GROUNDING_REDUCE)
        iv = (s, e) if s is not None else None
        return iv, text, flag
