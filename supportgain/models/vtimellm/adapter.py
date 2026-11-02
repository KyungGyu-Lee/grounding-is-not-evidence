"""VTimeLLM adapter (Vicuna-7B-v1.5 backbone) — video-path intervention.

VTimeLLM = ViT-Time-LLM (arxiv:2311.18445). CLIP ViT-L/14 encodes a fixed
budget of 100 uniformly-sampled frames into a feature sequence that feeds a
Vicuna decoder via a learned mm-projector + two LoRA stages.

Integration (same adapter contract as the other adapters):
- FIXED frame budget (100) -> keep/remove interventions are done at the
  VIDEO-PATH level (crop the evidence window with ffmpeg, then re-sample 100
  frames from the crop), NOT by subsetting the 100 full-video frames. This
  keeps the model in-distribution (always 100 frames) and gives a dense view
  of the evidence -- same rationale as VideoMind / GVLLM.
- QA prompt: SHARED (supportgain.config.prompts.build_qa_prompt), parsed by the
  shared parser.
- Grounding: SHARED GROUNDING_CORE (question-conditioned, NO answer leak) with
  VTimeLLM's native output format (two 0..99 frame indices "from X to Y",
  de-normalised by duration/100).
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
from supportgain.config.prompts import GROUNDING_CORE
from supportgain.core.parser import parse_answer

from supportgain.paths import VTIMELLM_ROOT as VTIMELLM_REPO
CKPT_DIR = os.path.join(VTIMELLM_REPO, 'checkpoints')
CLIP_PATH = os.path.join(CKPT_DIR, 'clip', 'ViT-L-14.pt')
MM_PROJECTOR = os.path.join(CKPT_DIR, 'vtimellm-vicuna-v1-5-7b-stage1', 'mm_projector.bin')
STAGE2 = os.path.join(CKPT_DIR, 'vtimellm-vicuna-v1-5-7b-stage2')
STAGE3 = os.path.join(CKPT_DIR, 'vtimellm-vicuna-v1-5-7b-stage3')
MODEL_BASE = os.path.join(CKPT_DIR, 'vicuna-7b-v1.5')

N_FRAMES = 100  # VTimeLLM's training-time frame count


def _ensure_on_path():
    if VTIMELLM_REPO not in sys.path:
        sys.path.insert(0, VTIMELLM_REPO)


def _build_qa_prompt(question: str, choices: List[str]) -> str:
    opts = "\n".join(f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(choices))
    return (f"<video>\nQuestion: {question}\nOptions:\n{opts}\n"
            "Answer with the letter only.")


def _build_grounding_prompt(question: str) -> str:
    # SHARED GROUNDING_CORE (question-conditioned, no answer leak) wrapped with
    # the <video> sentinel + VTimeLLM's native "which frames" output trigger so
    # the model emits its trained "from X to Y" (0..99) format.
    core = GROUNDING_CORE.format(question=question.strip())
    return f"<video>\n{core}\nDuring which frames does this evidence appear?"


class Adapter(ModelAdapter):
    name = 'vtimellm'

    # ---- lifecycle --------------------------------------------------------
    def load(self, device: str = 'cuda:0') -> None:
        _ensure_on_path()
        from vtimellm.model.builder import load_pretrained_model
        from vtimellm.utils import disable_torch_init
        import clip
        from torchvision.transforms import (
            Compose, Resize, CenterCrop, Normalize, InterpolationMode)
        from easydict import EasyDict as edict

        disable_torch_init()
        args = edict({'clip_path': CLIP_PATH, 'model_base': MODEL_BASE,
                      'pretrain_mm_mlp_adapter': MM_PROJECTOR,
                      'stage2': STAGE2, 'stage3': STAGE3})
        tokenizer, model, _ = load_pretrained_model(args, args.stage2, args.stage3)
        self.model = model.to(device).to(torch.float16).eval()
        self.clip_model, _ = clip.load(CLIP_PATH, device=device)
        self.clip_model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.transform = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711))])

    def supports_video_path_intervention(self) -> bool:
        return True

    # ---- frame decoding ---------------------------------------------------
    def _decode_n(self, video_path: str
                  ) -> Tuple[np.ndarray, np.ndarray, float]:
        import decord
        decord.bridge.set_bridge('native')
        vr = decord.VideoReader(video_path, num_threads=1)
        total = len(vr)
        fps = float(vr.get_avg_fps())
        duration = total / max(fps, 1e-6)
        n = min(N_FRAMES, total)
        idx = np.linspace(0, total - 1, num=n, dtype=int).tolist()
        frames = vr.get_batch(idx).asnumpy().astype(np.uint8)
        ts = np.array([i / fps for i in idx], dtype=np.float32)
        return frames, ts, duration

    def decode_video(self, video_path: str
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
        return self._decode_n(video_path)

    # ---- CLIP encode ------------------------------------------------------
    @torch.inference_mode()
    def _encode(self, frames: np.ndarray) -> torch.Tensor:
        if frames is None or len(frames) == 0:
            frames = np.zeros((N_FRAMES, 224, 224, 3), dtype=np.uint8)
        images = torch.from_numpy(frames.transpose((0, 3, 1, 2))).float() / 255.0
        images = self.transform(images).to(torch.float16).to(self.device)
        return self.clip_model.encode_image(images)

    @torch.inference_mode()
    def _generate(self, features: torch.Tensor, prompt: str,
                  max_new_tokens: int = 64) -> str:
        from vtimellm.inference import inference
        return inference(self.model, features, prompt, self.tokenizer)

    # ---- ffmpeg helpers (video-path intervention) -------------------------
    def _even(self):
        return ['-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2']

    def _crop(self, path, s, e):
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(s),
                        '-to', str(e), '-i', path, *self._even(),
                        '-c:v', 'libx264', '-an', tmp],
                       check=True, timeout=300)
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
        if len(segs) == 1:
            return segs[0]
        lst = tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False)
        for sg in segs:
            lst.write(f"file '{sg}'\n")
        lst.close()
        out = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat',
                        '-safe', '0', '-i', lst.name, '-c', 'copy', out],
                       check=True, timeout=300)
        for sg in segs:
            os.unlink(sg)
        os.unlink(lst.name)
        return out

    def _frames_for(self, video_path, spec):
        """Return the 100-frame stack for a given intervention spec."""
        if spec is None:                                   # full video
            return self._decode_n(video_path)[0]
        kind = spec.get('kind')
        iv = spec.get('interval')
        if kind == 'blind':
            return np.zeros((N_FRAMES, 224, 224, 3), dtype=np.uint8)
        if iv is None:
            return None
        s, e = float(iv[0]), float(iv[1])
        dur = self._decode_n(video_path)[2]
        try:
            if kind and 'keep' in kind:
                tmp = self._crop(video_path, s, e)
            elif kind and 'remove' in kind:
                tmp = self._complement(video_path, s, e, dur)
            else:
                return None
            if tmp is None:
                return None
            frames = self._decode_n(tmp)[0]
            os.unlink(tmp)
            return frames
        except Exception:
            return None

    # ---- public inference -------------------------------------------------
    def predict_qa(self, frames: np.ndarray, question: str,
                   choices: List[str]) -> Tuple[int, str]:
        raw = self._generate(self._encode(frames),
                             _build_qa_prompt(question, choices), max_new_tokens=32)
        return parse_answer(raw, choices), raw

    def predict_qa_video(self, video_path: str, intervention_spec: Optional[dict],
                         question: str, choices: List[str]) -> Tuple[int, str]:
        frames = self._frames_for(video_path, intervention_spec)
        if frames is None:
            return -1, ''
        raw = self._generate(self._encode(frames),
                             _build_qa_prompt(question, choices), max_new_tokens=32)
        return parse_answer(raw, choices), raw

    def predict_grounding(self, frames: np.ndarray, question: str,
                          predicted_answer: str, duration: float,
                          video_path: Optional[str] = None,
                          ) -> Tuple[Optional[Tuple[float, float]], str, Optional[str]]:
        # SHARED CORE, question-conditioned (predicted_answer intentionally unused).
        raw = self._generate(self._encode(frames),
                             _build_grounding_prompt(question), max_new_tokens=64)
        m = re.search(r'(\d{1,3})\s+(?:to|and|-)\s+(\d{1,3})', raw)
        if not m:
            return None, raw, 'parse_fail'
        s_norm, e_norm = float(m.group(1)), float(m.group(2))
        if e_norm < s_norm:
            s_norm, e_norm = e_norm, s_norm
        s = max(0.0, min(duration, s_norm * duration / 100.0))
        e = max(0.0, min(duration, e_norm * duration / 100.0))
        return (s, e), raw, None
