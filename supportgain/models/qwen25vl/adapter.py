"""Qwen2.5-VL-7B-Instruct adapter.

Native sampling: variable frame count (fps × duration, clipped to
[min_frames, max_frames]). The model speaks free-form text for both QA
and grounding, so we use the default TEF prompts and parse with the
shared utilities.
"""
import os
from typing import Tuple, Optional, List

import numpy as np
import torch
from PIL import Image

from supportgain.models.base import ModelAdapter
from supportgain.config.prompts import build_qa_prompt, build_grounding_prompt
from supportgain.core.video_io import decode_at_fps
from supportgain.core.intervals import parse_grounding
from supportgain.core.parser import parse_answer

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
DEFAULT_FPS = 2.0
DEFAULT_MAX_FRAMES = 128
DEFAULT_MIN_FRAMES = 4


class Adapter(ModelAdapter):
    name = 'qwen25vl'

    def __init__(self, fps: float = DEFAULT_FPS,
                 max_frames: int = DEFAULT_MAX_FRAMES,
                 min_frames: int = DEFAULT_MIN_FRAMES):
        self.fps = fps
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.model = None
        self.processor = None
        self.device = None
        self._eff_fps = None  # effective fps of the decoded frames = n/duration.
        #   Qwen2.5-VL encodes ABSOLUTE time via second_per_grid_ts =
        #   temporal_patch_size / fps. For pre-decoded frames the processor
        #   ignores the message `fps`, defaulting second_per_grid_ts to 1.0, so a
        #   150 s clip sampled to 128 frames is perceived as 64 s and grounding
        #   collapses to the front (only bites when max_frames binds, i.e.
        #   duration > 64 s: 100% of QVH, ~17% of NExT). We pass this effective
        #   fps as a processor kwarg so each grid maps to real seconds; the value
        #   is constant per video and survives step3 cropping (uniform decode).

    # ---- lifecycle -----------------------------------------------------

    def load(self, device: str = 'cuda:0') -> None:
        os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            device_map={'': device},
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.device = device

    # ---- video ---------------------------------------------------------

    def decode_video(self, video_path: str
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
        frames, ts, dur = decode_at_fps(video_path, fps=self.fps,
                                        max_frames=self.max_frames,
                                        min_frames=self.min_frames)
        self._eff_fps = (len(frames) / dur) if (dur and len(frames)) else None
        return frames, ts, dur

    # ---- internal generate --------------------------------------------

    @torch.inference_mode()
    def _generate(self, frames: np.ndarray, prompt_text: str,
                  max_new_tokens: int = 64) -> str:
        if frames is None or len(frames) == 0:
            return ''
        pil_frames = [Image.fromarray(f) for f in frames]
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'video', 'video': pil_frames},
                {'type': 'text', 'text': prompt_text},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Pass effective fps as a processor kwarg (the message `fps` is ignored
        # for pre-decoded frames) so second_per_grid_ts encodes real seconds.
        fps_kw = {'fps': [float(self._eff_fps)]} if self._eff_fps else {}
        inputs = self.processor(
            text=[text], images=None, videos=[pil_frames],
            return_tensors='pt', padding=True, **fps_kw,
        )
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
        input_len = inputs['input_ids'].shape[1]
        gen = out[:, input_len:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]

    # ---- inference -----------------------------------------------------

    def predict_qa(self, frames: np.ndarray, question: str,
                   choices: List[str]) -> Tuple[int, str]:
        prompt = build_qa_prompt(question, choices)
        text = self._generate(frames, prompt, max_new_tokens=16)
        return parse_answer(text, choices), text

    def predict_grounding(self, frames: np.ndarray, question: str,
                          predicted_answer: str, duration: float,
                          video_path: Optional[str] = None,
                          ) -> Tuple[Optional[Tuple[float, float]], str, Optional[str]]:
        # video_path is unused — Qwen does single-call free-form grounding.
        # Question-conditioned: predicted_answer is intentionally NOT used.
        prompt = build_grounding_prompt(question, duration)
        text = self._generate(frames, prompt, max_new_tokens=64)
        s, e, flag = parse_grounding(text, duration)
        iv = (s, e) if s is not None else None
        return iv, text, flag
