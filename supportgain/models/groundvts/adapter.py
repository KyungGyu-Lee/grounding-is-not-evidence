"""TEF adapter for GroundVTS-Q (Qwen2.5-VL-7B + Visual Token Sampling).

GroundVTS-Q = Qwen2.5-VL-7B backbone + a query-guided Visual Token Sampling
(VTS) module + grounding fine-tuning (Stage 3 LoRA). It ingests a video the
same way as vanilla Qwen2.5-VL (processor + process_vision_info), so this
adapter mirrors the qwen25vl adapter and only differs in:

  * load()            : builds the custom VTS_Qwen2_5_VL via GroundVTS'
                        build_model (Stage1_base + Stage2_3 grounding LoRA,
                        merge_adapter=True). VTS stays ON (vts_ratio=None ->
                        config default 0.5). The GroundVTS repo ships its own
                        top-level `models` package that collides with TEF's, so
                        we swap sys.modules around the import (HawkEye/GVLLM
                        pattern).
  * predict_grounding : SHARED GROUNDING_CORE (question-conditioned) + the
                        model's NATIVE output directive ("X to Y seconds"),
                        parsed with GroundVTS' own regex. predicted_answer is
                        intentionally NOT used (no answer leakage).

QA uses the SHARED MCQ prompt (build_qa_prompt) + the common letter parser,
identical to every other TEF model. Frame sampling matches the qwen25vl
baseline (decode_at_fps, fps=2.0) so the only difference vs the Qwen baseline
is the VTS module + grounding-FT weights -> isolates the grounding effect.
"""
import contextlib
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from supportgain.models.base import ModelAdapter
from supportgain.config.prompts import build_qa_prompt, GROUNDING_CORE
from supportgain.config.settings import GROUNDING_REDUCE
from supportgain.core.video_io import decode_at_fps
from supportgain.core.intervals import finalize_interval
from supportgain.core.parser import parse_answer

from supportgain.paths import GROUNDVTS_ROOT as GVTS_ROOT, ROOT as TEF_ROOT
WEIGHT = os.environ.get('TEF_GROUNDVTS_CKPT',
                        os.path.join(GVTS_ROOT, 'weight_path', 'GroundVTS-Q'))
STAGE1_BASE = os.path.join(WEIGHT, 'Stage1_base')
STAGE2_3_ADAPTER = os.path.join(WEIGHT, 'Stage2_3_adapter')

# Native GroundVTS pixel budget (eval/infer_auto.py).
MIN_PIXELS = 36 * 28 * 28
MAX_PIXELS = 64 * 28 * 28

DEFAULT_FPS = 2.0
DEFAULT_MAX_FRAMES = 128
DEFAULT_MIN_FRAMES = 4

# Top-level package that must resolve to GroundVTS (not TEF) during import.
_CONFLICTING_TOPLEVEL = ('models',)


@contextlib.contextmanager
def _gvts_scope():
    """Temporarily make GroundVTS' top-level `models` importable instead of TEF's."""
    saved = {}
    for k in list(sys.modules.keys()):
        for prefix in _CONFLICTING_TOPLEVEL:
            if k == prefix or k.startswith(prefix + '.'):
                saved[k] = sys.modules.pop(k)
                break
    saved_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if p != TEF_ROOT]
    sys.path.insert(0, GVTS_ROOT)
    cwd = os.getcwd()
    os.chdir(GVTS_ROOT)
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
    name = 'groundvts'

    def __init__(self, fps: float = DEFAULT_FPS,
                 max_frames: int = DEFAULT_MAX_FRAMES,
                 min_frames: int = DEFAULT_MIN_FRAMES,
                 vts_ratio: Optional[float] = None):
        self.fps = fps
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.vts_ratio = vts_ratio  # None -> config default (0.5 = VTS on)
        # --- ablation knobs (env) ---
        # GVTS_LORA=0  -> load Stage1_base only (no grounding-FT LoRA)
        self.use_lora = os.environ.get('GVTS_LORA', '1') != '0'
        # GVTS_MAXPIX=0 -> don't set pixel budget (Qwen default, larger)
        self._minpix = int(os.environ.get('GVTS_MINPIX', MIN_PIXELS))
        self._maxpix = int(os.environ.get('GVTS_MAXPIX', MAX_PIXELS))
        # GVTS_NATIVE=1 -> use GroundVTS' own infer_auto prompts (QA: nextqa
        # 'multimodal' style; grounding: nextgqa 'grounding' style). Invariants
        # kept fixed: question-conditioned grounding, NO answer leakage.
        self.native = os.environ.get('GVTS_NATIVE', '0') == '1'
        self.model = None
        self.processor = None
        self.device = None
        self._process_vision_info = None
        self._eff_fps = None  # n_frames/duration -> correct second_per_grid_ts.
        #   GroundVTS is a Qwen2.5-VL-7B backbone and calls process_vision_info
        #   WITHOUT return_video_kwargs and the processor WITHOUT an fps kwarg, so
        #   for pre-decoded frames second_per_grid_ts defaults to 1.0 and a 150 s
        #   clip sampled to 128 frames is perceived as 64 s (grounding collapses
        #   to the front; bites when max_frames binds, duration>64 s). We pass the
        #   effective fps as a processor kwarg so each grid maps to real seconds.

    # ---- lifecycle -----------------------------------------------------
    def load(self, device: str = 'cuda:0') -> None:
        # vts_ratio override via env, so run_parallel can launch an ablation
        # without editing code (e.g. VTS_RATIO=1.0 -> VTS off).
        env_ratio = os.environ.get('VTS_RATIO')
        if env_ratio:
            self.vts_ratio = float(env_ratio)
        # Pre-import heavy libs OUTSIDE the _gvts_scope so they are cached before
        # the top-level `models` namespace points at GroundVTS. Otherwise peft's
        # lazy `from transformers import PreTrainedModel` runs while `models` is
        # GroundVTS' namespace package and transformers' inspect.getfile() on it
        # raises "is a built-in module".
        import peft  # noqa: F401
        import transformers.modeling_utils  # noqa: F401
        from transformers import PreTrainedModel  # noqa: F401
        with _gvts_scope():
            from models.vts_qwen2_5_vl.builder import build_model
            model, processor = build_model(
                base_model_path=STAGE1_BASE,
                lora_adapter_paths=(STAGE2_3_ADAPTER if self.use_lora else None),
                merge_adapter=True,
                device=device,
                dtype=torch.bfloat16,
                vts_ratio=self.vts_ratio,
            )
        from qwen_vl_utils import process_vision_info
        self._process_vision_info = process_vision_info
        self.model = model.eval()
        self.processor = processor
        self.device = device
        print(f'[groundvts] loaded on {device} (vts_ratio={self.vts_ratio})',
              flush=True)

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
        vid = {'type': 'video', 'video': pil_frames, 'fps': self.fps}
        if self._maxpix > 0:  # GVTS_MAXPIX=0 -> Qwen default budget
            vid['min_pixels'] = self._minpix
            vid['max_pixels'] = self._maxpix
        messages = [{
            'role': 'user',
            'content': [vid, {'type': 'text', 'text': prompt_text}],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = self._process_vision_info(messages)
        # Pass effective fps to the processor so second_per_grid_ts encodes real
        # seconds (process_vision_info + the message `fps` do NOT set it for
        # pre-decoded frames). VTS operates on vision tokens and is orthogonal.
        fps_kw = {'fps': [float(self._eff_fps)]} if self._eff_fps else {}
        inputs = self.processor(text=[text], images=images, videos=videos,
                                return_tensors='pt', padding=True, **fps_kw)
        inputs = inputs.to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False)
        input_len = inputs['input_ids'].shape[1]
        gen = out[:, input_len:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]

    # ---- grounding parse (GroundVTS native "X to Y seconds") -----------
    @staticmethod
    def _parse_xy(response: str) -> Optional[Tuple[float, float]]:
        pat = (r'(\d+(?:\.\d+)?)\s*(?:s|seconds|sec)?\s*(?:and|to|-|–|—)\s*'
               r'(\d+(?:\.\d+)?)\s*(?:s|seconds|sec)?')
        m = re.search(pat, response.lower())
        if m:
            return float(m.group(1)), float(m.group(2))
        nums = re.findall(r'\d+(?:\.\d+)?', response)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        return None

    # ---- inference -----------------------------------------------------
    @staticmethod
    def _native_qa_prompt(question: str, choices: List[str]) -> str:
        # GroundVTS native mcq style (eval/infer_auto.py, do_answering branch).
        p = question.strip() + '\nOptions:'
        for idx, opt in enumerate(choices):
            if opt:
                p += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
        p += '\nPlease only give the best option.'
        return p

    def predict_qa(self, frames: np.ndarray, question: str,
                   choices: List[str]) -> Tuple[int, str]:
        if self.native:
            prompt = self._native_qa_prompt(question, choices)
            text = self._generate(frames, prompt, max_new_tokens=32)
        else:
            prompt = build_qa_prompt(question, choices)
            text = self._generate(frames, prompt, max_new_tokens=16)
        return parse_answer(text, choices), text

    def predict_grounding(self, frames: np.ndarray, question: str,
                          predicted_answer: str, duration: float,
                          video_path: Optional[str] = None,
                          ) -> Tuple[Optional[Tuple[float, float]], str, Optional[str]]:
        # Question-conditioned grounding (predicted_answer intentionally unused,
        # no answer leakage) — invariant kept in BOTH prompt modes.
        if self.native:
            # GroundVTS native grounding prompt (eval/infer_auto.py do_grounding).
            # query = the question (nextgqa uses parse_query(question)).
            prompt = (f"Based on the video, please provide the start and end "
                      f"timestamps (in seconds) for the moment described by: "
                      f"'{question.strip()}'. Respond in the format "
                      f"'X to Y seconds'. If unsure, make your best guess.")
        else:
            # SHARED GROUNDING_CORE + native output directive.
            core = GROUNDING_CORE.format(question=question.strip())
            dur = f"\nThe video is {duration:.1f} seconds long." if duration else ""
            prompt = (core + dur + "\nRespond in the format 'X to Y seconds'.")
        text = self._generate(frames, prompt, max_new_tokens=64)
        xy = self._parse_xy(text)
        spans = [(xy[0], xy[1])] if xy else []
        s, e, flag = finalize_interval(spans, duration, reduce=GROUNDING_REDUCE)
        iv = (s, e) if s is not None else None
        return iv, text, flag
