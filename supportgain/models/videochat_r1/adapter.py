"""VideoChat-R1-7B adapter (QVH-GQA only).

VideoChat-R1_7B is a GRPO-RFT fine-tune of Qwen2.5-VL-7B-Instruct, so this
adapter forks the qwen25vl adapter: SAME backbone, SAME processor, SAME
frame-subset intervention path (NO predict_qa_video — control parity with the
Qwen2.5-VL row; step3 keep/remove/blind go through apply_intervention on the
decoded frames exactly like qwen25vl).

Two protocols, split by `protocol` ('A' | 'C'):
  A (independent)  : QA one call -> a_full ; grounding a separate call
                     (question-conditioned, predicted_answer NOT used).
                     Identical flow to the other 6 models.
  C (joint)        : ONE call emits both the answer <answer> and the evidence
                     <glue>. a_full comes from THAT call; predict_grounding
                     returns the glue cached from the SAME forward.

SHARED (byte-identical to every model): the QA task text (build_qa_prompt),
the grounding task text (GROUNDING_CORE), letter parsing (utils.parser.
parse_answer), and finalize_interval. NATIVE (this model only): the output
DIRECTIVE appended after the shared task, tag stripping, and the glue/interval
decoder. RFT tags <think>/<timestep> are ALWAYS stripped and NEVER used for the
interval (only <answer> -> letter, <glue> -> interval).

Cache safety (C): key = (question, frame fingerprint) as a stand-in for
(video_id, question_id) — the adapter contract does not pass the vqid. A cache
MISS or key MISMATCH in predict_grounding RAISES (never a silent None), and the
returned flag is 'glue_from_cache' (stored by step1 as predicted_flag) so a
post-hoc check can confirm every C grounding came from the joint forward.

DECISION (step3 prompt for protocol C) — the joint answer+glue prompt is used
ONLY for the step1 a_full+E_hat forward. Every step3 intervention (keep / blind /
random-keep) uses the PLAIN QA prompt, byte-identical to protocol A. Reason is
CONTROL, not cost: A and C must differ in exactly one thing — how E_hat is
produced (independent grounding call vs joint glue) — so that any SG_A vs SG_C
gap has a single cause. If step3 also ran the joint prompt, a second difference
(intervention-stage prompt) would confound the comparison. It also removes the
joint-prompt artifact seen in the smoke: on short intervened clips the model
emitted '<answer></answer>' (empty) + only <glue>, a ~5% step3 letter "failure"
that is a prompt artifact, not a parse failure. The switch is triggered by
step3_intervention.run() calling begin_step3() at entry (hasattr-guarded)."""
import os
import re
import hashlib
from typing import Tuple, Optional, List

import numpy as np
import torch
from PIL import Image

from supportgain.models.base import ModelAdapter
from supportgain.config.prompts import build_qa_prompt, GROUNDING_CORE
from supportgain.core.video_io import decode_at_fps
from supportgain.core.intervals import parse_grounding, finalize_interval
from supportgain.core.parser import parse_answer

# local snapshot (Qwen2.5-VL-7B architecture, apache-2.0)
MODEL_ID = os.environ.get('VCR1_MODEL', 'OpenGVLab/VideoChat-R1_7B')
DEFAULT_FPS = 2.0            # == qwen25vl (control parity)
DEFAULT_MAX_FRAMES = 128     # == qwen25vl (control parity; author eval allows <=768, NOT used)
DEFAULT_MIN_FRAMES = 4

# NATIVE output directives (appended after SHARED task text; not verbatim author).
GND_DIRECTIVE = ("\nThe video is {dur:.1f} seconds long."
                 "\nProvide the start and end time in seconds within "
                 "<answer></answer> tags in the format 'start to end'.")
JOINT_DIRECTIVE = ("\nProvide the answer letter within <answer></answer> tags. "
                   "At the same time, within <glue></glue> tags, give the time "
                   "period(s) in seconds that your answer is based on, in the "
                   "format [(s1, e1), (s2, e2), ...].")

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_TAG_RE = {t: re.compile(rf'<{t}>(.*?)</{t}>', re.DOTALL | re.IGNORECASE)
           for t in ('answer', 'glue')}
_SPAN_RE = re.compile(r'\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?')


def _strip_think(t: str) -> str:
    return _THINK_RE.sub(' ', t or '')


def _tag(t: str, name: str) -> Optional[str]:
    m = _TAG_RE[name].search(t or '')
    return m.group(1).strip() if m else None


def _letter_src(raw: str) -> str:
    """Native preprocessing shared by full/keep/rand/blind: drop <think>, take
    <answer> inner if present, else the think-stripped text. Letter is then
    extracted by the COMMON parser (no model-specific letter parser)."""
    t = _strip_think(raw)
    ans = _tag(t, 'answer')
    return ans if ans is not None else t


def _parse_glue_spans(glue_text: str) -> List[Tuple[float, float]]:
    """Native decode of <glue>[(s1,e1),...]</glue> into a span list (raw, not
    finalized — finalize needs duration, applied in predict_grounding)."""
    if not glue_text:
        return []
    return [(float(a), float(b)) for a, b in _SPAN_RE.findall(glue_text)]


class Adapter(ModelAdapter):
    name = 'videochat_r1'
    protocol = 'A'            # subclass sets 'C'

    def __init__(self, fps: float = DEFAULT_FPS,
                 max_frames: int = DEFAULT_MAX_FRAMES,
                 min_frames: int = DEFAULT_MIN_FRAMES):
        self.fps = fps
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.model = None
        self.processor = None
        self.device = None
        self._glue_cache = None   # C only: {'key','spans','raw','n_spans','glue_absent'}
        self._step3_mode = False  # C only: flipped by begin_step3() -> plain QA
        self._eff_fps = None      # frames-per-second the DECODED frames actually
        #   span = n_frames / duration. Qwen2.5-VL encodes ABSOLUTE time via
        #   second_per_grid_ts = temporal_patch_size / fps; if we let it default
        #   (fps ignored for pre-decoded frames -> second_per_grid_ts=1.0), a
        #   150 s clip sampled to 128 frames is perceived as 128/2 = 64 s and
        #   grounding collapses to the front. We pass this effective fps to the
        #   processor so each temporal grid maps to real seconds. It is the same
        #   value for cropped step3 clips because uniform decode preserves the
        #   inter-frame spacing.

    # ---- stage hook ----------------------------------------------------
    def begin_step3(self) -> None:
        """Called once by step3_intervention.run() at entry. In protocol C it
        switches predict_qa to the PLAIN QA prompt for all intervention forwards
        (see module docstring: control decision). No-op semantics for protocol A
        (A is already plain)."""
        self._step3_mode = True

    # ---- lifecycle -----------------------------------------------------
    def load(self, device: str = 'cuda:0') -> None:
        os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2', device_map={'': device},
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.device = device

    # ---- video ---------------------------------------------------------
    def decode_video(self, video_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
        frames, ts, dur = decode_at_fps(video_path, fps=self.fps,
                                        max_frames=self.max_frames, min_frames=self.min_frames)
        # effective fps of the returned frames (constant per video, preserved
        # under step3 cropping) -> correct second_per_grid_ts. See __init__.
        self._eff_fps = (len(frames) / dur) if (dur and len(frames)) else None
        return frames, ts, dur

    # ---- internal generate --------------------------------------------
    @torch.inference_mode()
    def _generate(self, frames: np.ndarray, prompt_text: str,
                  max_new_tokens: int = 256) -> str:
        if frames is None or len(frames) == 0:
            return ''
        pil_frames = [Image.fromarray(f) for f in frames]
        messages = [{'role': 'user', 'content': [
            {'type': 'video', 'video': pil_frames},
            {'type': 'text', 'text': prompt_text}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # Pass the effective fps to the processor (NOT the message dict, which is
        # ignored for pre-decoded frames) so second_per_grid_ts encodes real
        # seconds. Fallback: default (== old behaviour) only if unknown.
        fps_kw = {'fps': [float(self._eff_fps)]} if self._eff_fps else {}
        inputs = self.processor(text=[text], images=None, videos=[pil_frames],
                                return_tensors='pt', padding=True, **fps_kw)
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                   do_sample=False)
        gen = out[:, inputs['input_ids'].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]

    # ---- cache fingerprint (C) ----------------------------------------
    @staticmethod
    def _fp(frames: np.ndarray, question: str) -> str:
        if frames is None or len(frames) == 0:
            body = b''
        else:
            step = max(1, len(frames) // 4)
            body = np.ascontiguousarray(frames[::step]).tobytes()
        return hashlib.md5(body + str(question).encode('utf-8')).hexdigest()

    # ---- QA ------------------------------------------------------------
    def predict_qa(self, frames: np.ndarray, question: str,
                   choices: List[str]) -> Tuple[int, str]:
        # Protocol C step1: ONE joint forward emits <answer> (a_full) + <glue>
        # (E_hat, cached for predict_grounding). Only here — NOT in step3.
        if self.protocol == 'C' and not self._step3_mode:
            prompt = build_qa_prompt(question, choices) + JOINT_DIRECTIVE
            raw = self._generate(frames, prompt, max_new_tokens=384)
            t = _strip_think(raw)
            glue = _tag(t, 'glue')
            spans = _parse_glue_spans(glue)
            self._glue_cache = {
                'key': self._fp(frames, question), 'spans': spans, 'raw': raw,
                'n_spans': len(spans), 'glue_absent': glue is None}
            return parse_answer(_letter_src(raw), choices), raw
        # Protocol A (all calls) AND protocol C step3 interventions: identical
        # PLAIN QA prompt (control — see module docstring / begin_step3).
        prompt = build_qa_prompt(question, choices)
        raw = self._generate(frames, prompt, max_new_tokens=256)
        return parse_answer(_letter_src(raw), choices), raw

    # ---- grounding -----------------------------------------------------
    def predict_grounding(self, frames: np.ndarray, question: str,
                          predicted_answer: str, duration: float,
                          video_path: Optional[str] = None,
                          ) -> Tuple[Optional[Tuple[float, float]], str, Optional[str]]:
        if self.protocol == 'C':
            key = self._fp(frames, question)
            if self._glue_cache is None:
                raise RuntimeError('videochat_r1[C]: glue cache MISS — '
                                   'predict_qa must be called before predict_grounding')
            if self._glue_cache['key'] != key:
                raise RuntimeError('videochat_r1[C]: glue cache KEY MISMATCH — '
                                   'predict_qa/predict_grounding misalignment')
            s, e, fflag = finalize_interval(self._glue_cache['spans'], duration, 'longest')
            iv = (s, e) if s is not None else None
            flag = 'glue_from_cache' + (('+' + fflag) if fflag else '')
            return iv, self._glue_cache['raw'], flag
        # protocol A — question-conditioned, predicted_answer NOT used
        prompt = GROUNDING_CORE.format(question=question.strip()) + \
            GND_DIRECTIVE.format(dur=float(duration))
        raw = self._generate(frames, prompt, max_new_tokens=256)
        t = _strip_think(raw)
        gsrc = _tag(t, 'answer')
        gsrc = gsrc if gsrc is not None else t
        s, e, flag = parse_grounding(gsrc, duration)
        return (s, e) if s is not None else None, raw, flag
