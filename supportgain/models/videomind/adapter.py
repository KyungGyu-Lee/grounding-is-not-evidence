"""TEF adapter for VideoMind-2B (Qwen2-VL + Chain-of-LoRA roles).

Non-circular TEF setup (a_full is a FULL-video answer, NOT VideoMind's native
grounded answer):
  * predict_grounding : VideoMind *grounder* role (register-token localization)
                        -> E_hat. Question-conditioned (query = the question).
  * predict_qa_video  : VideoMind *answerer* (= base Qwen2-VL, no answerer
                        adapter in the 2B ckpt) on the intervened video.
                        VideoMind ingests a video file with a native time-crop
                        (video_start/video_end), so interventions are at the
                        video-file level (like the GVLLM adapter):
                          keep   -> answer on [start, end] (native crop)
                          remove -> ffmpeg complement, then answer on it
                          full   -> answer on [0, duration]

VideoMind ships its own top-level `videomind` package (no clash with TEF's
`models`), so a plain sys.path insert is enough.
"""
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
from supportgain.config.settings import GROUNDING_REDUCE
from supportgain.core.parser import parse_answer
from supportgain.core.intervals import finalize_interval

from supportgain.paths import VIDEOMIND_ROOT as VM_ROOT
MODEL_PATH = os.environ.get('TEF_VIDEOMIND_CKPT',
                            os.path.join(VM_ROOT, 'model_zoo', 'VideoMind-2B'))

# nextgqa answerer settings (videomind/dataset/sub_classes/nextgqa.py defaults)
FPS = 2.0
MAX_FRAMES = 32
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
# grounder gets a denser view of the whole video
GND_MAX_FRAMES = 100
GND_MIN_PIXELS = 36 * 28 * 28
GND_MAX_PIXELS = 64 * 28 * 28


class Adapter(ModelAdapter):
    name = 'videomind'

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = 'cuda:0'
        self._pvi = None
        self._GND_PROMPT = None

    def supports_video_path_intervention(self) -> bool:
        return True

    # ---- load -------------------------------------------------------------
    def load(self, device: str = 'cuda:0'):
        self.device = device
        if VM_ROOT not in sys.path:
            sys.path.insert(0, VM_ROOT)
        from videomind.model.builder import build_model
        from videomind.constants import GROUNDER_PROMPT
        from videomind.dataset.utils import process_vision_info

        # config.base_model_path is relative ('model_zoo/...'); resolve from VM_ROOT
        _cwd = os.getcwd()
        os.chdir(VM_ROOT)
        try:
            model, processor = build_model(MODEL_PATH, device=device)
        finally:
            os.chdir(_cwd)
        self.model = model.to(device).eval()
        self.processor = processor
        self._pvi = process_vision_info
        self._GND_PROMPT = GROUNDER_PROMPT
        print(f'[videomind] loaded on {device}', flush=True)

    # ---- video ------------------------------------------------------------
    def decode_video(self, video_path: str):
        """Only duration is needed (video-path intervention). frames are a
        placeholder (this adapter re-reads the video file internally)."""
        import decord
        vr = decord.VideoReader(video_path, num_threads=1)
        dur = len(vr) / float(vr.get_avg_fps())
        placeholder = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        return placeholder, np.zeros(1), float(dur)

    # ---- generation helpers ----------------------------------------------
    def _build_video(self, video_path, s, e, max_frames, min_pixels, max_pixels):
        vid = {
            'type': 'video', 'video': video_path, 'num_threads': 1,
            'min_pixels': min_pixels, 'max_pixels': max_pixels,
            'max_frames': max_frames, 'fps': FPS,
        }
        if s is not None:
            vid['video_start'] = float(s)
            vid['video_end'] = float(e)
        return vid

    @torch.inference_mode()
    def _answer(self, video_path, s, e, prompt_text):
        messages = [{
            'role': 'user',
            'content': [
                self._build_video(video_path, s, e, MAX_FRAMES, MIN_PIXELS, MAX_PIXELS),
                {'type': 'text', 'text': prompt_text},
            ],
        }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        text += 'Best Option: ('
        images, videos = self._pvi(messages)
        data = self.processor(text=[text], images=images, videos=videos,
                              return_tensors='pt').to(self.device)
        with self.model.disable_adapter():  # answerer = base model
            out = self.model.generate(
                **data, do_sample=False, temperature=None, top_p=None, top_k=None,
                repetition_penalty=None, max_new_tokens=32)
        gen = out[0, data.input_ids.size(1):]
        return self.processor.decode(gen, skip_special_tokens=True)

    def _ffmpeg_complement(self, path, iv, duration):
        s, e, dur = float(iv[0]), float(iv[1]), float(duration)
        segs = []
        if s > 0.1:
            segs.append((0.0, s))
        if e < dur - 0.1:
            segs.append((e, dur))
        if not segs:
            return None, None
        tmpdir = tempfile.mkdtemp(prefix='vm_rm_')
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
                return None, None
            files.append(out)
        if len(files) == 1:
            return files[0], sum(b - a for a, b in segs)
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
            return None, None
        return out, sum(b - a for a, b in segs)

    # ---- QA prompt (VideoMind native mcq) --------------------------------
    @staticmethod
    def _qa_prompt(question, choices):
        p = question.strip() + '\nOptions:'
        for idx, opt in enumerate(choices):
            p += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
        p += '\nPlease only give the best option.'
        return p

    # ---- TEF contract -----------------------------------------------------
    def predict_qa(self, frames, question, choices):
        raise NotImplementedError('videomind uses predict_qa_video')

    def predict_qa_video(self, video_path, intervention_spec, question, choices):
        prompt = self._qa_prompt(question, choices)
        s, e = 0.0, None  # None end -> whole video handled by decode inside
        tmp = None
        # need duration for full/remove
        _, _, dur = self.decode_video(video_path)
        s, e = 0.0, dur
        if intervention_spec is not None and intervention_spec.get('kind') not in (None, 'full'):
            kind = intervention_spec['kind']
            iv = intervention_spec.get('interval')
            if kind == 'keep':
                if iv is None:
                    return -1, ''
                s, e = float(iv[0]), float(iv[1])
            elif kind == 'remove':
                if iv is None:
                    return -1, ''
                tmp, tmp_dur = self._ffmpeg_complement(video_path, iv, dur)
                if tmp is None:
                    return -1, ''
                video_path, s, e = tmp, 0.0, tmp_dur
            elif kind == 'blind':
                # no clean video-path blind; treat as a 0.1s sliver (near-blind)
                s, e = 0.0, min(0.2, dur)
        try:
            text = self._answer(video_path, s, e, prompt)
        finally:
            if tmp:
                shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)
        return parse_answer(text, choices), text

    # ---- grounding (VideoMind grounder role, register tokens) ------------
    @torch.inference_mode()
    def predict_grounding(self, frames, question, predicted_answer, duration,
                          video_path: Optional[str] = None):
        # question-conditioned: query = the question (predicted_answer unused)
        prompt = self._GND_PROMPT.format(question.strip())
        messages = [{
            'role': 'user',
            'content': [
                self._build_video(video_path, None, None, GND_MAX_FRAMES,
                                  GND_MIN_PIXELS, GND_MAX_PIXELS),
                {'type': 'text', 'text': prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images, videos = self._pvi(messages)
        data = self.processor(text=[text], images=images, videos=videos,
                              return_tensors='pt').to(self.device)
        self.model.base_model.disable_adapter_layers()
        self.model.base_model.enable_adapter_layers()
        self.model.set_adapter('grounder')
        out = self.model.generate(
            **data, do_sample=False, temperature=None, top_p=None, top_k=None,
            repetition_penalty=None, max_new_tokens=256)
        gen = out[0, data.input_ids.size(1):]
        raw = self.processor.decode(gen, clean_up_tokenization_spaces=False)

        spans = []
        if len(self.model.reg) > 0:
            blob = self.model.reg[0].cpu().float()
            pred = (blob[:, :2] * float(duration)).clamp(min=0, max=float(duration))
            top = pred[0].tolist()  # highest-confidence span
            s, e = min(top), max(top)
            spans = [(s, e)]
        s, e, flag = finalize_interval(spans, duration, reduce=GROUNDING_REDUCE)
        iv = (s, e) if s is not None else None
        return iv, raw, flag
