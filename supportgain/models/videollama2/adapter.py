"""VideoLLaMA2-7B (Mistral) as a full TEF grounding model.

Fixed 8-frame uniform sampler -> VIDEO-PATH interventions (keep = 8 frames in
[s,e]; remove = 8 frames in the complement via ffmpeg; blind = black clip).
Grounding = prompt the general VLM to localize the evidence window in seconds
(SHARED GROUNDING_CORE + native seconds trigger); it is not a native grounder,
so this measures a general VLM's localization ability. Calls only the upstream
videollama2 API (process_video / mm_infer, greedy). Repo unmodified.
"""
import os, re, tempfile, subprocess
from typing import List, Optional, Tuple

import numpy as np
from supportgain.models.base import ModelAdapter
from supportgain.config.prompts import (normalize_question, normalize_option,
                             build_qa_prompt, GROUNDING_CORE)
from supportgain.core.parser import parse_answer

MODEL_PATH = os.environ.get('TEF_VIDEOLLAMA2_CKPT',
                            'DAMO-NLP-SG/VideoLLaMA2-7B')


class Adapter(ModelAdapter):
    name = 'videollama2'

    def load(self, device: str = 'cuda:0') -> None:
        from videollama2 import model_init
        self.model, self.processor, self.tokenizer = model_init(MODEL_PATH)
        self.model = self.model.eval()
        self.device = device

    def supports_video_path_intervention(self) -> bool:
        return True

    # ---- decode (duration only; model re-samples internally) --------------
    def decode_video(self, video_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
        import decord
        vr = decord.VideoReader(video_path, num_threads=1)
        dur = len(vr) / max(float(vr.get_avg_fps()), 1e-6)
        return np.zeros((1, 2, 2, 3), np.uint8), np.zeros(1), float(dur)

    # ---- ffmpeg helpers ---------------------------------------------------
    def _even(self):
        return ['-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2']

    def _crop(self, path, s, e):
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(s),
                            '-to', str(e), '-i', path, *self._even(),
                            '-c:v', 'libx264', '-an', tmp], capture_output=True)
        return tmp if r.returncode == 0 and os.path.exists(tmp) else None

    def _complement(self, path, s, e, dur):
        parts = []
        if s > 0.05:
            parts.append((0.0, s))
        if e < dur - 0.05:
            parts.append((e, dur))
        if not parts:
            return None
        segs = [self._crop(path, a, b) for a, b in parts]
        if any(x is None for x in segs):
            for x in segs:
                x and os.unlink(x)
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
        return out if r.returncode == 0 and os.path.exists(out) else None

    def _black(self, dur):
        d = max(1.0, min(float(dur), 8.0))
        out = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                            '-i', f'color=black:s=224x224:r=8:d={d:.1f}',
                            '-c:v', 'libx264', out], capture_output=True)
        return out if r.returncode == 0 and os.path.exists(out) else None

    # ---- core inference ---------------------------------------------------
    def _infer(self, video_tensor, prompt, mnt=32):
        from videollama2 import mm_infer
        return mm_infer(video_tensor, prompt, model=self.model,
                        tokenizer=self.tokenizer, modal='video',
                        do_sample=False, max_new_tokens=mnt)

    def _qa(self, video_path, s, e, question, choices):
        ch = [normalize_option(str(c)) for c in choices]
        vid = self.processor['video'](video_path, s=float(s), e=float(e))
        raw = self._infer(vid, build_qa_prompt(normalize_question(str(question)), ch))
        return parse_answer(str(raw), ch), str(raw)

    # backward-compat helper used by the R_oracle-keep runner
    def predict_keep(self, video_path, s, e, question, choices):
        return self._qa(video_path, s, e, question, choices)

    def predict_qa(self, frames, question, choices):
        raise NotImplementedError('videollama2 uses predict_qa_video')

    def predict_qa_video(self, video_path, spec, question, choices):
        if spec is None or spec.get('kind') in (None, 'full'):
            _, _, dur = self.decode_video(video_path)
            return self._qa(video_path, 0.0, dur, question, choices)
        kind = spec['kind']; iv = spec.get('interval')
        _, _, dur = self.decode_video(video_path)
        if kind == 'blind':
            bp = self._black(dur)
            if bp is None:
                return -1, ''
            try:
                return self._qa(bp, 0.0, min(dur, 8.0), question, choices)
            finally:
                os.path.exists(bp) and os.unlink(bp)
        if iv is None:
            return -1, ''
        s, e = float(iv[0]), float(iv[1])
        if 'keep' in kind:
            return self._qa(video_path, s, e, question, choices)
        if 'remove' in kind:
            cp = self._complement(video_path, s, e, dur)
            if cp is None:
                return -1, ''
            try:
                _, _, cd = self.decode_video(cp)
                return self._qa(cp, 0.0, cd, question, choices)
            finally:
                os.path.exists(cp) and os.unlink(cp)
        return -1, ''

    # ---- grounding (question-conditioned; predicted_answer unused) --------
    def predict_grounding(self, frames, question, predicted_answer, duration,
                          video_path: Optional[str] = None):
        core = GROUNDING_CORE.format(question=normalize_question(str(question)))
        prompt = (f"{core}\nThe video is {float(duration):.0f} seconds long. "
                  "Answer ONLY with the start and end time in seconds of that "
                  "segment, formatted as 'START - END'.")
        vid = self.processor['video'](video_path, s=0.0, e=float(duration))
        raw = self._infer(vid, prompt, mnt=32)
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to|~|and|,)\s*(\d+(?:\.\d+)?)', str(raw))
        if not m:
            return None, str(raw), 'parse_fail'
        s, e = float(m.group(1)), float(m.group(2))
        if e < s:
            s, e = e, s
        s = max(0.0, min(duration, s)); e = max(0.0, min(duration, e))
        return (s, e), str(raw), None
