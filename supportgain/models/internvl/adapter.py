"""InternVL2.5-8B (InternLM2.5) as a full TEF grounding model.

Fixed 8-frame uniform sampler (bound=[s,e]) -> VIDEO-PATH interventions (keep =
8 frames in [s,e]; remove = complement via ffmpeg; blind = black clip).
Grounding = prompt the general VLM to localize the evidence in seconds (SHARED
GROUNDING_CORE + native seconds trigger). Frame helpers copied verbatim from the
model README; the upstream InternVL2_5-8B checkpoint is unmodified.
"""
import os, re, tempfile, subprocess
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from decord import VideoReader, cpu

from supportgain.models.base import ModelAdapter
from supportgain.config.prompts import (normalize_question, normalize_option,
                             build_qa_prompt, GROUNDING_CORE)
from supportgain.core.parser import parse_answer

MODEL_PATH = os.environ.get('TEF_INTERNVL_CKPT', 'OpenGVLab/InternVL2_5-8B')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def find_closest_aspect_ratio(ar, target_ratios, w, h, image_size):
    best_diff, best, area = float('inf'), (1, 1), w * h
    for r in target_ratios:
        d = abs(ar - r[0] / r[1])
        if d < best_diff:
            best_diff, best = d, r
        elif d == best_diff and area > 0.5 * image_size * image_size * r[0] * r[1]:
            best = r
    return best


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    w, h = image.size
    ratios = sorted(set((i, j) for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if min_num <= i * j <= max_num), key=lambda x: x[0] * x[1])
    tar = find_closest_aspect_ratio(w / h, ratios, w, h, image_size)
    tw, th, blocks = image_size * tar[0], image_size * tar[1], tar[0] * tar[1]
    resized = image.resize((tw, th))
    out = [resized.crop(((i % (tw // image_size)) * image_size,
                         (i // (tw // image_size)) * image_size,
                         ((i % (tw // image_size)) + 1) * image_size,
                         ((i // (tw // image_size)) + 1) * image_size))
           for i in range(blocks)]
    if use_thumbnail and len(out) != 1:
        out.append(image.resize((image_size, image_size)))
    return out


def get_index(bound, fps, max_frame, first_idx=0, num_segments=8):
    start, end = (bound[0], bound[1]) if bound else (-1e5, 1e5)
    si = max(first_idx, round(start * fps)); ei = min(round(end * fps), max_frame)
    seg = float(ei - si) / num_segments
    return np.array([int(si + seg / 2 + np.round(seg * i)) for i in range(num_segments)])


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=8):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame, fps = len(vr) - 1, float(vr.get_avg_fps())
    tf = build_transform(input_size)
    idx = get_index(bound, fps, max_frame, 0, num_segments)
    pvs, npl = [], []
    for fi in idx:
        img = Image.fromarray(vr[int(fi)].asnumpy()).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pv = torch.stack([tf(t) for t in tiles])
        npl.append(pv.shape[0]); pvs.append(pv)
    return torch.cat(pvs), npl


class Adapter(ModelAdapter):
    name = 'internvl'
    N_SEG = 8

    def load(self, device: str = 'cuda:0') -> None:
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, use_fast=False)
        self.device = device

    def supports_video_path_intervention(self) -> bool:
        return True

    def decode_video(self, video_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        dur = len(vr) / max(float(vr.get_avg_fps()), 1e-6)
        return np.zeros((1, 2, 2, 3), np.uint8), np.zeros(1), float(dur)

    # ---- ffmpeg helpers ---------------------------------------------------
    def _crop(self, path, s, e):
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(s),
                            '-to', str(e), '-i', path,
                            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
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
                            '-i', f'color=black:s=448x448:r=8:d={d:.1f}',
                            '-c:v', 'libx264', out], capture_output=True)
        return out if r.returncode == 0 and os.path.exists(out) else None

    # ---- core inference ---------------------------------------------------
    @torch.inference_mode()
    def _infer(self, video_path, bound, prompt_body, mnt=32):
        pv, npl = load_video(video_path, bound=bound, num_segments=self.N_SEG, max_num=1)
        pv = pv.to(torch.bfloat16).cuda()
        prefix = ''.join(f'Frame{i+1}: <image>\n' for i in range(len(npl)))
        gen = dict(max_new_tokens=mnt, do_sample=False)
        return str(self.model.chat(self.tokenizer, pv, prefix + prompt_body, gen,
                                   num_patches_list=npl, history=None))

    def _qa(self, video_path, bound, question, choices):
        ch = [normalize_option(str(c)) for c in choices]
        raw = self._infer(video_path, bound, build_qa_prompt(normalize_question(str(question)), ch))
        return parse_answer(raw, ch), raw

    def predict_keep(self, video_path, s, e, question, choices):   # R_oracle runner
        return self._qa(video_path, [float(s), float(e)], question, choices)

    def predict_qa(self, frames, question, choices):
        raise NotImplementedError('internvl uses predict_qa_video')

    def predict_qa_video(self, video_path, spec, question, choices):
        _, _, dur = self.decode_video(video_path)
        if spec is None or spec.get('kind') in (None, 'full'):
            return self._qa(video_path, None, question, choices)
        kind = spec['kind']; iv = spec.get('interval')
        if kind == 'blind':
            bp = self._black(dur)
            if bp is None:
                return -1, ''
            try:
                return self._qa(bp, None, question, choices)
            finally:
                os.path.exists(bp) and os.unlink(bp)
        if iv is None:
            return -1, ''
        s, e = float(iv[0]), float(iv[1])
        if 'keep' in kind:
            return self._qa(video_path, [s, e], question, choices)
        if 'remove' in kind:
            cp = self._complement(video_path, s, e, dur)
            if cp is None:
                return -1, ''
            try:
                return self._qa(cp, None, question, choices)
            finally:
                os.path.exists(cp) and os.unlink(cp)
        return -1, ''

    def predict_grounding(self, frames, question, predicted_answer, duration,
                          video_path: Optional[str] = None):
        core = GROUNDING_CORE.format(question=normalize_question(str(question)))
        body = (f"{core}\nThe video is {float(duration):.0f} seconds long. "
                "Answer ONLY with the start and end time in seconds of that "
                "segment, formatted as 'START - END'.")
        raw = self._infer(video_path, None, body, mnt=32)
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to|~|and|,)\s*(\d+(?:\.\d+)?)', raw)
        if not m:
            return None, raw, 'parse_fail'
        s, e = float(m.group(1)), float(m.group(2))
        if e < s:
            s, e = e, s
        s = max(0.0, min(duration, s)); e = max(0.0, min(duration, e))
        return (s, e), raw, None
