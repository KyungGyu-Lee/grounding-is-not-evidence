# -*- coding: utf-8 -*-
"""gen_aux_numbers.py -- reproduce paper scalars that build_paper_numbers does
NOT emit (IoU-binned answer-reproduction failure, IoU@0.5, VTimeLLM option-A).
Canonical trees: NExT <- results_v3 (tsfix); QVH <- results_v5. Reads only.

  python analysis/gen_aux_numbers.py

Definitions:
  * answer-reproduction failure at IoU>=t = fraction of grounder predictions
    with IoU>=t where predicted_keep != a_full (Ret(pred)==0). C2 basis, pooled.
  * IoU@0.5 = per-model fraction of C2 predictions with IoU>=0.5.
  * VTimeLLM option-A = fraction of a_full == index 0, over ALL questions.
"""
import json, glob, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

GNX = [('Qwen', 'qwen25vl'), ('VideoMind', 'videomind'), ('GroundVTS', 'groundvts'),
       ('GVLLM', 'grounded_videollm'), ('VTimeLLM', 'vtimellm')]
GQ = GNX + [('TimeChat', 'timechat'), ('VideoChat-R1', 'videochat_r1')]


def _ls(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        d.update(json.load(open(f)))
    return d


BDIR = {'NExT': 'nextgqa', 'QVH': 'qvhme'}


def paths(bench, dr):
    d = f'outputs/{BDIR[bench]}/{dr}'
    return f'{d}/step1.json', f'{d}/step3.json'


def rnorm(bench):
    return json.load(open(f'flags/{BDIR[bench]}/r_norm.json'))['r_norm']


def iou_fail(bench, models, C2, thr):
    """pooled: among predictions with IoU>=thr on C2, fraction Ret(pred)==0."""
    num = den = 0
    for _, dr in models:
        s1p, s3p = paths(bench, dr)
        s1, s3 = json.load(open(s1p)), json.load(open(s3p))
        for v, rec in s3.items():
            if v not in C2 or 'error' in rec:
                continue
            p = rec.get('preds', {}); af = rec.get('a_full')
            if af is None or af < 0 or 'predicted_keep' not in p:
                continue
            iou = s1.get(v, {}).get('iou')
            if iou is None or iou < thr:
                continue
            den += 1; num += (1 - int(p['predicted_keep'] == af))
    return num, den


def iou_at(bench, dr, C2, thr=0.5):
    s1 = json.load(open(paths(bench, dr)[0]))
    ious = [s1[v]['iou'] for v in C2 if v in s1 and s1[v].get('iou') is not None]
    return np.mean([i >= thr for i in ious]), len(ious)


def optionA(bench, dr):
    s1 = json.load(open(paths(bench, dr)[0]))
    af = [v.get('a_full') for v in s1.values() if v.get('a_full') is not None and v.get('a_full') >= 0]
    return np.mean([a == 0 for a in af]), len(af)


def main():
    for bench, models in [('NExT', GNX), ('QVH', GQ)]:
        C2 = set(rnorm(bench))
        print(f'== {bench} (C2, grounders pooled) ==')
        for thr in [0.5, 0.7, 0.9]:
            n, d = iou_fail(bench, models, C2, thr)
            print(f'  IoU>={thr}: Ret(pred)==0 = {n/d:.4f}  (N={d})')
    print('== IoU@0.5 (per-model, C2) ==')
    for bench, models in [('NExT', GNX), ('QVH', GQ)]:
        C2 = set(rnorm(bench))
        for disp, dr in models:
            r, n = iou_at(bench, dr, C2)
            print(f'  {bench:5s} {disp:13s}: {r:.4f} (N={n})')
    print('== VTimeLLM option-A (a_full==0, ALL) ==')
    for bench in ['NExT', 'QVH']:
        r, n = optionA(bench, 'vtimellm')
        print(f'  {bench:5s}: {r:.4f} (N={n})')


if __name__ == '__main__':
    main()
