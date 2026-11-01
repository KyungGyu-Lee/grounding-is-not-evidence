"""B1-1 — re-query Qwen2.5-VL on the QVH-ME C3 set with two non-cropping
variants of the Keep intervention.

Nothing about the protocol changes except how the window is applied to the
decoded frame stack. The prompt, the greedy decode and the A-E parsing are the
pipeline's own: we call `adapter.predict_qa`, which is exactly what
`pipeline/step3_intervention.py` calls, and the transform is the pipeline's own
`utils.interventions.apply_intervention`:

    CROP   (paper, stored)  phase2 'keep'  on the stored window
                            -> frames inside the window only; count shrinks
    MASK   (this script)    phase1 'keep'  on the stored window
                            -> full frame grid, frames outside the window
                               replaced by RGB(0,0,0); count/positions/fps
                               identical to the full-video query
    MARGIN (this script)    phase2 'keep'  on the window grown by +/-5 s
                            -> crop, but with temporal context on both sides

Windows are READ from `inputs.jsonl` (built by b1_build_inputs.py from the
stored step3 record); none are resampled. Blind is not re-run: it sees no
frames, so Ret(none) is shared by all three conditions.

Usage
-----
    # preview only, no model, no GPU
    python analysis/camera_ready/b1_requery.py --preview 3

    # smoke test: 10 questions, 1 GPU, both variants
    CUDA_VISIBLE_DEVICES=0 python analysis/camera_ready/b1_requery.py \
        --shard 0 --nshard 1 --limit 10 --tag smoke

    # full run: one process per GPU
    for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python \
        analysis/camera_ready/b1_requery.py --shard $g --nshard 4 & done

Resumable: a (vqid, variant, window_type) already on disk is skipped.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from supportgain.core.interventions import apply_intervention  # noqa: E402

OUT_DIR = f'{ROOT}/analysis/B1_masking'
INPUTS = f'{OUT_DIR}/inputs.jsonl'
VIDEO_DIR = os.environ.get('VIDEO_DIR', f'{ROOT}/data/qvh_gqa/videos')

# variant -> intervention phase. 'keep' kind for both; the only difference is
# whether the frames outside the window are blacked out (phase1) or dropped
# (phase2), and MARGIN's window is the pre-widened one from inputs.jsonl.
PHASE_OF = {'mask': 'phase1', 'margin': 'phase2'}


def video_path(video_id):
    return os.path.join(VIDEO_DIR, f'{video_id}.mp4')


def load_inputs(variants, limit=None):
    """Group inputs.jsonl by question, preserving the stored window order."""
    by_q = {}
    with open(INPUTS) as f:
        for line in f:
            r = json.loads(line)
            if r['variant'] not in variants:
                continue
            by_q.setdefault(r['vqid'], []).append(r)
    order = sorted(by_q)
    if limit:
        order = order[:limit]
    return order, by_q


def done_keys(path):
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            keys.add((r['vqid'], r['variant'], r['window_type']))
    return keys


# --------------------------------------------------------------- preview mode
def contact_sheet(frames, cols=16, cell=96):
    """Grid montage so the blacked-out region is visible at a glance."""
    from PIL import Image
    n = len(frames)
    rows = (n + cols - 1) // cols
    h = int(cell * frames.shape[1] / frames.shape[2])
    sheet = Image.new('RGB', (cols * cell, rows * h), (40, 40, 40))
    for i in range(n):
        im = Image.fromarray(frames[i]).resize((cell, h))
        sheet.paste(im, ((i % cols) * cell, (i // cols) * h))
    return sheet


def run_preview(n_questions, variants):
    """Dump full / MASK / MARGIN frame grids for a few questions. No model."""
    from supportgain.models.qwen25vl.adapter import Adapter
    order, by_q = load_inputs(variants)
    order = order[:n_questions]
    pdir = f'{OUT_DIR}/preview'
    os.makedirs(pdir, exist_ok=True)
    ad = Adapter()                       # decode only; weights never loaded
    meta = []
    for vqid in order:
        rows = by_q[vqid]
        vp = video_path(rows[0]['video_id'])
        frames, ts, dur = ad.decode_video(vp)
        d = f'{pdir}/{vqid}'
        os.makedirs(d, exist_ok=True)
        contact_sheet(frames).save(f'{d}/full.png')
        entry = {'vqid': vqid, 'video_id': rows[0]['video_id'],
                 'duration': float(dur), 'n_frames_full': int(len(frames)),
                 'eff_fps_full': float(len(frames) / dur), 'windows': []}
        for r in rows:
            if r['window_type'] != 'pred':
                continue
            iv = tuple(r['interval_applied'])
            mod = apply_intervention(PHASE_OF[r['variant']], frames, ts, iv, 'keep')
            contact_sheet(mod).save(f'{d}/{r["variant"]}_pred.png')
            in_ev = (ts >= iv[0]) & (ts < iv[1])
            nz = [int(i) for i in np.where(mod.reshape(len(mod), -1).max(1) > 0)[0]]
            entry['windows'].append({
                'variant': r['variant'],
                'interval_src': r['interval_src'],
                'interval_applied': r['interval_applied'],
                'margin_added_s': [round(r['interval_src'][0] - r['interval_applied'][0], 3),
                                   round(r['interval_applied'][1] - r['interval_src'][1], 3)],
                'n_frames_out': int(len(mod)),
                'n_frames_in_window': int(in_ev.sum()),
                'n_frames_nonblack': len(nz),
                'png': f'{r["variant"]}_pred.png',
            })
        meta.append(entry)
        print(f'{vqid}: {len(frames)} frames, {dur:.1f}s -> {d}', flush=True)
    json.dump(meta, open(f'{pdir}/preview_meta.json', 'w'), indent=2)
    print(json.dumps(meta, indent=2))


# ------------------------------------------------------------------ main run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default='mask,margin')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0,
                    help='only the first N questions (smoke test)')
    ap.add_argument('--tag', default='',
                    help='output filename suffix, e.g. "smoke"')
    ap.add_argument('--preview', type=int, default=0,
                    help='dump frame grids for N questions and exit (no model)')
    args = ap.parse_args()
    variants = tuple(args.variants.split(','))
    assert set(variants) <= set(PHASE_OF), variants

    if args.preview:
        run_preview(args.preview, variants)
        return

    order, by_q = load_inputs(variants, args.limit or None)
    mine = [v for i, v in enumerate(order) if i % args.nshard == args.shard]
    suf = f'_{args.tag}' if args.tag else ''
    paths = {v: f'{OUT_DIR}/responses_{v}{suf}_shard{args.shard}.jsonl'
             for v in variants}
    have = {v: done_keys(paths[v]) for v in variants}
    os.makedirs(OUT_DIR, exist_ok=True)

    from supportgain.models.qwen25vl.adapter import Adapter
    ad = Adapter()
    ad.load(device='cuda:0')
    print(f'[b1] shard {args.shard}/{args.nshard}  {len(mine)} questions  '
          f'variants={",".join(variants)}', flush=True)

    fhs = {v: open(paths[v], 'a') for v in variants}
    t0, nq = time.time(), 0
    for vqid in mine:
        rows = by_q[vqid]
        todo = [r for r in rows
                if (vqid, r['variant'], r['window_type']) not in have[r['variant']]]
        if not todo:
            continue
        vp = video_path(rows[0]['video_id'])
        if not os.path.exists(vp):
            print(f'[b1] MISSING VIDEO {vp}', flush=True)
            continue
        frames, ts, dur = ad.decode_video(vp)
        # MASK and MARGIN in order within the shard, each window in stored order.
        for variant in variants:
            for r in [x for x in todo if x['variant'] == variant]:
                iv = tuple(r['interval_applied'])
                mod = apply_intervention(PHASE_OF[variant], frames, ts, iv, 'keep')
                # Frames whose sampled timestamp falls inside the window. For a
                # window shorter than the frame spacing this can be 0: CROP then
                # has nothing to show (stored pred = -1), and MASK blacks out
                # the WHOLE clip, so its answer carries no window content and
                # must not be read as a response to the window.
                n_in_win = int(((ts >= iv[0]) & (ts < iv[1])).sum())
                if mod is None or len(mod) == 0:
                    pred, raw, flag = -1, '', 'empty_or_no_interval'
                else:
                    pred, raw = ad.predict_qa(mod, r['question'], r['choices'])
                    flag = None
                fhs[variant].write(json.dumps({
                    'vqid': vqid,
                    'variant': variant,
                    'window_type': r['window_type'],
                    'response': raw,
                    'parsed': int(pred),
                    'interval_src': r['interval_src'],
                    'interval_applied': r['interval_applied'],
                    'n_frames_in': int(len(mod)) if mod is not None else 0,
                    'n_frames_in_window': n_in_win,
                    'all_black': bool(variant == 'mask' and n_in_win == 0),
                    'n_frames_full': int(len(frames)),
                    'a_full': r['a_full'],
                    'answer_id': r['answer_id'],
                    'flag': flag,
                }) + '\n')
            fhs[variant].flush()
        nq += 1
        el = time.time() - t0
        print(f'[b1] {nq}/{len(mine)} {vqid} +{len(todo)} queries '
              f'({el:.0f}s, {el / max(nq, 1):.1f}s/question)', flush=True)
    for fh in fhs.values():
        fh.close()
    print(f'[b1] shard {args.shard} done: {nq} questions, '
          f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
