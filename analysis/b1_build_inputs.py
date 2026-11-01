"""B1-0 — build the re-query input list for the masking/margin variants.

Reads STORED Qwen2.5-VL QVH-ME results only (no model, no GPU, no new random
sampling): the C3 membership from `set_masks('QVH')` and, per question, the
predicted interval and the five stored random windows from

    outputs/qvhme/qwen25vl/step3.json           ('predicted_iv', 'r_pred')

Emits one line per re-query:  241 questions x (1 pred + 5 rand) x 2 variants
= 2892 queries, so CROP (stored) and MASK/MARGIN pair up question-by-question
AND window-by-window.

        python analysis/b1_build_inputs.py
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{ROOT}/paper')
import build_paper_numbers as bpn  # noqa: E402  (chdir()s to the repo root)

sys.path.insert(0, ROOT)
from supportgain.config.prompts import normalize_option  # noqa: E402

OUT_DIR = f'{ROOT}/analysis/B1_masking'
QVH_CSV = f'{ROOT}/data/qvh_gqa_regen/full2/val_qvh_gqa_2stage_corrected.csv'
MARGIN_K = 5.0          # +/- seconds for the MARGIN variant
NDRAW = 5
VARIANTS = ('mask', 'margin')


def widen(iv, k, duration):
    """MARGIN interval: grow by k s on each side, clipped to [0, duration]."""
    s, e = iv
    return [max(0.0, s - k), min(float(duration), e + k)]


def main():
    masks, _ = bpn.set_masks('QVH')
    c3 = masks['C3']

    s3 = json.load(open(bpn._s3('QVH', 'qwen25vl')))
    s1 = json.load(open(bpn._s1('QVH', 'qwen25vl')))
    df = pd.read_csv(QVH_CSV)
    df['vqid'] = df['video_id'].astype(str) + '_' + df['qid'].astype(str)
    meta = df.set_index('vqid')

    os.makedirs(OUT_DIR, exist_ok=True)
    rows, skipped = [], []
    for vqid in sorted(c3):
        rec = s3.get(vqid)
        if rec is None or 'error' in rec:
            skipped.append((vqid, 'no step3 record'))
            continue
        preds = rec.get('preds', {})
        need = ['predicted_keep'] + [f'r_pred_keep_{k}' for k in range(NDRAW)]
        if any(lbl not in preds for lbl in need):
            skipped.append((vqid, 'incomplete stored keep labels'))
            continue
        if rec['predicted_iv'] is None or any(w is None for w in rec['r_pred']):
            skipped.append((vqid, 'missing stored window'))
            continue

        m = meta.loc[vqid]
        # step3 does not store the question; step1 holds the same normalized
        # string step3 fed the model (step3: `question = s1['question']`).
        question = s1[vqid]['question']
        choices = [normalize_option(m[f'a{i}']) for i in range(5)]
        dur = float(rec['duration'])

        windows = [('pred', rec['predicted_iv'], preds['predicted_keep'])]
        for k in range(NDRAW):
            windows.append((f'rand_{k}', rec['r_pred'][k],
                            preds[f'r_pred_keep_{k}']))

        for variant in VARIANTS:
            for wtype, iv, crop_pred in windows:
                applied = list(iv) if variant == 'mask' else widen(iv, MARGIN_K, dur)
                rows.append({
                    'vqid': vqid,
                    'video_id': str(m['video_id']),
                    'qid': int(m['qid']),
                    'variant': variant,
                    'window_type': wtype,
                    'interval_src': [float(iv[0]), float(iv[1])],
                    'interval_applied': [float(applied[0]), float(applied[1])],
                    'duration': dur,
                    'a_full': int(rec['a_full']),
                    'answer_id': int(rec['answer_id']),
                    'crop_parsed': int(crop_pred),   # stored CROP pred, for pairing
                    'question': question,
                    'choices': choices,
                })

    out = f'{OUT_DIR}/inputs.jsonl'
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    nq = len({r['vqid'] for r in rows})
    print(f'C3 = {len(c3)}  usable = {nq}  skipped = {len(skipped)}')
    for v, why in skipped:
        print(f'  SKIP {v}: {why}')
    for variant in VARIANTS:
        sub = [r for r in rows if r['variant'] == variant]
        per_q = {r['vqid']: 0 for r in sub}
        for r in sub:
            per_q[r['vqid']] += 1
        assert set(per_q.values()) == {6}, sorted(set(per_q.values()))
        print(f'  {variant:7} {len(sub)} queries, 6 windows/question')
    assert len(rows) == nq * 6 * len(VARIANTS)
    print(f'wrote {len(rows)} lines -> {out}')

    # MARGIN sanity: every applied interval is the stored one grown by <= K per
    # side, and strictly wider unless clipped at a clip boundary.
    mg = [r for r in rows if r['variant'] == 'margin']
    bad = [r for r in mg
           if not (r['interval_applied'][0] <= r['interval_src'][0]
                   and r['interval_applied'][1] >= r['interval_src'][1]
                   and r['interval_src'][0] - r['interval_applied'][0] <= MARGIN_K + 1e-9
                   and r['interval_applied'][1] - r['interval_src'][1] <= MARGIN_K + 1e-9
                   and r['interval_applied'][0] >= -1e-9
                   and r['interval_applied'][1] <= r['duration'] + 1e-9)]
    print(f'MARGIN interval check: {len(bad)} violations')
    clipped_lo = sum(1 for r in mg if r['interval_applied'][0] == 0.0)
    clipped_hi = sum(1 for r in mg
                     if abs(r['interval_applied'][1] - r['duration']) < 1e-9)
    print(f'  clipped at 0 s: {clipped_lo}   clipped at duration: {clipped_hi}')
    assert not bad


if __name__ == '__main__':
    main()
