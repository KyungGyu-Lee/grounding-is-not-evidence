"""Step 2: Segment preparation.

For each sample produced by step 1, build:
  R_pred[k]   - random spans of length |E_hat|   (k = 0 .. N-1)
  R_oracle[k] - random spans of length |E_star|  (k = 0 .. N-1)

Uses deterministic per-vqid seeds so re-running gives identical
intervals. Saves to results/<model>/step2.json.
"""
import json
import os
from supportgain.config.settings import RESULTS_DIR, N_RANDOM, OVERLAP_THR, MAX_TRIES
from supportgain.core.intervals import random_intervals


def run(model_name: str, n_random: int = N_RANDOM,
        step1_path: str = None, out_path: str = None):
    if step1_path is None:
        step1_path = os.path.join(RESULTS_DIR, model_name, 'step1.json')
    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, model_name, 'step2.json')

    step1 = json.load(open(step1_path))
    seg = {}
    all_overlaps, all_fallbacks = [], []   # for aggregate reporting
    for vqid, r in step1.items():
        if 'error' in r:
            continue
        dur = r['duration']
        pred_iv = r.get('predicted_iv')
        ora_iv = r.get('oracle_iv')

        pred_len = (pred_iv[1] - pred_iv[0]) if pred_iv else 0.0
        ora_len = (ora_iv[1] - ora_iv[0]) if ora_iv else 0.0

        # Random controls must avoid BOTH the model's evidence (E_hat) and the
        # GT evidence (E_star) — PDF §2 overlap control via IoP threshold.
        forbidden = [pred_iv, ora_iv]

        if pred_len > 0:
            r_pred, m_pred = random_intervals(
                vqid + '_pred', pred_len, dur, N=n_random, forbidden=forbidden,
                overlap_thr=OVERLAP_THR, max_tries=MAX_TRIES, return_meta=True)
        else:
            r_pred = [None] * n_random
            m_pred = [{'overlap': None, 'fallback': False} for _ in range(n_random)]

        if ora_len > 0:
            r_oracle, m_oracle = random_intervals(
                vqid + '_oracle', ora_len, dur, N=n_random, forbidden=forbidden,
                overlap_thr=OVERLAP_THR, max_tries=MAX_TRIES, return_meta=True)
        else:
            r_oracle = [None] * n_random
            m_oracle = [{'overlap': None, 'fallback': False} for _ in range(n_random)]

        seg[vqid] = {
            'r_pred': [list(iv) if iv else None for iv in r_pred],
            'r_oracle': [list(iv) if iv else None for iv in r_oracle],
            'pred_len': pred_len,
            'oracle_len': ora_len,
            'r_pred_overlap':    [m['overlap'] for m in m_pred],
            'r_pred_fallback':   [m['fallback'] for m in m_pred],
            'r_oracle_overlap':  [m['overlap'] for m in m_oracle],
            'r_oracle_fallback': [m['fallback'] for m in m_oracle],
        }
        for m in (m_pred + m_oracle):
            if m['overlap'] is not None:
                all_overlaps.append(m['overlap'])
                all_fallbacks.append(bool(m['fallback']))

    json.dump(seg, open(out_path, 'w'))
    print(f'Saved {len(seg)} segment sets -> {out_path}', flush=True)

    # Aggregate provenance for the random control — report alongside results so
    # reviewers can judge the validity of the control (PDF §2).
    n_spans = len(all_overlaps)
    meta = {
        'model': model_name,
        'N_RANDOM': n_random,
        'overlap_thr': OVERLAP_THR,
        'max_tries': MAX_TRIES,
        'forbidden': ['E_hat', 'E_star'],
        'overlap_metric': 'IoP = |R ∩ E| / |R|',
        'n_random_spans': n_spans,
        'fallback_rate': (sum(all_fallbacks) / n_spans) if n_spans else None,
        'mean_final_overlap': (sum(all_overlaps) / n_spans) if n_spans else None,
        'max_final_overlap': (max(all_overlaps) if n_spans else None),
    }
    meta_path = os.path.join(os.path.dirname(out_path), 'step2_meta.json')
    json.dump(meta, open(meta_path, 'w'), indent=2)
    print(f'Saved random-control provenance -> {meta_path}', flush=True)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    return seg
