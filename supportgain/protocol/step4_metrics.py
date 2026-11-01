"""Step 4: Metric computation.

Reads step1 + step3 JSONs, normalizes records into the shape expected
by utils.metrics.compute_all(), and saves the metric dict to
results/<model>/<phase>/step4_metrics.json.
"""
import json
import os
from supportgain.config.settings import RESULTS_DIR, N_RANDOM
from supportgain.core import metrics as M
from supportgain.protocol.step3_intervention import present_families

# Metrics that need the 'gt' family (E_star + R_oracle). When those labels are
# absent (e.g. a --families self pass), computing them yields garbage (division
# by an ~eps denominator), so we blank them instead of emitting fake numbers.
GT_DEPENDENT_KEYS = ['GT-Suff@Correct', 'GT-Nece@Correct', 'GT-CTEF@Correct',
                     'CGA', 'OP-Rescue']


def _gt_covered(records):
    """True only if every record carries the full gt family."""
    return bool(records) and all('gt' in r.get('_families', []) for r in records)


def _guard(metric_dict, records):
    """Blank gt-dependent metrics when the gt family is not fully present."""
    covered = _gt_covered(records)
    metric_dict['_gt_covered'] = covered
    if not covered:
        for k in GT_DEPENDENT_KEYS:
            metric_dict[k] = None
    return metric_dict


def _normalize_records(step1, step3):
    """Build per-sample records of the shape utils.metrics expects."""
    out = []
    for vqid, s3 in step3.items():
        if 'error' in s3 or 'preds' not in s3:
            continue
        s1 = step1.get(vqid, {})
        if not s1 or 'error' in s1:
            continue
        preds = s3['preds']
        # Family coverage: trust step3's explicit marker, else infer from the
        # raw label presence using the record's own N (robust to N changes).
        n_r = len(s3.get('r_pred', [])) or N_RANDOM
        fams = s3.get('families_present') or present_families(preds, n_r)
        rec = {
            'vqid': vqid,
            'type': s3['type'],
            'answer_id': s3['answer_id'],
            'a_full': s3['a_full'],
            'duration': s3['duration'],
            'predicted_iv': s3['predicted_iv'],
            'oracle_iv': s3['oracle_iv'],
            '_families': fams,
            'preds': {
                'predicted_keep':   preds.get('predicted_keep', -1),
                'predicted_remove': preds.get('predicted_remove', -1),
                'oracle_keep':      preds.get('oracle_keep', -1),
                'oracle_remove':    preds.get('oracle_remove', -1),
                'blind':            preds.get('blind', -1),
                # Use the record's own N (n_r), not the global N_RANDOM: a step3
                # file written with a different N must still read correctly
                # (padding to N_RANDOM with -1 would corrupt the random averages).
                'r_pred_keep':   [preds.get(f'r_pred_keep_{k}',   -1) for k in range(n_r)],
                'r_pred_remove': [preds.get(f'r_pred_remove_{k}', -1) for k in range(n_r)],
                'r_oracle_keep':   [preds.get(f'r_oracle_keep_{k}',   -1) for k in range(n_r)],
                'r_oracle_remove': [preds.get(f'r_oracle_remove_{k}', -1) for k in range(n_r)],
            },
        }
        out.append(rec)
    return out


def run(model_name: str, phase: str = 'phase1'):
    step1_path = os.path.join(RESULTS_DIR, model_name, 'step1.json')
    step3_path = os.path.join(RESULTS_DIR, model_name, phase, 'step3.json')
    out_path   = os.path.join(RESULTS_DIR, model_name, phase, 'step4_metrics.json')

    step1 = json.load(open(step1_path))
    step3 = json.load(open(step3_path))
    records = _normalize_records(step1, step3)

    overall = _guard(M.compute_all(records), records)

    # Per-type breakdown
    per_type = {}
    types = sorted(set(r['type'] for r in records))
    for t in types:
        sub = [r for r in records if r['type'] == t]
        per_type[t] = _guard(M.compute_all(sub), sub)

    out = {
        'model': model_name,
        'phase': phase,
        'N_random': N_RANDOM,
        'overall': overall,
        'per_type': per_type,
    }
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f'Saved metrics -> {out_path}', flush=True)
    # also print to console
    print(json.dumps(overall, indent=2))
    return out
