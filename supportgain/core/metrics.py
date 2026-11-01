"""Metric computation as defined in temporal_evidence_faithfulness_metrics.pdf.

  Self-CTEF        (target = a_full,  R_pred only)        — §3
  GT-CTEF@Correct  (target = y,       on full_correct,
                     uses R_pred AND R_oracle)            — §4
  CGA              FullAcc * GT-CTEF@Correct              — §5
  OP-Rescue        (on full_wrong subset)                 — §6
  Remove Random Stability                                 — §6
  Raw GT-Suff / Raw GT-Nece                               — §6 (diagnostic)
"""
from typing import Sequence, Optional

EPS = 1e-8


# ---------------------------------------------------------------------
# Helpers: 'K_t(E)' and 'M_t(E)' per sample
# ---------------------------------------------------------------------

def _match(pred, target):
    """1 if both are valid (>=0) and equal, else 0. Invalid (-1) → 0."""
    return int(pred is not None and pred == target and pred >= 0)


def K_target(pred_keep, target):
    """K_t(E)  ← Keep(E) answer matches target t."""
    return _match(pred_keep, target)


def M_target(pred_remove, target):
    """M_t(E)  ← Remove(E) answer matches target t."""
    return _match(pred_remove, target)


def _avg_over_seeds(preds: Sequence[int], target: int):
    """Average of _match over K seeds."""
    if not preds:
        return None
    return sum(_match(p, target) for p in preds) / len(preds)


# ---------------------------------------------------------------------
# Aggregate metrics over a dataset (list of sample records)
# ---------------------------------------------------------------------

def _aggregate(records, getter):
    """Mean over records, skipping None values."""
    vals = [getter(r) for r in records]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _clip01(x):
    if x is None:
        return 0.0
    return max(0.0, min(1.0, x))


def _harmonic_mean(a, b):
    if a is None or b is None:
        return None
    if a <= 0 and b <= 0:
        return 0.0
    return 2 * a * b / (a + b + EPS)


# ---------------------------------------------------------------------
# §3 Self-CTEF (target = a_full)
# ---------------------------------------------------------------------

def self_suff(records):
    """ ( K_a_full(E_hat) - K_a_full(R_pred) ) / ( 1 - K_a_full(R_pred) + eps )"""
    k_e = _aggregate(records, lambda r: K_target(r['preds']['predicted_keep'], r['a_full']))
    k_r = _aggregate(records, lambda r: _avg_over_seeds(r['preds']['r_pred_keep'], r['a_full']))
    if k_e is None or k_r is None:
        return None
    return (k_e - k_r) / (1.0 - k_r + EPS)


def self_nece(records):
    """ ( M_a_full(R_pred) - M_a_full(E_hat) ) / ( M_a_full(R_pred) + eps )"""
    m_e = _aggregate(records, lambda r: M_target(r['preds']['predicted_remove'], r['a_full']))
    m_r = _aggregate(records, lambda r: _avg_over_seeds(r['preds']['r_pred_remove'], r['a_full']))
    if m_e is None or m_r is None:
        return None
    return (m_r - m_e) / (m_r + EPS)


def self_ctef(records):
    return _harmonic_mean(_clip01(self_suff(records)), _clip01(self_nece(records)))


# ---------------------------------------------------------------------
# §4 GT-CTEF@Correct (subset: a_full == y)
# ---------------------------------------------------------------------

def _correct_subset(records):
    return [r for r in records if r['a_full'] == r['answer_id'] and r['a_full'] >= 0]


def gt_suff_correct(records):
    """ ( K_y(E_hat) - K_y(R_pred) ) / ( K_y(E_star) - K_y(R_oracle) + eps )"""
    cr = _correct_subset(records)
    if not cr:
        return None
    k_pred = _aggregate(cr, lambda r: K_target(r['preds']['predicted_keep'], r['answer_id']))
    k_rpred = _aggregate(cr, lambda r: _avg_over_seeds(r['preds']['r_pred_keep'], r['answer_id']))
    k_star = _aggregate(cr, lambda r: K_target(r['preds']['oracle_keep'], r['answer_id']))
    k_roracle = _aggregate(cr, lambda r: _avg_over_seeds(r['preds']['r_oracle_keep'], r['answer_id']))
    if None in (k_pred, k_rpred, k_star, k_roracle):
        return None
    num = k_pred - k_rpred
    den = k_star - k_roracle + EPS
    return num / den


def gt_nece_correct(records):
    """ ( M_y(R_pred) - M_y(E_hat) ) / ( M_y(R_oracle) - M_y(E_star) + eps )"""
    cr = _correct_subset(records)
    if not cr:
        return None
    m_pred = _aggregate(cr, lambda r: M_target(r['preds']['predicted_remove'], r['answer_id']))
    m_rpred = _aggregate(cr, lambda r: _avg_over_seeds(r['preds']['r_pred_remove'], r['answer_id']))
    m_star = _aggregate(cr, lambda r: M_target(r['preds']['oracle_remove'], r['answer_id']))
    m_roracle = _aggregate(cr, lambda r: _avg_over_seeds(r['preds']['r_oracle_remove'], r['answer_id']))
    if None in (m_pred, m_rpred, m_star, m_roracle):
        return None
    num = m_rpred - m_pred
    den = m_roracle - m_star + EPS
    return num / den


def gt_ctef_correct(records):
    return _harmonic_mean(_clip01(gt_suff_correct(records)),
                          _clip01(gt_nece_correct(records)))


# ---------------------------------------------------------------------
# §5 CGA
# ---------------------------------------------------------------------

def full_acc(records):
    return _aggregate(records, lambda r: _match(r['a_full'], r['answer_id']))


def cga(records):
    fa = full_acc(records)
    gt = gt_ctef_correct(records)
    if fa is None or gt is None:
        return None
    return fa * gt


# ---------------------------------------------------------------------
# §6 OP-Rescue (subset: a_full != y)
# ---------------------------------------------------------------------

def _wrong_subset(records):
    return [r for r in records if r['a_full'] != r['answer_id']]


def op_rescue(records):
    wr = _wrong_subset(records)
    if not wr:
        return None
    k_pred = _aggregate(wr, lambda r: K_target(r['preds']['predicted_keep'], r['answer_id']))
    k_rpred = _aggregate(wr, lambda r: _avg_over_seeds(r['preds']['r_pred_keep'], r['answer_id']))
    k_star = _aggregate(wr, lambda r: K_target(r['preds']['oracle_keep'], r['answer_id']))
    k_roracle = _aggregate(wr, lambda r: _avg_over_seeds(r['preds']['r_oracle_keep'], r['answer_id']))
    if None in (k_pred, k_rpred, k_star, k_roracle):
        return None
    return (k_pred - k_rpred) / (k_star - k_roracle + EPS)


# ---------------------------------------------------------------------
# §6 Remove Random Stability
# ---------------------------------------------------------------------

def remove_random_stability(records):
    """Average of M_t(R_pred) on the full set, target = y."""
    return _aggregate(records, lambda r: _avg_over_seeds(r['preds']['r_pred_remove'], r['answer_id']))


# ---------------------------------------------------------------------
# Diagnostic raw metrics (kept as appendix in the paper)
# ---------------------------------------------------------------------

def raw_gt_suff(records):
    """Acc( Keep(E_hat), y ) - Acc( full, y ) is the original PDF form,
    but we report it as Acc(Keep, y) for the diagnostic table."""
    return _aggregate(records, lambda r: K_target(r['preds']['predicted_keep'], r['answer_id']))


def raw_gt_nece(records):
    """Acc(full, y) - Acc(Remove(E_hat), y)"""
    fa = full_acc(records)
    rem = _aggregate(records, lambda r: M_target(r['preds']['predicted_remove'], r['answer_id']))
    if fa is None or rem is None:
        return None
    return fa - rem


# ---------------------------------------------------------------------
# Length-related diagnostics
# ---------------------------------------------------------------------

def mean_pred_video_len_ratio(records):
    def f(r):
        iv = r.get('predicted_iv')
        if iv is None:
            return None
        return (iv[1] - iv[0]) / max(r['duration'], EPS)
    return _aggregate(records, f)


def mean_pred_gt_len_ratio(records):
    def f(r):
        iv = r.get('predicted_iv')
        gt = r.get('oracle_iv')
        if iv is None or gt is None:
            return None
        gt_len = gt[1] - gt[0]
        if gt_len <= 0:
            return None
        return (iv[1] - iv[0]) / gt_len
    return _aggregate(records, f)


# ---------------------------------------------------------------------
# Public summary function (called from supportgain.protocol.step4)
# ---------------------------------------------------------------------

def compute_all(records):
    """Return a dict of every metric defined above."""
    return {
        # §3
        'Self-Suff': self_suff(records),
        'Self-Nece': self_nece(records),
        'Self-CTEF': self_ctef(records),
        # §4
        'GT-Suff@Correct': gt_suff_correct(records),
        'GT-Nece@Correct': gt_nece_correct(records),
        'GT-CTEF@Correct': gt_ctef_correct(records),
        # §5
        'FullAcc': full_acc(records),
        'CGA': cga(records),
        # §6
        'OP-Rescue': op_rescue(records),
        'RemoveRandomStability': remove_random_stability(records),
        'Raw-GT-Suff': raw_gt_suff(records),
        'Raw-GT-Nece': raw_gt_nece(records),
        # length diagnostics
        'Pred/Video Len': mean_pred_video_len_ratio(records),
        'Pred/GT Len': mean_pred_gt_len_ratio(records),
        # subset sizes
        'N_total': len(records),
        'N_full_correct': len(_correct_subset(records)),
        'N_full_wrong': len(_wrong_subset(records)),
    }
