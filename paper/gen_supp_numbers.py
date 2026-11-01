# -*- coding: utf-8 -*-
"""gen_supp_numbers.py — supplementary scalars that build_paper_numbers does NOT
emit: the measurability-restricted analyses (supp. Sec. A), the random-control
fallback table (Table S16), the SG/SG_gt ratio column (Table S1) and the paired
Support-Gain difference quoted in Sec. 6.1.

  python paper/gen_supp_numbers.py        # prints every value + PASS/FAIL gate

Reads stored outputs only; no model is re-run.

MEASURABLE POSITIVE CONTROL (the dagger rule).
  A (model, benchmark) pair is measurable iff the 95% bootstrap interval on
  SG_gt lies ABOVE zero, i.e. its LOWER bound is > 0 -- the same test
  build_paper_numbers.py applies before it reports ESR (see the ESR_undefined
  branch there). An interval that merely EXCLUDES zero is not enough: QVH-ME
  VTimeLLM on C2 is [-.077, -.012], entirely below zero, and is still not
  interpretable. The rule is read off the stored figdata, and the script
  asserts that C2 and C3 agree for every model so the set is a property of the
  (model, benchmark) pair rather than of the cut.
    NExT-GQA : Qwen2.5-VL, VideoMind, GroundVTS, Grounded-VideoLLM   (VTimeLLM out)
    QVH-ME   : the same four + VideoChat-R1            (VTimeLLM, TimeChat out)

RANDOM-CONTROL FALLBACK (Table S16).
  * Prediction-matched: step2.json is not part of the release, so the per-draw
    flag is RECONSTRUCTED from the stored r_pred intervals with the rule in
    core/intervals.random_intervals -- a draw is a fallback when its IoP
    overlap with E_hat or E* reaches OVERLAP_THR, or when the span covers the
    whole clip. Validated against the original run's stored
    `r_pred_fallback`: 11,290 / 11,290 draws agree for Qwen2.5-VL on NExT-GQA.
    The denominator counts every draw slot (models x questions x 5), including
    the slots left empty when |E_hat| <= 0.
  * Annotation-matched: the flag is stored directly in the released
    rand_gt_keep files, which carry the shared control (identical segments for
    all three reference models; asserted below).
    CAVEAT -- the released rand_gt_keep covers the 1,507 NExT-GQA questions
    that have a full-video reference output, whereas the paper's row is taken
    over C1 (1,526). The 19 extra questions have |E*|/T between .42 and .50, so
    no length-matched window can avoid E*; all five of their draws fall back and
    they contribute 95 of the paper's 96 fallback draws. Over the released
    questions the rate is therefore .01% / .07%, not 1.26% / 1.31%. This script
    gates the released-basis value and prints both.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_paper_numbers as B          # noqa: E402  (also chdirs to the repo root)
from supportgain.config.settings import OVERLAP_THR   # noqa: E402
from supportgain.core.intervals import overlap_coverage  # noqa: E402

RET_RAND_THR = 0.6        # the stricter residual-support diagnostic (supp. Sec. A)
B_BOOT = B.B_BOOT
SEED = B.SEED

FIGDATA = {'NExT': 'paper/figdata_nextgqa.json', 'QVH': 'paper/figdata_qvhme.json'}
DISP = {'Qwen': 'Qwen2.5-VL', 'VideoMind': 'VideoMind', 'GroundVTS': 'GroundVTS',
        'GVLLM': 'Grounded-VideoLLM', 'VTimeLLM': 'VTimeLLM', 'TimeChat': 'TimeChat',
        'VideoChat-R1': 'VideoChat-R1'}

# ------------------------------------------------- values printed in the paper
PAPER = {
    'spearman_all':  {('NExT', 'C2'): .091, ('NExT', 'C3'): .166,
                      ('QVH', 'C2'): .283, ('QVH', 'C3'): .344},
    'spearman_meas': {('NExT', 'C2'): .099, ('NExT', 'C3'): .193,
                      ('QVH', 'C2'): .281, ('QVH', 'C3'): .338},
    'residual_all':  {'NExT': 19.0, 'QVH': 11.6},
    'residual_meas': {'NExT': 18.0, 'QVH': 12.0},
    'fallback_pred': {'NExT': (9.98, 10.11), 'QVH': (3.18, 3.28)},
    # released-basis annotation-matched (see the module docstring for the 1,507
    # vs 1,526 caveat; the paper prints 1.26 / 1.31 for NExT-GQA over C1).
    'fallback_gt':   {'NExT': (0.01, 0.07), 'QVH': (0.00, 0.00)},
    'fallback_gt_paper_C1': {'NExT': (1.26, 1.31), 'QVH': (0.00, 0.00)},
    'ratio': {('NExT', 'Qwen'): .72, ('NExT', 'VideoMind'): .46,
              ('NExT', 'GVLLM'): .27, ('NExT', 'GroundVTS'): .39,
              ('QVH', 'VideoChat-R1'): .82, ('QVH', 'Qwen'): .74,
              ('QVH', 'GVLLM'): .62, ('QVH', 'VideoMind'): .69,
              ('QVH', 'GroundVTS'): .30},
    'paired_diff': (+.030, -.015, +.075),
    'max_corr_shift': .027,
}

_FAIL = []


def gate(tag, got, want, tol=5e-4, fmt='{:.4f}'):
    ok = got is not None and abs(got - want) <= tol
    if not ok:
        _FAIL.append((tag, got, want))
    print(f"  {'PASS' if ok else 'FAIL'}  {tag:52} got={fmt.format(got)}  paper={want}")
    return ok


def spearman(x, y):
    return float(pd.Series(np.asarray(x, float)).corr(
        pd.Series(np.asarray(y, float)), method='spearman'))


# --------------------------------------------------------------- measurability
def measurable_sets(fig):
    """{bench: [models]} whose SG_gt 95% interval lies above zero on every cut."""
    out = {}
    for bench in ('NExT', 'QVH'):
        G = [g[0] for g in B.GROUNDERS[bench]]
        keep = []
        for m in G:
            flags = []
            for cut in ('C2', 'C3'):
                lo = fig[bench][cut]['positive_control'][m]['SG_GT_CI'][0]
                flags.append(lo > 0)
            assert flags[0] == flags[1], (
                f'{bench} {m}: measurability differs between C2 and C3 '
                f'({flags}); the dagger mark is a per-(model, benchmark) property')
            if flags[0]:
                keep.append(m)
        out[bench] = keep
    return out


# ------------------------------------------------------------------- fallbacks
def fallback_pred(bench):
    """Reconstructed prediction-matched fallback: (draw %, item %, counts)."""
    n_fb = n_draw = n_item_fb = n_item = 0
    for disp, dr in B.GROUNDERS[bench]:
        s3 = json.load(open(B._s3(bench, dr)))
        for rec in s3.values():
            if 'error' in rec:
                continue
            forbidden = [iv for iv in (rec.get('predicted_iv'), rec.get('oracle_iv')) if iv]
            dur = rec['duration']
            draws = rec.get('r_pred') or []
            hit = [max((overlap_coverage(tuple(R), tuple(f)) for f in forbidden), default=0.0)
                   >= OVERLAP_THR - 1e-12 or (R[1] - R[0]) >= dur - 1e-6
                   for R in draws if R]
            n_fb += sum(hit)
            n_draw += len(draws) if draws else 5
            n_item += 1
            n_item_fb += any(hit)
    return 100 * n_fb / n_draw, 100 * n_item_fb / n_item, (n_fb, n_draw, n_item_fb, n_item)


def fallback_gt(bench):
    """Annotation-matched fallback from the released shared control."""
    dr = 'qwen25vl'
    rk = B._ls(f'outputs/{B.BDIR[bench]}/{dr}/rand_gt_keep*.json')
    for other in ('internvl', 'videollama2'):
        o = B._ls(f'outputs/{B.BDIR[bench]}/{other}/rand_gt_keep*.json')
        shared = all([t['segment'] for t in rk[k]] == [t['segment'] for t in o[k]]
                     for k in set(rk) & set(o))
        assert shared, f'{bench}: {other} does not share the annotation-matched segments'
    n_fb = n_draw = n_item_fb = 0
    for trials in rk.values():
        flags = [bool(t.get('fallback_flag')) for t in trials]
        n_draw += len(flags); n_fb += sum(flags); n_item_fb += any(flags)
    return 100 * n_fb / n_draw, 100 * n_item_fb / len(rk), (n_fb, n_draw, n_item_fb, len(rk))


# ------------------------------------------------------------------------ main
def main():
    fig = {b: json.load(open(FIGDATA[b]))[b] for b in ('NExT', 'QVH')}
    meas = measurable_sets(fig)
    data = {b: dict(zip(('masks', 'rn'), B.set_masks(b))) for b in ('NExT', 'QVH')}
    for b in data:
        data[b]['df'] = B.load_samples(b)

    print('== measurable positive control (SG_gt 95% CI lower bound > 0) ==')
    for b in ('NExT', 'QVH'):
        allg = [g[0] for g in B.GROUNDERS[b]]
        print(f'  {b:5} measurable: {[DISP[m] for m in meas[b]]}')
        print(f'  {b:5} excluded  : {[DISP[m] for m in allg if m not in meas[b]]}')

    print('\n== supp. Sec. A -- pooled Spearman(IoU, SG) ==')
    shifts = []
    for b in ('NExT', 'QVH'):
        for cut in ('C2', 'C3'):
            ids = data[b]['masks'][cut]; df = data[b]['df']
            allg = [g[0] for g in B.GROUNDERS[b]]
            g_all = df[df.vqid.isin(ids) & df.model.isin(allg)].dropna(subset=['iou'])
            g_ms = df[df.vqid.isin(ids) & df.model.isin(meas[b])].dropna(subset=['iou'])
            r_all = spearman(g_all.iou.values, g_all.SG.values)
            r_ms = spearman(g_ms.iou.values, g_ms.SG.values)
            gate(f'Spearman(IoU,SG) {b} {cut} all models', r_all, PAPER['spearman_all'][(b, cut)])
            gate(f'Spearman(IoU,SG) {b} {cut} measurable', r_ms, PAPER['spearman_meas'][(b, cut)])
            shifts.append(abs(round(r_all, 3) - round(r_ms, 3)))
    gate('Sec. 6.3 "changes these correlations by at most"',
         max(shifts), PAPER['max_corr_shift'], tol=5e-4, fmt='{:.3f}')

    print('\n== supp. Sec. A -- residual random-window support on C3 '
          f'(Ret(none)=0 and Ret(rand)>={RET_RAND_THR}) ==')
    for b in ('NExT', 'QVH'):
        ids = data[b]['masks']['C3']; df = data[b]['df']
        allg = [g[0] for g in B.GROUNDERS[b]]
        for tag, models, want in (('all models', allg, PAPER['residual_all'][b]),
                                  ('measurable', meas[b], PAPER['residual_meas'][b])):
            g = df[df.vqid.isin(ids) & df.model.isin(models)]
            h = g[(g.Kbl == 0) & (g.KR >= RET_RAND_THR - 1e-9)]
            gate(f'residual random support {b} {tag}', 100 * len(h) / len(g), want,
                 tol=.05, fmt='{:.2f}')

    print('\n== Table S16 -- random-control fallback ==')
    for b in ('NExT', 'QVH'):
        d, i, (a, c, e, f) = fallback_pred(b)
        print(f'       prediction-matched {b}: {a}/{c} draws, {e}/{f} items')
        gate(f'fallback prediction-matched {b} draw', d, PAPER['fallback_pred'][b][0],
             tol=.005, fmt='{:.3f}')
        gate(f'fallback prediction-matched {b} item', i, PAPER['fallback_pred'][b][1],
             tol=.005, fmt='{:.3f}')
    for b in ('NExT', 'QVH'):
        d, i, (a, c, e, f) = fallback_gt(b)
        print(f'       annotation-matched {b}: {a}/{c} draws, {e}/{f} items '
              f'(released basis; paper row is over C1 = '
              f'{"1,526" if b == "NExT" else "763"})')
        gate(f'fallback annotation-matched {b} draw', d, PAPER['fallback_gt'][b][0],
             tol=.005, fmt='{:.3f}')
        gate(f'fallback annotation-matched {b} item', i, PAPER['fallback_gt'][b][1],
             tol=.005, fmt='{:.3f}')
    if PAPER['fallback_gt']['NExT'] != PAPER['fallback_gt_paper_C1']['NExT']:
        print('       NOTE  the paper prints 1.26% / 1.31% for NExT-GQA, taken over C1 '
              '(1,526).\n'
              '             19 of those questions have no full-video reference output, are '
              'absent\n'
              '             from the release, and carry 95 of the 96 fallback draws '
              '(|E*|/T = .42-.50,\n'
              '             so no length-matched window can avoid E*). See the module '
              'docstring.')

    print('\n== Table S1 -- SG / SG_gt ratio on C2 (from the raw values, not the '
          'rounded cells) ==')
    for b in ('NExT', 'QVH'):
        for m in meas[b]:
            sg = fig[b]['C2']['components'][m]['SG']
            sggt = fig[b]['C2']['positive_control'][m]['SG_GT']
            gate(f'ratio {b} {DISP[m]}', sg / sggt, PAPER['ratio'][(b, m)],
                 tol=.005, fmt='{:.4f}')

    print('\n== Sec. 6.1 -- paired SG difference, VideoChat-R1 - Qwen2.5-VL, QVH-ME C2 ==')
    ids = sorted(data['QVH']['masks']['C2']); df = data['QVH']['df']
    a = df[df.model == 'VideoChat-R1'].set_index('vqid').SG.reindex(ids).values
    q = df[df.model == 'Qwen'].set_index('vqid').SG.reindex(ids).values
    d = a - q
    assert not np.isnan(d).any(), 'both models must cover every C2 question'
    idx = np.random.default_rng(SEED).integers(0, len(d), (B_BOOT, len(d)))
    boot = d[idx].mean(1)
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    print(f'       N={len(d)}  rng=default_rng({SEED})  B={B_BOOT}')
    gate('paired SG difference (point)', float(d.mean()), PAPER['paired_diff'][0],
         tol=5e-4, fmt='{:+.4f}')
    gate('paired SG difference CI lower', float(lo), PAPER['paired_diff'][1],
         tol=5e-4, fmt='{:+.4f}')
    gate('paired SG difference CI upper', float(hi), PAPER['paired_diff'][2],
         tol=5e-4, fmt='{:+.4f}')

    print()
    if _FAIL:
        print(f'*** {len(_FAIL)} supplementary gate(s) FAILED ***')
        for tag, got, want in _FAIL:
            print(f'    {tag}: got {got!r}, paper {want!r}')
        raise SystemExit(1)
    print('supplementary numbers OK (measurability, Spearman, residual support, '
          'fallback, ratio, paired CI)')


if __name__ == '__main__':
    main()
