# -*- coding: utf-8 -*-
"""a4_loo_reference.py — leave-one-reference-model-out sensitivity (Table S9)
and the Qwen2.5-VL figures quoted in supp. Sec. A.1.

  python analysis/a4_loo_reference.py

Qwen2.5-VL defines part of the C1-C3 reference ensemble and is also an
evaluated grounding model, so the conditions are recomputed with each reference
model removed in turn. Read-only: b, u and r are rebuilt from the same stored
per-model outputs the main pipeline uses, and the grounder Support Gains are
reused unchanged.

With two reference models C2 keeps its meaning -- at least one additional
reference model becomes correct with the full video -- so the cut moves from
u - b >= 1/3 to u - b >= 1/2.

Every bootstrap here is numpy.random.default_rng(0) with B = 5000 over
questions, matching paper/build_paper_numbers.py. The questions are SORTED
before resampling: the bootstrap index matrix is fixed, so the realized
percentiles depend on the order of the input vector, and iterating the C3 set
directly makes the interval depend on PYTHONHASHSEED. The supplement quotes
[.107, .266] and [.337, .487], which is one such unsorted run; the
deterministic sorted values are [.108, .265] and [.336, .487]. Point estimates
(.187 / .412) are order-invariant and unaffected.

Ported from the original run's analysis/loo_reference_sensitivity.py with the
file layout remapped to the released tree and the question order pinned; the
arithmetic is otherwise unchanged.
"""
import json
import os
import sys

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'paper'))
import build_paper_numbers as B          # noqa: E402  (also chdirs to the repo root)

EPS = B.MEAS_EPS
C3T = B.C3_THRESHOLD
SEED = B.SEED
B_BOOT = B.B_BOOT
REFS = ['q', 'i', 'v']
REFDIR = {'q': 'qwen25vl', 'i': 'internvl', 'v': 'videollama2'}
BLIND = {'q': 'qwen25vl', 'i': 'internvl2_5_8b', 'v': 'videollama2_7b'}

VARIANTS = [('Base', REFS), ('w/o Qwen', ['i', 'v']),
            ('w/o InternVL', ['q', 'v']), ('w/o VideoLLaMA2', ['q', 'i'])]

# ---------------------------------------------------------- expected (paper)
TABLE_S9 = {   # (bench, variant) -> (C2 N, C3 N, median rho, C3 mean SG)
    ('NExT', 'Base'): (1116, 248, .93, .124),
    ('NExT', 'w/o Qwen'): (962, 197, 1.00, .088),
    ('NExT', 'w/o InternVL'): (982, 217, 1.00, .124),
    ('NExT', 'w/o VideoLLaMA2'): (965, 233, .90, .135),
    ('QVH', 'Base'): (427, 241, .40, .212),
    ('QVH', 'w/o Qwen'): (338, 179, .40, .207),
    ('QVH', 'w/o InternVL'): (366, 218, .30, .225),
    ('QVH', 'w/o VideoLLaMA2'): (403, 240, .40, .218),
}
# Deterministic (sorted-question) bootstrap; see the module docstring for why
# these differ from the supplement's [.107, .266] / [.337, .487] in the third
# decimal. The point estimates are the supplement's.
QWEN_LOO = {'NExT': (.187, .108, .265), 'QVH': (.412, .336, .487)}
FUNNEL = {'NExT': (2268, 1526, 1116, 248), 'QVH': (763, 560, 427, 241)}

_FAIL = []


def gate(tag, got, want, tol=5e-4, fmt='{:.3f}'):
    ok = got is not None and abs(got - want) <= tol
    if not ok:
        _FAIL.append((tag, got, want))
    print(f"  {'PASS' if ok else 'FAIL'}  {tag:46} got={fmt.format(got)}  paper={want}")


def boot_ci(vals, seed=SEED, nboot=B_BOOT):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), (nboot, len(vals)))
    m = vals[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ------------------------------------------------- reference-model quantities
def load_ref(bench):
    """Per-reference-model blind / full-video / annotation-matched-random correctness."""
    bd = B.BDIR[bench]
    bl = {k: json.load(open(f'flags/{bd}/blind_{BLIND[k]}.json')) for k in REFS}
    if bench == 'NExT':
        s1q = json.load(open('outputs/nextgqa/qwen25vl/step1.json'))
        u = {'q': {k: int(v['a_full'] == v['answer_id']) for k, v in s1q.items()}}
    else:
        u = {'q': {k: v['correct'] for k, v in
                   B._ls(f'flags/{bd}/fullvideo_qwen25vl_shard[0-9]*.json').items()}}
    for k in ('i', 'v'):
        u[k] = {q: r['correct'] for q, r in
                B._ls(f'flags/{bd}/fullvideo_{REFDIR[k]}_shard[0-9]*.json').items()}

    def rmean(trials):
        vals = [t['correct'] for t in trials if not t.get('skip')]
        return float(np.mean(vals)) if vals else np.nan

    r = {}
    for k in REFS:
        src = B._ls(f'outputs/{bd}/{REFDIR[k]}/rand_gt_keep*.json')
        r[k] = {q: rmean(t) for q, t in src.items()}
    ALL = set(bl['q']) & set(bl['i']) & set(bl['v'])
    b = {k: {q: bl[k][q]['correct'] for q in bl[k]} for k in REFS}
    basis = ALL.intersection(*[set(u[k]) for k in REFS], *[set(r[k]) for k in REFS])
    return dict(ALL=ALL, basis=basis, b=b, u=u, r=r,
                rn=json.load(open(f'flags/{bd}/r_norm.json'))['r_norm'])


def sets_for(D, subset, c2_thr):
    """C1 / C2 / C3 and rho for one reference subset."""
    def bm(q): return np.mean([D['b'][k][q] for k in subset])
    def um(q): return np.mean([D['u'][k][q] for k in subset])
    def rm(q): return np.mean([D['r'][k][q] for k in subset])
    C1 = {q for q in D['ALL'] if bm(q) < 1.0 - EPS}
    C2, rho = set(), {}
    for q in D['basis']:
        den = um(q) - bm(q)
        if den >= c2_thr - EPS:
            C2.add(q)
            rho[q] = (rm(q) - bm(q)) / den
    C3 = {q for q in C2 if rho[q] <= C3T + EPS}
    med = float(np.median([rho[q] for q in C2])) if C2 else np.nan
    return dict(C1=C1, C2=C2, C3=C3, rho=rho, med=med)


def load_sg(bench):
    """{display model: {vqid: SG}} with build_paper_numbers' validity filter."""
    out = {}
    for disp, dr in B.GROUNDERS[bench]:
        d = {}
        for v, rec in json.load(open(B._s3(bench, dr))).items():
            if 'error' in rec:
                continue
            p = rec.get('preds', {}); af = rec.get('a_full')
            if af is None or af < 0 or 'predicted_keep' not in p or 'blind' not in p:
                continue
            rk = [p[f'r_pred_keep_{k}'] for k in range(5) if f'r_pred_keep_{k}' in p]
            if not rk:
                continue
            d[v] = int(p['predicted_keep'] == af) - sum(int(x == af) for x in rk) / len(rk)
        out[disp] = d
    return out


def mean_sg(sg, models, ids):
    per = []
    for m in models:
        vals = [sg[m][v] for v in ids if v in sg[m]]
        if vals:
            per.append(float(np.mean(vals)))
    return float(np.mean(per)) if per else np.nan


def main():
    print('== baseline validation (must reproduce before the leave-one-out is trusted) ==')
    store = {}
    for bench in ('NExT', 'QVH'):
        D = load_ref(bench)
        sg = load_sg(bench)
        base = sets_for(D, REFS, 1 / 3)
        store[bench] = (D, sg, base)
        a, c1, c2, c3 = FUNNEL[bench]
        gate(f'{bench} funnel ALL', float(len(D['ALL'])), float(a), .5, '{:.0f}')
        gate(f'{bench} funnel C1', float(len(base['C1'])), float(c1), .5, '{:.0f}')
        gate(f'{bench} funnel C2', float(len(base['C2'])), float(c2), .5, '{:.0f}')
        gate(f'{bench} funnel C3', float(len(base['C3'])), float(c3), .5, '{:.0f}')
        stored_C2 = set(D['rn'])
        stored_C3 = {q for q in D['rn'] if D['rn'][q] <= C3T + EPS}
        same = (base['C2'] == stored_C2 and base['C3'] == stored_C3)
        print(f"  {'PASS' if same else 'FAIL'}  {bench} recomputed C2/C3 "
              f"set-identical to the released flags")
        if not same:
            _FAIL.append((f'{bench} C2/C3 set identity', False, True))

    print('\n== Table S9 -- leave-one-reference-model-out ==')
    print(f"  {'bench':5} {'ensemble':16} {'C2 N':>6} {'C3 N':>6} {'med rho':>8} {'C3 SG':>7}")
    for bench in ('NExT', 'QVH'):
        D, sg, _ = store[bench]
        for name, subset in VARIANTS:
            thr = 1 / 3 if len(subset) == 3 else 1 / 2
            S = sets_for(D, subset, thr)
            m = mean_sg(sg, [g[0] for g in B.GROUNDERS[bench]], S['C3'])
            print(f"  {bench:5} {name:16} {len(S['C2']):6d} {len(S['C3']):6d} "
                  f"{S['med']:8.3f} {m:7.3f}")
            w = TABLE_S9[(bench, name)]
            gate(f'S9 {bench} {name} C2 N', float(len(S['C2'])), float(w[0]), .5, '{:.0f}')
            gate(f'S9 {bench} {name} C3 N', float(len(S['C3'])), float(w[1]), .5, '{:.0f}')
            gate(f'S9 {bench} {name} median rho', S['med'], w[2], .005)
            gate(f'S9 {bench} {name} C3 mean SG', m, w[3], .0005)

    print('\n== supp. A.1 -- Qwen2.5-VL on C3 defined without its own reference outputs ==')
    for bench in ('NExT', 'QVH'):
        D, sg, _ = store[bench]
        # sorted: the bootstrap percentiles depend on the input order
        ids = sorted(sets_for(D, ['i', 'v'], 1 / 2)['C3'])
        vals = [sg['Qwen'][v] for v in ids if v in sg['Qwen']]
        m = float(np.mean(vals))
        lo, hi = boot_ci(vals)
        print(f'  {bench:5} N={len(vals):4d}  rng=default_rng({SEED})  B={B_BOOT}')
        w = QWEN_LOO[bench]
        gate(f'A.1 {bench} Qwen C3 SG', m, w[0], .0005)
        gate(f'A.1 {bench} Qwen C3 CI lower', lo, w[1], .0005)
        gate(f'A.1 {bench} Qwen C3 CI upper', hi, w[2], .0005)

    print()
    if _FAIL:
        print(f'*** {len(_FAIL)} leave-one-out gate(s) FAILED ***')
        for tag, got, want in _FAIL:
            print(f'    {tag}: got {got!r}, paper {want!r}')
        raise SystemExit(1)
    print('leave-one-reference-model-out OK (Table S9 and supp. A.1)')


if __name__ == '__main__':
    main()
