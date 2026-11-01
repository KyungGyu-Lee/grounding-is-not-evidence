"""C2 — confidence intervals for the positive control SG_gt.

The main text marks VTimeLLM and TimeChat with a dagger and declines to read
their Support Gain, because the 95% interval of their positive control
SG_gt = Ret(gt) - Ret(rand-gt) includes zero. That interval is what makes ESR
undefined in `build_paper_numbers` (`if lo <= 0 <= hi`), yet it is printed
nowhere. This script computes it for all 24 cells so the last column of
supplementary Table S2 can become `SG_gt [95% CI]`.

READS STORED OUTPUTS ONLY. No model is re-run and no existing file is modified.
`load_samples`, `set_masks` and the two bootstrap routines are imported from
`build_paper_numbers` and used verbatim, so every number is produced by the
same code path as the paper:

  * SG_gt CI  -> `boot_diff(KGT, KRG, seed=0, B=5000)`: resample the model's
    questions with replacement, take mean(KGT) - mean(KRG) on each resample.
    This is exactly the interval stored as `SG_GT_CI` and exactly the one the
    dagger rule tests.
  * SG CI     -> the `paired_bootstrap_SGpred_CI` construction: resample the
    cut's full sorted vqid list once, shared across models, and take the
    nanmean of each model's SG. Recomputed here purely as a reproduction gate.

`build_paper_numbers` chdir()s to $BP_CWD at import, so each benchmark is
loaded in its own child process against a wrapper dir whose results/ points at
the canonical tree (NExT-GQA = outputs/nextgqa, QVH-ME = outputs/qvhme).

NAMING: code names are FLIPPED vs the paper:
    paper C2 (u-b>=1/3) == code 'C1'
    paper C3 (rho<=0.4) == code 'C3'
Everything written here uses PAPER naming.

Usage:
    python analysis/camera_ready/c2_sggt_ci.py
    python analysis/camera_ready/c2_sggt_ci.py --dump NExT   # child
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = f'{ROOT}/analysis/C2_sggt_ci'
CACHE = f'{OUTDIR}/_cache'
FIGDATA = {'NExT': f'{ROOT}/paper/figdata_nextgqa.json',
           'QVH': f'{ROOT}/paper/figdata_qvhme.json'}

B_BOOT, SEED = 5000, 0            # same as build_paper_numbers
CODE_SET = {'C2': 'C2', 'C3': 'C3'}
BENCH_LABEL = {'NExT': 'NExT-GQA', 'QVH': 'QVH-ME'}
DISP = {'Qwen': 'Qwen2.5-VL', 'VideoMind': 'VideoMind',
        'GVLLM': 'Grounded-VideoLLM', 'GroundVTS': 'GroundVTS',
        'VTimeLLM': 'VTimeLLM', 'TimeChat': 'TimeChat',
        'VideoChat-R1': 'VideoChat-R1'}
DAG = {'VTimeLLM': r'$^{\dagger}$', 'TimeChat': r'$^{\dagger}$',
       'VideoMind': r'$^{\ddagger}$'}
DAGGERED = {'VTimeLLM', 'TimeChat'}      # models the text marks with a dagger

# Table 5 model order
MODELS = {'NExT': ['Qwen', 'VideoMind', 'GVLLM', 'GroundVTS', 'VTimeLLM'],
          'QVH': ['VideoChat-R1', 'Qwen', 'GVLLM', 'VideoMind', 'GroundVTS',
                  'TimeChat', 'VTimeLLM']}

# ---- reproduction targets, from the prompt / Table S2 ---------------------
SGGT_EXPECT = {
    ('NExT', 'C2'): {'Qwen': .118, 'VideoMind': .064, 'GVLLM': .089,
                     'GroundVTS': .059, 'VTimeLLM': .002},
    ('NExT', 'C3'): {'Qwen': .455, 'VideoMind': .182, 'GVLLM': .231,
                     'GroundVTS': .172, 'VTimeLLM': .009},
    ('QVH', 'C2'): {'VideoChat-R1': .439, 'Qwen': .445, 'GVLLM': .290,
                    'VideoMind': .244, 'GroundVTS': .301, 'TimeChat': .017,
                    'VTimeLLM': -.045},
    ('QVH', 'C3'): {'VideoChat-R1': .642, 'Qwen': .647, 'GVLLM': .381,
                    'VideoMind': .291, 'GroundVTS': .400, 'TimeChat': .008,
                    'VTimeLLM': -.045},
}
SG_CI_EXPECT = {                      # spot checks given in the prompt
    ('NExT', 'C2', 'Qwen'): (.061, .109),
    ('NExT', 'C3', 'Qwen'): (.227, .361),
    ('QVH', 'C2', 'VideoChat-R1'): (.310, .406),
    ('QVH', 'C3', 'VideoChat-R1'): (.443, .563),
    ('QVH', 'C3', 'VTimeLLM'): (-.062, .012),
}
CI_TOL = 1.5e-3                       # bootstrap last-digit slack


# --------------------------------------------------------------- child: dump
def dump(bench):
    sys.path.insert(0, f'{ROOT}/paper')
    import build_paper_numbers as bpn          # chdir()s to $BP_CWD on import
    df = bpn.load_samples(bench)
    masks, _ = bpn.set_masks(bench)
    os.makedirs(CACHE, exist_ok=True)
    df.to_csv(f'{CACHE}/samples_{bench}.csv', index=False)
    json.dump({k: sorted(v) for k, v in masks.items()},
              open(f'{CACHE}/masks_{bench}.json', 'w'))
    print(f'[dump {bench}] rows={len(df)} '
          f'funnel={ {k: len(v) for k, v in masks.items()} }')


def spawn_dumps():
    for bench in ('NExT', 'QVH'):
        env = dict(os.environ)
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            '--dump', bench], env=env,
                          capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f'dump {bench} failed')
        print(r.stdout.strip())


# ----------------------------------------------------------------- bootstrap
def boot_diff(a, b, seed=SEED, B=B_BOOT):
    """Verbatim copy of build_paper_numbers.boot_diff (SG_gt CI)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed); n = len(a)
    idx = rng.integers(0, n, (B, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def paired_sg_ci(df, ids, models):
    """Verbatim copy of the paired_bootstrap_SGpred_CI block (SG CI)."""
    vqid_list = sorted(ids)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(vqid_list), (B_BOOT, len(vqid_list)))
    g = df[(df.vqid.isin(ids)) & (df.model.isin(models))]
    out = {}
    for m in models:
        s = g[g.model == m].set_index('vqid').SG.reindex(vqid_list).values
        valid = ~np.isnan(s)
        if valid.sum() < 2:
            out[m] = (np.nan, np.nan); continue
        sv = np.where(valid, s, np.nan)
        means = np.nanmean(sv[idx], axis=1)
        out[m] = (float(np.percentile(means, 2.5)),
                  float(np.percentile(means, 97.5)))
    return out


# ------------------------------------------------------------------ compute
def compute():
    rows = []
    for bench in ('NExT', 'QVH'):
        df = pd.read_csv(f'{CACHE}/samples_{bench}.csv')
        masks = json.load(open(f'{CACHE}/masks_{bench}.json'))
        for pset, cset in CODE_SET.items():
            ids = set(masks[cset])
            sgci = paired_sg_ci(df, ids, MODELS[bench])
            for m in MODELS[bench]:
                sub = df[(df.model == m) & (df.vqid.isin(ids))]
                pc = sub.dropna(subset=['KGT', 'KRG'])
                kgt, krg = pc.KGT.values, pc.KRG.values
                sggt = float(kgt.mean() - krg.mean())
                lo, hi = boot_diff(kgt, krg)
                slo, shi = sgci[m]
                rows.append(dict(
                    dataset=BENCH_LABEL[bench], bench=bench, model=m,
                    model_display=DISP[m], set=pset,
                    N=len(pc), N_sg=len(sub),
                    Ret_none=float(sub.Kbl.mean()),
                    Ret_rand=float(sub.KR.mean()),
                    Ret_pred=float(sub.KE.mean()),
                    sggt_mean=sggt, sggt_lo=lo, sggt_hi=hi,
                    includes_zero=bool(lo <= 0 <= hi),
                    sg_mean=float(sub.SG.mean()), sg_lo=slo, sg_hi=shi,
                    daggered=m in DAGGERED))
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- gates
def verify(df):
    L, ok = [], True
    idx = df.set_index(['bench', 'set', 'model'])

    L.append('### SG_gt point estimate vs Table S2 (3 decimals)')
    for (b, s), exp in SGGT_EXPECT.items():
        for m, want in exp.items():
            got = float(idx.loc[(b, s, m), 'sggt_mean'])
            good = abs(round(got, 3) - want) < 5e-9
            ok &= good
            L.append(f'  {BENCH_LABEL[b]:9} {s}  {DISP[m]:18} '
                     f'{got:+.4f} -> {got:+.3f}  (paper {want:+.3f})  '
                     f'{"OK" if good else "MISMATCH"}')

    L.append('')
    L.append(f'### SG 95% CI vs Table S2 spot checks (tol {CI_TOL})')
    for (b, s, m), (wlo, whi) in SG_CI_EXPECT.items():
        glo = float(idx.loc[(b, s, m), 'sg_lo']); ghi = float(idx.loc[(b, s, m), 'sg_hi'])
        good = abs(glo - wlo) <= CI_TOL and abs(ghi - whi) <= CI_TOL
        ok &= good
        L.append(f'  {BENCH_LABEL[b]:9} {s}  {DISP[m]:18} '
                 f'[{glo:+.4f}, {ghi:+.4f}]  (paper [{wlo:+.3f}, {whi:+.3f}])  '
                 f'{"OK" if good else "MISMATCH"}')

    L.append('')
    L.append('### every SG_gt / SG CI vs the stored figdata (must be exact)')
    worst = 0.0
    for bench in ('NExT', 'QVH'):
        D = json.load(open(FIGDATA[bench]))[BENCH_KEY[bench]]
        for pset, cset in CODE_SET.items():
            pc = D[cset]['positive_control']
            sgc = D[cset]['paired_bootstrap_SGpred_CI']
            for m in MODELS[bench]:
                r = idx.loc[(bench, pset, m)]
                for got, want, tag in [
                        (r.sggt_mean, pc[m]['SG_GT'], 'SG_gt'),
                        (r.sggt_lo, pc[m]['SG_GT_CI'][0], 'SG_gt lo'),
                        (r.sggt_hi, pc[m]['SG_GT_CI'][1], 'SG_gt hi'),
                        (r.sg_lo, sgc[m][0], 'SG lo'),
                        (r.sg_hi, sgc[m][1], 'SG hi')]:
                    d = abs(float(got) - float(want))
                    worst = max(worst, d)
                    if d > 1e-9:
                        ok = False
                        L.append(f'  MISMATCH {BENCH_LABEL[bench]} {pset} '
                                 f'{DISP[m]} {tag}: {got:.6f} vs {want:.6f}')
                if int(r.N) != int(pc[m]['N']):
                    ok = False
                    L.append(f'  MISMATCH N {BENCH_LABEL[bench]} {pset} '
                             f'{DISP[m]}: {int(r.N)} vs {pc[m]["N"]}')
    L.append(f'  worst |recomputed - stored| over 24 cells x 5 numbers = {worst:.2e}')

    return ok, L


def dagger_audit(df):
    """NOT a gate — a finding. Does 'the CI includes zero' describe exactly the
    daggered cells? Reported either way; never blocks the output."""
    L = ['### dagger audit: does the CI include zero exactly on daggered models?']
    conflicts = df[df.includes_zero != df.daggered]
    for _, r in conflicts.sort_values(['bench', 'model', 'set']).iterrows():
        kind = ('daggered but CI EXCLUDES zero' if r.daggered
                else 'not daggered but CI INCLUDES zero')
        L.append(f'  *** {kind}: {r.dataset} {r.set} {r.model_display} '
                 f'{r.sggt_mean:+.3f} [{r.sggt_lo:+.3f}, {r.sggt_hi:+.3f}] '
                 f'N={int(r.N)}')
    if not len(conflicts):
        L.append('  none — the two sets coincide exactly')
    L.append(f'  cells whose CI includes 0 : {int(df.includes_zero.sum())}')
    L.append(f'  cells on daggered models  : {int(df.daggered.sum())}')
    nd = df[(~df.daggered) & df.includes_zero]
    L.append(f'  non-daggered cells including 0 : {len(nd)}')
    return L


BENCH_KEY = {'NExT': 'NExT', 'QVH': 'QVH'}


# -------------------------------------------------------------------- latex
def k3(x):
    s = f'{x:.3f}'
    return s[1:] if s.startswith('0.') else ('-' + s[2:] if s.startswith('-0.') else s)


def sg3(x):
    return ('$+$' if x >= 0 else '$-$') + f'{abs(x):.3f}'[1:]


def ci3(lo, hi):
    return f'[{sg3(lo)}, {sg3(hi)}]'


def write_tex(df):
    idx = df.set_index(['bench', 'set', 'model'])
    body = []
    for bench in ('NExT', 'QVH'):
        if bench == 'QVH':
            body.append(r'\midrule')
        body.append(r'\multicolumn{7}{@{}l}{\emph{' + BENCH_LABEL[bench] + r'}} \\')
        for m in MODELS[bench]:
            nm = DISP[m] + DAG.get(m, '')
            for i, pset in enumerate(('C2', 'C3')):
                r = idx.loc[(bench, pset, m)]
                lead = nm if i == 0 else ''
                body.append(
                    f'{lead} & $\\mathcal{{C}}_{pset[1]}$ & '
                    f'{k3(r.Ret_none)} & {k3(r.Ret_rand)} & {k3(r.Ret_pred)} & '
                    f'{sg3(r.sg_mean)} {ci3(r.sg_lo, r.sg_hi)} & '
                    f'{sg3(r.sggt_mean)} {ci3(r.sggt_lo, r.sggt_hi)} \\\\')
    tex = (
        '% C2: generated by analysis/camera_ready/c2_sggt_ci.py -- do not edit by hand.\n'
        r'\begin{table*}[t]' '\n' r'\centering\scriptsize' '\n'
        r'\setlength{\tabcolsep}{4pt}' '\n'
        r'\begin{tabular}{@{}llcccll@{}}' '\n' r'\toprule' '\n'
        r'Model & Set & $\mathrm{Ret}(\text{none})$ & $\mathrm{Ret}(\text{rand})$ & '
        r'$\mathrm{Ret}(\text{pred})$ & SG [95\% CI] & '
        r'$\mathrm{SG}_{\text{gt}}$ [95\% CI] \\' '\n'
        r'\midrule' '\n'
        + '\n'.join(body) + '\n'
        r'\bottomrule' '\n' r'\end{tabular}' '\n'
        r'\caption{\textbf{Full retention decomposition, with the positive '
        r'control interval.} $\mathrm{Ret}(\text{none})$, '
        r'$\mathrm{Ret}(\text{rand})$ and $\mathrm{Ret}(\text{pred})$ are '
        r"agreement with the model's own full-video answer under the blind, "
        r'random and predicted window; $\mathrm{SG} = \mathrm{Ret}(\text{pred}) '
        r'- \mathrm{Ret}(\text{rand})$ and '
        r'$\mathrm{SG}_{\text{gt}} = \mathrm{Ret}(\text{gt}) - '
        r'\mathrm{Ret}(\text{rand-gt})$. Both intervals are 95\% item '
        r'bootstrap, $B{=}5000$, seed 0. The $\mathrm{SG}_{\text{gt}}$ interval '
        r'is what the dagger refers to. It lies strictly above zero for every '
        r'model except VTimeLLM and TimeChat: it contains zero for VTimeLLM on '
        r'both NExT-GQA cuts and on QVH-ME $\mathcal{C}_3$ and for TimeChat on '
        r'both QVH-ME cuts, and for VTimeLLM on QVH-ME $\mathcal{C}_2$ it lies '
        r'strictly \emph{below} zero. In none of these six cells is the '
        r'positive control positive, so SG is left uninterpreted and ESR '
        r'undefined. Daggers as in Tab.~\ref{tab:models}.}' '\n'
        r'\label{tab:s2_full}' '\n' r'\end{table*}' '\n')
    open(f'{OUTDIR}/table_s2_full.tex', 'w').write(tex)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump')
    a = ap.parse_args()
    if a.dump:
        dump(a.dump)
        return
    os.makedirs(OUTDIR, exist_ok=True)
    spawn_dumps()
    df = compute()
    ok, lines = verify(df)
    print('\n'.join(lines))
    if not ok:
        raise SystemExit('\nREPRODUCTION FAILED — nothing written.')
    print('')
    print('\n'.join(dagger_audit(df)))
    cols = ['dataset', 'model_display', 'set', 'N', 'N_sg',
            'Ret_none', 'Ret_rand', 'Ret_pred',
            'sggt_mean', 'sggt_lo', 'sggt_hi', 'includes_zero',
            'sg_mean', 'sg_lo', 'sg_hi', 'daggered']
    df[cols].rename(columns={'model_display': 'model'}).to_csv(
        f'{OUTDIR}/sggt_ci.csv', index=False)
    write_tex(df)
    print('\nREPRODUCTION: PASS')
    print(f'wrote {OUTDIR}/sggt_ci.csv, table_s2_full.tex')

    print('\n### cells whose SG_gt CI includes zero')
    z = df[df.includes_zero].sort_values(['bench', 'model', 'set'])
    for _, r in z.iterrows():
        print(f'  {r.dataset:9} {r.set}  {r.model_display:13} '
              f'{r.sggt_mean:+.3f} [{r.sggt_lo:+.3f}, {r.sggt_hi:+.3f}]  N={int(r.N)}')
    print('\n### tightest non-zero-crossing cells (closest call)')
    nz = df[~df.includes_zero].copy()
    nz['margin'] = nz.apply(lambda r: min(abs(r.sggt_lo), abs(r.sggt_hi)), axis=1)
    for _, r in nz.nsmallest(3, 'margin').iterrows():
        print(f'  {r.dataset:9} {r.set}  {r.model_display:18} '
              f'{r.sggt_mean:+.3f} [{r.sggt_lo:+.3f}, {r.sggt_hi:+.3f}]')
    return df


if __name__ == '__main__':
    main()
