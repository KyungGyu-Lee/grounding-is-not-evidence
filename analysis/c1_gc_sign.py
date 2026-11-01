"""C1 — grounded-correct answers split by the sign of SG, at three decimals.

Paper Table 8 (main text, C3) and Table S4 (supplement, C2) report, among
answers that are *conventionally* grounded-correct

    a_full == ground truth   AND   IoP(pred, gt) >= 0.5,

the share with SG > 0 (SGA), SG = 0 (UGA_{=0}) and SG < 0 (UGA_{<0}). The
NExT-GQA block is currently printed at two decimals, which makes the three
shares sum to 1.010 for VideoMind/C3. This script recomputes both blocks the
way the QVH-ME block was computed, at three decimals.

READS STORED OUTPUTS ONLY. No model is re-run, no existing file is modified.
`build_paper_numbers.load_samples` and `set_masks` are imported and reused
verbatim, so `correct`, `iop` and `SG` are bit-identical to the paper — in
particular the IoP fallback for an empty or unparsed predicted window is
whatever step 1 stored (NaN there fails `iop >= 0.5` and the question simply
is not grounded-correct).

`build_paper_numbers` chdir()s to the repository root at import and reads the
stored outputs (NExT-GQA = outputs/nextgqa, QVH-ME = outputs/qvhme); each
benchmark is still loaded in its own child process.

NAMING: build_paper_numbers uses the paper's names throughout:
    C1 = prior-exclusion (b < 1), C2 = answerability (u - b >= 1/3),
    C3 = evidence-specificity (rho <= 0.4).  CODE_SET below is the identity
map, kept so the paper-side and figdata-side names stay visibly paired.

Usage:
    python analysis/c1_gc_sign.py
    python analysis/c1_gc_sign.py --dump NExT   # child
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = f'{ROOT}/analysis/C1_gc_sign'
CACHE = f'{OUTDIR}/_cache'

EPS = 1e-9                      # |SG| < EPS counts as SG = 0
IOP_GC = 0.5

CODE_SET = {'C2': 'C2', 'C3': 'C3'}          # figdata now uses paper names
BENCH_LABEL = {'NExT': 'NExT-GQA', 'QVH': 'QVH-ME'}
DISP = {'Qwen': 'Qwen2.5-VL', 'VideoMind': 'VideoMind',
        'GVLLM': 'Grounded-VideoLLM', 'GroundVTS': 'GroundVTS',
        'VTimeLLM': 'VTimeLLM', 'TimeChat': 'TimeChat',
        'VideoChat-R1': 'VideoChat-R1'}
DAG = {'VTimeLLM': r'$^{\dagger}$', 'TimeChat': r'$^{\dagger}$',
       'VideoMind': r'$^{\ddagger}$'}

# every evaluated grounder, for gc_sign.csv
MODELS = {'NExT': ['Qwen', 'VideoMind', 'GVLLM', 'GroundVTS', 'VTimeLLM'],
          'QVH': ['VideoChat-R1', 'GVLLM', 'Qwen', 'VideoMind', 'TimeChat',
                  'GroundVTS', 'VTimeLLM']}
# row order of the two tables (main text drops VTimeLLM/TimeChat)
ROWS_C3 = {'NExT': ['Qwen', 'VideoMind', 'GVLLM', 'GroundVTS'],
           'QVH': ['VideoChat-R1', 'GVLLM', 'Qwen', 'VideoMind', 'GroundVTS']}
ROWS_C2 = {'NExT': ['Qwen', 'VideoMind', 'GVLLM', 'GroundVTS', 'VTimeLLM'],
           'QVH': ['VideoChat-R1', 'GVLLM', 'Qwen', 'VideoMind', 'TimeChat',
                   'GroundVTS', 'VTimeLLM']}

# ---- reproduction targets -------------------------------------------------
# N_gc, every (bench, set, model) the paper prints
N_GC_EXPECT = {
    ('NExT', 'C3'): {'Qwen': 100, 'VideoMind': 57, 'GVLLM': 50, 'GroundVTS': 52},
    ('NExT', 'C2'): {'Qwen': 397, 'VideoMind': 307, 'GVLLM': 197,
                     'GroundVTS': 281, 'VTimeLLM': 114},
    ('QVH', 'C3'): {'VideoChat-R1': 160, 'GVLLM': 97, 'Qwen': 159,
                    'VideoMind': 112, 'GroundVTS': 61},
    ('QVH', 'C2'): {'VideoChat-R1': 283, 'GVLLM': 185, 'Qwen': 283,
                    'VideoMind': 215, 'TimeChat': 47, 'GroundVTS': 103,
                    'VTimeLLM': 35},
}
# current paper proportions (tables_v2.tex): (p_pos, p_zero, p_neg)
PAPER_P = {
    ('NExT', 'C2'): {'Qwen': (.30, .62, .07), 'VideoMind': (.29, .61, .10),
                     'GVLLM': (.28, .60, .11), 'GroundVTS': (.27, .58, .15),
                     'VTimeLLM': (.16, .77, .07)},
    ('NExT', 'C3'): {'Qwen': (.68, .24, .08), 'VideoMind': (.60, .32, .09),
                     'GVLLM': (.52, .44, .04), 'GroundVTS': (.48, .27, .25),
                     'VTimeLLM': (.25, .75, .00)},
    ('QVH', 'C2'): {'VideoChat-R1': (.714, .223, .064), 'GVLLM': (.703, .216, .081),
                    'Qwen': (.682, .233, .085), 'VideoMind': (.674, .260, .065),
                    'TimeChat': (.617, .298, .085), 'GroundVTS': (.583, .301, .117),
                    'VTimeLLM': (.143, .829, .029)},
    ('QVH', 'C3'): {'VideoChat-R1': (.863, .100, .037), 'GVLLM': (.804, .134, .062),
                    'Qwen': (.799, .126, .075), 'VideoMind': (.786, .161, .054),
                    'TimeChat': (.682, .273, .045), 'GroundVTS': (.639, .295, .066),
                    'VTimeLLM': (.250, .750, .000)},
}
# §6.3 sentence, as currently written
PAPER_UGA_RANGE = {'NExT': (0.32, 0.52), 'QVH': (0.14, 0.36)}


# --------------------------------------------------------------- child: dump
def dump(bench):
    sys.path.insert(0, f'{ROOT}/paper')
    import build_paper_numbers as bpn          # chdir()s to the repo root on import
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


# ------------------------------------------------------------------ compute
def compute():
    rows = []
    for bench in ('NExT', 'QVH'):
        df = pd.read_csv(f'{CACHE}/samples_{bench}.csv')
        masks = json.load(open(f'{CACHE}/masks_{bench}.json'))
        for pset, cset in CODE_SET.items():
            ids = set(masks[cset])
            for m in MODELS[bench]:
                sub = df[(df.model == m) & (df.vqid.isin(ids))]
                gc = sub[(sub.correct == 1) & (sub.iop >= IOP_GC)]
                sg = gc.SG.values.astype(float)
                n = len(sg)
                n_zero = int((np.abs(sg) < EPS).sum())
                n_pos = int((sg >= EPS).sum())
                n_neg = int((sg <= -EPS).sum())
                assert n_pos + n_zero + n_neg == n, (bench, pset, m)
                rows.append(dict(
                    dataset=BENCH_LABEL[bench], bench=bench, model=m,
                    model_display=DISP[m], set=pset, N_gc=n,
                    n_pos=n_pos, n_zero=n_zero, n_neg=n_neg,
                    p_pos=round(n_pos / n, 3) if n else np.nan,
                    p_zero=round(n_zero / n, 3) if n else np.nan,
                    p_neg=round(n_neg / n, 3) if n else np.nan))
    return pd.DataFrame(rows)


def verify(df):
    """Hard gate. Returns (ok, lines)."""
    L, ok = [], True
    idx = df.set_index(['bench', 'set', 'model'])

    L.append('### N_gc vs the paper')
    for (b, s), exp in N_GC_EXPECT.items():
        for m, want in exp.items():
            got = int(idx.loc[(b, s, m), 'N_gc'])
            good = got == want
            ok &= good
            L.append(f'  {BENCH_LABEL[b]:9} {s}  {DISP[m]:18} N_gc {got:4d} '
                     f'(paper {want:4d})  {"OK" if good else "MISMATCH"}')

    L.append('')
    L.append('### QVH-ME proportions must match the paper to 3 decimals')
    for s in ('C2', 'C3'):
        for m in MODELS['QVH']:
            got = tuple(float(idx.loc[('QVH', s, m), c])
                        for c in ('p_pos', 'p_zero', 'p_neg'))
            want = PAPER_P[('QVH', s)][m]
            good = all(abs(g - w) < 5e-4 for g, w in zip(got, want))
            ok &= good
            L.append(f'  QVH-ME    {s}  {DISP[m]:18} '
                     f'{got[0]:.3f}/{got[1]:.3f}/{got[2]:.3f} '
                     f'(paper {want[0]:.3f}/{want[1]:.3f}/{want[2]:.3f})  '
                     f'{"OK" if good else "MISMATCH"}')

    L.append('')
    L.append('### NExT-GQA proportions must match the paper when rounded to 2 decimals')
    for s in ('C2', 'C3'):
        for m in MODELS['NExT']:
            got = tuple(float(idx.loc[('NExT', s, m), c])
                        for c in ('p_pos', 'p_zero', 'p_neg'))
            want = PAPER_P[('NExT', s)][m]
            good = all(abs(round(g, 2) - w) < 5e-9 for g, w in zip(got, want))
            ok &= good
            L.append(f'  NExT-GQA  {s}  {DISP[m]:18} '
                     f'{got[0]:.3f}/{got[1]:.3f}/{got[2]:.3f} '
                     f'-> {round(got[0],2):.2f}/{round(got[1],2):.2f}/{round(got[2],2):.2f} '
                     f'(paper {want[0]:.2f}/{want[1]:.2f}/{want[2]:.2f})  '
                     f'{"OK" if good else "MISMATCH"}')

    L.append('')
    L.append('### three shares sum to 1.000 +/- .001')
    worst = 0.0
    for _, r in df.iterrows():
        tot = r.p_pos + r.p_zero + r.p_neg
        worst = max(worst, abs(tot - 1.0))
        if abs(tot - 1.0) > 1e-3 + 1e-9:          # spec: 1.000 +/- .001, inclusive
            ok = False
            L.append(f'  SUM FAIL {r.dataset} {r.set} {r.model_display}: {tot:.3f}')
    L.append(f'  worst |sum - 1.000| over {len(df)} cells = {worst:.3f}')
    old_worst = max(abs(sum(v) - 1.0)
                    for d in PAPER_P.values() for v in d.values())
    L.append(f'  (same quantity in the current paper tables: {old_worst:.3f})')
    return ok, L


# -------------------------------------------------------------------- latex
def p3(x):
    return f'{x:.3f}'.replace('0.', '.', 1)


def block(df, bench, pset, rows):
    idx = df.set_index(['bench', 'set', 'model'])
    out = [r'\multicolumn{5}{@{}l}{\emph{' + BENCH_LABEL[bench]
           + ', ' + pset.replace('C', r'$\mathcal{C}_') + '$}} \\\\']
    for m in rows:
        r = idx.loc[(bench, pset, m)]
        out.append(f'{DISP[m]}{DAG.get(m, "")} & {int(r.N_gc)} & '
                   f'{p3(r.p_pos)} & {p3(r.p_zero)} & {p3(r.p_neg)} \\\\')
    return out


def write_tex(df):
    head = (r'\begin{tabular}{@{}lrrrr@{}}' '\n' r'\toprule' '\n'
            r'Model & $N_{gc}$ & SG\,$>$\,0 & SG\,$=$\,0 & SG\,$<$\,0 \\'
            '\n' r'\midrule')
    cap_common = (
        r'Conventionally grounded-correct answers ($a_{\text{full}}$ equals the '
        r'ground-truth answer and $\mathrm{IoP}(\hat{E}, E^{*}) \geq 0.5$), split by '
        r'the sign of the Support Gain: SG\,$>$\,0 is SGA, SG\,$=$\,0 is '
        r'$\textsc{uga}_{=0}$ (the predicted window and a random window are '
        r'indistinguishable) and SG\,$<$\,0 is $\textsc{uga}_{<0}$ (the predicted '
        r'window is worse). Shares are of $N_{gc}$, each rounded independently to '
        r'three decimals, so a row may sum to $0.999$ or $1.001$; '
        r'$\mathrm{Ret}(\text{rand})$ averages five draws, so SG is a multiple of '
        r'$0.2$. Daggers as in Tab.~\ref{tab:models}.')

    c3 = ['% C1: generated by analysis/camera_ready/c1_gc_sign.py -- do not edit by hand.',
          r'\begin{table}[t]', r'\centering\small',
          r'\setlength{\tabcolsep}{4pt}', head]
    for b in ('NExT', 'QVH'):
        if b == 'QVH':
            c3.append(r'\midrule')
        c3 += block(df, b, 'C3', ROWS_C3[b])
    c3 += [r'\bottomrule', r'\end{tabular}',
           r'\caption{\textbf{Grounded answers without measurable support '
           r'($\mathcal{C}_3$).} ' + cap_common +
           r' VTimeLLM and TimeChat are omitted: their positive control is not '
           r'measurable (App.~\ref{sec:extra_tables} gives every model on '
           r'$\mathcal{C}_2$). Several $\mathcal{C}_3$ cells rest on fewer than '
           r'60 answers.}',
           r'\label{tab:gc_sign_c3}', r'\end{table}']
    open(f'{OUTDIR}/table_gc_c3.tex', 'w').write('\n'.join(c3) + '\n')

    c2 = ['% C1: generated by analysis/camera_ready/c1_gc_sign.py -- do not edit by hand.',
          r'\begin{table}[t]', r'\centering\footnotesize',
          r'\setlength{\tabcolsep}{4pt}', head]
    for b in ('NExT', 'QVH'):
        if b == 'QVH':
            c2.append(r'\midrule')
        c2 += block(df, b, 'C2', ROWS_C2[b])
    c2 += [r'\bottomrule', r'\end{tabular}',
           r'\caption{\textbf{Grounded answers without measurable support '
           r'($\mathcal{C}_2$), all models.} ' + cap_common +
           r' This is the $\mathcal{C}_2$ counterpart of '
           r'Tab.~\ref{tab:gc_sign_c3} and additionally lists VTimeLLM and '
           r'TimeChat.}',
           r'\label{tab:gc_sign_c2}', r'\end{table}']
    open(f'{OUTDIR}/table_gc_c2.tex', 'w').write('\n'.join(c2) + '\n')


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
    df.to_csv(f'{OUTDIR}/gc_sign.csv', index=False)
    write_tex(df)
    print('\nREPRODUCTION: PASS')
    print(f'wrote {OUTDIR}/gc_sign.csv, table_gc_c3.tex, table_gc_c2.tex')

    # §6.3 UGA range on C3, grounders with a measurable positive control
    print('\n### Sec 6.3 UGA share on C3 (VTimeLLM/TimeChat excluded)')
    idx = df.set_index(['bench', 'set', 'model'])
    for b in ('NExT', 'QVH'):
        vals = {m: float(idx.loc[(b, 'C3', m), 'p_zero']
                         + idx.loc[(b, 'C3', m), 'p_neg']) for m in ROWS_C3[b]}
        lo, hi = min(vals.values()), max(vals.values())
        plo, phi = PAPER_UGA_RANGE[b]
        print(f'  {BENCH_LABEL[b]:9} ' +
              '  '.join(f'{DISP[m]} {v:.3f}' for m, v in vals.items()))
        print(f'  {"":9} new range {lo * 100:.1f}-{hi * 100:.1f}%  '
              f'(paper {plo * 100:.0f}-{phi * 100:.0f}%)  '
              f'rounded {round(lo * 100):.0f}-{round(hi * 100):.0f}%')
    return df


if __name__ == '__main__':
    main()
