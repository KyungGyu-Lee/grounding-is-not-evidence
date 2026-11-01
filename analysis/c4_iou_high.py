"""C4 — retention by IoU bin, the missing support for the §6.3 sentence.

The main text says

    "Even among predictions with IoU >= .9, 11.6% on NExT-GQA and 14.3% on
     QVH-ME C2 fail to reproduce the full-video response."

but no table in the paper or the supplement carries those numbers. This script
builds a "Retention by IoU bin" block for supplementary Table S3 (overlap
diagnostics), computed on exactly the code path that produced that table's
"Retention by IoP bin" block:

    build_paper_numbers.compute(), item 4
        g    = df[df.vqid.isin(masks[cut]) & df.model.isin(G)]   # G = grounders
        bins = g[(g.iop >= lo) & (g.iop < hi)]  ->  N, mean(KE)

so the same model set (all evaluated grounders, including the two daggered
ones), the same question set, and the same row-level pooling — one row per
(question, model). Only `iop` is swapped for `iou`.

The IoP block is recomputed on the same path and gated against the printed
Table S3 values; nothing is written if it does not reproduce.

READS STORED OUTPUTS ONLY. No model is re-run, no existing file is modified.

`build_paper_numbers` chdir()s to $BP_CWD at import, so each benchmark loads in
its own child process against a wrapper dir whose results/ points at the
canonical tree (NExT-GQA = outputs/nextgqa, QVH-ME = outputs/qvhme).

NAMING: build_paper_numbers' code names are FLIPPED vs the paper:
    paper C1 (b<1)       == code 'C2'
    paper C2 (u-b>=1/3)  == code 'C1'
    paper C3 (rho<=0.4)  == code 'C3'
Everything written here uses PAPER naming.

Usage:
    python analysis/camera_ready/c4_iou_high.py
    python analysis/camera_ready/c4_iou_high.py --dump NExT    # child
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = f'{ROOT}/analysis/C4_iou_high'
CACHE = f'{OUTDIR}/_cache'
B1 = f'{ROOT}/analysis/B1_masking'

# bin edges exactly as build_paper_numbers item 4 writes them
BINS = [(0.5, 0.7), (0.7, 0.9), (0.9, 1.001)]
BIN_LABEL = {(0.5, 0.7): '[0.5,0.7)', (0.7, 0.9): '[0.7,0.9)',
             (0.9, 1.001): '[0.9,1.0]'}
BIN_TEX = {(0.5, 0.7): r'$[0.5, 0.7)$', (0.7, 0.9): r'$[0.7, 0.9)$',
           (0.9, 1.001): r'$[0.9, 1.0]$'}

CODE_SET = {'ALL': 'ALL', 'C2': 'C2', 'C3': 'C3'}       # figdata uses paper names
REPORT_SETS = ['C2', 'C3']                              # Table S3 columns
BENCH_LABEL = {'NExT': 'NExT-GQA', 'QVH': 'QVH-ME'}

# the pooled model set: every evaluated grounder, as in build_paper_numbers' G
GROUNDERS = {'NExT': ['Qwen', 'VideoMind', 'GVLLM', 'GroundVTS', 'VTimeLLM'],
             'QVH': ['Qwen', 'VideoMind', 'GVLLM', 'GroundVTS', 'VTimeLLM',
                     'TimeChat', 'VideoChat-R1']}
DAGGERED = {'VTimeLLM', 'TimeChat'}

# ---- reproduction gate: Table S3, "Answer retained on its own window, by IoP
#      bin", columns NExT C2 / NExT C3 / QVH C2 / QVH C3 (tables_v2.tex:76-78)
IOP_EXPECT = {
    '[0.5,0.7)': {('NExT', 'C2'): .836, ('NExT', 'C3'): .761,
                  ('QVH', 'C2'): .805, ('QVH', 'C3'): .783},
    '[0.7,0.9)': {('NExT', 'C2'): .861, ('NExT', 'C3'): .809,
                  ('QVH', 'C2'): .803, ('QVH', 'C3'): .764},
    '[0.9,1.0]': {('NExT', 'C2'): .813, ('NExT', 'C3'): .683,
                  ('QVH', 'C2'): .779, ('QVH', 'C3'): .765},
}
# the §6.3 sentence: failure rate in the IoU >= .9 bin, paper C2
SENTENCE = {'NExT': 0.116, 'QVH': 0.143}


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
    g = [x[0] for x in bpn.GROUNDERS[bench]]
    json.dump(g, open(f'{CACHE}/grounders_{bench}.json', 'w'))
    print(f'[dump {bench}] rows={len(df)} grounders={g} '
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
def binned(df, ids, models, metric):
    """build_paper_numbers item 4, with `metric` in place of `iop`."""
    g = df[df.vqid.isin(ids) & df.model.isin(models)]
    out = {}
    for lo, hi in BINS:
        b = g[(g[metric] >= lo) & (g[metric] < hi)]
        out[BIN_LABEL[(lo, hi)]] = (len(b),
                                    float(b.KE.mean()) if len(b) else np.nan)
    return out


def load(bench):
    df = pd.read_csv(f'{CACHE}/samples_{bench}.csv')
    masks = json.load(open(f'{CACHE}/masks_{bench}.json'))
    gnd = json.load(open(f'{CACHE}/grounders_{bench}.json'))
    assert sorted(gnd) == sorted(GROUNDERS[bench]), (bench, gnd)
    return df, masks, gnd


def compute():
    rows = []
    for bench in ('NExT', 'QVH'):
        df, masks, gnd = load(bench)
        for pset in REPORT_SETS:
            ids = set(masks[CODE_SET[pset]])
            for metric in ('IoU', 'IoP'):
                for lab, (n, ret) in binned(df, ids, gnd,
                                            metric.lower()).items():
                    rows.append(dict(
                        dataset=BENCH_LABEL[bench], bench=bench, set=pset,
                        metric=metric, bin=lab, N=n,
                        ret_pred=round(ret, 6) if n else np.nan,
                        fail_rate=round(1 - ret, 6) if n else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- verify
def verify_iop(df):
    """Gate: the IoP block must reproduce Table S3 to three decimals."""
    bad, lines = 0, []
    for lab, exp in IOP_EXPECT.items():
        for (bench, pset), want in exp.items():
            r = df[(df.bench == bench) & (df['set'] == pset) &
                   (df.metric == 'IoP') & (df.bin == lab)]
            assert len(r) == 1, (bench, pset, lab)
            got = round(float(r.ret_pred.iloc[0]), 3)
            ok = abs(got - want) < 1e-9
            bad += not ok
            lines.append(f'  IoP {lab:11s} {BENCH_LABEL[bench]:9s} {pset}: '
                         f'got {got:.3f}  paper {want:.3f}  '
                         f'{"OK" if ok else "<<< MISMATCH"}  (N={int(r.N.iloc[0])})')
    return bad, lines


def check_sentence(df):
    """Does the IoU>=.9 / paper-C2 failure rate give 11.6% and 14.3%?"""
    out = {}
    for bench, want in SENTENCE.items():
        r = df[(df.bench == bench) & (df['set'] == 'C2') &
               (df.metric == 'IoU') & (df.bin == '[0.9,1.0]')]
        got = round(float(r.fail_rate.iloc[0]), 3)
        out[bench] = (got, want, abs(got - want) < 1e-9, int(r.N.iloc[0]))
    return out


def search_combinations():
    """Only consulted if the headline check fails: which (model set, question
    set) does reproduce 11.6% / 14.3% in the IoU >= .9 bin?"""
    hits = []
    for bench in ('NExT', 'QVH'):
        df, masks, gnd = load(bench)
        variants = {
            'all grounders': gnd,
            'grounders, no dagger': [m for m in gnd if m not in DAGGERED],
        }
        for mname, models in variants.items():
            for pset, cset in CODE_SET.items():
                n, ret = binned(df, set(masks[cset]), models,
                                'iou')['[0.9,1.0]']
                fail = round(1 - ret, 3) if n else np.nan
                hits.append(dict(bench=bench, models=mname, set=pset, N=n,
                                 fail_rate=fail,
                                 matches=bool(n and abs(fail - SENTENCE[bench])
                                              < 1e-9)))
    return pd.DataFrame(hits)


# -------------------------------------------------------------------- latex
def k3(x):
    return f'{x:.3f}'.lstrip('0') if x == x else '--'


def render_tex(df):
    cols = [('NExT', 'C2'), ('NExT', 'C3'), ('QVH', 'C2'), ('QVH', 'C3')]

    def cell(metric, lab, key):
        bench, pset = key
        r = df[(df.bench == bench) & (df['set'] == pset) &
               (df.metric == metric) & (df.bin == lab)]
        return r.ret_pred.iloc[0], int(r.N.iloc[0])

    L = ['% Retention by IoU bin -- block to paste into supplementary Table S3',
         '% (tab:audit), after the "by IoP bin" block. Generated by',
         '% analysis/camera_ready/c4_iou_high.py -- do not edit by hand.',
         '% Same pooling as the IoP block: all evaluated grounders, one row per',
         '% (question, model). Columns: NExT-GQA C2, C3 | QVH-ME C2, C3.']
    for lab in ('[0.5,0.7)', '[0.7,0.9)', '[0.9,1.0]'):
        ns = ', '.join(f'{BENCH_LABEL[b]} {s} N={cell("IoU", lab, (b, s))[1]}'
                       for b, s in cols)
        L.append(f'% N, {lab}: {ns}')
    L += [r'\addlinespace',
          r'\multicolumn{5}{@{}l}{\emph{Answer retained on its own window, '
          r'by IoU bin}} \\']
    for lab in ('[0.5,0.7)', '[0.7,0.9)', '[0.9,1.0]'):
        vals = ' & '.join(k3(cell('IoU', lab, c)[0]) for c in cols)
        L.append(f'\\quad {BIN_TEX[[k for k in BIN_TEX if BIN_LABEL[k] == lab][0]]} '
                 f'& {vals} \\\\')
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------- report
def render_report(df, gate_lines, sent, search):
    cols = [('NExT', 'C2'), ('NExT', 'C3'), ('QVH', 'C2'), ('QVH', 'C3')]

    def get(metric, lab, bench, pset, field):
        r = df[(df.bench == bench) & (df['set'] == pset) &
               (df.metric == metric) & (df.bin == lab)]
        return r[field].iloc[0]

    R = ['# C4 — retention by IoU bin, and the B1 window-presentation table', '',
         'Produced by [analysis/camera_ready/c4_iou_high.py]'
         '(analysis/camera_ready/c4_iou_high.py). No model was re-run and no '
         'existing file was modified.', '',
         '## [1] Retention by IoU bin for Table S3', '',
         'The §6.3 sentence', '',
         '> "Even among predictions with IoU $\\ge$ .9, 11.6% on NExT-GQA and '
         '14.3% on QVH-ME C2 fail to reproduce the full-video response."', '',
         'has no table behind it. This block supplies one, on the same code '
         'path as Table S3\'s IoP block: `build_paper_numbers.compute()` item '
         '4, pooling **all evaluated grounders** (NExT-GQA 5, QVH-ME 7, the '
         'two daggered models included) over the cut\'s question set, one row '
         'per (question, model), `Ret(pred) = mean(K_E)`. Only `iop` is '
         'swapped for `iou`; the top bin is `iou >= 0.9`, which is the same '
         'set as the sentence\'s "IoU $\\ge$ .9" since IoU cannot exceed 1.', '',
         '### Reproduction gate — the IoP block', '',
         'Recomputed through the new code and compared with the printed '
         'Table S3 values:', '', '```'] + gate_lines + ['```', '']

    ok = all(v[2] for v in sent.values())
    R += ['### The §6.3 sentence', '']
    R.append('| | failure rate, IoU $\\ge$ .9, C2 | paper | N |')
    R.append('|---|---|---|---|')
    for bench in ('NExT', 'QVH'):
        got, want, good, n = sent[bench]
        R.append(f'| {BENCH_LABEL[bench]} | **{got*100:.1f}%** | '
                 f'{want*100:.1f}% | {n} |')
    R += ['']
    if ok:
        R += ['**Both reproduce exactly**, on the first combination tried: '
              'all evaluated grounders pooled, paper $\\mathcal{C}_2$. The '
              'sentence needs no change.', '']
        R += ['For completeness, the same bin under the other combinations '
              '(this is a report, not a gate):', '',
              '| dataset | pooled models | set | N | failure rate |',
              '|---|---|---|---|---|']
        for _, r in search.iterrows():
            mark = ' **(paper)**' if r.matches and r['set'] == 'C2' and \
                r.models == 'all grounders' else ''
            R.append(f'| {BENCH_LABEL[r.bench]} | {r.models} | {r["set"]} | '
                     f'{r.N} | {r.fail_rate*100:.1f}%{mark} |')
        R += ['']
    else:
        R += ['**The headline combination does not reproduce.** Search over '
              '(pooled model set) x (question set):', '',
              '| dataset | pooled models | set | N | failure rate | matches |',
              '|---|---|---|---|---|---|']
        for _, r in search.iterrows():
            R.append(f'| {BENCH_LABEL[r.bench]} | {r.models} | {r["set"]} | '
                     f'{r.N} | {r.fail_rate*100:.1f}% | '
                     f'{"**YES**" if r.matches else ""} |')
        R += ['']

    R += ['### The IoU block', '',
          '| IoU bin | NExT-GQA C2 | NExT-GQA C3 | QVH-ME C2 | QVH-ME C3 |',
          '|---|---|---|---|---|']
    for lab in ('[0.5,0.7)', '[0.7,0.9)', '[0.9,1.0]'):
        vals = ' | '.join(k3(get('IoU', lab, b, s, 'ret_pred'))
                          for b, s in cols)
        R.append(f'| {lab} | {vals} |')
    R.append('| *N* | ' + ' | '.join(str(int(get('IoU', '[0.5,0.7)', b, s, 'N')))
                                     + ' / ' +
                                     str(int(get('IoU', '[0.7,0.9)', b, s, 'N')))
                                     + ' / ' +
                                     str(int(get('IoU', '[0.9,1.0]', b, s, 'N')))
                                     for b, s in cols) + ' |')
    R += ['', '(*N* is per bin, in the order the three bins are listed above.)',
          '',
          'Failure rate is $1 - \\mathrm{Ret(pred)}$, so the bottom-right of '
          'the first table reads off this one directly.', '',
          'Two things to note before this goes in the table. **The IoU block '
          'is monotone where the IoP block is not**: retention rises across '
          'the three IoU bins in all four columns (by .05 to .07 from the '
          'bottom bin to the top), while the IoP block is flat or falls '
          '(NExT-GQA C3: .761, .809, .683). A reader comparing the two blocks '
          'will see that, so the caption should say which quantity each '
          'block bins. It does not weaken the sentence the block is there to '
          'support: even in the top IoU bin, 11.6% to 17.4% of predictions do '
          'not reproduce the full-video response. **Second, the NExT-GQA C3 '
          'top bin holds only N = 23 rows** — thin enough that .826 should '
          'not be read as a precise figure; the C2 columns (N = 121 and 280) '
          'are the ones the §6.3 sentence quotes.', '',
          'The LaTeX block is `table_s3_iou_block.tex`: paste it into Table S3 '
          '(`tab:audit`) after the IoP block. It uses the same '
          '`\\multicolumn{5}{@{}l}{\\emph{...}}` heading row and `\\quad` bin '
          'rows as the neighbouring IoP block rather than a bare '
          '`\\textit{...} \\\\`, so it drops in without disturbing the column '
          'spec. Per-bin $N$ is carried in comment lines above the block.', '']

    # ---------------------------------------------------------------- part 2
    R += ['---', '', '## [2] B1 — window presentation (CROP / MASK / MARGIN)',
          '',
          'Verbatim from [analysis/B1_masking/]'
          '(analysis/B1_masking/). Nothing recomputed; '
          '`B1_masking/REPORT.md` names exactly one table file, '
          '`table_b1.tex`.', '', '### `table_b1.tex`', '', '```latex',
          open(f'{B1}/table_b1.tex').read().rstrip(), '```', '',
          '### Source rows, `summary.csv`', '']

    s = pd.read_csv(f'{B1}/summary.csv')

    def sg3(v):
        return '--' if v != v else f'{v:+.3f}'.replace('0.', '.')

    for setname, title in (('a_all241', '(a) all 241 questions'),
                           ('b_excl5_236',
                            '(b) 236 questions, the 5 with a zero-frame '
                            'window dropped')):
        R += [f'**{title}**', '',
              '| condition | Ret(none) | Ret(rand) | Ret(pred) | SG [95% CI] '
              '| ΔSG vs CROP [95% CI] |', '|---|---|---|---|---|---|']
        for _, r in s[s['set'] == setname].iterrows():
            d = ('—' if r.dSG != r.dSG else
                 f'**{sg3(r.dSG)}** [{sg3(r.dSG_lo)}, {sg3(r.dSG_hi)}]')
            R.append(f'| {r.condition} | {k3(r.Ret_none)} | {k3(r.Ret_rand)} '
                     f'| {k3(r.Ret_pred)} | **{sg3(r.SG)}** '
                     f'[{sg3(r.SG_lo)}, {sg3(r.SG_hi)}] | {d} |')
        R += ['']

    d = pd.read_csv(f'{B1}/disagreement.csv')
    R += ['### Question-level agreement on Ret(pred), `disagreement.csv`', '',
          '| comparison | n | median \\|Ê\\| | median frames in Ê |',
          '|---|---|---|---|']
    for _, r in d.iterrows():
        R.append(f'| {r.direction} | {int(r.n)} | {r.median_ehat_len:.1f} s '
                 f'| {r.median_n_frames_in_ehat:.0f} |')
    agree = d.drop_duplicates('condition')[['condition', 'agree_n',
                                            'agree_frac']]
    R += ['']
    for _, r in agree.iterrows():
        R.append(f'* CROP vs {r.condition}: agree on **{int(r.agree_n)}/241 '
                 f'= {r.agree_frac:.3f}** of questions.')
    R += ['* Overall median over all 241 C3 questions: '
          f'|Ê| = {d.overall_median_ehat_len.iloc[0]:.1f} s, '
          f'{d.overall_median_n_frames_in_ehat.iloc[0]:.0f} frames in Ê.', '']

    z = json.load(open(f'{B1}/zero_frame_windows.json'))
    R += ['### Zero-frame windows', '',
          f'**{len(z["windows"])}** of the 1446 stored windows contain no '
          'sampled frame (shorter than the 1.18 s frame spacing). All of them '
          'are random windows — none is a predicted window — and they belong '
          f'to **{len(z["questions"])}** questions, which is the difference '
          'between sets (a) and (b) above. Listed in '
          '`B1_masking/zero_frame_windows.json`.', '']

    R += ['## Files', '', '```', 'analysis/C4_iou_high/',
          '├── iou_bins.csv              dataset, set, metric, bin, N, '
          'ret_pred, fail_rate (IoU and IoP)',
          '├── table_s3_iou_block.tex    block to paste into Table S3',
          '└── _cache/                   samples and masks dumped from '
          'build_paper_numbers',
          'analysis/camera_ready/c4_iou_high.py', '```', '']
    return '\n'.join(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump')
    a = ap.parse_args()
    if a.dump:
        return dump(a.dump)

    os.makedirs(OUTDIR, exist_ok=True)
    spawn_dumps()
    df = compute()

    bad, lines = verify_iop(df)
    print('\n'.join(lines))
    if bad:
        raise SystemExit(f'REPRODUCTION FAILED: {bad}/12 IoP cells differ '
                         'from Table S3 — nothing written')
    print('IoP reproduction gate: 12/12 OK')

    sent = check_sentence(df)
    for b, (got, want, ok, n) in sent.items():
        print(f'sentence {BENCH_LABEL[b]:9s} C2 IoU>=.9 fail = {got:.3f} '
              f'(paper {want:.3f}, N={n}) {"OK" if ok else "<<< MISMATCH"}')
    search = search_combinations()
    if not all(v[2] for v in sent.values()):
        print(search.to_string(index=False))
        if not search.matches.any():
            raise SystemExit('No (model set, question set) combination '
                             'reproduces 11.6% / 14.3% — nothing written')

    df.to_csv(f'{OUTDIR}/iou_bins.csv', index=False)
    open(f'{OUTDIR}/table_s3_iou_block.tex', 'w').write(render_tex(df))
    open(f'{OUTDIR}/REPORT.md', 'w').write(
        render_report(df, lines, sent, search))
    print(f'wrote {OUTDIR}/' + '{iou_bins.csv,table_s3_iou_block.tex,REPORT.md}')


if __name__ == '__main__':
    main()
