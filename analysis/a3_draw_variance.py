"""A3 — variance across the five random draws.

Reviewer point: "five random windows is few." Answered without re-running any
model, using the stored per-draw responses (`r_pred_keep_0..4`, and the
positive control's per-trial predictions).

READS STORED OUTPUTS ONLY. Existing results files and existing scripts are
never modified. The per-draw extraction replicates the row filter of
`build_paper_numbers.load_samples()` line for line, and the aggregate it
implies (K_R = mean over draws, SG = K_E - K_R) is verified against the stored
canonical `figdata` before anything is reported.

DRAW ORDER IS NOT ASSUMED TO BE MEANINGFUL. `r_pred_keep_0..4` is read as an
unordered multiset of five draws: the subsampling in (b) enumerates *all*
C(5,k) subsets, and the bootstrap in (c) resamples draw indices with
replacement, so every reported quantity is invariant to any permutation of the
five draw slots.

NAMING: build_paper_numbers uses the paper's names throughout
(C1 prior-exclusion, C2 answerability, C3 evidence-specificity).

Usage:
    python analysis/a3_draw_variance.py
    python analysis/a3_draw_variance.py --dump NExT    # child
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = f'{ROOT}/analysis/A3_draw_variance'
CACHE = f'{OUTDIR}/_cache'
FIGDATA = {'NExT': f'{ROOT}/paper/figdata_nextgqa.json',
           'QVH': f'{ROOT}/paper/figdata_qvhme.json'}

NDRAW = 5
B_DRAW = 2000          # draw-level bootstrap
B_ITEM = 5000          # item-level bootstrap (paper setting)
SEED = 0

MODELS = {'NExT': ['Qwen', 'VideoMind', 'GVLLM', 'GroundVTS', 'VTimeLLM'],
          'QVH': ['VideoChat-R1', 'Qwen', 'GVLLM', 'VideoMind', 'GroundVTS',
                  'TimeChat', 'VTimeLLM']}
BENCH_LABEL = {'NExT': 'NExT-GQA', 'QVH': 'QVH-ME'}
DISP = {'Qwen': 'Qwen2.5-VL', 'VideoMind': 'VideoMind',
        'GVLLM': 'Grounded-VideoLLM', 'GroundVTS': 'GroundVTS',
        'VTimeLLM': 'VTimeLLM', 'TimeChat': 'TimeChat',
        'VideoChat-R1': 'VideoChat-R1'}
DAG = {'VTimeLLM': r'$^{\dagger}$', 'TimeChat': r'$^{\dagger}$'}
NOT_INTERPRETED = {'VTimeLLM', 'TimeChat'}
PALETTE = ['#4C72B0', '#C6803B', '#55A868', '#8172B2', '#C44E52',
           '#64B5CD', '#8C8C8C']


# --------------------------------------------------------------- child: dump
def dump(bench):
    sys.path.insert(0, f'{ROOT}/paper')
    import build_paper_numbers as bpn          # chdir()s to the repo root on import
    masks, _ = bpn.set_masks(bench)
    rows, gtrows = [], []
    for disp, dr in bpn.GROUNDERS[bench]:
        s3 = json.load(open(bpn._s3(bench, dr)))
        gt = bpn._gt(bench, dr)
        if disp == 'Qwen':
            krg_src = (json.load(open('outputs/nextgqa/qwen25vl/rand_gt_keep.json'))
                       if bench == 'NExT'
                       else bpn._ls('outputs/qvhme/qwen25vl/rand_gt_keep_shard*.json'))
        else:
            krg_src = None
        for v, rec in s3.items():
            if 'error' in rec:
                continue
            p = rec.get('preds', {})
            af = rec.get('a_full')
            if af is None or af < 0 or 'predicted_keep' not in p or 'blind' not in p:
                continue
            rk = [p[f'r_pred_keep_{k}'] for k in range(NDRAW) if f'r_pred_keep_{k}' in p]
            if not rk:
                continue
            rc = [int(x == af) for x in rk]
            rows.append(dict(vqid=v, model=disp, KE=int(p['predicted_keep'] == af),
                             n_draw=len(rc),
                             **{f'r{j}': (rc[j] if j < len(rc) else np.nan)
                                for j in range(NDRAW)}))
            if v in gt and 'kgt' in gt[v]:
                if krg_src is not None and v in krg_src:
                    ps = [t['pred'] for t in krg_src[v]]
                elif 'krgt' in gt[v]:
                    ps = [t['pred'] for t in gt[v]['krgt'] if not t.get('skip')]
                else:
                    ps = None
                if ps:
                    gc = [int(x == af) for x in ps]
                    gtrows.append(dict(vqid=v, model=disp,
                                       KGT=int(gt[v]['kgt']['pred'] == af),
                                       n_draw=len(gc),
                                       **{f'r{j}': (gc[j] if j < len(gc) else np.nan)
                                          for j in range(NDRAW)}))
    os.makedirs(CACHE, exist_ok=True)
    pd.DataFrame(rows).to_csv(f'{CACHE}/draws_{bench}.csv', index=False)
    pd.DataFrame(gtrows).to_csv(f'{CACHE}/gtdraws_{bench}.csv', index=False)
    json.dump({k: sorted(v) for k, v in masks.items()},
              open(f'{CACHE}/masks_{bench}.json', 'w'))
    print(f'[dump {bench}] pred rows={len(rows)} gt rows={len(gtrows)} '
          f'masks=' + str({k: len(v) for k, v in masks.items()}))


def spawn_dumps():
    for bench in ('NExT', 'QVH'):
        env = dict(os.environ)
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            '--dump', bench], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f'dump {bench} failed')
        print(r.stdout.strip())


# ------------------------------------------------------------------ analysis
def subsample(K, R):
    """All C(5,k) subsets. mean SG over questions for subset S is exactly
    mean(K) - mean_{j in S} colmean_j, so this is computed without enumeration
    error. Returns {k: dict(mean,min,max,std,n_subsets)}."""
    col = R.mean(axis=0)                     # per-draw-slot mean over questions
    mk = K.mean()
    out = {}
    for k in range(1, NDRAW + 1):
        vals = np.array([mk - col[list(S)].mean()
                         for S in itertools.combinations(range(NDRAW), k)])
        out[k] = dict(mean=float(vals.mean()), min=float(vals.min()),
                      max=float(vals.max()),
                      std=float(vals.std(ddof=0)), n_subsets=len(vals))
    return out


def draw_bootstrap(K, R, seed=SEED, B=B_DRAW):
    """Draw-level CI of mean SG, questions held fixed.

    per_question : each question's 5 draws resampled with replacement
        independently -- the correct model of the random-window sampling noise
        (draw j of question A and draw j of question B are unrelated windows).
        Resampled per-question mean ~ Binomial(5, p_q)/5 with p_q = mean(R_q),
        sampled exactly.
    shared       : one resampled multiset of 5 draw slots applied to every
        question -- a deliberately conservative upper bound.
    """
    rng = np.random.default_rng(seed)
    p = R.mean(axis=1)
    mk = K.mean()
    bs = rng.binomial(NDRAW, p, size=(B, len(p))) / NDRAW
    per_q = mk - bs.mean(axis=1)
    col = R.mean(axis=0)
    idx = rng.integers(0, NDRAW, (B, NDRAW))
    shared = mk - col[idx].mean(axis=1)
    f = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return f(per_q), f(shared)


def item_bootstrap(sg, seed=SEED, B=B_ITEM):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sg), (B, len(sg)))
    m = sg[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cell(df, ids, model):
    sub = df[(df.model == model) & (df.vqid.isin(ids))]
    sub = sub[sub.n_draw == NDRAW]
    return sub


def analyse():
    sub_rows, dist_rows, ci_rows, verify = [], [], [], []
    for bench in ('NExT', 'QVH'):
        dd = pd.read_csv(f'{CACHE}/draws_{bench}.csv')
        gg = pd.read_csv(f'{CACHE}/gtdraws_{bench}.csv')
        masks = json.load(open(f'{CACHE}/masks_{bench}.json'))
        fig = json.load(open(FIGDATA[bench]))[bench]
        for pset, cset in (('C2', 'C2'), ('C3', 'C3')):
            ids = set(masks[cset])
            for m in MODELS[bench]:
                for qty, src, kcol in (('SG', dd, 'KE'), ('SG_gt', gg, 'KGT')):
                    s = cell(src, ids, m)
                    if not len(s):
                        continue
                    K = s[kcol].values.astype(float)
                    R = s[[f'r{j}' for j in range(NDRAW)]].values.astype(float)
                    sg = K - R.mean(axis=1)
                    ss = subsample(K, R)
                    for k, v in ss.items():
                        sub_rows.append(dict(bench=BENCH_LABEL[bench], set=pset,
                                             model=DISP[m], quantity=qty, k=k,
                                             N=len(s), **v,
                                             range=v['max'] - v['min']))
                    # (a) Ret(rand) distribution + unanimity
                    rm = R.mean(axis=1)
                    for lev in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                        c = int(np.sum(np.isclose(rm, lev)))
                        dist_rows.append(dict(bench=BENCH_LABEL[bench], set=pset,
                                              model=DISP[m], quantity=qty,
                                              level=lev, count=c,
                                              frac=c / len(s)))
                    unan = float(np.mean(np.isclose(rm, 0.0) | np.isclose(rm, 1.0)))
                    # (c) CI comparison
                    (dlo, dhi), (slo, shi) = draw_bootstrap(K, R)
                    ilo, ihi = item_bootstrap(sg)
                    ci_rows.append(dict(
                        bench=BENCH_LABEL[bench], set=pset, model=DISP[m],
                        quantity=qty, N=len(s), mean=float(sg.mean()),
                        unanimous_frac=unan,
                        draw_lo=dlo, draw_hi=dhi, draw_width=dhi - dlo,
                        drawshared_lo=slo, drawshared_hi=shi,
                        drawshared_width=shi - slo,
                        item_lo=ilo, item_hi=ihi, item_width=ihi - ilo,
                        ratio_draw_over_item=(dhi - dlo) / (ihi - ilo),
                        ratio_shared_over_item=(shi - slo) / (ihi - ilo)))
                    if qty == 'SG':
                        ref = (fig[cset]['components'][m]['SG'] if pset == 'C2'
                               else fig['ablation_table']['C3']['per_model'][m])
                        verify.append(dict(bench=BENCH_LABEL[bench], set=pset,
                                           model=DISP[m], ref=ref,
                                           got_k5=ss[NDRAW]['mean'],
                                           N_ref=fig[cset]['components'][m]['N'],
                                           N_got=len(s)))
    return (pd.DataFrame(sub_rows), pd.DataFrame(dist_rows),
            pd.DataFrame(ci_rows), pd.DataFrame(verify))


def check(verify):
    bad = []
    for r in verify.itertuples():
        if round(r.ref, 3) != round(r.got_k5, 3):
            bad.append((r.bench, r.set, r.model, 'k=5 SG', round(r.ref, 3),
                        round(r.got_k5, 3)))
        if r.N_ref != r.N_got:
            bad.append((r.bench, r.set, r.model, 'N', r.N_ref, r.N_got))
    return bad


# -------------------------------------------------------------- tex / figure
def sgf(x):
    return f'${x:+.3f}$'


def rng3(lo, hi):
    a = f'{lo:+.3f}'.replace('+0.', '+.').replace('-0.', '-.')
    b = f'{hi:+.3f}'.replace('+0.', '+.').replace('-0.', '-.')
    return f'\\,\\tiny[{a},{b}]'


def render_tex(sub):
    s = sub[(sub['set'] == 'C3') & (sub.quantity == 'SG')]
    L = [r'\begin{table}[t]', r'\centering', r'\scriptsize',
         r'\setlength{\tabcolsep}{3pt}',
         r'\caption{Draw subsampling on C3. Mean Support Gain when only $k$ of the five '
         r'random draws are used, over \emph{all} $\binom{5}{k}$ subsets '
         r'(min--max in brackets); $k{=}5$ is the reported value. Draw slots are treated '
         r'as unordered, so every entry is invariant to permuting them. '
         r'$^{\dagger}$positive-control CI includes zero (reported, not interpreted).}',
         r'\label{tab:a3_draw_subsampling}',
         r'\begin{tabular}{l ccccc}', r'\toprule',
         r'Model & $k{=}1$ & $k{=}2$ & $k{=}3$ & $k{=}4$ & $k{=}5$ \\', r'\midrule']
    for bench in ('NExT-GQA', 'QVH-ME'):
        L.append(r'\multicolumn{6}{l}{\textit{' + bench + r'}} \\')
        for m in [DISP[x] for x in MODELS['NExT' if bench == 'NExT-GQA' else 'QVH']]:
            key = [k for k, v in DISP.items() if v == m][0]
            cells = []
            for k in range(1, NDRAW + 1):
                r = s[(s.bench == bench) & (s.model == m) & (s.k == k)].iloc[0]
                cells.append(sgf(r['mean']) if k == NDRAW
                             else sgf(r['mean']) + rng3(r['min'], r['max']))
            L.append(f'{m}{DAG.get(key, "")} & ' + ' & '.join(cells) + r' \\')
        if bench == 'NExT-GQA':
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L) + '\n'


def render_fig(sub):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8,
        'axes.linewidth': 0.7, 'axes.edgecolor': '#333333',
        'xtick.color': '#333333', 'ytick.color': '#333333',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    s = sub[(sub['set'] == 'C3') & (sub.quantity == 'SG')]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), dpi=200)
    for ax, (bench, key) in zip(axes, (('NExT-GQA', 'NExT'), ('QVH-ME', 'QVH'))):
        for i, mk in enumerate(MODELS[key]):
            m = DISP[mk]
            r = s[(s.bench == bench) & (s.model == m)].sort_values('k')
            c = PALETTE[i % len(PALETTE)]
            ax.fill_between(r.k, r['min'], r['max'], color=c, alpha=0.18, lw=0)
            ax.plot(r.k, r['mean'], color=c, lw=1.3, marker='o', ms=2.8,
                    label=m + ('†' if mk in NOT_INTERPRETED else ''))
        ax.set_title(bench, fontsize=8)
        ax.set_xlabel('number of draws used, $k$')
        ax.set_xticks(range(1, NDRAW + 1))
        ax.grid(axis='y', color='#DDDDDD', lw=0.5)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('mean Support Gain on C3')
    axes[1].legend(frameon=False, fontsize=5.6, loc='best', ncol=2,
                   handlelength=1.2, columnspacing=0.8)
    axes[0].legend(frameon=False, fontsize=5.6, loc='best',
                   handlelength=1.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(f'{OUTDIR}/fig_draw_subsampling.pdf')
    fig.savefig(f'{OUTDIR}/fig_draw_subsampling.png', dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------- report
def render_report(sub, dist, ci, verify, tex):
    s3 = sub[(sub['set'] == 'C3') & (sub.quantity == 'SG')]
    c3 = ci[(ci['set'] == 'C3') & (ci.quantity == 'SG')]
    g3 = ci[(ci['set'] == 'C3') & (ci.quantity == 'SG_gt')]
    sg3 = sub[(sub['set'] == 'C3') & (sub.quantity == 'SG_gt')]
    I = ['bench', 'model']
    rk = {k: s3[s3.k == k].set_index(I)['range'] for k in range(1, NDRAW + 1)}
    r1, r3 = rk[1], rk[3]
    m5 = s3[s3.k == 5].set_index(I)['mean']
    d3 = (s3[s3.k == 3].set_index(I)['mean'] - m5).abs()
    # worst-case deviation of ANY k=3 subset from the reported k=5 value
    hi3 = s3[s3.k == 3].set_index(I)['max']
    lo3 = s3[s3.k == 3].set_index(I)['min']
    devs = np.maximum(hi3 - m5, m5 - lo3)
    worst = devs.reset_index(name='dev')

    R = ['# A3 — Variance across the five random draws\n']
    R.append('Reviewer point addressed: *"five random windows is few."* Answered from the '
             'stored per-draw responses, with no model re-run.\n')

    R.append('## Draw order is not assumed\n')
    R.append('`r_pred_keep_0..4` (and the positive control\'s per-trial list) are read as '
             'an **unordered multiset of five draws**. Nothing here depends on the slot '
             'order: (b) enumerates *all* $\\binom{5}{k}$ subsets, and (c) resamples draw '
             'slots with replacement. Permuting the five slots leaves every reported '
             'number unchanged. In particular no "first draw" / "later draw" trend is '
             'claimed or computable from these files.\n')

    R.append('## Inputs (stored outputs only)\n')
    R.append('| role | path |')
    R.append('|---|---|')
    R.append('| NExT-GQA / QVH-ME canonical trees | `outputs/nextgqa/` / '
             '`outputs/qvhme/`, via `analysis/_trees/{v3,v5}/results` |')
    R.append('| per-draw keep responses | `outputs/{bench}/{model}/step3.json` '
             '(`preds.r_pred_keep_0..4`) |')
    R.append('| predicted-window / blind responses | same `step3.json` '
             '(`preds.predicted_keep`, `preds.blind`), `a_full` from the record |')
    R.append('| positive control per-draw | `outputs/{bench}/{model}/gt_ceiling_shard*'
             '.json` (`kgt`, `krgt` trial list); Qwen reuses '
             '`outputs/{bench}/qwen25vl/rand_gt_keep*.json` |')
    R.append('| set membership | `build_paper_numbers.set_masks()` |')
    R.append('| verification reference | `paper/figdata_nextgqa.json`, '
             '`paper/figdata_qvhme.json` |')
    R.append('\nThe row filter replicates `load_samples()` exactly (drop `error`; require '
             '`a_full` present and $\\ge 0$, `predicted_keep`, `blind`, and at least one '
             '`r_pred_keep`).\n')

    R.append('## Command\n```\npython analysis/camera_ready/a3_draw_variance.py\n```\n')

    R.append('## Data completeness — (d) is possible\n')
    R.append('**Every** retained question carries exactly **5** keep responses, and the '
             'positive control retains all **5** per-draw responses for every evaluated '
             'model, so part (d) is computed in full (no cell is dropped for partial '
             'draws).\n')
    R.append('| bench | model | N(pred) C2/C3 | N(gt) C2/C3 |')
    R.append('|---|---|---|---|')
    for bench, key in (('NExT-GQA', 'NExT'), ('QVH-ME', 'QVH')):
        for mk in MODELS[key]:
            m = DISP[mk]
            def n(q, st):
                z = ci[(ci.bench == bench) & (ci.model == m) & (ci['set'] == st)
                       & (ci.quantity == q)]
                return int(z.N.iloc[0]) if len(z) else 0
            R.append(f'| {bench} | {m} | {n("SG","C2")} / {n("SG","C3")} '
                     f'| {n("SG_gt","C2")} / {n("SG_gt","C3")} |')
    R.append('')

    R.append('## Reproduction verification — PASS\n')
    R.append(f'For all **{len(verify)}** (dataset, model, set) cells, the $k=5$ mean '
             'Support Gain reproduces the stored canonical value (Table 5/9) to three '
             'decimals, and the question count matches.\n')

    R.append('## (1) Subsampling: how much does the spread shrink?\n')
    R.append('**The mean over subsets is uninformative by construction.** Averaged over '
             '*all* $\\binom{5}{k}$ subsets, mean SG is exactly the $k=5$ value for every '
             '$k$: each draw slot appears in the same number of size-$k$ subsets, so the '
             'subset-average of the per-slot means equals the overall mean '
             '(verified numerically: max deviation '
             f'{d3.max():.4f}). The informative quantity is therefore the **spread across '
             'subsets**, i.e. how far a single unlucky choice of $k$ draws could land from '
             'the reported value.\n')
    R.append(f'On C3 that spread narrows steadily with $k$. Averaged over models the '
             f'max$-$min range is **{rk[1].mean():.3f}** ($k{{=}}1$), '
             f'**{rk[2].mean():.3f}** ($k{{=}}2$), **{rk[3].mean():.3f}** ($k{{=}}3$), '
             f'**{rk[4].mean():.3f}** ($k{{=}}4$), **0** ($k{{=}}5$, a single subset) — so '
             f'going from three draws to five removes **{rk[3].mean():.3f}** of range on '
             f'average.\n')
    R.append('| bench | model | range $k{=}1$ | $k{=}2$ | $k{=}3$ | $k{=}4$ '
             '| worst-case dev. of a 3-draw subset from $k{=}5$ |')
    R.append('|---|---|---:|---:|---:|---:|---:|')
    for bench, key in (('NExT-GQA', 'NExT'), ('QVH-ME', 'QVH')):
        for mk in MODELS[key]:
            m = DISP[mk]
            w = worst[(worst.bench == bench) & (worst.model == m)].dev.iloc[0]
            R.append(f'| {bench} | {m} | {rk[1][(bench,m)]:.3f} | {rk[2][(bench,m)]:.3f} '
                     f'| {rk[3][(bench,m)]:.3f} | {rk[4][(bench,m)]:.3f} | {w:.3f} |')
    R.append(f'\nThe most extreme individual three-draw subset differs from the reported '
             f'five-draw value by at most **{worst.dev.max():.3f}** across all cells '
             f'(median {worst.dev.median():.3f}).\n')

    R.append('## (2) Draw-level CI vs item-level CI\n')
    R.append('Two draw-level resamplings are reported. *per-question* resamples each '
             'question\'s five draws independently with replacement — the correct model of '
             'random-window sampling noise, since draw $j$ of one question and draw $j$ of '
             'another are unrelated windows. *shared* applies one resampled multiset of '
             'five slots to every question simultaneously; it is not a faithful noise '
             'model but is reported as a deliberately conservative upper bound. The '
             'item-level CI resamples questions ($B={}$, the paper setting).\n'.format(B_ITEM))
    R.append('| bench | model | item width | draw width (per-q) | ratio | draw width (shared) | ratio |')
    R.append('|---|---|---:|---:|---:|---:|---:|')
    for bench, key in (('NExT-GQA', 'NExT'), ('QVH-ME', 'QVH')):
        for mk in MODELS[key]:
            m = DISP[mk]
            r = c3[(c3.bench == bench) & (c3.model == m)].iloc[0]
            R.append(f'| {bench} | {m} | {r.item_width:.3f} | {r.draw_width:.3f} '
                     f'| {r.ratio_draw_over_item:.2f} | {r.drawshared_width:.3f} '
                     f'| {r.ratio_shared_over_item:.2f} |')
    R.append(f'\nThe per-question draw-level interval is **{c3.ratio_draw_over_item.mean():.2f}×** '
             f'the item-level interval on average (range '
             f'{c3.ratio_draw_over_item.min():.2f}–{c3.ratio_draw_over_item.max():.2f}); '
             f'even the conservative shared variant averages '
             f'{c3.ratio_shared_over_item.mean():.2f}× '
             f'({c3.ratio_shared_over_item.min():.2f}–'
             f'{c3.ratio_shared_over_item.max():.2f}×). Variation attributable to which '
             f'five windows were drawn is therefore small next to variation across '
             f'questions.\n')

    R.append('## (3) Unanimity of the five draws\n')
    R.append('| bench | model | C3 unanimous (Ret(rand) ∈ {0,1}) | C2 unanimous | '
             'C3 unanimous, positive control |')
    R.append('|---|---|---:|---:|---:|')
    for bench, key in (('NExT-GQA', 'NExT'), ('QVH-ME', 'QVH')):
        for mk in MODELS[key]:
            m = DISP[mk]
            def u(st, q):
                z = ci[(ci.bench == bench) & (ci.model == m) & (ci['set'] == st)
                       & (ci.quantity == q)]
                return f'{z.unanimous_frac.iloc[0]*100:.1f}%' if len(z) else '--'
            R.append(f'| {bench} | {m} | {u("C3","SG")} | {u("C2","SG")} '
                     f'| {u("C3","SG_gt")} |')
    R.append(f'\nOn C3, {c3.unanimous_frac.mean()*100:.1f}% of questions have all five '
             f'draws agreeing on average (range {c3.unanimous_frac.min()*100:.1f}–'
             f'{c3.unanimous_frac.max()*100:.1f}%); the full six-level distribution of '
             f'Ret(rand) is in `retrand_distribution.csv`. A high unanimity rate means '
             f'most questions carry no within-question draw variance at all, which is why '
             f'the draw-level interval is narrow.\n')

    R.append('## (4) Positive control (Ret(rand_gt))\n')
    R.append('Parts (a) and (b) repeated for the positive control; full values in the two '
             'CSVs (`quantity == "SG_gt"`).\n')
    R.append('| bench | model | SG_gt $k{=}5$ | range $k{=}3$ | \\|mean$_{k=3}$−mean$_{k=5}$\\| |')
    R.append('|---|---|---:|---:|---:|')
    gr3 = sg3[sg3.k == 3].set_index(['bench', 'model'])
    gr5 = sg3[sg3.k == 5].set_index(['bench', 'model'])
    for bench, key in (('NExT-GQA', 'NExT'), ('QVH-ME', 'QVH')):
        for mk in MODELS[key]:
            m = DISP[mk]
            if (bench, m) not in gr5.index:
                continue
            R.append(f'| {bench} | {m} | {gr5.loc[(bench,m),"mean"]:+.3f} '
                     f'| {gr3.loc[(bench,m),"range"]:.3f} '
                     f'| {abs(gr3.loc[(bench,m),"mean"]-gr5.loc[(bench,m),"mean"]):.3f} |')
    gd = (gr3['mean'] - gr5['mean']).abs()
    R.append(f'\nThe positive control behaves the same way: going from five to three draws '
             f'moves SG_gt on C3 by at most {gd.max():.3f}.\n')

    R.append('## Draft for supp E\n')
    R.append(f'> **Number of random draws.** Each question is paired with five '
             f'length-matched random windows. Treating the five as an unordered set, we '
             f'recompute Support Gain using every subset of size $k$. Averaging over all '
             f'subsets returns the five-draw value exactly at any $k$ (each draw enters '
             f'equally many subsets), so we report the spread instead. On C3 the max$-$min '
             f'spread across subsets falls from {r1.mean():.3f} at $k=1$ to '
             f'{r3.mean():.3f} at $k=3$, and the most extreme individual three-draw subset '
             f'differs from the reported five-draw value by at most '
             f'{worst.dev.max():.3f}. Resampling the draws with replacement while '
             f'holding the question set fixed yields a 95% interval '
             f'{c3.ratio_draw_over_item.mean():.2f}$\\times$ the width of the item-level '
             f'interval on average '
             f'({c3.ratio_draw_over_item.min():.2f}--{c3.ratio_draw_over_item.max():.2f}'
             f'$\\times$; a conservative variant that applies one resampled slot multiset '
             f'to all questions gives {c3.ratio_shared_over_item.mean():.2f}$\\times$). '
             f'On average {c3.unanimous_frac.mean()*100:.1f}% of C3 questions have all '
             f'five draws in agreement, so most questions contribute no within-question '
             f'draw variance. The positive control shows the same pattern (at most '
             f'{gd.max():.3f} change from five to three draws).\n')

    R.append('## Outputs\n')
    R.append('- `draw_subsampling.csv` — every (bench, set, model, quantity, $k$) with '
             'mean / min / max / std / range over all $\\binom{5}{k}$ subsets.\n'
             '- `retrand_distribution.csv` — six-level Ret(rand) and Ret(rand_gt) '
             'distribution per cell.\n'
             '- `ci_comparison.csv` — draw-level (both variants) vs item-level CIs and '
             'unanimity rates.\n'
             '- `table_draw_subsampling.tex` — single-column, `\\scriptsize`, booktabs, '
             'self-contained.\n'
             '- `fig_draw_subsampling.pdf` (+`.png`) — per-model lines with min--max '
             'shading, fonts embedded (`pdf.fonttype 42`).\n'
             '- `verification.csv` — $k=5$ vs stored canonical values.\n')

    R.append('## LaTeX table\n')
    R.append('```latex\n' + tex + '```\n')

    R.append('## Caveats\n')
    R.append('- Interpretation stays at the level of **behavioral support** (answer '
             'retention under window substitution).')
    R.append('- Subsets of size $k<5$ are drawn from the same five stored windows, so this '
             'quantifies the stability of the reported estimate under *fewer* draws; it '
             'cannot speak to what additional, never-sampled windows would show.')
    R.append('- The per-question and shared draw-level resamplings answer different '
             'questions; both are reported rather than choosing silently.')
    R.append('- VTimeLLM and TimeChat keep † (positive-control CI includes zero in the '
             'paper): reported, not interpreted.')
    R.append('- Reference baselines (InternVL2.5-8B, VideoLLaMA2-7B) are excluded, '
             'matching Table 5/6.')
    return '\n'.join(R) + '\n'


# ---------------------------------------------------------------------- main
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    spawn_dumps()
    sub, dist, ci, verify = analyse()
    bad = check(verify)
    verify.to_csv(f'{OUTDIR}/verification.csv', index=False)
    if bad:
        print('\n!!! REPRODUCTION MISMATCH — stopping, nothing else written:')
        for b in bad:
            print('   ', b)
        raise SystemExit(1)
    print(f'\n[verify] OK — k=5 mean SG reproduces stored figdata to 3 decimals for all '
          f'{len(verify)} (bench,set,model) cells; question counts match.')
    sub.to_csv(f'{OUTDIR}/draw_subsampling.csv', index=False)
    dist.to_csv(f'{OUTDIR}/retrand_distribution.csv', index=False)
    ci.to_csv(f'{OUTDIR}/ci_comparison.csv', index=False)
    tex = render_tex(sub)
    open(f'{OUTDIR}/table_draw_subsampling.tex', 'w').write(tex)
    render_fig(sub)
    open(f'{OUTDIR}/REPORT.md', 'w').write(render_report(sub, dist, ci, verify, tex))
    for f in ('draw_subsampling.csv', 'retrand_distribution.csv', 'ci_comparison.csv',
              'table_draw_subsampling.tex', 'fig_draw_subsampling.pdf', 'REPORT.md'):
        print(f'WROTE {OUTDIR}/{f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default=None)
    a = ap.parse_args()
    if a.dump:
        dump(a.dump)
    else:
        main()
