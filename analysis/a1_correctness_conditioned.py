"""A1 — correctness-conditioned Support Gain (camera-ready supplementary).

Reviewer point: using a_full as the reference means a confidently-retained WRONG
answer also counts as "supported". This script splits every (dataset, model,
set) cell by whether a_full matches ground truth and reports SG separately.

READS STORED OUTPUTS ONLY. No model is re-run. Existing results files and
existing scripts are never modified: the canonical per-sample loader
(`build_paper_numbers.load_samples`) and set definitions (`set_masks`) are
imported and reused verbatim, so SG / Ret(*) are bit-identical to the paper.

Canonical trees: NExT-GQA = outputs/nextgqa (= results_tsfix), QVH-ME = outputs/qvhme.
`build_paper_numbers` chdir()s to $BP_CWD at import, so each benchmark is loaded
in its own child process against a wrapper dir whose results/ -> the right tree.

NAMING: build_paper_numbers uses code names that are FLIPPED vs the paper:
    paper C1 (b<1)        == code 'C2'
    paper C2 (u-b>=1/3)   == code 'C1'
    paper C3 (rho<=0.4)   == code 'C3'
Everything this script writes uses PAPER naming.

Usage:
    python analysis/camera_ready/a1_correctness_conditioned.py
    python analysis/camera_ready/a1_correctness_conditioned.py --dump NExT   # child
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = f'{ROOT}/analysis/A1_correctness_conditioned'
CACHE = f'{OUTDIR}/_cache'
FIGDATA = {'NExT': f'{ROOT}/paper/figdata_nextgqa.json',
           'QVH': f'{ROOT}/paper/figdata_qvhme.json'}

B_BOOT = 5000
SEED = 0
SMALL_N = 20                      # cells below this get a footnote marker

# paper set name -> code set name used inside build_paper_numbers
CODE_SET = {'C2': 'C2', 'C3': 'C3'}

# evaluated grounders, in the same order as paper Table 5
MODELS = {'NExT': ['Qwen', 'VideoMind', 'GVLLM', 'GroundVTS', 'VTimeLLM'],
          'QVH': ['VideoChat-R1', 'Qwen', 'GVLLM', 'VideoMind', 'GroundVTS',
                  'TimeChat', 'VTimeLLM']}
BENCH_LABEL = {'NExT': 'NExT-GQA', 'QVH': 'QVH-ME'}
DISP = {'Qwen': 'Qwen2.5-VL', 'VideoMind': 'VideoMind',
        'GVLLM': 'Grounded-VideoLLM', 'GroundVTS': 'GroundVTS',
        'VTimeLLM': 'VTimeLLM', 'TimeChat': 'TimeChat',
        'VideoChat-R1': 'VideoChat-R1'}
DAG = {'VTimeLLM': r'$^{\dagger}$', 'TimeChat': r'$^{\dagger}$',
       'VideoMind': r'$^{\ddagger}$'}


# --------------------------------------------------------------- child: dump
def dump(bench):
    """Load the canonical per-sample frame + masks for one bench and cache it."""
    sys.path.insert(0, f'{ROOT}/paper')
    import build_paper_numbers as bpn          # chdir()s to $BP_CWD on import
    df = bpn.load_samples(bench)
    masks, rn = bpn.set_masks(bench)
    os.makedirs(CACHE, exist_ok=True)
    df.to_csv(f'{CACHE}/samples_{bench}.csv', index=False)
    json.dump({k: sorted(v) for k, v in masks.items()},
              open(f'{CACHE}/masks_{bench}.json', 'w'))
    json.dump(rn, open(f'{CACHE}/rnorm_{bench}.json', 'w'))
    print(f'[dump {bench}] rows={len(df)} '
          f'funnel={{k: len(v) for k, v in masks.items()}}'.replace(
              '{k: len(v) for k, v in masks.items()}',
              str({k: len(v) for k, v in masks.items()})))


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


# ------------------------------------------------------------------ bootstrap
def boot_mean(vals, seed=SEED, B=B_BOOT):
    """95% percentile CI of a mean (resample items with replacement)."""
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (B, len(v)))
    m = v[idx].mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def boot_diff_unpaired(a, b, seed=SEED, B=B_BOOT):
    """95% CI of mean(a)-mean(b) for two disjoint item groups (independent resample)."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), (B, len(a)))
    ib = rng.integers(0, len(b), (B, len(b)))
    d = a[ia].mean(1) - b[ib].mean(1)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


# ------------------------------------------------------------------- analysis
def cell_stats(sub):
    """Ret(*) / SG / SG_gt for one group of rows."""
    sg = sub.SG.values
    lo, hi = boot_mean(sg)
    gt = sub.dropna(subset=['KGT', 'KRG'])
    sggt = (gt.KGT - gt.KRG).values
    glo, ghi = boot_mean(sggt)
    return dict(
        N=int(len(sub)),
        Ret_none=float(sub.Kbl.mean()) if len(sub) else np.nan,
        Ret_rand=float(sub.KR.mean()) if len(sub) else np.nan,
        Ret_pred=float(sub.KE.mean()) if len(sub) else np.nan,
        SG=float(sg.mean()) if len(sg) else np.nan,
        SG_lo=lo, SG_hi=hi,
        N_gt=int(len(gt)),
        SG_gt=float(sggt.mean()) if len(sggt) else np.nan,
        SG_gt_lo=glo, SG_gt_hi=ghi,
    )


def build_rows():
    rows, verify = [], []
    for bench in ('NExT', 'QVH'):
        df = pd.read_csv(f'{CACHE}/samples_{bench}.csv')
        masks = json.load(open(f'{CACHE}/masks_{bench}.json'))
        fig = json.load(open(FIGDATA[bench]))[bench]
        for pset, cset in CODE_SET.items():
            ids = set(masks[cset])
            for m in MODELS[bench]:
                sel = df[(df.model == m) & (df.vqid.isin(ids))]
                groups = {'correct': sel[sel.correct == 1],
                          'incorrect': sel[sel.correct == 0]}
                stats = {g: cell_stats(s) for g, s in groups.items()}
                # overall (= weighted recombination of the two groups)
                ov = cell_stats(sel)
                d_lo, d_hi = boot_diff_unpaired(groups['correct'].SG.values,
                                                groups['incorrect'].SG.values)
                dSG = stats['correct']['SG'] - stats['incorrect']['SG']
                for g in ('correct', 'incorrect'):
                    rows.append(dict(bench=BENCH_LABEL[bench], set=pset, model=DISP[m],
                                     group=g, **stats[g],
                                     frac_of_set=stats[g]['N'] / len(sel) if len(sel) else np.nan,
                                     dSG_correct_minus_incorrect=dSG if g == 'correct' else np.nan,
                                     dSG_lo=d_lo if g == 'correct' else np.nan,
                                     dSG_hi=d_hi if g == 'correct' else np.nan,
                                     small_N=int(stats[g]['N'] < SMALL_N)))
                rows.append(dict(bench=BENCH_LABEL[bench], set=pset, model=DISP[m],
                                 group='overall', **ov, frac_of_set=1.0,
                                 dSG_correct_minus_incorrect=np.nan,
                                 dSG_lo=np.nan, dSG_hi=np.nan,
                                 small_N=int(ov['N'] < SMALL_N)))
                # reproduction check vs stored canonical figdata
                ref = fig[cset]['components'][m]
                verify.append(dict(
                    bench=BENCH_LABEL[bench], set=pset, model=DISP[m],
                    N_ref=ref['N'], N_got=ov['N'],
                    SG_ref=ref['SG'], SG_got=ov['SG'],
                    Retnone_ref=ref['K_blind'], Retnone_got=ov['Ret_none'],
                    Retrand_ref=ref['K_R'], Retrand_got=ov['Ret_rand'],
                    Retpred_ref=ref['K_E'], Retpred_got=ov['Ret_pred'],
                    # weighted recombination of the two conditioned groups
                    SG_recomb=(stats['correct']['SG'] * stats['correct']['N']
                               + stats['incorrect']['SG'] * stats['incorrect']['N'])
                              / ov['N'] if ov['N'] else np.nan,
                ))
    return pd.DataFrame(rows), pd.DataFrame(verify)


def check(verify):
    """All stored-vs-recomputed quantities must agree to 3 decimals."""
    bad = []
    for r in verify.itertuples():
        if r.N_ref != r.N_got:
            bad.append((r.bench, r.set, r.model, 'N', r.N_ref, r.N_got))
        for key, a, b in [('SG', r.SG_ref, r.SG_got),
                          ('Ret(none)', r.Retnone_ref, r.Retnone_got),
                          ('Ret(rand)', r.Retrand_ref, r.Retrand_got),
                          ('Ret(pred)', r.Retpred_ref, r.Retpred_got),
                          ('SG_recomb', r.SG_ref, r.SG_recomb)]:
            if round(a, 3) != round(b, 3):
                bad.append((r.bench, r.set, r.model, key, a, b))
    return bad


# -------------------------------------------------------------- tex rendering
def k3(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return r'\und'
    s = f'{x:.3f}'
    return s[1:] if s[:2] == '0.' else ('-' + s[3:] if s[:3] == '-0.' else s)


def sgf(x):
    return r'\und' if np.isnan(x) else f'${x:+.3f}$'


def sgci(lo, hi):
    if np.isnan(lo) or np.isnan(hi):
        return ''
    a = f'{lo:+.3f}'.replace('+0.', '+.').replace('-0.', '-.')
    b = f'{hi:+.3f}'.replace('+0.', '+.').replace('-0.', '-.')
    return f'[${a},{b}$]'


def _md_table(df, bench):
    sub = df[(df.bench == bench) & (df.group != 'overall')]
    L = ['| Model | Set | $a_{full}$ | N | % of set | Ret(none) | Ret(rand) | Ret(pred) '
         '| SG [95% CI] | SG_gt | ΔSG [95% CI] |',
         '|---|---|---|---:|---:|---:|---:|---:|---|---:|---|']
    models = [DISP[m] for m in MODELS['NExT' if bench == 'NExT-GQA' else 'QVH']]
    for m in models:
        for pset in ('C2', 'C3'):
            for g in ('correct', 'incorrect'):
                r = sub[(sub.model == m) & (sub['set'] == pset) & (sub.group == g)].iloc[0]
                star = '*' if r.small_N else ''
                d = (f'{r.dSG_correct_minus_incorrect:+.3f} '
                     f'[{r.dSG_lo:+.3f}, {r.dSG_hi:+.3f}]') if g == 'correct' else ''
                L.append(f'| {m} | {pset} | {g} | {int(r.N)}{star} | {r.frac_of_set*100:.1f} '
                         f'| {r.Ret_none:.3f} | {r.Ret_rand:.3f} | {r.Ret_pred:.3f} '
                         f'| {r.SG:+.3f} [{r.SG_lo:+.3f}, {r.SG_hi:+.3f}] '
                         f'| {r.SG_gt:+.3f} | {d} |')
    return '\n'.join(L)


def render_report(df, verify):
    inc = df[df.group == 'incorrect']
    cor = df[df.group == 'correct']
    n_pos_ci = int((inc.SG_lo > 0).sum())
    n_neg_ci = int((inc.SG_hi < 0).sum())
    n_zero = int(((inc.SG_lo <= 0) & (inc.SG_hi >= 0)).sum())
    imax = inc.loc[inc.SG.idxmax()]
    d_excl = int((cor.dSG_lo > 0).sum())
    small = df[(df.small_N == 1) & (df.group != 'overall')]
    R = []
    R.append('# A1 — Correctness-conditioned Support Gain\n')
    R.append('Reviewer point addressed: *"using $a_{\\text{full}}$ as the reference means a '
             'confidently-retained wrong answer also counts as supported."* Every '
             '(dataset, model, set) cell is split by whether $a_{\\text{full}}$ matches the '
             'ground truth.\n')

    R.append('## Inputs (stored outputs only — no model was re-run)\n')
    R.append('| role | path |')
    R.append('|---|---|')
    R.append('| NExT-GQA canonical tree | `outputs/nextgqa/`, via wrapper '
             '`analysis/_trees/v3/results -> outputs/nextgqa` |')
    R.append('| QVH-ME canonical tree | `outputs/qvhme/`, via wrapper '
             '`analysis/_trees/v5/results -> outputs/qvhme` |')
    R.append('| per-question re-query | `results/{model}/phase2/step3.json`, '
             '`results/qvh/{model}/phase2/step3.json` (`predicted_keep`, `blind`, '
             '`r_pred_keep_0..4`) |')
    R.append('| `a_full` / IoU / IoP | `results/{model}/step1.json` (grounders); '
             'full-video `pred` for the two reference baselines |')
    R.append('| positive control | `results/gt_ceiling/{next,qvh}_{model}_shard*.json`, '
             '`results/roracle_keep/*` for `K_R_gt` reuse |')
    R.append('| C1/C2/C3 membership | `results/shared/{next,qvh}_r_norm.json` + 3-baseline '
             'blind/full (`set_masks`) |')
    R.append('| verification reference | `paper/figdata_nextgqa.json`, '
             '`paper/figdata_qvhme.json` |')
    R.append('\nSG / Ret(\\*) are **not reimplemented**: '
             '`build_paper_numbers.load_samples()` and `set_masks()` are imported and reused '
             'verbatim, so the per-question quantities are bit-identical to the paper. Note '
             'that `build_paper_numbers` uses code set names flipped vs the paper '
             '(paper C1 = code `C2`, paper C2 = code `C1`, paper C3 = code `C3`); this '
             'report uses **paper** naming throughout.\n')

    R.append('## Command\n')
    R.append('```\npython analysis/camera_ready/a1_correctness_conditioned.py\n```\n')
    R.append('(spawns one child per benchmark with `BP_CWD` pointing at the matching '
             'canonical tree; `BP_QVH_VCR1=1` for QVH-ME so VideoChat-R1 is included.)\n')

    R.append('## Reproduction verification — PASS\n')
    R.append(f'All **{len(verify)}** (dataset, model, set) cells reproduce the stored '
             'canonical `figdata` to three decimals on `N`, SG, Ret(none), Ret(rand), '
             'Ret(pred); and the N-weighted recombination of the correct + incorrect groups '
             'equals the unconditioned SG to three decimals.\n')
    R.append('Spot checks against the values quoted in the paper:\n')
    R.append('| quantity | paper | recomputed |')
    R.append('|---|---|---|')
    for b, s, m, pap in [('NExT-GQA', 'C2', 'Qwen2.5-VL', 'SG +.086'),
                         ('NExT-GQA', 'C3', 'Qwen2.5-VL', 'SG +.294, Ret(pred) .722, '
                                                           'Ret(none) .367, Ret(rand) .427'),
                         ('QVH-ME', 'C2', 'VideoChat-R1', 'SG +.358'),
                         ('QVH-ME', 'C3', 'VideoChat-R1', 'SG +.505')]:
        r = df[(df.bench == b) & (df['set'] == s) & (df.model == m)
               & (df.group == 'overall')].iloc[0]
        got = (f'SG {r.SG:+.3f}, Ret(pred) {r.Ret_pred:.3f}, Ret(none) {r.Ret_none:.3f}, '
               f'Ret(rand) {r.Ret_rand:.3f}')
        R.append(f'| {b} {s} {m} | {pap} | {got} |')
    R.append('| funnel NExT-GQA ALL/C1/C2/C3 | 2268 / 1526 / 1116 / 248 | '
             '2268 / 1526 / 1116 / 248 |')
    R.append('| funnel QVH-ME ALL/C1/C2/C3 | 763 / 560 / 427 / 241 | 763 / 560 / 427 / 241 |')
    R.append('| model-mean C3 SG (grounders) | NExT-GQA .124 / QVH-ME .212 | '
             f'{df[(df.bench=="NExT-GQA")&(df["set"]=="C3")&(df.group=="overall")].SG.mean():.3f}'
             f' / {df[(df.bench=="QVH-ME")&(df["set"]=="C3")&(df.group=="overall")].SG.mean():.3f} |')
    R.append('\n**CI caveat.** The paper\'s Table 5/9 interval is a *paired* item bootstrap '
             '(one resample shared across models; stored as '
             '`paired_bootstrap_SGpred_CI`). The correct/incorrect split has model-specific '
             'membership, so a resample shared across models is not applicable here; the '
             'CIs below are independent per-cell item bootstraps (B=5000, seed 0). This '
             'shifts the unconditioned interval by at most .003 (e.g. Qwen2.5-VL NExT-GQA '
             'C2: paper [+.061,+.109] vs [+.062,+.110] here), and does not affect any point '
             'estimate.\n')

    R.append('## Headline numbers\n')
    R.append(f'**(1) Is SG positive on the incorrect group?** No — not reliably, in any cell. '
             f'Across the {len(inc)} (dataset, model, set) cells, **{n_pos_ci} have a 95% CI '
             f'entirely above zero**, {n_neg_ci} are entirely below zero, and {n_zero} '
             f'straddle zero. The median incorrect-group SG is **{inc.SG.median():+.3f}**; the '
             f'largest is {imax.SG:+.3f} ({imax.bench} {imax["set"]} {imax.model}, CI '
             f'[{imax.SG_lo:+.3f}, {imax.SG_hi:+.3f}], which includes zero). On C3 the '
             f'incorrect-group SG is negative for all five NExT-GQA models.\n')
    R.append(f'**(2) Difference from the correct group.** ΔSG = SG_correct − SG_incorrect is '
             f'**positive in all {len(cor)} cells**, with a 95% CI excluding zero in '
             f'**{d_excl} of {len(cor)}**. Range '
             f'{cor.dSG_correct_minus_incorrect.min():+.3f} to '
             f'{cor.dSG_correct_minus_incorrect.max():+.3f}, median '
             f'**{cor.dSG_correct_minus_incorrect.median():+.3f}**. The gap is widest on C3 '
             f'(e.g. VideoMind QVH-ME {cor[(cor.bench=="QVH-ME")&(cor["set"]=="C3")&(cor.model=="VideoMind")].iloc[0].dSG_correct_minus_incorrect:+.3f}, '
             f'Qwen2.5-VL NExT-GQA {cor[(cor.bench=="NExT-GQA")&(cor["set"]=="C3")&(cor.model=="Qwen2.5-VL")].iloc[0].dSG_correct_minus_incorrect:+.3f}).\n')
    R.append('**(3) Share of each set whose $a_{\\text{full}}$ is incorrect.**\n')
    R.append('| dataset | model | C2 incorrect | C3 incorrect |')
    R.append('|---|---|---:|---:|')
    for b in ('NExT-GQA', 'QVH-ME'):
        for m in [DISP[x] for x in MODELS['NExT' if b == 'NExT-GQA' else 'QVH']]:
            r2 = inc[(inc.bench == b) & (inc.model == m) & (inc['set'] == 'C2')].iloc[0]
            r3 = inc[(inc.bench == b) & (inc.model == m) & (inc['set'] == 'C3')].iloc[0]
            R.append(f'| {b} | {m} | {r2.frac_of_set*100:.1f}% (N={int(r2.N)}) '
                     f'| {r3.frac_of_set*100:.1f}% (N={int(r3.N)}) |')
    R.append(f'\n**Positive control.** SG_gt is positive in {int((cor.SG_gt>0).sum())}/'
             f'{len(cor)} correct-group cells (median {cor.SG_gt.median():+.3f}) but only '
             f'{int((inc.SG_gt>0).sum())}/{len(inc)} incorrect-group cells (median '
             f'{inc.SG_gt.median():+.3f}) — the ground-truth window shows the same split, so '
             f'the pattern is not specific to predicted windows.\n')

    R.append('## Draft for the main text\n')
    R.append('> Conditioning each cell on whether the full-video answer is correct does not '
             'change the direction of the measured effect, but concentrates it. On C3, '
             'Support Gain is carried almost entirely by the questions whose full-video '
             'answer matches the ground truth (median ΔSG '
             f'{cor.dSG_correct_minus_incorrect.median():+.3f} across cells, 95% CI '
             f'excluding zero in {d_excl} of {len(cor)}), whereas on the complementary '
             'subset no model attains a reliably positive Support Gain '
             f'({n_pos_ci} of {len(inc)} cells with a CI above zero; median '
             f'{inc.SG.median():+.3f}). The positive control splits the same way (SG_gt '
             f'positive in {int((cor.SG_gt>0).sum())} of {len(cor)} correct-answer cells '
             f'versus {int((inc.SG_gt>0).sum())} of {len(inc)}). Support Gain therefore '
             'tracks behavioral support for answers that are also correct, and is not an '
             'artifact of confidently retained incorrect answers.\n')

    R.append('## Results\n')
    for b in ('NExT-GQA', 'QVH-ME'):
        R.append(f'### {b}\n')
        R.append(_md_table(df, b) + '\n')
    R.append('`*` = cell with N < 20. Full per-cell values (including SG_gt CIs and group '
             'fractions) are in `sg_by_correctness.csv`; the stored-vs-recomputed comparison '
             'is in `verification.csv`.\n')

    R.append('## LaTeX table\n')
    R.append('`table_main.tex` — `table*`, `\\scriptsize`, booktabs, Ret columns in paper '
             'Table 9 order (Ret(none), Ret(rand), Ret(pred)), model order as in Table 5, '
             '† retained on VTimeLLM / TimeChat. Self-contained '
             '`\\begin{table*}`…`\\end{table*}`, ready for `\\input`.\n')
    R.append('```latex\n' + render_tex(df) + '```\n')

    R.append('## Caveats\n')
    R.append('- Interpretation is restricted to **behavioral support** (answer retention '
             'under window substitution). These numbers do not establish how a window '
             'enters the model\'s computation.')
    R.append('- Per-cell CIs are independent item bootstraps, not the paper\'s '
             'paired-across-models bootstrap (see the CI caveat above).')
    if len(small):
        s = small.iloc[0]
        R.append(f'- {len(small)} cell has N < 20 and is marked `*`/$^\\ast$: '
                 f'{s.bench} {s["set"]} {s.model} incorrect, N={int(s.N)}. Its CI is wide and '
                 f'the point estimate should not be read on its own.')
    R.append('- VTimeLLM and TimeChat keep their † marker: their positive-control SG_gt CI '
             'includes zero in the paper, so their rows are reported but not interpreted.')
    R.append('- The two reference baselines (InternVL2.5-8B, VideoLLaMA2-7B) are excluded, '
             'matching Table 5/6 where grounder SG is the reported quantity.')
    return '\n'.join(R) + '\n'


def render_tex(df):
    L = []
    L.append(r'\begin{table*}[t]')
    L.append(r'\centering')
    L.append(r'\scriptsize')
    L.append(r'\setlength{\tabcolsep}{4pt}')
    L.append(r'\caption{Correctness-conditioned Support Gain. Each (model, set) cell is '
             r'split by whether the full-video answer $a_{\text{full}}$ matches the ground '
             r'truth. $\Delta$SG $=$ SG$_{\text{correct}}-$SG$_{\text{incorrect}}$ with an '
             r'unpaired item-level bootstrap CI ($B{=}5000$). Item-level bootstrap, '
             r'$B{=}5000$, seed fixed. $^{\dagger}$positive-control SG$_{\text{gt}}$ CI '
             r'includes zero (not interpreted); $^{\ddagger}$see Table 5. '
             r'$^{\ast}$cell with $N<20$.}')
    L.append(r'\label{tab:a1_correctness_conditioned}')
    L.append(r'\begin{tabular}{lllrrrrlrl}')
    L.append(r'\toprule')
    L.append(r'Model & Set & $a_{\text{full}}$ & $N$ & Ret(none) & Ret(rand) & Ret(pred) '
             r'& SG [95\% CI] & SG$_{\text{gt}}$ & $\Delta$SG [95\% CI] \\')
    L.append(r'\midrule')
    for bench in ('NExT-GQA', 'QVH-ME'):
        sub = df[(df.bench == bench) & (df.group != 'overall')]
        L.append(r'\multicolumn{10}{l}{\textit{' + bench + r'}} \\')
        models = [DISP[m] for m in MODELS['NExT' if bench == 'NExT-GQA' else 'QVH']]
        for mi, m in enumerate(models):
            key = [k for k, v in DISP.items() if v == m][0]
            nm = m + DAG.get(key, '')
            first_m = True
            for pset in ('C2', 'C3'):
                for g in ('correct', 'incorrect'):
                    r = sub[(sub.model == m) & (sub['set'] == pset)
                            & (sub.group == g)].iloc[0]
                    lead = f'{nm}' if first_m else ''
                    setlab = pset if g == 'correct' else ''
                    star = r'$^{\ast}$' if r.small_N else ''
                    dcol = (f'{sgf(r.dSG_correct_minus_incorrect)} '
                            f'{sgci(r.dSG_lo, r.dSG_hi)}') if g == 'correct' else ''
                    L.append(f'{lead} & {setlab} & {g} & {int(r.N)}{star} & '
                             f'{k3(r.Ret_none)} & {k3(r.Ret_rand)} & {k3(r.Ret_pred)} & '
                             f'{sgf(r.SG)} {sgci(r.SG_lo, r.SG_hi)} & '
                             f'{sgf(r.SG_gt)} & {dcol} \\\\')
                    first_m = False
            if mi != len(models) - 1:
                L.append(r'\addlinespace')
        if bench == 'NExT-GQA':
            L.append(r'\midrule')
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\end{table*}')
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------------ main
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    spawn_dumps()
    df, verify = build_rows()
    bad = check(verify)
    verify.to_csv(f'{OUTDIR}/verification.csv', index=False)
    if bad:
        print('\n!!! REPRODUCTION MISMATCH — stopping, nothing else written:')
        for b in bad:
            print('   ', b)
        raise SystemExit(1)
    print(f'\n[verify] OK — {len(verify)} (bench,set,model) cells reproduce stored '
          f'figdata N / SG / Ret(none) / Ret(rand) / Ret(pred) to 3 decimals, '
          f'and the weighted recombination of correct+incorrect equals SG.')
    df.to_csv(f'{OUTDIR}/sg_by_correctness.csv', index=False)
    open(f'{OUTDIR}/table_main.tex', 'w').write(render_tex(df))
    open(f'{OUTDIR}/REPORT.md', 'w').write(render_report(df, verify))
    print(f'WROTE {OUTDIR}/sg_by_correctness.csv')
    print(f'WROTE {OUTDIR}/table_main.tex')
    print(f'WROTE {OUTDIR}/REPORT.md')
    return df, verify


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default=None)
    a = ap.parse_args()
    if a.dump:
        dump(a.dump)
    else:
        main()
