"""B1-2 — CROP vs MASK vs MARGIN on QVH-ME C3, Qwen2.5-VL.

Reads the stored CROP step3 record and the two re-query files written by
b1_requery.py. Scoring is the paper's, taken from
`build_paper_numbers.load_samples`:

    Ret(pred) = 1[keep_pred == a_full]
    Ret(rand) = mean_k 1[rand_k_pred == a_full]
    Ret(none) = 1[blind_pred == a_full]        (shared: blind sees no frames)
    SG        = Ret(pred) - Ret(rand)

`a_full` is the model's own full-video answer and is identical across the three
conditions, so the three are directly comparable question by question.

Two question sets are reported throughout:
  (a) all 241 C3 questions
  (b) the 236 left after dropping the 5 questions that own one of the 9
      windows which select zero frames on the uniform 128-frame grid

In (a) the 9 all-black MASK rows are scored NOT RETAINED, the same rule CROP's
`empty_or_no_interval` rows get, so the two conditions stay comparable; the
model's recorded answer there is a response to an entirely black clip and is
not interpreted. In (b) they are gone.

Bootstrap: item bootstrap over the sorted vqid list, B=5000, seed=0, ONE shared
resample across the three conditions — the same construction as the paper's
`paired_bootstrap_SGpred_CI`, so the per-condition CIs match the paper's
convention and the Delta-SG CIs are paired.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f'{ROOT}/analysis/B1_masking'
CROP_S3 = f'{ROOT}/outputs/qvhme/qwen25vl/step3.json'
B_BOOT, SEED = 5000, 0
NDRAW = 5
WTYPES = ['pred'] + [f'rand_{k}' for k in range(NDRAW)]
CONDS = ['CROP', 'MASK', 'MARGIN']
# Table 5, Qwen2.5-VL / QVH-ME / C3
PAPER = {'Ret_none': 0.303, 'Ret_rand': 0.293, 'Ret_pred': 0.747, 'SG': 0.454}


def crop_label(w):
    return 'predicted_keep' if w == 'pred' else f'r_pred_keep_{w.split("_")[1]}'


# --------------------------------------------------------------- build frame
def build():
    s3 = json.load(open(CROP_S3))
    inputs = [json.loads(l) for l in open(f'{OUT}/inputs.jsonl')]
    questions = sorted({r['vqid'] for r in inputs})
    zf = json.load(open(f'{OUT}/zero_frame_windows.json'))
    zset = {(w['vqid'], w['window_type']) for w in zf['windows']}
    zq = set(zf['questions'])

    resp = {}
    for v in ('mask', 'margin'):
        for r in (json.loads(l) for l in open(f'{OUT}/responses_{v}.jsonl')):
            resp[(v, r['vqid'], r['window_type'])] = r

    rows = []
    for q in questions:
        rec = s3[q]
        af = rec['a_full']
        kbl = int(rec['preds']['blind'] == af)
        # |E_hat| and how many sampled frames fall inside it
        ehat = rec['predicted_iv']
        ehl = float(ehat[1] - ehat[0])
        nfw = resp[('mask', q, 'pred')]['n_frames_in_window']
        row = {'vqid': q, 'a_full': af, 'answer_id': rec['answer_id'],
               'correct': int(af == rec['answer_id']),
               'ehat_len': ehl, 'n_frames_in_ehat': nfw,
               'Ret_none': kbl, 'has_zero_frame_window': q in zq}
        for cond in CONDS:
            ret = {}
            for w in WTYPES:
                if cond == 'CROP':
                    p = rec['preds'][crop_label(w)]
                    k = int(p == af)          # -1 never equals a_full -> 0
                else:
                    rr = resp[(cond.lower(), q, w)]
                    k = int(rr['parsed'] == af)
                    # all-black MASK input: score as CROP's empty window does
                    if cond == 'MASK' and rr.get('all_black'):
                        assert (q, w) in zset
                        k = 0
                ret[w] = k
            row[f'{cond}_Ret_pred'] = ret['pred']
            row[f'{cond}_Ret_rand'] = np.mean([ret[f'rand_{k}'] for k in range(NDRAW)])
            row[f'{cond}_SG'] = row[f'{cond}_Ret_pred'] - row[f'{cond}_Ret_rand']
        rows.append(row)
    return pd.DataFrame(rows), zf


# ----------------------------------------------------------------- bootstrap
def boot_idx(n):
    return np.random.default_rng(SEED).integers(0, n, (B_BOOT, n))


def ci(vals, idx):
    m = np.asarray(vals, float)[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ci_paired_diff(a, b, idx):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ------------------------------------------------------------------ analysis
def summarize(df, label):
    d = df.sort_values('vqid')
    idx = boot_idx(len(d))
    out = []
    for cond in CONDS:
        sg = d[f'{cond}_SG'].values
        lo, hi = ci(sg, idx)
        r = {'set': label, 'n': len(d), 'condition': cond,
             'Ret_none': d.Ret_none.mean(),
             'Ret_rand': d[f'{cond}_Ret_rand'].mean(),
             'Ret_pred': d[f'{cond}_Ret_pred'].mean(),
             'SG': sg.mean(), 'SG_lo': lo, 'SG_hi': hi}
        if cond == 'CROP':
            r.update(dSG=np.nan, dSG_lo=np.nan, dSG_hi=np.nan)
        else:
            dlo, dhi = ci_paired_diff(sg, d['CROP_SG'].values, idx)
            r.update(dSG=sg.mean() - d.CROP_SG.mean(), dSG_lo=dlo, dSG_hi=dhi)
        out.append(r)
    return pd.DataFrame(out)


def disagreement(df, cond):
    """Question-level Ret(pred) agreement between CROP and `cond`."""
    a, b = df['CROP_Ret_pred'].values, df[f'{cond}_Ret_pred'].values
    same = int((a == b).sum())
    rows = []
    for name, m in [(f'agree (CROP == {cond})', a == b),
                    (f'CROP kept, {cond} not', (a == 1) & (b == 0)),
                    (f'CROP not, {cond} kept', (a == 0) & (b == 1)),
                    (f'any disagreement', a != b)]:
        s = df[m]
        rows.append({'condition': cond, 'direction': name, 'n': int(m.sum()),
                     'median_ehat_len': float(s.ehat_len.median()) if len(s) else np.nan,
                     'median_n_frames_in_ehat':
                         float(s.n_frames_in_ehat.median()) if len(s) else np.nan})
    return same, len(df), pd.DataFrame(rows)


def fmt(x, d=3):
    return f'{x:.{d}f}'.replace('0.', '.', 1) if abs(x) < 1 else f'{x:.{d}f}'


def signed(x, d=3):
    s = f'{abs(x):.{d}f}'
    s = s.replace('0.', '.', 1) if abs(x) < 1 else s
    return ('$+$' if x >= 0 else '$-$') + s


def main():
    df, zf = build()
    os.makedirs(OUT, exist_ok=True)

    df_a = df
    df_b = df[~df.has_zero_frame_window]
    assert len(df_a) == 241 and len(df_b) == 236, (len(df_a), len(df_b))

    # ---- reproduction gate: (a) CROP must be Table 5
    sa = summarize(df_a, 'a_all241')
    crop = sa[sa.condition == 'CROP'].iloc[0]
    got = {'Ret_none': crop.Ret_none, 'Ret_rand': crop.Ret_rand,
           'Ret_pred': crop.Ret_pred, 'SG': crop.SG}
    print('--- reproduction check (set a, CROP) vs Table 5')
    bad = []
    for k, v in PAPER.items():
        ok = abs(got[k] - v) < 5e-4
        print(f'    {k:9} got {got[k]:.4f}  paper {v:.3f}  {"OK" if ok else "MISMATCH"}')
        if not ok:
            bad.append(k)
    if bad:
        raise SystemExit(f'reproduction FAILED on {bad}; nothing written')

    sb = summarize(df_b, 'b_excl5_236')
    summary = pd.concat([sa, sb], ignore_index=True)
    summary.to_csv(f'{OUT}/summary.csv', index=False)
    print('\n--- summary')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    df.to_csv(f'{OUT}/per_question.csv', index=False)

    # ---- question-level agreement
    print('\n--- question-level Ret(pred) agreement (set a, N=241)')
    print(f'    overall median |E_hat| = {df_a.ehat_len.median():.1f} s, '
          f'median frames in E_hat = {df_a.n_frames_in_ehat.median():.0f}')
    dis = []
    for cond in ('MASK', 'MARGIN'):
        same, n, t = disagreement(df_a, cond)
        print(f'    CROP vs {cond}: agree {same}/{n} = {same / n:.3f}')
        print(t.to_string(index=False))
        t.insert(0, 'agree_n', same)
        t.insert(1, 'agree_frac', same / n)
        dis.append(t)
    dis = pd.concat(dis, ignore_index=True)
    dis['overall_median_ehat_len'] = df_a.ehat_len.median()
    dis['overall_median_n_frames_in_ehat'] = df_a.n_frames_in_ehat.median()
    dis.to_csv(f'{OUT}/disagreement.csv', index=False)

    write_tex(sa, sb)
    print(f'\nwrote {OUT}/summary.csv, per_question.csv, disagreement.csv, table_b1.tex')
    return sa, sb, dis, df_a


def write_tex(sa, sb):
    A = sa.set_index('condition')
    B = sb.set_index('condition')
    disp = {'CROP': 'CROP (paper)', 'MASK': 'MASK',
            'MARGIN': r'MARGIN $\pm$5\,s'}
    lines = []
    for c in CONDS:
        a, b = A.loc[c], B.loc[c]
        d = ('---' if c == 'CROP' else
             f'{signed(a.dSG)} [{signed(a.dSG_lo)}, {signed(a.dSG_hi)}]')
        lines.append(
            f'{disp[c]} & {fmt(a.Ret_none)} & {fmt(a.Ret_rand)} & {fmt(a.Ret_pred)} & '
            f'{signed(a.SG)} [{signed(a.SG_lo)}, {signed(a.SG_hi)}] ({signed(b.SG)}) & {d} \\\\')
    tex = r"""% B1-2: cropping-perturbation control. Qwen2.5-VL, QVH-ME C3.
% Generated by analysis/camera_ready/b1_analyze.py -- do not edit by hand.
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\caption{\textbf{The Keep intervention without cropping.} Qwen2.5-VL on the
QVH-ME $\mathcal{C}_3$ set ($N=241$). CROP (the protocol used in the paper)
shows only the frames inside the window, so the frame count shrinks;
MASK keeps the full 128-frame grid and replaces every frame outside the window
with black, so frame count, frame positions and the absolute-time encoding are
identical to the full-video query; MARGIN crops, but to the window grown by
5\,s on each side. Ret(none) sees no frames and is therefore shared. The
parenthesised SG is the same quantity on the 236 questions left after dropping
the 5 that own a window shorter than the 1.18\,s frame spacing. 95\% CIs:
item bootstrap, $B{=}5000$, seed 0, one resample shared across conditions, so
the $\Delta$SG intervals are paired. Nine random windows fall between grid
points: CROP has no frames to show there and scores them not retained, and the
nine corresponding MASK inputs are entirely black, so they are scored by the
same rule and their responses are not interpreted.}
\label{tab:b1_masking}
\begin{tabular}{lcccrr}
\toprule
Condition & Ret(none) & Ret(rand) & Ret(pred) & SG [95\% CI] & $\Delta$SG vs.\ CROP [CI] \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f'{OUT}/table_b1.tex', 'w').write(tex)


if __name__ == '__main__':
    main()
