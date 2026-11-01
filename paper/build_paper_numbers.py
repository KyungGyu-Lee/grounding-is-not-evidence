"""build_paper_numbers.py — single reproducible builder for ALL paper numbers.

Replaces the two builder-less static artifacts (figdata.json, master_*.csv) and
pins every previously-inline analysis to one script. READS stored per-question
results only; never re-runs a model.

Paths below are repo-relative; {bench} is nextgqa or qvhme.

INPUTS (stored values only):
  outputs/{bench}/{model}/step3.json            per grounder + reference model
  outputs/{bench}/{model}/step1.json            iou/iop/a_full
  outputs/{bench}/{model}/gt_ceiling_shard*.json            K_GT / Ret(rand_gt)
  outputs/{bench}/{qwen25vl,internvl,videollama2}/rand_gt_keep*.json
                                                Ret(rand_gt) reuse (3 reference)
  flags/{bench}/blind_{qwen25vl,internvl2_5_8b,videollama2_7b}.json      b
  flags/{bench}/fullvideo_{model}_shard*.json                            u
  flags/{bench}/r_norm.json                     C2 membership + rho

OUTPUT:
  paper/figdata_{bench}.json
  paper/tables/master_{bench}.csv, paper/tables/ablation_table2_*.csv

SET DEFINITIONS (benchmark-level, single function `set_masks`; PAPER naming,
NExT-GQA / QVH-ME):
  ALL = all step3 vqids                                        (2268 / 763)
  C1  = prior-exclusion     : mean3(blind correct) < 1.0       (1526 / 560)
  C2  = answerability       : mean3(u - b) >= 1/3              (1116 / 427)
        (verified == stored r_norm keys; assertion in code)
  C3  = evidence-specificity: C2 AND rho <= C3_THRESHOLD        (248 / 241)
  Nested: C3 subset of C2 subset of C1.
  3 reference models = Qwen2.5-VL + InternVL2.5-8B + VideoLLaMA2-7B.
  ** Percentile (bottom-N%) cuts are NOT used ** — a 30% cut landed on the
  r_norm=0.6 tie mass (71 tied), which produced the 0.105/0.0994/0.101 triple
  mismatch. A fixed threshold removes that ambiguity.

MISSING / NaN POLICY:
  * step3 record with 'error' key -> dropped.
  * SG needs a_full (not None, >=0), 'predicted_keep', 'blind', >=1 r_pred_keep.
    K_R = mean over the r_pred_keep trials actually present (nominally 5).
  * IoP/IoU from step1; if None the sample is excluded ONLY from IoP-based items
    (bins / Spearman / IoP=1.00 / dIoP pairs), still used for SG/components.
  * K = agreement with the model's OWN a_full (grounder: step1 a_full;
    baseline: full-video pred).  correct = 1[a_full == answer_id].
  * K_GT/K_R_gt from gt_ceiling; Qwen & the two reference models reuse the
    stored rand_gt_keep for K_R_gt (byte-identical, determinism verified).
  * Bootstrap: resample vqids WITH REPLACEMENT, seed=0, B=5000; paired = one
    resample shared across models.
  * Every reported quantity carries its N.  Model averages given BOTH as
    mean-of-per-model-means and pooled (equal iff balanced).
"""
import os, sys, json, glob, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

C3_THRESHOLD = 0.4             # paper notation: rho <= 0.4 (use +EPS below, float-safe)
MEAS_EPS = 1e-9               # rho at 0.4 straddles float (0.399..986 ~ 0.400..001); u-b multiples of 1/3
B_BOOT = 5000
SEED = 0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)              # every data path below is repo-relative
BDIR = {'NExT': 'nextgqa', 'QVH': 'qvhme'}
BENCHES = os.environ.get('SG_BENCHES', 'NExT,QVH').split(',')

_QVH_G = [('Qwen', 'qwen25vl'), ('VideoMind', 'videomind'),
          ('GroundVTS', 'groundvts'), ('GVLLM', 'grounded_videollm'),
          ('VTimeLLM', 'vtimellm'), ('TimeChat', 'timechat')]
_QVH_G.append(('VideoChat-R1', 'videochat_r1'))
GROUNDERS = {
    'NExT': [('Qwen', 'qwen25vl'), ('VideoMind', 'videomind'),
             ('GroundVTS', 'groundvts'), ('GVLLM', 'grounded_videollm'),
             ('VTimeLLM', 'vtimellm')],
    'QVH':  _QVH_G,
}
BASELINES = [('InternVL', 'internvl'), ('VideoLLaMA2', 'videollama2')]


def _ls(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        d.update(json.load(open(f)))
    return d


def _s3(bench, dr):
    return f'outputs/{BDIR[bench]}/{dr}/step3.json'


def _s1(bench, dr):
    return f'outputs/{BDIR[bench]}/{dr}/step1.json'


def _gt(bench, dr):
    return _ls(f'outputs/{BDIR[bench]}/{dr}/gt_ceiling_shard*.json')


def iv_len(iv):
    return (iv[1] - iv[0]) if iv else np.nan


# ---------------------------------------------------------------- set masks
def set_masks(bench):
    """Return dict ALL/C1/C2/C3 -> set(vqid) in PAPER naming, plus r_norm.

    C1 = prior-exclusion (b < 1), C2 = answerability (u - b >= 1/3),
    C3 = evidence-specificity (rho <= 0.4). Nested C3 < C2 < C1.
    """
    if bench == 'NExT':
        bl = {k: json.load(open(f'flags/nextgqa/blind_{f}.json'))
              for k, f in [('q', 'qwen25vl'), ('i', 'internvl2_5_8b'), ('v', 'videollama2_7b')]}
        s1q = json.load(open('outputs/nextgqa/qwen25vl/step1.json'))
        full_q = {k: int(v['a_full'] == v['answer_id']) for k, v in s1q.items()}
        full_i = _ls('flags/nextgqa/fullvideo_internvl_shard[0-9]*.json')
        full_v = _ls('flags/nextgqa/fullvideo_videollama2_shard[0-9]*.json')
        rn = json.load(open('flags/nextgqa/r_norm.json'))['r_norm']
    else:
        bl = {k: json.load(open(f'flags/qvhme/blind_{f}.json'))
              for k, f in [('q', 'qwen25vl'), ('i', 'internvl2_5_8b'), ('v', 'videollama2_7b')]}
        full_q = {k: v['correct'] for k, v in _ls('flags/qvhme/fullvideo_qwen25vl_shard[0-9]*.json').items()}
        full_i = _ls('flags/qvhme/fullvideo_internvl_shard[0-9]*.json')
        full_v = _ls('flags/qvhme/fullvideo_videollama2_shard[0-9]*.json')
        rn = json.load(open('flags/qvhme/r_norm.json'))['r_norm']
    ALL = set(bl['q'])
    def bmean(i): return (bl['q'][i]['correct'] + bl['i'][i]['correct'] + bl['v'][i]['correct']) / 3
    C1 = {i for i in ALL if bmean(i) < 1.0}
    basis = set(full_q) & set(full_i) & set(full_v) & ALL
    def fmean(i): return (full_q[i] + full_i[i]['correct'] + full_v[i]['correct']) / 3
    C2 = {i for i in basis if fmean(i) - bmean(i) >= 1/3 - MEAS_EPS}   # u-b >= 1/3
    assert C2 == set(rn), f'{bench}: C2(u-b>=1/3) != r_norm keys ({len(C2)} vs {len(rn)})'
    C3 = {i for i in C2 if rn[i] <= C3_THRESHOLD + MEAS_EPS}           # rho <= 0.4
    return {'ALL': ALL, 'C1': C1, 'C2': C2, 'C3': C3}, rn


# ------------------------------------------------------- per-sample dataframe
def load_samples(bench):
    rn = json.load(open(f'flags/{BDIR[bench]}/r_norm.json'))['r_norm']
    rows = []
    models = GROUNDERS[bench] + BASELINES
    for disp, dr in models:
        is_base = disp in [b[0] for b in BASELINES]
        s3 = json.load(open(_s3(bench, dr)))
        # step1 loaded for EVERY model (grounders + baselines) so baseline IoU/IoP
        # are available for the model-level Spearman sensitivity (baseline in/out).
        # NOTE: baseline a_full still = full-video pred (afb), NOT step1 a_full.
        s1 = json.load(open(_s1(bench, dr)))
        # a_full for baselines = full-video pred
        if is_base:
            if bench == 'NExT':
                afb = {k: v['pred'] for k, v in _ls(f'flags/nextgqa/fullvideo_{dr}_shard[0-9]*.json').items()}
            else:
                afb = {k: v['pred'] for k, v in _ls(f'flags/qvhme/fullvideo_{dr}_shard[0-9]*.json').items()}
        gt = _gt(bench, dr)
        # K_R_gt reuse sources
        if disp == 'Qwen':
            krg_src = (json.load(open('outputs/nextgqa/qwen25vl/rand_gt_keep.json')) if bench == 'NExT'
                       else _ls('outputs/qvhme/qwen25vl/rand_gt_keep_shard*.json'))
        elif is_base:
            krg_src = (json.load(open(f'outputs/nextgqa/{dr}/rand_gt_keep.json')) if bench == 'NExT'
                       else _ls(f'outputs/qvhme/{dr}/rand_gt_keep_shard*.json'))
        else:
            krg_src = None  # from gt_ceiling krgt
        for v, rec in s3.items():
            if 'error' in rec:
                continue
            p = rec.get('preds', {})
            af = afb.get(v) if is_base else rec.get('a_full')
            if af is None or af < 0 or 'predicted_keep' not in p or 'blind' not in p:
                continue
            rk = [p[f'r_pred_keep_{k}'] for k in range(5) if f'r_pred_keep_{k}' in p]
            if not rk:
                continue
            KE = int(p['predicted_keep'] == af)
            KR = sum(int(x == af) for x in rk) / len(rk)
            Kbl = int(p['blind'] == af)
            iou = rec.get('predicted_iv'); ehl = iv_len(rec.get('predicted_iv'))
            gtl = iv_len(rec.get('oracle_iv'))
            one = s1.get(v, {})
            iou_v = one.get('iou'); iop_v = one.get('iop')
            answer_id = rec.get('answer_id')
            # K_GT / K_R_gt
            KGT = KRG = np.nan
            if v in gt and 'kgt' in gt[v]:
                KGT = int(gt[v]['kgt']['pred'] == af)
                if krg_src is not None and v in krg_src:
                    ps = [t['pred'] for t in krg_src[v]]
                elif not is_base and disp != 'Qwen' and 'krgt' in gt[v]:
                    ps = [t['pred'] for t in gt[v]['krgt'] if not t.get('skip')]
                else:
                    ps = None
                if ps:
                    KRG = sum(int(x == af) for x in ps) / len(ps)
            rows.append(dict(bench=bench, vqid=v, model=disp, is_base=is_base,
                             a_full=int(af), answer_id=int(answer_id),
                             correct=int(af == answer_id),
                             iou=iou_v, iop=iop_v, ehl=ehl, gtl=gtl,
                             KE=KE, KR=round(KR, 6), Kbl=Kbl, SG=round(KE - KR, 6),
                             KGT=KGT, KRG=KRG, rnorm=rn.get(v, np.nan)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- helpers
def mean_pair(series_by_model):
    """(mean-of-means, pooled) — series_by_model: dict model->np.array."""
    perm = [a.mean() for a in series_by_model.values() if len(a)]
    pooled = np.concatenate([a for a in series_by_model.values() if len(a)]) if series_by_model else np.array([])
    return (float(np.mean(perm)) if perm else np.nan,
            float(pooled.mean()) if len(pooled) else np.nan)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(pd.Series(x).corr(pd.Series(y), method='spearman'))


def boot_diff(a, b, seed=SEED, B=B_BOOT):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed); n = len(a); idx = rng.integers(0, n, (B, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def boot_prop(vals, seed=SEED, B=B_BOOT):
    """95% CI of a proportion/mean by resampling with replacement (B, seed)."""
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), (B, len(vals)))
    m = vals[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_ratio(num_a, num_b, den_a, den_b, seed=SEED, B=B_BOOT):
    """CI of (mean(num_a)-mean(num_b)) / (mean(den_a)-mean(den_b)), paired."""
    n = len(num_a)
    if n < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed); idx = rng.integers(0, n, (B, n))
    num_a = np.asarray(num_a, float); num_b = np.asarray(num_b, float)
    den_a = np.asarray(den_a, float); den_b = np.asarray(den_b, float)
    sgp = num_a[idx].mean(1) - num_b[idx].mean(1)
    sgg = den_a[idx].mean(1) - den_b[idx].mean(1)
    r = np.where(np.abs(sgg) > 1e-9, sgp / sgg, np.nan)
    r = r[~np.isnan(r)]
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))) if len(r) > 200 else (np.nan, np.nan)


# --------------------------------------------------------------- computations
def compute(bench, df, masks, rn):
    G = [g[0] for g in GROUNDERS[bench]]
    ALLM = G + [b[0] for b in BASELINES]
    out = {}
    # 1. funnel N
    out['funnel'] = {k: len(masks[k]) for k in ['ALL', 'C1', 'C2', 'C3']}
    # 1b. ablation table over ALL 4 cuts (Table 2): grounder mean SG, both
    #     mean-of-means and pooled, N on every cut. Baselines excluded (their SG
    #     is undefined by design — item 5), so this is a grounders-only ablation.
    out['ablation_table'] = {}
    for cut in ['ALL', 'C1', 'C2', 'C3']:
        ids = masks[cut]
        gA = df[(df.vqid.isin(ids)) & (df.model.isin(G))]
        sbm = {m: gA[gA.model == m].SG.values for m in G}
        mom, pooled = mean_pair(sbm)
        out['ablation_table'][cut] = dict(
            mean_of_means=mom, pooled=pooled, N_vqid=len(ids), N_rows=len(gA),
            per_model={m: (float(gA[gA.model == m].SG.mean())
                           if len(gA[gA.model == m]) else None) for m in G},
            N_per_model={m: int(len(gA[gA.model == m])) for m in G})
    # 'ALL' is carried through the SAME code path as C2/C3 so the audit table
    # (IoP=1.00 block, Spearman) has an unfiltered column; nothing downstream
    # reads out['ALL'] except that table, and out['C2']/out['C3'] are untouched.
    for cut in ['ALL', 'C2', 'C3']:
        ids = masks[cut]
        g = df[(df.vqid.isin(ids)) & (df.model.isin(G))]
        gb = df[(df.vqid.isin(ids)) & (df.model.isin(ALLM))]
        cutd = {}
        # 2. ablation mean SG (grounders)
        sbm = {m: g[g.model == m].SG.values for m in G}
        mom, pooled = mean_pair(sbm)
        cutd['ablation_SG'] = dict(mean_of_means=mom, pooled=pooled,
                                   N_vqid=len(ids), N_rows=len(g))
        # 3. |dIoP|<=0.1 pairs (grounders, both IoP>=0.5; and both>=0.8)
        import itertools
        npair = ks = np8 = ks8 = 0
        for v, gg in g.groupby('vqid'):
            rr = gg[gg.iop >= 0.5][['iop', 'KE']].values
            for (i1, k1), (i2, k2) in itertools.combinations(rr, 2):
                if abs(i1 - i2) <= 0.1:
                    npair += 1; ks += (k1 != k2)
                    if i1 >= 0.8 and i2 >= 0.8:
                        np8 += 1; ks8 += (k1 != k2)
        p = g[g.iop >= 0.5].KE.mean()
        cutd['dIoP_pairs'] = dict(n_pairs=int(npair), discord=float(ks / npair) if npair else np.nan,
                                  n_pairs_hi=int(np8), discord_hi=float(ks8 / np8) if np8 else np.nan,
                                  baseline_2p1p=float(2 * p * (1 - p)), p_KE=float(p))
        # 4. IoP bins K_E=1
        bins = {}
        for lo, hi in [(0.5, 0.7), (0.7, 0.9), (0.9, 1.001)]:
            b = g[(g.iop >= lo) & (g.iop < hi)]
            bins[f'[{lo},{hi if hi < 1 else 1.0})'] = dict(N=len(b), KE1=float(b.KE.mean()) if len(b) else np.nan)
        cutd['IoP_bins'] = bins
        # 5. IoP=1.00
        s1 = g[g.iop >= 0.999]
        cutd['IoP_eq1'] = dict(N=len(s1),
                               KE0_rate=float((s1.KE == 0).mean()) if len(s1) else np.nan,
                               SGle0_rate=float((s1.SG <= 0).mean()) if len(s1) else np.nan)
        # 6. Spearman(IoP,SG): per-model / pooled / model-level(mIoU & mIoP,SG)
        #    Model-level computed BOTH over grounders only and grounders+baselines
        #    (baseline IoU/IoP now loaded; baseline SG is raw KE-KR even though it
        #    is display-blanked in components — this is the sensitivity the user
        #    asked for). Each carries n = number of model points.
        per = {m: spearman(g[g.model == m].iop.values, g[g.model == m].SG.values) for m in G}
        pooled_sp = spearman(g.iop.values, g.SG.values)

        def _mlvl(models, metric):
            xs, ys = [], []
            for m in models:
                sub = gb[gb.model == m]
                mv = sub[metric].dropna()
                if not len(mv):
                    continue
                xs.append(float(mv.mean())); ys.append(float(sub.SG.mean()))
            rho = spearman(xs, ys)
            return (None if rho != rho else rho), len(xs)

        ml = {}
        for tag, mods in [('grounders', G), ('with_baseline', ALLM)]:
            for metric in ['iop', 'iou']:
                rho, n = _mlvl(mods, metric)
                ml[f'm{"IoP" if metric=="iop" else "IoU"}_{tag}'] = dict(rho=rho, n=n)
        cutd['spearman'] = dict(per_model={m: (None if per[m] != per[m] else per[m]) for m in G},
                                pooled=(None if pooled_sp != pooled_sp else pooled_sp),
                                model_level=ml,
                                # back-compat scalar keys (grounders-only)
                                model_level_mIoP=ml['mIoP_grounders']['rho'],
                                model_level_mIoU=ml['mIoU_grounders']['rho'],
                                N_rows=len(g))
        # 7. positive control per model (grounders + baselines)
        pc = {}
        for m in ALLM:
            mm = gb[gb.model == m].dropna(subset=['KGT', 'KRG'])
            if not len(mm):
                continue
            kgt = mm.KGT.values; krg = mm.KRG.values
            sgg = kgt.mean() - krg.mean()
            lo, hi = boot_diff(kgt, krg, seed=SEED)
            rec = dict(N=len(mm), K_R_gt=float(krg.mean()), K_GT=float(kgt.mean()),
                       SG_GT=float(sgg), SG_GT_CI=[lo, hi])
            if m in G:
                mmp = gb[gb.model == m]
                sgp = mmp.KE.mean() - mmp.KR.mean()
                rec['SG_pred'] = float(sgp)
                # ESR is FORCED undefined unless the positive control is
                # POSITIVE, i.e. the whole SG_GT 95% CI lies above 0. The old
                # test ('CI includes 0') let an interval lying entirely BELOW
                # zero through, which produced ESR = SG_pred/SG_GT as a ratio of
                # two negatives (VTimeLLM, QVH-ME C2: 0.136). Matches the printed
                # tables, which mark VTimeLLM and TimeChat undefined by name.
                if not (lo > 0):
                    rec['ESR'] = None
                    rec['ESR_CI'] = None
                    rec['ESR_undefined'] = ('SG_GT 95% CI does not lie above 0'
                                            if hi < 0 else
                                            'SG_GT 95% CI includes 0')
                else:
                    rec['ESR'] = float(sgp / sgg) if abs(sgg) > 1e-9 else None
                    elo, ehi = boot_ratio(mm.KE.values, mm.KR.values,
                                          mm.KGT.values, mm.KRG.values, seed=SEED)
                    rec['ESR_CI'] = [elo, ehi]
            pc[m] = rec
        cutd['positive_control'] = pc
        # 8. per-model SG + components (grounders + baselines)
        #    Baseline SG is BLANKED (item 5): InternVL/VideoLLaMA2 are measurability
        #    instruments (blind/full), not evaluated grounders. Their E_hat is
        #    incidental/degenerate (VideoLLaMA2 QVH: 46% full-video E_hat), so
        #    SG_pred = K_E - K_R conflates a length artifact with grounding and is
        #    undefined by design. K_blind/K_R/K_E are kept (raw, for transparency);
        #    SG_GT (positive control) is retained separately (GT-based, does NOT
        #    use their E_hat).
        base_names = [b[0] for b in BASELINES]
        comp = {}
        for m in ALLM:
            mm = gb[gb.model == m]
            is_b = m in base_names
            comp[m] = dict(N=len(mm), K_blind=float(mm.Kbl.mean()), K_R=float(mm.KR.mean()),
                           K_E=float(mm.KE.mean()),
                           SG=(None if is_b else float(mm.SG.mean())))
            if is_b:
                comp[m]['SG_note'] = ('undefined: baseline not evaluated as grounder '
                                      '(E_hat degenerate; SG conflates length artifact)')
        cutd['components'] = comp
        # 9. SGA / UGA(A:SG=0, B:SG<0) — all models incl baselines.
        #    Bootstrap 95% CI (B=5000, seed) on SGA / UGA_A / UGA_B so the
        #    uncertainty of small N_gc cells (e.g. QVH C3) is visible.
        sga = {}
        for m in ALLM:
            mm = gb[gb.model == m]
            gc = mm[(mm.correct == 1) & (mm.iop >= 0.5)]
            n = len(gc)
            sga[m] = dict(N_grounded_correct=n,
                          SGA=float((gc.SG > 0).mean()) if n else np.nan,
                          SGA_CI=boot_prop((gc.SG > 0).astype(float).values) if n else [np.nan, np.nan],
                          UGA_A_eq0=float((gc.SG == 0).mean()) if n else np.nan,
                          UGA_A_CI=boot_prop((gc.SG == 0).astype(float).values) if n else [np.nan, np.nan],
                          UGA_B_lt0=float((gc.SG < 0).mean()) if n else np.nan,
                          UGA_B_CI=boot_prop((gc.SG < 0).astype(float).values) if n else [np.nan, np.nan],
                          UGA_total=float((gc.SG <= 0).mean()) if n else np.nan)
        cutd['SGA_UGA'] = sga
        # 10. Type2 / Case(b) / hardcore (grounders)
        t2 = g[(g.rnorm <= 0.4) & (g.Kbl == 0) & (g.KR >= 0.6) & (g.KE == 1) & (g.iop >= 0.5) & (g.correct == 1)]
        cb = g[(g.correct == 1) & (g.iop >= 0.5) & (g.KE == 0) & (g.KR > 0)]
        hc = cb[(cb.iop >= 0.8) & (cb.ehl >= 2) & (cb.rnorm < C3_THRESHOLD)]
        cutd['type2'] = dict(instances=len(t2), vqids=int(t2.vqid.nunique()),
                             pct_inst=float(len(t2) / len(g)) if len(g) else np.nan)
        cutd['case_b'] = dict(instances=len(cb), vqids=int(cb.vqid.nunique()),
                              pct_grounded_correct=float(len(cb) / len(g[(g.correct == 1) & (g.iop >= 0.5)]))
                              if len(g[(g.correct == 1) & (g.iop >= 0.5)]) else np.nan)
        cutd['hardcore'] = dict(instances=len(hc), vqids=int(hc.vqid.nunique()))
        # 11. length-bias 4 diagnostics (grounders)
        ehl = g.ehl.dropna(); gtl = g.gtl.dropna()
        rpred_len = []  # mean |R_pred| per row from step3 not stored here; use ehl-match proxy skipped
        lb = {}
        lb['mean_Ehat_len'] = float(ehl.mean()); lb['mean_GT_len'] = float(gtl.mean())
        lb['Ehat_minus_GT'] = float(ehl.mean() - gtl.mean())
        lb['spearman_Ehat_SG'] = spearman(g.dropna(subset=['ehl']).ehl.values, g.dropna(subset=['ehl']).SG.values)
        ke1 = g[(g.KE == 1)].ehl.dropna(); ke0 = g[(g.KE == 0)].ehl.dropna()
        lb['mean_Ehat_KE1'] = float(ke1.mean()) if len(ke1) else np.nan
        lb['mean_Ehat_KE0'] = float(ke0.mean()) if len(ke0) else np.nan
        lb['KE0_minus_KE1_len'] = lb['mean_Ehat_KE0'] - lb['mean_Ehat_KE1']
        cutd['length_bias'] = lb
        # 12. paired bootstrap CI of SG_pred per model (shared resample across models)
        vqid_list = sorted(ids)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(vqid_list), (B_BOOT, len(vqid_list)))
        # per-model SG indexed by vqid
        sgmap = {m: g[g.model == m].set_index('vqid').SG for m in G}
        pboot = {}
        for m in G:
            s = sgmap[m].reindex(vqid_list).values
            valid = ~np.isnan(s)
            if valid.sum() < 2:
                pboot[m] = [np.nan, np.nan]; continue
            sv = np.where(valid, s, np.nan)
            samp = sv[idx]
            means = np.nanmean(samp, axis=1)
            pboot[m] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
        cutd['paired_bootstrap_SGpred_CI'] = pboot
        out[cut] = cutd
    return out


# -------------------------------------------------------------------- outputs
def main():
    figdata_v2 = {'meta': dict(C3_THRESHOLD=C3_THRESHOLD, B_BOOT=B_BOOT, SEED=SEED,
                               baseline3='Qwen+InternVL+VideoLLaMA2')}
    master_rows = {'NExT': [], 'QVH': []}
    for bench in BENCHES:
        masks, rn = set_masks(bench)
        df = load_samples(bench)
        res = compute(bench, df, masks, rn)
        figdata_v2[bench] = res
        # master csv rows (per model x cut, components + positive control)
        G = [g[0] for g in GROUNDERS[bench]] + [b[0] for b in BASELINES]
        for cut in ['C2', 'C3']:
            comp = res[cut]['components']; pc = res[cut]['positive_control']
            for m in G:
                c = comp.get(m, {}); p = pc.get(m, {})
                master_rows[bench].append(dict(
                    bench=bench, cut=cut, model=m, N=c.get('N'),
                    K_blind=c.get('K_blind'), K_R=c.get('K_R'), K_E=c.get('K_E'), SG=c.get('SG'),
                    K_R_gt=p.get('K_R_gt'), K_GT=p.get('K_GT'), SG_GT=p.get('SG_GT'),
                    SG_pred=p.get('SG_pred'), ESR=p.get('ESR')))
    os.makedirs('paper/tables', exist_ok=True)
    for b in BENCHES:
        json.dump({'meta': figdata_v2['meta'], b: figdata_v2[b]},
                  open(f'paper/figdata_{BDIR[b]}.json', 'w'), indent=1, default=float)
    for bench, fn in [(b, f'master_{BDIR[b]}.csv') for b in BENCHES]:
        cols = ['bench', 'cut', 'model', 'N', 'K_blind', 'K_R', 'K_E', 'SG',
                'K_R_gt', 'K_GT', 'SG_GT', 'SG_pred', 'ESR']
        with open(f'paper/tables/{fn}', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for r in master_rows[bench]:
                w.writerow({k: r.get(k) for k in cols})
    # Table 2 ablation CSV (ALL/C1/C2/C3 x bench): grounder mean SG
    _abn = 'ablation_table2_' + '_'.join(BDIR[b] for b in BENCHES) + '.csv'
    with open(f'paper/tables/{_abn}', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['bench', 'cut', 'N_vqid', 'N_rows', 'mean_of_means_SG', 'pooled_SG'])
        for bench in BENCHES:
            for cut in ['ALL', 'C1', 'C2', 'C3']:
                a = figdata_v2[bench]['ablation_table'][cut]
                w.writerow([bench, cut, a['N_vqid'], a['N_rows'],
                            f"{a['mean_of_means']:.3f}", f"{a['pooled']:.3f}"])
    for b in BENCHES:
        print(f'WROTE paper/figdata_{BDIR[b]}.json, paper/tables/master_{BDIR[b]}.csv')
    print(f'WROTE paper/tables/{_abn}')
    return figdata_v2


if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--bench', choices=['nextgqa', 'qvhme', 'all'], default='all')
    _a = _ap.parse_args()
    if _a.bench != 'all':
        BENCHES = [b for b, d in BDIR.items() if d == _a.bench]
    main()
