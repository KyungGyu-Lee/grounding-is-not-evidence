# C2 — confidence intervals for the positive control SG_gt

Produced by [analysis/camera_ready/c2_sggt_ci.py](analysis/camera_ready/c2_sggt_ci.py).
No model was re-run and no existing file was modified. `load_samples`,
`set_masks` and both bootstrap routines are imported or copied verbatim from
`build_paper_numbers`, so every number comes out of the paper's own code path:

* **SG_gt CI** — `boot_diff(KGT, KRG, seed=0, B=5000)`: resample the model's
  questions with replacement, take `mean(KGT) − mean(KRG)` on each resample.
  This is the interval stored as `SG_GT_CI` and the one the dagger rule tests
  (`if lo <= 0 <= hi: ESR undefined`).
* **SG CI** — the `paired_bootstrap_SGpred_CI` construction: resample the cut's
  full sorted vqid list once, shared across models, nanmean of each model's SG.
  Recomputed only as a reproduction gate.

24 cells: NExT-GQA × 5 models × {C2, C3} and QVH-ME × 7 models × {C2, C3}.

## (1) Reproduction

**All gates pass.** The script writes nothing if any fails.

* **SG_gt point estimates** — all 24 match Table S2 to three decimals
  (NExT-GQA C2/C3: Qwen +.118/+.455, VideoMind +.064/+.182,
  Grounded-VideoLLM +.089/+.231, GroundVTS +.059/+.172, VTimeLLM +.002/+.009;
  QVH-ME C2/C3: VideoChat-R1 +.439/+.642, Qwen +.445/+.647,
  Grounded-VideoLLM +.290/+.381, VideoMind +.244/+.291, GroundVTS +.301/+.400,
  TimeChat +.017/+.008, VTimeLLM −.045/−.045).
* **SG CI spot checks** — all five match: Qwen NExT-GQA C2 [+.061, +.109] and
  C3 [+.227, +.361]; VideoChat-R1 QVH-ME C2 [+.310, +.406] and
  C3 [+.443, +.563]; VTimeLLM QVH-ME C3 [−.062, +.012].
* **Against the stored `figdata`** — every SG_gt, both SG_gt bounds, both SG
  bounds and N, for all 24 cells (120 numbers): **worst deviation 0.00e+00**.
  The recomputation is bit-identical, so the seed and the resampling scheme are
  confirmed to be the paper's, not merely close.

## (2) Cells whose SG_gt CI includes zero

Five of 24, not the expected six:

| Dataset | Set | Model | N | SG_gt [95% CI] |
|---|---|---|---|---|
| NExT-GQA | C2 | VTimeLLM† | 1110 | +.002 [−.015, +.019] |
| NExT-GQA | C3 | VTimeLLM† | 246 | +.009 [−.033, +.050] |
| QVH-ME | C2 | TimeChat† | 420 | +.017 [−.034, +.070] |
| QVH-ME | C3 | TimeChat† | 236 | +.008 [−.069, +.086] |
| QVH-ME | C3 | VTimeLLM† | 236 | −.045 [−.092, +.002] |

**No non-daggered cell includes zero.** All 18 cells of the other five models
have an SG_gt interval strictly above zero; the closest calls are
GroundVTS NExT-GQA C2 (+.059 [+.035, +.082]) and VideoMind NExT-GQA C2
(+.064 [+.039, +.089]), both comfortably clear.

### ⚠ One daggered cell does NOT include zero

> **QVH-ME, C2, VTimeLLM†: SG_gt = −.045, 95% CI [−.077, −.012] — the interval
> lies entirely BELOW zero.**

This is the sixth expected cell, and the footnote's stated reason does not hold
for it. Two things follow.

**The printed tables are correct.** `gen_tables.py:t6block` sets
`und = m in ('VTimeLLM','TimeChat')`, so ESR is forced to `\und` by model name,
not by the CI test. The cell prints as undefined in the paper.

**The stored `figdata` disagrees with the printed table for this one cell.**
Because `build_paper_numbers` applies the CI test literally, this cell is the
only daggered one where the test passes, so it carries a numeric
`ESR = 0.136` and `ESR_undefined = None`, while every other daggered cell has
`ESR = None`. That 0.136 is `SG_pred / SG_gt = (−.0061) / (−.0449)`: a ratio of
two negative numbers, which would read as "13.6% of the ceiling" while both
the measured support and the ceiling point the wrong way. It is never printed,
but it is in the released `figdata` and would mislead anyone reading that file.

**Suggested wording.** The dagger does not need weakening — it needs stating as
what is actually true of all six cells, namely that the positive control is
*not positive*:

> † The positive control is not positive on this benchmark: the 95% interval of
> SG_gt contains zero (VTimeLLM on both NExT-GQA cuts and on QVH-ME C3;
> TimeChat on both QVH-ME cuts) or lies entirely below it (VTimeLLM on QVH-ME
> C2). Support Gain is not interpreted and ESR is undefined for these models.

The caption of the regenerated `table_s2_full.tex` already uses this phrasing.

## (3) For the Limitations paragraph

The daggered models' positive controls, ready to quote:

**VTimeLLM** — NExT-GQA: +.002 [−.015, +.019] on C2 and +.009 [−.033, +.050]
on C3; QVH-ME: −.045 [−.077, −.012] on C2 and −.045 [−.092, +.002] on C3.

**TimeChat** — QVH-ME: +.017 [−.034, +.070] on C2 and +.008 [−.069, +.086]
on C3.

One-sentence form:

> VTimeLLM and TimeChat fail the positive control — replacing the predicted
> window with the ground-truth window does not raise retention above the
> matched random baseline (VTimeLLM: +.002 [−.015, +.019] on NExT-GQA C2,
> +.009 [−.033, +.050] on C3, −.045 [−.077, −.012] on QVH-ME C2 and
> −.045 [−.092, +.002] on C3; TimeChat: +.017 [−.034, +.070] and
> +.008 [−.069, +.086] on the two QVH-ME cuts) — so we do not interpret their
> Support Gain and leave their ESR undefined.

For contrast, the strongest controls are Qwen2.5-VL on QVH-ME C3
(+.647 [+.589, +.702]) and VideoChat-R1 on QVH-ME C3 (+.642 [+.586, +.695]).

## Note on an adjacent number

Regenerating Table S2 surfaced one other boundary value, outside this task's
scope: GroundVTS on NExT-GQA C2 has SG = +.023 with CI **[+.000, +.045]** —
the 2.5th percentile is exactly 0.000 in the stored figdata as well. It prints
as `$+$.000`, which is correct but reads oddly; worth a glance before the
camera-ready.

## Files

```
outputs/camera_ready/C2_sggt_ci/
├── sggt_ci.csv        24 rows: dataset, model, set, N, N_sg, Ret(none/rand/pred),
│                      sggt_mean/lo/hi, includes_zero, sg_mean/lo/hi, daggered
├── table_s2_full.tex  full Table S2 rewritten, \scriptsize, booktabs,
│                      dataset blocks, C2/C3 rows per model, Table 5 model order,
│                      last column now SG_gt [95% CI]
└── _cache/            per-sample frames and masks dumped from build_paper_numbers
analysis/camera_ready/c2_sggt_ci.py
```

`N` is the positive-control row count (questions with both KGT and KRG);
`N_sg` is the SG row count. They differ by at most 7 — a handful of questions
lack a gt_ceiling record — which is why the CSV carries both.
