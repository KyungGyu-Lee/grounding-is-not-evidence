# A3 — Variance across the five random draws

Reviewer point addressed: *"five random windows is few."* Answered from the stored per-draw responses, with no model re-run.

## Draw order is not assumed

`r_pred_keep_0..4` (and the positive control's per-trial list) are read as an **unordered multiset of five draws**. Nothing here depends on the slot order: (b) enumerates *all* $\binom{5}{k}$ subsets, and (c) resamples draw slots with replacement. Permuting the five slots leaves every reported number unchanged. In particular no "first draw" / "later draw" trend is claimed or computable from these files.

## Inputs (stored outputs only)

| role | path |
|---|---|
| NExT-GQA / QVH-ME canonical trees | `outputs/nextgqa/` / `outputs/qvhme/`, via `analysis/_trees/{v3,v5}/results` |
| per-draw keep responses | `outputs/{bench}/{model}/step3.json` (`preds.r_pred_keep_0..4`) |
| predicted-window / blind responses | same `step3.json` (`preds.predicted_keep`, `preds.blind`), `a_full` from the record |
| positive control per-draw | `outputs/{bench}/{model}/gt_ceiling_shard*.json` (`kgt`, `krgt` trial list); Qwen reuses `outputs/{bench}/qwen25vl/rand_gt_keep*.json` |
| set membership | `build_paper_numbers.set_masks()` |
| verification reference | `paper/figdata_nextgqa.json`, `paper/figdata_qvhme.json` |

The row filter replicates `load_samples()` exactly (drop `error`; require `a_full` present and $\ge 0$, `predicted_keep`, `blind`, and at least one `r_pred_keep`).

## Command
```
python analysis/camera_ready/a3_draw_variance.py
```

## Data completeness — (d) is possible

**Every** retained question carries exactly **5** keep responses, and the positive control retains all **5** per-draw responses for every evaluated model, so part (d) is computed in full (no cell is dropped for partial draws).

| bench | model | N(pred) C2/C3 | N(gt) C2/C3 |
|---|---|---|---|
| NExT-GQA | Qwen2.5-VL | 1116 / 248 | 1113 / 246 |
| NExT-GQA | VideoMind | 1116 / 248 | 1113 / 246 |
| NExT-GQA | Grounded-VideoLLM | 1116 / 248 | 1112 / 246 |
| NExT-GQA | GroundVTS | 1116 / 248 | 1113 / 246 |
| NExT-GQA | VTimeLLM | 1113 / 248 | 1110 / 246 |
| QVH-ME | VideoChat-R1 | 427 / 241 | 420 / 236 |
| QVH-ME | Qwen2.5-VL | 427 / 241 | 420 / 236 |
| QVH-ME | Grounded-VideoLLM | 427 / 241 | 420 / 236 |
| QVH-ME | VideoMind | 427 / 241 | 420 / 236 |
| QVH-ME | GroundVTS | 427 / 241 | 420 / 236 |
| QVH-ME | TimeChat | 427 / 241 | 420 / 236 |
| QVH-ME | VTimeLLM | 426 / 241 | 419 / 236 |

## Reproduction verification — PASS

For all **24** (dataset, model, set) cells, the $k=5$ mean Support Gain reproduces the stored canonical value (Table 5/9) to three decimals, and the question count matches.

## (1) Subsampling: how much does the spread shrink?

**The mean over subsets is uninformative by construction.** Averaged over *all* $\binom{5}{k}$ subsets, mean SG is exactly the $k=5$ value for every $k$: each draw slot appears in the same number of size-$k$ subsets, so the subset-average of the per-slot means equals the overall mean (verified numerically: max deviation 0.0000). The informative quantity is therefore the **spread across subsets**, i.e. how far a single unlucky choice of $k$ draws could land from the reported value.

On C3 that spread narrows steadily with $k$. Averaged over models the max$-$min range is **0.055** ($k{=}1$), **0.038** ($k{=}2$), **0.025** ($k{=}3$), **0.014** ($k{=}4$), **0** ($k{=}5$, a single subset) — so going from three draws to five removes **0.025** of range on average.

| bench | model | range $k{=}1$ | $k{=}2$ | $k{=}3$ | $k{=}4$ | worst-case dev. of a 3-draw subset from $k{=}5$ |
|---|---|---:|---:|---:|---:|---:|
| NExT-GQA | Qwen2.5-VL | 0.044 | 0.032 | 0.022 | 0.011 | 0.011 |
| NExT-GQA | VideoMind | 0.048 | 0.030 | 0.020 | 0.012 | 0.010 |
| NExT-GQA | Grounded-VideoLLM | 0.052 | 0.034 | 0.023 | 0.013 | 0.012 |
| NExT-GQA | GroundVTS | 0.060 | 0.034 | 0.023 | 0.015 | 0.012 |
| NExT-GQA | VTimeLLM | 0.060 | 0.038 | 0.026 | 0.015 | 0.014 |
| QVH-ME | VideoChat-R1 | 0.087 | 0.058 | 0.039 | 0.022 | 0.020 |
| QVH-ME | Qwen2.5-VL | 0.054 | 0.041 | 0.028 | 0.013 | 0.015 |
| QVH-ME | Grounded-VideoLLM | 0.075 | 0.044 | 0.029 | 0.019 | 0.015 |
| QVH-ME | VideoMind | 0.054 | 0.046 | 0.030 | 0.013 | 0.017 |
| QVH-ME | GroundVTS | 0.037 | 0.033 | 0.022 | 0.009 | 0.012 |
| QVH-ME | TimeChat | 0.062 | 0.037 | 0.025 | 0.016 | 0.013 |
| QVH-ME | VTimeLLM | 0.029 | 0.025 | 0.017 | 0.007 | 0.010 |

The most extreme individual three-draw subset differs from the reported five-draw value by at most **0.020** across all cells (median 0.013).

## (2) Draw-level CI vs item-level CI

Two draw-level resamplings are reported. *per-question* resamples each question's five draws independently with replacement — the correct model of random-window sampling noise, since draw $j$ of one question and draw $j$ of another are unrelated windows. *shared* applies one resampled multiset of five slots to every question simultaneously; it is not a faithful noise model but is reported as a deliberately conservative upper bound. The item-level CI resamples questions ($B=5000$, the paper setting).

| bench | model | item width | draw width (per-q) | ratio | draw width (shared) | ratio |
|---|---|---:|---:|---:|---:|---:|
| NExT-GQA | Qwen2.5-VL | 0.138 | 0.031 | 0.22 | 0.027 | 0.19 |
| NExT-GQA | VideoMind | 0.111 | 0.033 | 0.30 | 0.027 | 0.25 |
| NExT-GQA | Grounded-VideoLLM | 0.105 | 0.031 | 0.30 | 0.031 | 0.30 |
| NExT-GQA | GroundVTS | 0.113 | 0.031 | 0.28 | 0.035 | 0.31 |
| NExT-GQA | VTimeLLM | 0.076 | 0.022 | 0.29 | 0.034 | 0.45 |
| QVH-ME | VideoChat-R1 | 0.121 | 0.034 | 0.28 | 0.050 | 0.41 |
| QVH-ME | Qwen2.5-VL | 0.128 | 0.034 | 0.27 | 0.034 | 0.27 |
| QVH-ME | Grounded-VideoLLM | 0.144 | 0.032 | 0.22 | 0.042 | 0.29 |
| QVH-ME | VideoMind | 0.151 | 0.036 | 0.24 | 0.037 | 0.25 |
| QVH-ME | GroundVTS | 0.116 | 0.036 | 0.31 | 0.027 | 0.23 |
| QVH-ME | TimeChat | 0.123 | 0.034 | 0.28 | 0.037 | 0.30 |
| QVH-ME | VTimeLLM | 0.072 | 0.018 | 0.25 | 0.021 | 0.29 |

The per-question draw-level interval is **0.27×** the item-level interval on average (range 0.22–0.31); even the conservative shared variant averages 0.29× (0.19–0.45×). Variation attributable to which five windows were drawn is therefore small next to variation across questions.

## (3) Unanimity of the five draws

| bench | model | C3 unanimous (Ret(rand) ∈ {0,1}) | C2 unanimous | C3 unanimous, positive control |
|---|---|---:|---:|---:|
| NExT-GQA | Qwen2.5-VL | 61.3% | 74.7% | 52.4% |
| NExT-GQA | VideoMind | 52.0% | 66.8% | 52.4% |
| NExT-GQA | Grounded-VideoLLM | 60.1% | 72.0% | 54.1% |
| NExT-GQA | GroundVTS | 60.1% | 69.0% | 58.5% |
| NExT-GQA | VTimeLLM | 78.6% | 82.5% | 79.7% |
| QVH-ME | VideoChat-R1 | 52.7% | 50.6% | 46.2% |
| QVH-ME | Qwen2.5-VL | 50.6% | 50.4% | 50.4% |
| QVH-ME | Grounded-VideoLLM | 55.6% | 52.9% | 47.0% |
| QVH-ME | VideoMind | 46.5% | 48.2% | 42.4% |
| QVH-ME | GroundVTS | 51.5% | 50.6% | 51.3% |
| QVH-ME | TimeChat | 52.3% | 48.5% | 55.9% |
| QVH-ME | VTimeLLM | 85.9% | 84.0% | 86.4% |

On C3, 58.9% of questions have all five draws agreeing on average (range 46.5–85.9%); the full six-level distribution of Ret(rand) is in `retrand_distribution.csv`. A high unanimity rate means most questions carry no within-question draw variance at all, which is why the draw-level interval is narrow.

## (4) Positive control (Ret(rand_gt))

Parts (a) and (b) repeated for the positive control; full values in the two CSVs (`quantity == "SG_gt"`).

| bench | model | SG_gt $k{=}5$ | range $k{=}3$ | \|mean$_{k=3}$−mean$_{k=5}$\| |
|---|---|---:|---:|---:|
| NExT-GQA | Qwen2.5-VL | +0.455 | 0.019 | 0.000 |
| NExT-GQA | VideoMind | +0.182 | 0.043 | 0.000 |
| NExT-GQA | Grounded-VideoLLM | +0.231 | 0.009 | 0.000 |
| NExT-GQA | GroundVTS | +0.172 | 0.012 | 0.000 |
| NExT-GQA | VTimeLLM | +0.009 | 0.015 | 0.000 |
| QVH-ME | VideoChat-R1 | +0.642 | 0.034 | 0.000 |
| QVH-ME | Qwen2.5-VL | +0.647 | 0.034 | 0.000 |
| QVH-ME | Grounded-VideoLLM | +0.381 | 0.014 | 0.000 |
| QVH-ME | VideoMind | +0.291 | 0.032 | 0.000 |
| QVH-ME | GroundVTS | +0.400 | 0.027 | 0.000 |
| QVH-ME | TimeChat | +0.008 | 0.028 | 0.000 |
| QVH-ME | VTimeLLM | -0.045 | 0.023 | 0.000 |

The positive control behaves the same way: going from five to three draws moves SG_gt on C3 by at most 0.000.

## Draft for supp E

> **Number of random draws.** Each question is paired with five length-matched random windows. Treating the five as an unordered set, we recompute Support Gain using every subset of size $k$. Averaging over all subsets returns the five-draw value exactly at any $k$ (each draw enters equally many subsets), so we report the spread instead. On C3 the max$-$min spread across subsets falls from 0.055 at $k=1$ to 0.025 at $k=3$, and the most extreme individual three-draw subset differs from the reported five-draw value by at most 0.020. Resampling the draws with replacement while holding the question set fixed yields a 95% interval 0.27$\times$ the width of the item-level interval on average (0.22--0.31$\times$; a conservative variant that applies one resampled slot multiset to all questions gives 0.29$\times$). On average 58.9% of C3 questions have all five draws in agreement, so most questions contribute no within-question draw variance. The positive control shows the same pattern (at most 0.000 change from five to three draws).

## Outputs

- `draw_subsampling.csv` — every (bench, set, model, quantity, $k$) with mean / min / max / std / range over all $\binom{5}{k}$ subsets.
- `retrand_distribution.csv` — six-level Ret(rand) and Ret(rand_gt) distribution per cell.
- `ci_comparison.csv` — draw-level (both variants) vs item-level CIs and unanimity rates.
- `table_draw_subsampling.tex` — single-column, `\scriptsize`, booktabs, self-contained.
- `fig_s2_draw_subsampling.pdf` (+`.png`) — per-model lines with min--max shading, fonts embedded (`pdf.fonttype 42`).
- `verification.csv` — $k=5$ vs stored canonical values.

## LaTeX table

```latex
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Draw subsampling on C3. Mean Support Gain when only $k$ of the five random draws are used, over \emph{all} $\binom{5}{k}$ subsets (min--max in brackets); $k{=}5$ is the reported value. Draw slots are treated as unordered, so every entry is invariant to permuting them. $^{\dagger}$positive-control CI includes zero (reported, not interpreted).}
\label{tab:a3_draw_subsampling}
\begin{tabular}{l ccccc}
\toprule
Model & $k{=}1$ & $k{=}2$ & $k{=}3$ & $k{=}4$ & $k{=}5$ \\
\midrule
\multicolumn{6}{l}{\textit{NExT-GQA}} \\
Qwen2.5-VL & $+0.294$\,\tiny[+.270,+.315] & $+0.294$\,\tiny[+.278,+.310] & $+0.294$\,\tiny[+.284,+.305] & $+0.294$\,\tiny[+.289,+.300] & $+0.294$ \\
VideoMind & $+0.130$\,\tiny[+.101,+.149] & $+0.130$\,\tiny[+.115,+.145] & $+0.130$\,\tiny[+.120,+.140] & $+0.130$\,\tiny[+.125,+.137] & $+0.130$ \\
Grounded-VideoLLM & $+0.127$\,\tiny[+.101,+.153] & $+0.127$\,\tiny[+.111,+.145] & $+0.127$\,\tiny[+.114,+.137] & $+0.127$\,\tiny[+.120,+.133] & $+0.127$ \\
GroundVTS & $+0.061$\,\tiny[+.028,+.089] & $+0.061$\,\tiny[+.044,+.079] & $+0.061$\,\tiny[+.050,+.073] & $+0.061$\,\tiny[+.054,+.070] & $+0.061$ \\
VTimeLLM$^{\dagger}$ & $+0.006$\,\tiny[-.020,+.040] & $+0.006$\,\tiny[-.012,+.026] & $+0.006$\,\tiny[-.008,+.017] & $+0.006$\,\tiny[-.003,+.012] & $+0.006$ \\
\midrule
\multicolumn{6}{l}{\textit{QVH-ME}} \\
VideoChat-R1 & $+0.505$\,\tiny[+.469,+.556] & $+0.505$\,\tiny[+.475,+.533] & $+0.505$\,\tiny[+.487,+.526] & $+0.505$\,\tiny[+.493,+.515] & $+0.505$ \\
Qwen2.5-VL & $+0.454$\,\tiny[+.432,+.485] & $+0.454$\,\tiny[+.436,+.477] & $+0.454$\,\tiny[+.438,+.466] & $+0.454$\,\tiny[+.446,+.460] & $+0.454$ \\
Grounded-VideoLLM & $+0.230$\,\tiny[+.195,+.270] & $+0.230$\,\tiny[+.207,+.251] & $+0.230$\,\tiny[+.216,+.245] & $+0.230$\,\tiny[+.220,+.239] & $+0.230$ \\
VideoMind & $+0.186$\,\tiny[+.162,+.216] & $+0.186$\,\tiny[+.166,+.212] & $+0.186$\,\tiny[+.169,+.199] & $+0.186$\,\tiny[+.178,+.192] & $+0.186$ \\
GroundVTS & $+0.123$\,\tiny[+.108,+.145] & $+0.123$\,\tiny[+.108,+.141] & $+0.123$\,\tiny[+.111,+.133] & $+0.123$\,\tiny[+.117,+.127] & $+0.123$ \\
TimeChat$^{\dagger}$ & $+0.013$\,\tiny[-.008,+.054] & $+0.013$\,\tiny[-.004,+.033] & $+0.013$\,\tiny[+.000,+.025] & $+0.013$\,\tiny[+.003,+.019] & $+0.013$ \\
VTimeLLM$^{\dagger}$ & $-0.025$\,\tiny[-.041,-.012] & $-0.025$\,\tiny[-.039,-.015] & $-0.025$\,\tiny[-.032,-.015] & $-0.025$\,\tiny[-.028,-.021] & $-0.025$ \\
\bottomrule
\end{tabular}
\end{table}
```

## Caveats

- Interpretation stays at the level of **behavioral support** (answer retention under window substitution).
- Subsets of size $k<5$ are drawn from the same five stored windows, so this quantifies the stability of the reported estimate under *fewer* draws; it cannot speak to what additional, never-sampled windows would show.
- The per-question and shared draw-level resamplings answer different questions; both are reported rather than choosing silently.
- VTimeLLM and TimeChat keep † (positive-control CI includes zero in the paper): reported, not interpreted.
- Reference baselines (InternVL2.5-8B, VideoLLaMA2-7B) are excluded, matching Table 5/6.
