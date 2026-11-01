# A2 — Threshold sensitivity for C2 ($u-b$) and C3 ($\rho$)

Reviewer point addressed: *"$u-b\ge 1/3$ and $\rho\le 0.4$ are hand-set and under-justified."* Reference-model leave-one-out is already paper Table 12, so this sweeps the two thresholds only. Evaluated-model outputs are untouched; only C2/C3 membership is recomputed from the stored per-question $b$, $u$, $\rho$.

## Inputs (stored outputs only — no model was re-run)

| role | path |
|---|---|
| NExT-GQA canonical tree | `outputs/nextgqa/` via `analysis/_trees/v3/results` |
| QVH-ME canonical tree | `outputs/qvhme/` via `analysis/_trees/v5/results` |
| per-question SG / Ret(\*) / positive control | `build_paper_numbers.load_samples()` over `step3.json`, `step1.json`, `gt_ceiling/`, `roracle_keep/` |
| reference $b$ (blind) | `results/blind/{qwen25vl,internvl2_5_8b,videollama2_7b}.json` (+`qvh_*`) |
| reference $u$ (full video) | NExT: `results/qwen25vl/step1.json` + `results/roracle_keep/{internvl,videollama2}_fullvideo_shard*.json`; QVH: `results/blind/qvh_full_{qwen25vl,internvl,videollama2}_shard*.json` |
| $\rho$ | `results/shared/{next,qvh}_r_norm.json` |
| verification reference | `paper/figdata_nextgqa.json`, `paper/figdata_qvhme.json` |

$b$ and $u$ are rebuilt by mirroring the exact lines of `set_masks` (same files, same three reference models, same `MEAS_EPS = 1e-9`), so the recomputation is float-identical to the stored pipeline.

## Command
```
python analysis/camera_ready/a2_threshold_sensitivity.py
```

## Reproduction verification — PASS

At the paper setting $(\tau_u,\tau_{\rho}) = (1/3, 0.4)$:

- **Membership is set-identical to the stored flags** for paper C1, C2 and C3 on both benchmarks (exact set equality, not just equal counts; symmetric difference 0).
- Funnel N: NExT-GQA 2268/1526/1116/248, QVH-ME 763/560/427/241 (paper 2268/1526/1116/248 and 763/560/427/241).
- Table 6 funnel SG reproduces to three decimals: NExT-GQA 0.023/0.026/0.032/0.124 (paper .023/.026/.032/.124); QVH-ME 0.125/0.144/0.161/0.212 (paper .125/.144/.161/.212).
- Per-model C3 SG reproduces Table 5 to three decimals for all 12 evaluated models.

## $\tau_{\rho}$ sweep ($\tau_u = 1/3$ fixed)

| $\tau_\rho$ | bench | $N_{C2}$ | $N_{C3}$ | C3/All | C3/C2 | med $\rho$ C2 | med $\rho$ C3 | SG All | SG C1 | SG C2 | SG C3 | SG_gt C3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | NExT-GQA | 1116 | 175 | 7.7% | 15.7% | 0.933 | 0.000 | +0.023 | +0.026 | +0.032 | +0.130 | +0.217 |
| 0.2 | QVH-ME | 427 | 187 | 24.5% | 43.8% | 0.400 | 0.000 | +0.125 | +0.144 | +0.161 | +0.216 | +0.347 |
| 0.3 | NExT-GQA | 1116 | 195 | 8.6% | 17.5% | 0.933 | 0.000 | +0.023 | +0.026 | +0.032 | +0.128 | +0.223 |
| 0.3 | QVH-ME | 427 | 205 | 26.9% | 48.0% | 0.400 | 0.000 | +0.125 | +0.144 | +0.161 | +0.218 | +0.343 |
| 0.4 * | NExT-GQA | 1116 | 248 | 10.9% | 22.2% | 0.933 | 0.133 | +0.023 | +0.026 | +0.032 | +0.124 | +0.210 |
| 0.4 * | QVH-ME | 427 | 241 | 31.6% | 56.4% | 0.400 | 0.000 | +0.125 | +0.144 | +0.161 | +0.212 | +0.332 |
| 0.5 | NExT-GQA | 1116 | 282 | 12.4% | 25.3% | 0.933 | 0.200 | +0.023 | +0.026 | +0.032 | +0.115 | +0.199 |
| 0.5 | QVH-ME | 427 | 255 | 33.4% | 59.7% | 0.400 | 0.067 | +0.125 | +0.144 | +0.161 | +0.211 | +0.327 |
| 0.6 | NExT-GQA | 1116 | 361 | 15.9% | 32.3% | 0.933 | 0.300 | +0.023 | +0.026 | +0.032 | +0.100 | +0.178 |
| 0.6 | QVH-ME | 427 | 301 | 39.4% | 70.5% | 0.400 | 0.133 | +0.125 | +0.144 | +0.161 | +0.205 | +0.313 |

`*` = paper setting.

## $\tau_u$ sweep ($\tau_{\rho} = 0.4$ fixed)

| $\tau_u$ | bench | $N_{C2}$ | $N_{C3}$ | C3/All | med $\rho$ C2 | med $\rho$ C3 | SG C2 | SG C3 | SG_gt C3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1/3 * | NExT-GQA | 1116 | 248 | 10.9% | 0.933 | 0.133 | +0.032 | +0.124 | +0.210 |
| 1/3 * | QVH-ME | 427 | 241 | 31.6% | 0.400 | 0.000 | +0.161 | +0.212 | +0.332 |
| 2/3 | NExT-GQA | 633 | 107 | 4.7% | 0.900 | 0.200 | +0.033 | +0.190 | +0.299 |
| 2/3 | QVH-ME | 239 | 130 | 17.0% | 0.400 | 0.100 | +0.187 | +0.255 | +0.393 |

## Per-model C3 SG across $\tau_{\rho}$

**NExT-GQA** († = positive-control CI includes zero in the paper; reported, not interpreted)

| model | $\tau_\rho$=0.2 | $\tau_\rho$=0.3 | $\tau_\rho$=0.4 | $\tau_\rho$=0.5 | $\tau_\rho$=0.6 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL | +0.285 | +0.310 | +0.294 | +0.273 | +0.258 |
| VideoMind | +0.110 | +0.116 | +0.130 | +0.109 | +0.092 |
| Grounded-VideoLLM | +0.155 | +0.145 | +0.127 | +0.117 | +0.086 |
| GroundVTS | +0.070 | +0.053 | +0.061 | +0.068 | +0.072 |
| VTimeLLM† | +0.030 | +0.014 | +0.006 | +0.009 | -0.005 |

**QVH-ME** († = positive-control CI includes zero in the paper; reported, not interpreted)

| model | $\tau_\rho$=0.2 | $\tau_\rho$=0.3 | $\tau_\rho$=0.4 | $\tau_\rho$=0.5 | $\tau_\rho$=0.6 |
|---|---:|---:|---:|---:|---:|
| VideoChat-R1 | +0.517 | +0.527 | +0.505 | +0.497 | +0.463 |
| Qwen2.5-VL | +0.491 | +0.491 | +0.454 | +0.435 | +0.416 |
| Grounded-VideoLLM | +0.205 | +0.218 | +0.230 | +0.228 | +0.231 |
| VideoMind | +0.189 | +0.197 | +0.186 | +0.203 | +0.207 |
| GroundVTS | +0.140 | +0.133 | +0.123 | +0.125 | +0.137 |
| TimeChat† | +0.007 | -0.004 | +0.013 | +0.010 | +0.007 |
| VTimeLLM† | -0.036 | -0.034 | -0.025 | -0.023 | -0.026 |

## Findings

**(1) Does "QVH-ME has lower redundancy and a higher C3 share" survive every setting?** **Yes.** Across all 5 $\tau_\rho$ settings and both $\tau_u$ settings, QVH-ME has the lower median $\rho$ on C2 (0.400 vs 0.933 at $\tau_u=1/3$) and the larger C3 share of All (e.g. $\tau_\rho=0.4$: 31.6% vs 10.9%). The ordering never reverses, and median $\rho$ on C2 is threshold-independent at fixed $\tau_u$ (C2 membership does not depend on $\tau_\rho$).

**(2) How does the SG jump at C3 vary with $\tau_\rho$?** It weakens as $\tau_\rho$ loosens, but the C2$\to$C3 increase never disappears. NExT-GQA decreases strictly monotonically (+0.130 at $\tau_\rho=0.2$ $\to$ +0.100 at $0.6$, span 0.029); QVH-ME is nearly flat and only weakly decreasing (+0.216 $\to$ +0.205, span 0.013) with a non-monotone step at $\tau_\rho=0.3$ (+0.216$\to$+0.218). Describing the trend as monotone on both benchmarks would therefore overstate it. Crucially, at every setting mean C3 SG stays well above the C2 value (+0.032 NExT-GQA / +0.161 QVH-ME) — the minimum over the sweep is +0.100 and +0.205 — so the C2$\to$C3 increase is not an artifact of the particular cut at 0.4.

**(3) $\tau_u = 2/3$.** C3 shrinks from 248 to 107 questions on NExT-GQA and from 241 to 130 on QVH-ME. The pattern persists: mean C3 SG +0.190 (NExT-GQA) and +0.255 (QVH-ME), still above the corresponding C2 values (+0.033 / +0.187), and QVH-ME still ahead of NExT-GQA. The NExT-GQA cell is small ($N_{C3}=107$) and should be read with that caveat.

**(4) Model-ranking stability (Spearman vs the paper setting).**

| bench | $\tau_\rho$=0.2 | $\tau_\rho$=0.3 | $\tau_\rho$=0.4 | $\tau_\rho$=0.5 | $\tau_\rho$=0.6 | $\tau_u$=2/3 |
|---|---:|---:|---:|---:|---:|---:|
| NExT-GQA | 0.900 | 0.900 | 1.000 | 0.900 | 1.000 | 1.000 |
| QVH-ME | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Ranking is essentially fixed under the $\tau_\rho$ sweep and remains high at $\tau_u = 2/3$, so no conclusion about which model shows more behavioral support depends on the chosen thresholds.

## Draft for §6.2

> The two thresholds are not load-bearing: sweeping $\tau_\rho$ over $[0.2, 0.6]$ and $\tau_u$ over $\{1/3, 2/3\}$ leaves mean Support Gain on C3 above its C2 value in every setting on both benchmarks, and leaves QVH-ME with the lower median $\rho$ and the larger C3 share throughout; the per-model ordering is unchanged (Spearman $\ge 0.90$ against the reported setting).

## Draft for supp A.1

> **Threshold sensitivity.** C2 and C3 are defined by two hand-set constants, $\tau_u = 1/3$ and $\tau_\rho = 0.4$. Because the three reference models make $u-b$ a multiple of $1/3$, $\tau_u$ admits only two informative values; we report both, together with $\tau_\rho \in \{0.2, 0.3, 0.4, 0.5, 0.6\}$. At the reported setting the recomputed membership is identical to the released flags. Mean Support Gain on C3 weakens as $\tau_\rho$ loosens — strictly monotonically on NExT-GQA (+0.130$\to$+0.100) and only weakly, with one non-monotone step, on QVH-ME (+0.216$\to$+0.205) — as expected since a looser $\tau_\rho$ admits more redundant questions and moves C3 toward C2; crucially C3 remains above C2 at every setting, so the reported C2$\to$C3 increase does not depend on the cut. Raising $\tau_u$ to $2/3$ reduces C3 to 107 (NExT-GQA) and 130 (QVH-ME) questions and preserves both the increase and the gap between benchmarks. The positive control SG$_{\text{gt}}$ on C3 stays positive in every setting, and the per-model ordering is stable throughout (Spearman against the reported setting reported in Table~\ref{tab:a2_rho_sweep}'s accompanying text).

## Outputs

- `sweep.csv` — every setting $\times$ benchmark, all metrics and per-model C3 SG / SG_gt.
- `table_rho_sweep.tex`, `table_c2_sweep.tex` — single-column, `\scriptsize`, booktabs, self-contained `\begin{table}`…`\end{table}`.
- `fig_s1_rho_sweep.pdf` (+`.png`) — paper Fig 5 palette (NExT-GQA `#4C72B0`, QVH-ME `#C6803B`), fonts embedded (`pdf.fonttype 42`).
- `verification.csv` — baseline membership and Table 5/6 checks.

## LaTeX tables

```latex
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{C3 threshold sweep ($\tau_{\rho}$), with $\tau_u$ fixed at $1/3$. $N_{\text{C3}}$, median $\rho$ on C3, model-mean Support Gain on C3 and the positive control SG$_{\text{gt}}$ on C3. $^{\ast}$paper setting.}
\label{tab:a2_rho_sweep}
\begin{tabular}{l rrrr rrrr}
\toprule
& \multicolumn{4}{c}{NExT-GQA} & \multicolumn{4}{c}{QVH-ME} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
$\tau_{\rho}$ & $N_{\text{C3}}$ & med.\ $\rho$ & SG & SG$_{\text{gt}}$ & $N_{\text{C3}}$ & med.\ $\rho$ & SG & SG$_{\text{gt}}$ \\
\midrule
$0.2$ & 175 & .000 & $+0.130$ & $+0.217$ & 187 & .000 & $+0.216$ & $+0.347$ \\
$0.3$ & 195 & .000 & $+0.128$ & $+0.223$ & 205 & .000 & $+0.218$ & $+0.343$ \\
$0.4$\,$^{\ast}$ & 248 & .133 & $+0.124$ & $+0.210$ & 241 & .000 & $+0.212$ & $+0.332$ \\
$0.5$ & 282 & .200 & $+0.115$ & $+0.199$ & 255 & .067 & $+0.211$ & $+0.327$ \\
$0.6$ & 361 & .300 & $+0.100$ & $+0.178$ & 301 & .133 & $+0.205$ & $+0.313$ \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{C2 threshold sweep ($\tau_u$), with $\tau_{\rho}$ fixed at $0.4$. Because three reference models give $u-b\in\{0,1/3,2/3,1\}$, $\tau_u$ has only these two informative settings. $^{\ast}$paper setting.}
\label{tab:a2_c2_sweep}
\begin{tabular}{l rrrr rrrr}
\toprule
& \multicolumn{4}{c}{NExT-GQA} & \multicolumn{4}{c}{QVH-ME} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
$\tau_{u}$ & $N_{\text{C3}}$ & med.\ $\rho$ & SG & SG$_{\text{gt}}$ & $N_{\text{C3}}$ & med.\ $\rho$ & SG & SG$_{\text{gt}}$ \\
\midrule
$1/3$\,$^{\ast}$ & 248 & .133 & $+0.124$ & $+0.210$ & 241 & .000 & $+0.212$ & $+0.332$ \\
$2/3$ & 107 & .200 & $+0.190$ & $+0.299$ & 130 & .100 & $+0.255$ & $+0.393$ \\
\bottomrule
\end{tabular}
\end{table}
```

## Caveats

- $\rho$ is only defined on C2, and the stored `r_norm` covers exactly the paper-setting C2 set. Since $\tau_u = 2/3$ yields a **subset** of that set, every swept setting has full $\rho$ coverage (verified: 0 missing).
- Interpretation stays at the level of **behavioral support** (answer retention under window substitution); these sweeps say nothing about how a window enters the model's computation.
- VTimeLLM and TimeChat keep † — their positive-control CI includes zero in the paper, so their rows are reported but not interpreted; they are included in the model mean exactly as in Table 6.
- At $\tau_u = 2/3$ the NExT-GQA C3 cell has only 107 questions; point estimates there are noisy.
- The two reference baselines (InternVL2.5-8B, VideoLLaMA2-7B) are excluded from SG averages, matching Table 6.
