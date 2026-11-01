# A1 — Correctness-conditioned Support Gain

Reviewer point addressed: *"using $a_{\text{full}}$ as the reference means a confidently-retained wrong answer also counts as supported."* Every (dataset, model, set) cell is split by whether $a_{\text{full}}$ matches the ground truth.

## Inputs (stored outputs only — no model was re-run)

| role | path |
|---|---|
| NExT-GQA canonical tree | `outputs/nextgqa/`, via wrapper `analysis/_trees/v3/results -> outputs/nextgqa` |
| QVH-ME canonical tree | `outputs/qvhme/`, via wrapper `analysis/_trees/v5/results -> outputs/qvhme` |
| per-question re-query | `results/{model}/phase2/step3.json`, `results/qvh/{model}/phase2/step3.json` (`predicted_keep`, `blind`, `r_pred_keep_0..4`) |
| `a_full` / IoU / IoP | `results/{model}/step1.json` (grounders); full-video `pred` for the two reference baselines |
| positive control | `results/gt_ceiling/{next,qvh}_{model}_shard*.json`, `results/roracle_keep/*` for `K_R_gt` reuse |
| C1/C2/C3 membership | `results/shared/{next,qvh}_r_norm.json` + 3-baseline blind/full (`set_masks`) |
| verification reference | `paper/figdata_nextgqa.json`, `paper/figdata_qvhme.json` |

SG / Ret(\*) are **not reimplemented**: `build_paper_numbers.load_samples()` and `set_masks()` are imported and reused verbatim, so the per-question quantities are bit-identical to the paper. Note that `build_paper_numbers` uses code set names flipped vs the paper (paper C1 = code `C2`, paper C2 = code `C1`, paper C3 = code `C3`); this report uses **paper** naming throughout.

## Command

```
python analysis/camera_ready/a1_correctness_conditioned.py
```

(spawns one child per benchmark with `BP_CWD` pointing at the matching canonical tree; `BP_QVH_VCR1=1` for QVH-ME so VideoChat-R1 is included.)

## Reproduction verification — PASS

All **24** (dataset, model, set) cells reproduce the stored canonical `figdata` to three decimals on `N`, SG, Ret(none), Ret(rand), Ret(pred); and the N-weighted recombination of the correct + incorrect groups equals the unconditioned SG to three decimals.

Spot checks against the values quoted in the paper:

| quantity | paper | recomputed |
|---|---|---|
| NExT-GQA C2 Qwen2.5-VL | SG +.086 | SG +0.086, Ret(pred) 0.845, Ret(none) 0.375, Ret(rand) 0.759 |
| NExT-GQA C3 Qwen2.5-VL | SG +.294, Ret(pred) .722, Ret(none) .367, Ret(rand) .427 | SG +0.294, Ret(pred) 0.722, Ret(none) 0.367, Ret(rand) 0.427 |
| QVH-ME C2 VideoChat-R1 | SG +.358 | SG +0.358, Ret(pred) 0.803, Ret(none) 0.337, Ret(rand) 0.445 |
| QVH-ME C3 VideoChat-R1 | SG +.505 | SG +0.505, Ret(pred) 0.784, Ret(none) 0.303, Ret(rand) 0.279 |
| funnel NExT-GQA ALL/C1/C2/C3 | 2268 / 1526 / 1116 / 248 | 2268 / 1526 / 1116 / 248 |
| funnel QVH-ME ALL/C1/C2/C3 | 763 / 560 / 427 / 241 | 763 / 560 / 427 / 241 |
| model-mean C3 SG (grounders) | NExT-GQA .124 / QVH-ME .212 | 0.124 / 0.212 |

**CI caveat.** The paper's Table 5/9 interval is a *paired* item bootstrap (one resample shared across models; stored as `paired_bootstrap_SGpred_CI`). The correct/incorrect split has model-specific membership, so a resample shared across models is not applicable here; the CIs below are independent per-cell item bootstraps (B=5000, seed 0). This shifts the unconditioned interval by at most .003 (e.g. Qwen2.5-VL NExT-GQA C2: paper [+.061,+.109] vs [+.062,+.110] here), and does not affect any point estimate.

## Headline numbers

**(1) Is SG positive on the incorrect group?** No — not reliably, in any cell. Across the 24 (dataset, model, set) cells, **0 have a 95% CI entirely above zero**, 3 are entirely below zero, and 21 straddle zero. The median incorrect-group SG is **-0.017**; the largest is +0.160 (QVH-ME C3 VideoChat-R1, CI [-0.064, +0.384], which includes zero). On C3 the incorrect-group SG is negative for all five NExT-GQA models.

**(2) Difference from the correct group.** ΔSG = SG_correct − SG_incorrect is **positive in all 24 cells**, with a 95% CI excluding zero in **19 of 24**. Range +0.028 to +0.577, median **+0.196**. The gap is widest on C3 (e.g. VideoMind QVH-ME +0.577, Qwen2.5-VL NExT-GQA +0.458).

**(3) Share of each set whose $a_{\text{full}}$ is incorrect.**

| dataset | model | C2 incorrect | C3 incorrect |
|---|---|---:|---:|
| NExT-GQA | Qwen2.5-VL | 11.3% (N=126) | 13.3% (N=33) |
| NExT-GQA | VideoMind | 23.2% (N=259) | 37.5% (N=93) |
| NExT-GQA | Grounded-VideoLLM | 21.0% (N=234) | 31.0% (N=77) |
| NExT-GQA | GroundVTS | 26.2% (N=292) | 44.0% (N=109) |
| NExT-GQA | VTimeLLM | 62.5% (N=696) | 72.2% (N=179) |
| QVH-ME | VideoChat-R1 | 9.8% (N=42) | 10.4% (N=25) |
| QVH-ME | Qwen2.5-VL | 7.3% (N=31) | 7.9% (N=19) |
| QVH-ME | Grounded-VideoLLM | 28.6% (N=122) | 32.4% (N=78) |
| QVH-ME | VideoMind | 25.1% (N=107) | 30.3% (N=73) |
| QVH-ME | GroundVTS | 22.7% (N=97) | 26.6% (N=64) |
| QVH-ME | TimeChat | 57.1% (N=244) | 65.1% (N=157) |
| QVH-ME | VTimeLLM | 70.7% (N=301) | 74.3% (N=179) |

**Positive control.** SG_gt is positive in 24/24 correct-group cells (median +0.380) but only 2/24 incorrect-group cells (median -0.084) — the ground-truth window shows the same split, so the pattern is not specific to predicted windows.

## Draft for the main text

> Conditioning each cell on whether the full-video answer is correct does not change the direction of the measured effect, but concentrates it. On C3, Support Gain is carried almost entirely by the questions whose full-video answer matches the ground truth (median ΔSG +0.196 across cells, 95% CI excluding zero in 19 of 24), whereas on the complementary subset no model attains a reliably positive Support Gain (0 of 24 cells with a CI above zero; median -0.017). The positive control splits the same way (SG_gt positive in 24 of 24 correct-answer cells versus 2 of 24). Support Gain therefore tracks behavioral support for answers that are also correct, and is not an artifact of confidently retained incorrect answers.

## Results

### NExT-GQA

| Model | Set | $a_{full}$ | N | % of set | Ret(none) | Ret(rand) | Ret(pred) | SG [95% CI] | SG_gt | ΔSG [95% CI] |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Qwen2.5-VL | C2 | correct | 990 | 88.7 | 0.344 | 0.768 | 0.858 | +0.090 [+0.065, +0.116] | +0.143 | +0.036 [-0.050, +0.123] |
| Qwen2.5-VL | C2 | incorrect | 126 | 11.3 | 0.611 | 0.692 | 0.746 | +0.054 [-0.032, +0.137] | -0.076 |  |
| Qwen2.5-VL | C3 | correct | 215 | 86.7 | 0.326 | 0.366 | 0.721 | +0.355 [+0.283, +0.427] | +0.546 | +0.458 [+0.293, +0.630] |
| Qwen2.5-VL | C3 | incorrect | 33 | 13.3 | 0.636 | 0.830 | 0.727 | -0.103 [-0.267, +0.042] | -0.127 |  |
| VideoMind | C2 | correct | 857 | 76.8 | 0.697 | 0.790 | 0.837 | +0.046 [+0.021, +0.072] | +0.112 | +0.074 [+0.015, +0.135] |
| VideoMind | C2 | incorrect | 259 | 23.2 | 0.703 | 0.723 | 0.695 | -0.028 [-0.083, +0.027] | -0.098 |  |
| VideoMind | C3 | correct | 155 | 62.5 | 0.426 | 0.498 | 0.710 | +0.212 [+0.142, +0.280] | +0.374 | +0.218 [+0.109, +0.333] |
| VideoMind | C3 | incorrect | 93 | 37.5 | 0.720 | 0.759 | 0.753 | -0.006 [-0.092, +0.077] | -0.139 |  |
| Grounded-VideoLLM | C2 | correct | 882 | 79.0 | 0.362 | 0.810 | 0.845 | +0.035 [+0.010, +0.060] | +0.136 | +0.051 [-0.007, +0.108] |
| Grounded-VideoLLM | C2 | incorrect | 234 | 21.0 | 0.577 | 0.743 | 0.726 | -0.016 [-0.068, +0.033] | -0.088 |  |
| Grounded-VideoLLM | C3 | correct | 171 | 69.0 | 0.374 | 0.586 | 0.772 | +0.186 [+0.120, +0.253] | +0.386 | +0.191 [+0.086, +0.296] |
| Grounded-VideoLLM | C3 | incorrect | 77 | 31.0 | 0.636 | 0.771 | 0.766 | -0.005 [-0.086, +0.075] | -0.116 |  |
| GroundVTS | C2 | correct | 824 | 73.8 | 0.454 | 0.762 | 0.803 | +0.041 [+0.017, +0.066] | +0.079 | +0.071 [+0.016, +0.127] |
| GroundVTS | C2 | incorrect | 292 | 26.2 | 0.586 | 0.673 | 0.644 | -0.029 [-0.080, +0.021] | -0.000 |  |
| GroundVTS | C3 | correct | 139 | 56.0 | 0.446 | 0.465 | 0.583 | +0.118 [+0.040, +0.197] | +0.341 | +0.129 [+0.018, +0.242] |
| GroundVTS | C3 | incorrect | 109 | 44.0 | 0.550 | 0.662 | 0.651 | -0.011 [-0.088, +0.066] | -0.049 |  |
| VTimeLLM | C2 | correct | 417 | 37.5 | 0.576 | 0.905 | 0.921 | +0.016 [-0.011, +0.043] | +0.021 | +0.028 [-0.008, +0.064] |
| VTimeLLM | C2 | incorrect | 696 | 62.5 | 0.704 | 0.896 | 0.884 | -0.012 [-0.035, +0.011] | -0.010 |  |
| VTimeLLM | C3 | correct | 69 | 27.8 | 0.710 | 0.843 | 0.899 | +0.055 [+0.000, +0.113] | +0.147 | +0.068 [-0.003, +0.143] |
| VTimeLLM | C3 | incorrect | 179 | 72.2 | 0.693 | 0.891 | 0.877 | -0.013 [-0.063, +0.032] | -0.044 |  |

### QVH-ME

| Model | Set | $a_{full}$ | N | % of set | Ret(none) | Ret(rand) | Ret(pred) | SG [95% CI] | SG_gt | ΔSG [95% CI] |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| VideoChat-R1 | C2 | correct | 385 | 90.2 | 0.332 | 0.450 | 0.842 | +0.392 [+0.343, +0.438] | +0.483 | +0.339 [+0.155, +0.525] |
| VideoChat-R1 | C2 | incorrect | 42 | 9.8 | 0.381 | 0.400 | 0.452 | +0.052 [-0.124, +0.233] | +0.010 |  |
| VideoChat-R1 | C3 | correct | 216 | 89.6 | 0.296 | 0.274 | 0.819 | +0.545 [+0.485, +0.606] | +0.713 | +0.385 [+0.160, +0.615] |
| VideoChat-R1 | C3 | incorrect | 25 | 10.4 | 0.360 | 0.320 | 0.480 | +0.160 [-0.064, +0.384] | -0.017 |  |
| Qwen2.5-VL | C2 | correct | 396 | 92.7 | 0.333 | 0.455 | 0.808 | +0.354 [+0.306, +0.403] | +0.478 | +0.354 [+0.152, +0.559] |
| Qwen2.5-VL | C2 | incorrect | 31 | 7.3 | 0.484 | 0.419 | 0.419 | +0.000 [-0.206, +0.200] | -0.000 |  |
| Qwen2.5-VL | C3 | correct | 222 | 92.1 | 0.297 | 0.291 | 0.779 | +0.488 [+0.425, +0.552] | +0.705 | +0.436 [+0.166, +0.709] |
| Qwen2.5-VL | C3 | incorrect | 19* | 7.9 | 0.368 | 0.316 | 0.368 | +0.053 [-0.211, +0.316] | -0.106 |  |
| Grounded-VideoLLM | C2 | correct | 305 | 71.4 | 0.272 | 0.470 | 0.757 | +0.288 [+0.228, +0.345] | +0.465 | +0.376 [+0.261, +0.490] |
| Grounded-VideoLLM | C2 | incorrect | 122 | 28.6 | 0.492 | 0.457 | 0.369 | -0.089 [-0.187, +0.011] | -0.147 |  |
| Grounded-VideoLLM | C3 | correct | 163 | 67.6 | 0.196 | 0.312 | 0.706 | +0.394 [+0.315, +0.471] | +0.664 | +0.507 [+0.362, +0.652] |
| Grounded-VideoLLM | C3 | incorrect | 78 | 32.4 | 0.487 | 0.472 | 0.359 | -0.113 [-0.241, +0.010] | -0.205 |  |
| VideoMind | C2 | correct | 320 | 74.9 | 0.456 | 0.533 | 0.812 | +0.279 [+0.224, +0.333] | +0.413 | +0.442 [+0.318, +0.563] |
| VideoMind | C2 | incorrect | 107 | 25.1 | 0.439 | 0.480 | 0.318 | -0.163 [-0.269, -0.056] | -0.263 |  |
| VideoMind | C3 | correct | 168 | 69.7 | 0.399 | 0.389 | 0.750 | +0.361 [+0.286, +0.436] | +0.589 | +0.577 [+0.417, +0.732] |
| VideoMind | C3 | incorrect | 73 | 30.3 | 0.493 | 0.518 | 0.301 | -0.216 [-0.356, -0.079] | -0.403 |  |
| GroundVTS | C2 | correct | 330 | 77.3 | 0.485 | 0.444 | 0.567 | +0.122 [+0.072, +0.172] | +0.369 | +0.141 [+0.046, +0.235] |
| GroundVTS | C2 | incorrect | 97 | 22.7 | 0.412 | 0.400 | 0.381 | -0.019 [-0.103, +0.066] | +0.053 |  |
| GroundVTS | C3 | correct | 177 | 73.4 | 0.435 | 0.321 | 0.497 | +0.176 [+0.107, +0.245] | +0.537 | +0.201 [+0.081, +0.316] |
| GroundVTS | C3 | incorrect | 64 | 26.6 | 0.500 | 0.416 | 0.391 | -0.025 [-0.119, +0.072] | -0.010 |  |
| TimeChat | C2 | correct | 183 | 42.9 | 0.437 | 0.564 | 0.628 | +0.064 [-0.002, +0.136] | +0.290 | +0.102 [+0.012, +0.196] |
| TimeChat | C2 | incorrect | 244 | 57.1 | 0.561 | 0.616 | 0.578 | -0.038 [-0.096, +0.020] | -0.189 |  |
| TimeChat | C3 | correct | 84 | 34.9 | 0.417 | 0.414 | 0.583 | +0.169 [+0.067, +0.271] | +0.446 | +0.239 [+0.115, +0.361] |
| TimeChat | C3 | incorrect | 157 | 65.1 | 0.599 | 0.650 | 0.580 | -0.070 [-0.143, +0.001] | -0.225 |  |
| VTimeLLM | C2 | correct | 125 | 29.3 | 0.784 | 0.886 | 0.904 | +0.018 [-0.026, +0.058] | +0.026 | +0.034 [-0.022, +0.088] |
| VTimeLLM | C2 | incorrect | 301 | 70.7 | 0.880 | 0.913 | 0.897 | -0.016 [-0.051, +0.018] | -0.074 |  |
| VTimeLLM | C3 | correct | 62 | 25.7 | 0.871 | 0.874 | 0.919 | +0.045 [+0.003, +0.090] | +0.058 | +0.094 [+0.032, +0.158] |
| VTimeLLM | C3 | incorrect | 179 | 74.3 | 0.877 | 0.921 | 0.872 | -0.049 [-0.095, -0.004] | -0.079 |  |

`*` = cell with N < 20. Full per-cell values (including SG_gt CIs and group fractions) are in `sg_by_correctness.csv`; the stored-vs-recomputed comparison is in `verification.csv`.

## LaTeX table

`table_main.tex` — `table*`, `\scriptsize`, booktabs, Ret columns in paper Table 9 order (Ret(none), Ret(rand), Ret(pred)), model order as in Table 5, † retained on VTimeLLM / TimeChat. Self-contained `\begin{table*}`…`\end{table*}`, ready for `\input`.

```latex
\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\caption{Correctness-conditioned Support Gain. Each (model, set) cell is split by whether the full-video answer $a_{\text{full}}$ matches the ground truth. $\Delta$SG $=$ SG$_{\text{correct}}-$SG$_{\text{incorrect}}$ with an unpaired item-level bootstrap CI ($B{=}5000$). Item-level bootstrap, $B{=}5000$, seed fixed. $^{\dagger}$positive-control SG$_{\text{gt}}$ CI includes zero (not interpreted); $^{\ddagger}$see Table 5. $^{\ast}$cell with $N<20$.}
\label{tab:a1_correctness_conditioned}
\begin{tabular}{lllrrrrlrl}
\toprule
Model & Set & $a_{\text{full}}$ & $N$ & Ret(none) & Ret(rand) & Ret(pred) & SG [95\% CI] & SG$_{\text{gt}}$ & $\Delta$SG [95\% CI] \\
\midrule
\multicolumn{10}{l}{\textit{NExT-GQA}} \\
Qwen2.5-VL & C2 & correct & 990 & .344 & .768 & .858 & $+0.090$ [$+.065,+.116$] & $+0.143$ & $+0.036$ [$-.050,+.123$] \\
 &  & incorrect & 126 & .611 & .692 & .746 & $+0.054$ [$-.032,+.137$] & $-0.076$ &  \\
 & C3 & correct & 215 & .326 & .366 & .721 & $+0.355$ [$+.283,+.427$] & $+0.546$ & $+0.458$ [$+.293,+.630$] \\
 &  & incorrect & 33 & .636 & .830 & .727 & $-0.103$ [$-.267,+.042$] & $-0.127$ &  \\
\addlinespace
VideoMind$^{\ddagger}$ & C2 & correct & 857 & .697 & .790 & .837 & $+0.046$ [$+.021,+.072$] & $+0.112$ & $+0.074$ [$+.015,+.135$] \\
 &  & incorrect & 259 & .703 & .723 & .695 & $-0.028$ [$-.083,+.027$] & $-0.098$ &  \\
 & C3 & correct & 155 & .426 & .498 & .710 & $+0.212$ [$+.142,+.280$] & $+0.374$ & $+0.218$ [$+.109,+.333$] \\
 &  & incorrect & 93 & .720 & .759 & .753 & $-0.006$ [$-.092,+.077$] & $-0.139$ &  \\
\addlinespace
Grounded-VideoLLM & C2 & correct & 882 & .362 & .810 & .845 & $+0.035$ [$+.010,+.060$] & $+0.136$ & $+0.051$ [$-.007,+.108$] \\
 &  & incorrect & 234 & .577 & .743 & .726 & $-0.016$ [$-.068,+.033$] & $-0.088$ &  \\
 & C3 & correct & 171 & .374 & .586 & .772 & $+0.186$ [$+.120,+.253$] & $+0.386$ & $+0.191$ [$+.086,+.296$] \\
 &  & incorrect & 77 & .636 & .771 & .766 & $-0.005$ [$-.086,+.075$] & $-0.116$ &  \\
\addlinespace
GroundVTS & C2 & correct & 824 & .454 & .762 & .803 & $+0.041$ [$+.017,+.066$] & $+0.079$ & $+0.071$ [$+.016,+.127$] \\
 &  & incorrect & 292 & .586 & .673 & .644 & $-0.029$ [$-.080,+.021$] & $-0.000$ &  \\
 & C3 & correct & 139 & .446 & .465 & .583 & $+0.118$ [$+.040,+.197$] & $+0.341$ & $+0.129$ [$+.018,+.242$] \\
 &  & incorrect & 109 & .550 & .662 & .651 & $-0.011$ [$-.088,+.066$] & $-0.049$ &  \\
\addlinespace
VTimeLLM$^{\dagger}$ & C2 & correct & 417 & .576 & .905 & .921 & $+0.016$ [$-.011,+.043$] & $+0.021$ & $+0.028$ [$-.008,+.064$] \\
 &  & incorrect & 696 & .704 & .896 & .884 & $-0.012$ [$-.035,+.011$] & $-0.010$ &  \\
 & C3 & correct & 69 & .710 & .843 & .899 & $+0.055$ [$+.000,+.113$] & $+0.147$ & $+0.068$ [$-.003,+.143$] \\
 &  & incorrect & 179 & .693 & .891 & .877 & $-0.013$ [$-.063,+.032$] & $-0.044$ &  \\
\midrule
\multicolumn{10}{l}{\textit{QVH-ME}} \\
VideoChat-R1 & C2 & correct & 385 & .332 & .450 & .842 & $+0.392$ [$+.343,+.438$] & $+0.483$ & $+0.339$ [$+.155,+.525$] \\
 &  & incorrect & 42 & .381 & .400 & .452 & $+0.052$ [$-.124,+.233$] & $+0.010$ &  \\
 & C3 & correct & 216 & .296 & .274 & .819 & $+0.545$ [$+.485,+.606$] & $+0.713$ & $+0.385$ [$+.160,+.615$] \\
 &  & incorrect & 25 & .360 & .320 & .480 & $+0.160$ [$-.064,+.384$] & $-0.017$ &  \\
\addlinespace
Qwen2.5-VL & C2 & correct & 396 & .333 & .455 & .808 & $+0.354$ [$+.306,+.403$] & $+0.478$ & $+0.354$ [$+.152,+.559$] \\
 &  & incorrect & 31 & .484 & .419 & .419 & $+0.000$ [$-.206,+.200$] & $-0.000$ &  \\
 & C3 & correct & 222 & .297 & .291 & .779 & $+0.488$ [$+.425,+.552$] & $+0.705$ & $+0.436$ [$+.166,+.709$] \\
 &  & incorrect & 19$^{\ast}$ & .368 & .316 & .368 & $+0.053$ [$-.211,+.316$] & $-0.106$ &  \\
\addlinespace
Grounded-VideoLLM & C2 & correct & 305 & .272 & .470 & .757 & $+0.288$ [$+.228,+.345$] & $+0.465$ & $+0.376$ [$+.261,+.490$] \\
 &  & incorrect & 122 & .492 & .457 & .369 & $-0.089$ [$-.187,+.011$] & $-0.147$ &  \\
 & C3 & correct & 163 & .196 & .312 & .706 & $+0.394$ [$+.315,+.471$] & $+0.664$ & $+0.507$ [$+.362,+.652$] \\
 &  & incorrect & 78 & .487 & .472 & .359 & $-0.113$ [$-.241,+.010$] & $-0.205$ &  \\
\addlinespace
VideoMind$^{\ddagger}$ & C2 & correct & 320 & .456 & .533 & .812 & $+0.279$ [$+.224,+.333$] & $+0.413$ & $+0.442$ [$+.318,+.563$] \\
 &  & incorrect & 107 & .439 & .480 & .318 & $-0.163$ [$-.269,-.056$] & $-0.263$ &  \\
 & C3 & correct & 168 & .399 & .389 & .750 & $+0.361$ [$+.286,+.436$] & $+0.589$ & $+0.577$ [$+.417,+.732$] \\
 &  & incorrect & 73 & .493 & .518 & .301 & $-0.216$ [$-.356,-.079$] & $-0.403$ &  \\
\addlinespace
GroundVTS & C2 & correct & 330 & .485 & .444 & .567 & $+0.122$ [$+.072,+.172$] & $+0.369$ & $+0.141$ [$+.046,+.235$] \\
 &  & incorrect & 97 & .412 & .400 & .381 & $-0.019$ [$-.103,+.066$] & $+0.053$ &  \\
 & C3 & correct & 177 & .435 & .321 & .497 & $+0.176$ [$+.107,+.245$] & $+0.537$ & $+0.201$ [$+.081,+.316$] \\
 &  & incorrect & 64 & .500 & .416 & .391 & $-0.025$ [$-.119,+.072$] & $-0.010$ &  \\
\addlinespace
TimeChat$^{\dagger}$ & C2 & correct & 183 & .437 & .564 & .628 & $+0.064$ [$-.002,+.136$] & $+0.290$ & $+0.102$ [$+.012,+.196$] \\
 &  & incorrect & 244 & .561 & .616 & .578 & $-0.038$ [$-.096,+.020$] & $-0.189$ &  \\
 & C3 & correct & 84 & .417 & .414 & .583 & $+0.169$ [$+.067,+.271$] & $+0.446$ & $+0.239$ [$+.115,+.361$] \\
 &  & incorrect & 157 & .599 & .650 & .580 & $-0.070$ [$-.143,+.001$] & $-0.225$ &  \\
\addlinespace
VTimeLLM$^{\dagger}$ & C2 & correct & 125 & .784 & .886 & .904 & $+0.018$ [$-.026,+.058$] & $+0.026$ & $+0.034$ [$-.022,+.088$] \\
 &  & incorrect & 301 & .880 & .913 & .897 & $-0.016$ [$-.051,+.018$] & $-0.074$ &  \\
 & C3 & correct & 62 & .871 & .874 & .919 & $+0.045$ [$+.003,+.090$] & $+0.058$ & $+0.094$ [$+.032,+.158$] \\
 &  & incorrect & 179 & .877 & .921 & .872 & $-0.049$ [$-.095,-.004$] & $-0.079$ &  \\
\bottomrule
\end{tabular}
\end{table*}
```

## Caveats

- Interpretation is restricted to **behavioral support** (answer retention under window substitution). These numbers do not establish how a window enters the model's computation.
- Per-cell CIs are independent item bootstraps, not the paper's paired-across-models bootstrap (see the CI caveat above).
- 1 cell has N < 20 and is marked `*`/$^\ast$: QVH-ME C3 Qwen2.5-VL incorrect, N=19. Its CI is wide and the point estimate should not be read on its own.
- VTimeLLM and TimeChat keep their † marker: their positive-control SG_gt CI includes zero in the paper, so their rows are reported but not interpreted.
- The two reference baselines (InternVL2.5-8B, VideoLLaMA2-7B) are excluded, matching Table 5/6 where grounder SG is the reported quantity.
