# C1 — grounded-correct answers by SG sign, recomputed at three decimals

Produced by [analysis/camera_ready/c1_gc_sign.py](analysis/camera_ready/c1_gc_sign.py).
No model was re-run and no existing file was modified; `load_samples` and
`set_masks` from `build_paper_numbers` are imported and reused verbatim, so
`correct`, `iop` and `SG` are bit-identical to the paper. The IoP handling for
an empty or unparsed predicted window is whatever step 1 stored — a missing
IoP fails `iop >= 0.5`, so such a question is simply not grounded-correct.

Definitions, unchanged: a question is **grounded-correct** when `a_full` equals
the ground-truth answer *and* `IoP(Ê, E*) = |Ê ∩ E*| / |Ê| ≥ 0.5`.
`SG = Ret(pred) − Ret(rand)` with Ret(rand) averaged over 5 draws, so SG is a
multiple of 0.2; `|SG| < 1e-9` counts as SG = 0.

## (1) Reproduction

**All three gates pass. The script writes nothing if any of them fails.**

**N_gc — every value the paper prints, 21/21 exact:**

| | NExT-GQA C3 | NExT-GQA C2 | QVH-ME C3 | QVH-ME C2 |
|---|---|---|---|---|
| Qwen2.5-VL | 100 ✓ | 397 ✓ | 159 ✓ | 283 ✓ |
| VideoMind | 57 ✓ | 307 ✓ | 112 ✓ | 215 ✓ |
| Grounded-VideoLLM | 50 ✓ | 197 ✓ | 97 ✓ | 185 ✓ |
| GroundVTS | 52 ✓ | 281 ✓ | 61 ✓ | 103 ✓ |
| VTimeLLM | 16 | 114 ✓ | 16 | 35 ✓ |
| TimeChat | — | — | 22 | 47 ✓ |
| VideoChat-R1 | — | — | 160 ✓ | 283 ✓ |

(NExT-GQA C3 VTimeLLM = 16 is not in the main-text table; it matches
`tables_v2.tex`.)

**QVH-ME proportions — identical to the paper at three decimals, 14/14 cells**
(both sets, all seven models). Nothing in the QVH-ME block changes.

**NExT-GQA proportions — all 15 cells reproduce the current paper values when
rounded back to two decimals.** Example from the prompt: VideoMind C3 gives
34/57 = .596, 18/57 = .316, 5/57 = .088, which rounds to .60 / .32 / .09.

**Row sums.** All 10 NExT-GQA cells now sum to exactly 1.000. Five QVH-ME cells
sum to .999 or 1.001 from independent three-decimal rounding (VideoChat-R1 C2,
VideoMind C2, GroundVTS C2, VTimeLLM C2, VideoMind C3) — these are the paper's
existing values, unchanged. Worst deviation over all 24 cells is **.001**,
against **.010** in the current tables. The caption states this rather than
claiming every row sums to 1.000.

## (2) NExT-GQA: old vs new

Counts are exact; only the printed precision changes.

### C3 (main-text table)

| Model | N_gc | counts (>0 / =0 / <0) | old (2 dp) | **new (3 dp)** | row sum old → new |
|---|---|---|---|---|---|
| Qwen2.5-VL | 100 | 68 / 24 / 8 | .68 / .24 / .08 | **.680 / .240 / .080** | 1.000 → 1.000 |
| VideoMind‡ | 57 | 34 / 18 / 5 | .60 / .32 / .09 | **.596 / .316 / .088** | **1.010** → 1.000 |
| Grounded-VideoLLM | 50 | 26 / 22 / 2 | .52 / .44 / .04 | **.520 / .440 / .040** | 1.000 → 1.000 |
| GroundVTS | 52 | 25 / 14 / 13 | .48 / .27 / .25 | **.481 / .269 / .250** | 1.000 → 1.000 |
| VTimeLLM† | 16 | 4 / 12 / 0 | .25 / .75 / .00 | **.250 / .750 / .000** | 1.000 → 1.000 |

The 1.010 row is fixed: VideoMind's three shares are .596 / .316 / .088.

### C2 (supplementary table)

| Model | N_gc | counts (>0 / =0 / <0) | old (2 dp) | **new (3 dp)** |
|---|---|---|---|---|
| Qwen2.5-VL | 397 | 121 / 247 / 29 | .30 / .62 / .07 | **.305 / .622 / .073** |
| VideoMind‡ | 307 | 89 / 188 / 30 | .29 / .61 / .10 | **.290 / .612 / .098** |
| Grounded-VideoLLM | 197 | 56 / 119 / 22 | .28 / .60 / .11 | **.284 / .604 / .112** |
| GroundVTS | 281 | 77 / 163 / 41 | .27 / .58 / .15 | **.274 / .580 / .146** |
| VTimeLLM† | 114 | 18 / 88 / 8 | .16 / .77 / .07 | **.158 / .772 / .070** |

Across both NExT-GQA blocks no share moves by more than **.005** from its
printed two-decimal value, every two-decimal rounding is preserved, and the
model ordering within each block is unchanged, so no claim in the text is
affected.

## (3) The §6.3 sentence

> "Among models whose positive control is measurable, 32–52% of
> grounded-correct answers on NExT-GQA and 14–36% on QVH-ME have SG ≤ 0."

Recomputed as `p_zero + p_neg` on C3 over the models with a measurable positive
control (VTimeLLM and TimeChat excluded):

| NExT-GQA C3 | SG ≤ 0 | | QVH-ME C3 | SG ≤ 0 |
|---|---|---|---|---|
| Qwen2.5-VL | .320 | | VideoChat-R1 | .137 |
| VideoMind‡ | .404 | | Grounded-VideoLLM | .196 |
| Grounded-VideoLLM | .480 | | Qwen2.5-VL | .201 |
| GroundVTS | .519 | | VideoMind‡ | .215 |
| | | | GroundVTS | .361 |

* NExT-GQA: **32.0 – 51.9%** → rounds to **32–52%**
* QVH-ME: **13.7 – 36.1%** → rounds to **14–36%**

**The sentence does not change.** Both stated ranges are the correct rounding
of the three-decimal values. (If the camera-ready prefers one more digit, the
honest forms are "32.0–51.9%" and "13.7–36.1%".)

## Files

```
outputs/camera_ready/C1_gc_sign/
├── gc_sign.csv       24 rows: dataset, model, set, N_gc, n_pos/zero/neg, p_pos/zero/neg
├── table_gc_c3.tex   main-text form, \small, one column; NExT-GQA C3 + QVH-ME C3,
│                     VTimeLLM/TimeChat omitted
├── table_gc_c2.tex   supplementary form, \footnotesize; NExT-GQA C2 (+VTimeLLM†)
│                     + QVH-ME C2 (all seven)
└── _cache/           per-sample frames and masks dumped from build_paper_numbers
analysis/camera_ready/c1_gc_sign.py
```

Both `.tex` files are self-contained `table` environments with the column
header `Model | N_gc | SG>0 | SG=0 | SG<0` and a caption that names the SGA /
UGA notation so the text can keep using it.
