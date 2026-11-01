# B1-2 — is the Keep/Random gap specific to cropping?

Qwen2.5-VL, QVH-ME $\mathcal{C}_3$. Produced by
[analysis/camera_ready/b1_analyze.py](analysis/camera_ready/b1_analyze.py)
from the stored CROP run plus the 2892 re-queries of B1-1. One model, one
benchmark, one cut — nothing here generalises beyond that.

## Conditions

| | how the window reaches the model | frames |
|---|---|---|
| **CROP** | paper protocol, stored, not re-run: keep only the frames inside the window | median 14 of 128 |
| **MASK** | keep the full 128-frame grid, replace every frame outside the window with black | always 128 |
| **MARGIN ±5 s** | crop, but to the window grown by 5 s per side | median 22 |

`Ret(none)` is shared: blind sees no frames in any condition. Scoring is the
paper's (`Ret = 1[pred == a_full]`, `a_full` = the model's own full-video
answer, identical across conditions). CIs: item bootstrap, B=5000, seed 0,
**one resample shared across the three conditions**, so the ΔSG intervals are
paired — the same construction as `paired_bootstrap_SGpred_CI`.

**Reproduction gate (set a).** CROP recomputed from the stored record gives
Ret(none) .3029, Ret(rand) .2929, Ret(pred) .7469, SG +.4539 — matching Table 5's
.303 / .293 / .747 / +.454 to the third decimal. The script refuses to write
anything if this fails.

## Results

### (a) all 241 questions

| Condition | Ret(none) | Ret(rand) | Ret(pred) | SG [95% CI] | ΔSG vs. CROP [95% CI] |
|---|---|---|---|---|---|
| CROP (paper) | .303 | .293 | .747 | **+.454** [+.389, +.515] | — |
| MASK | .303 | .339 | .747 | **+.407** [+.340, +.474] | **−.046** [−.091, −.003] |
| MARGIN ±5 s | .303 | .315 | .801 | **+.486** [+.423, +.547] | **+.032** [−.009, +.077] |

### (b) 236 questions, the 5 with a zero-frame window dropped

| Condition | Ret(none) | Ret(rand) | Ret(pred) | SG [95% CI] | ΔSG vs. CROP [95% CI] |
|---|---|---|---|---|---|
| CROP (paper) | .309 | .297 | .758 | **+.462** [+.398, +.524] | — |
| MASK | .309 | .345 | .754 | **+.409** [+.343, +.476] | **−.053** [−.096, −.008] |
| MARGIN ±5 s | .309 | .320 | .814 | **+.493** [+.430, +.558] | **+.031** [−.011, +.075] |

Set (a) is the reproduction reference; (b) changes nothing qualitatively.

### The 9 zero-frame windows

Nine of the 1446 windows are shorter than the 1.18 s frame spacing
(150 s / 127) and fall between grid points, so no sampled frame lies inside
them. **All nine are random windows — none is a predicted window** — and they
belong to 5 questions (`-_s0sXOfS3w_60.0_210.0_3938`,
`23YBs2JxE-k_210.0_360.0_2093` ×4, `2X46BBkcCeY_210.0_360.0_8239`,
`S1DZb9SkrK8_60.0_210.0_1285` ×2, `h_XoA5-_8W0_510.0_660.0_7035`); lengths
0.4–1.1 s. Listed in `zero_frame_windows.json`.

CROP has nothing to show there and the stored run recorded
`pred = -1, flag = empty_or_no_interval`, which scores as not retained. The
nine corresponding MASK inputs are **entirely black** — the window contains no
frame, so `out[~in_ev] = 0` zeroes the whole clip. In set (a) they are scored
by CROP's rule (not retained) so the two conditions stay comparable, and the
model's recorded answer there is a response to a black clip and **is not
interpreted**. Set (b) removes those 5 questions outright. The responses are
kept in `responses_mask.jsonl` with `all_black: true`. MARGIN has no empty
input (minimum 5 frames).

## (1) Is SG positive with a CI excluding zero under MASK and MARGIN?

**Yes, in both variants and on both question sets.** The lower bound of every
SG interval is above +.34 — far from zero:

* MASK +.407 [+.340, +.474]; set (b) +.409 [+.343, +.476]
* MARGIN +.486 [+.423, +.547]; set (b) +.493 [+.430, +.558]

## (2) How large is the difference from CROP?

Small, and in opposite directions.

* **MASK − CROP = −.046 [−.091, −.003]**, (b) −.053 [−.096, −.008]. The
  interval excludes zero, but only just: its upper bound is −.003 in (a) and
  −.008 in (b), and the point estimate is 10 % of CROP's SG.
* **MARGIN − CROP = +.032 [−.009, +.077]**, (b) +.031 [−.011, +.075]. The
  interval contains zero.

The three conditions span .407 to .486, a range of .079 against an SG of
roughly .45.

**Where the MASK difference comes from.** In set (a) Ret(pred) is *identical*
under CROP and MASK (.747 both); the entire ΔSG sits in Ret(rand), which rises
from .293 to .339. So replacing the crop with a blacked-out full grid does not
change how often the predicted window's answer matches `a_full`; it makes the
*random* windows match slightly more often. MARGIN moves both terms but
Ret(pred) more (.747 → .801) than Ret(rand) (.293 → .315), which is why its SG
is the largest of the three.

## (3) Question-level agreement on Ret(pred)

| | agree | CROP kept, variant not | CROP not, variant kept |
|---|---|---|---|
| CROP vs MASK | 219/241 = **.909** | 11 | 11 |
| CROP vs MARGIN | 220/241 = **.913** | 4 | 17 |

CROP and MASK disagree on 22 questions and the two directions are exactly
balanced, which is why the aggregate Ret(pred) is unchanged. MARGIN's
disagreements are one-sided: 17 questions it retains and CROP does not,
against 4 the other way.

**The disagreements concentrate on short windows.**

| set | n | median \|Ê\| | median frames in Ê |
|---|---|---|---|
| all C3 questions | 241 | 16.4 s | 14 |
| CROP = MASK | 219 | 17.4 s | 15 |
| CROP ≠ MASK | 22 | **7.3 s** | **6** |
| — CROP kept, MASK not | 11 | 13.3 s | 11 |
| — CROP not, MASK kept | 11 | 6.1 s | 5 |
| CROP = MARGIN | 220 | 17.5 s | 15 |
| CROP ≠ MARGIN | 21 | **7.4 s** | **6** |
| — CROP kept, MARGIN not | 4 | 13.4 s | 12 |
| — CROP not, MARGIN kept | 17 | 7.4 s | 6 |

Questions where the conditions disagree have a predicted window less than half
the median length — about 6 sampled frames against 15. The asymmetry is
sharpest in the direction where CROP does not retain and the context-preserving
variant does (6.1 s for MASK, 7.4 s for MARGIN). Short windows are where the
crop leaves the fewest frames, and they are where the choice of transform
matters; on the other ~90 % of questions the three conditions give the same
question-level outcome.

## (4) Reading

Across all three ways of presenting the window — dropping the outside frames,
blacking them out at full frame count, or cropping with 5 s of context on each
side — SG stays between +.41 and +.49 with intervals well clear of zero, and no
pairwise difference reaches a tenth of SG itself. **The pattern is not specific
to cropping.** The only difference that survives its interval is MASK's, which
is small, negative, and located entirely in Ret(rand) rather than Ret(pred);
preserving the full temporal grid raises retention under random windows a
little, which narrows but does not close the gap. MARGIN, the condition
closest to CROP in input format, is indistinguishable from it. This is one
model on one benchmark cut, so it bounds the cropping artefact for
Qwen2.5-VL on QVH-ME $\mathcal{C}_3$ and nothing wider.

---

## Draft for the paper

**Limitations, first paragraph (one sentence).**

> Our Keep intervention presents the predicted window by cropping, which also
> shortens the input; re-running Qwen2.5-VL on the QVH-ME $\mathcal{C}_3$ set
> with two non-cropping variants — blacking out the frames outside the window
> at the full frame count, and cropping to the window plus 5 s of context on
> each side — leaves SG at $+.41$ and $+.49$ against $+.45$ for cropping, so
> the gap we report is not an artefact of the crop, though we verify this for
> one model on one benchmark only.

**Supplementary A paragraph.**

> **Is the Keep/Random gap an artefact of cropping?** Showing only the frames
> inside the predicted window shortens the input and changes its frame density,
> so a change in the answer could in principle follow from the crop rather than
> from what the window contains. We therefore re-queried Qwen2.5-VL on all 241
> QVH-ME $\mathcal{C}_3$ questions under two variants that do not crop away the
> context, reusing the stored predicted window and the five stored random
> windows for each question so that the conditions pair window by window
> (2892 re-queries; the cropped condition is the stored run and was not
> re-executed). MASK keeps the full 128-frame grid and replaces every frame
> outside the window with black, so the frame count, the frame positions and
> the absolute-time encoding match the full-video query exactly; MARGIN crops,
> but to the window grown by 5 s on each side. Ret(none) is shared, since the
> blind query sees no frames. SG is $+.454$ [$+.389$, $+.515$] under cropping,
> $+.407$ [$+.340$, $+.474$] under MASK and $+.486$ [$+.423$, $+.547$] under
> MARGIN (item bootstrap, $B{=}5000$, one resample shared across conditions).
> Paired against cropping, MASK differs by $-.046$ [$-.091$, $-.003$] and
> MARGIN by $+.032$ [$-.009$, $+.077$]; the MASK difference lies entirely in
> Ret(rand), which rises from $.293$ to $.339$, while Ret(pred) is unchanged at
> $.747$. At the question level the cropped and masked conditions give the same
> Ret(pred) on 219 of 241 questions, and the 22 disagreements concentrate on
> short windows (median $|\hat{E}|$ 7.3 s, about 6 sampled frames, against
> 16.4 s and 14 frames overall). Nine random windows are shorter than the
> 1.18 s frame spacing and contain no sampled frame; cropping has nothing to
> show there and scores them not retained, the nine masked inputs are entirely
> black and are scored by the same rule without interpreting their responses,
> and dropping the five questions that own them shifts every SG by at most
> $.008$ and no reported value by more than $.013$. The gap therefore survives
> both non-cropping presentations, for this
> model and this benchmark cut.

---

## Files

```
outputs/camera_ready/B1_masking/
├── summary.csv            (a)/(b) x {CROP, MASK, MARGIN}: Ret(*), SG, CIs, dSG
├── per_question.csv       241 rows x 3 conditions, + |E_hat|, frames in E_hat
├── disagreement.csv       question-level Ret(pred) agreement breakdown
├── table_b1.tex           one-column, \scriptsize, booktabs, self-contained
├── zero_frame_windows.json
├── responses_{mask,margin}.jsonl   1446 rows each (merged, verified in B1-1)
├── inputs.jsonl
analysis/{b1_build_inputs,b1_requery,b1_merge_check,b1_analyze}.py
```
