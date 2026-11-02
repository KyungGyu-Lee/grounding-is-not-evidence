# Grounding is Not Evidence: Measuring Evidence Support in Grounded Video Question Answering

**WACV 2027** · Kyunggyu Lee, Jiyoung Jung · University of Seoul

This repository reproduces the paper's results — Tables 2-8, the supplementary
tables, and Figures 4, 5, S1 and S2 — **from stored per-question model
outputs**. No model is re-run, no GPU is needed, and no video is downloaded.
`make verify` (or the commands in [Reproduction](#reproduction)) regenerates
`paper/tables.tex` byte-for-byte, and `make figures` redraws the four figures
pixel-for-pixel. Table 1 quotes published figures from the cited papers rather
than measurements of ours, and the few exceptions among the rest are listed
under [Three things the released outputs do not reproduce
exactly](#three-things-the-released-outputs-do-not-reproduce-exactly).

The code to *re-run* the models is here too (`supportgain/`), for anyone who
wants to add a model or re-derive the stored outputs; that path does need GPUs,
the videos, and the upstream model repositories.

One caveat for that path. The four protocol steps write wherever `$RESULTS_DIR`
points, but the auxiliary runners that *built* the measurability inputs —
`supportgain/protocol/{gt_ceiling,roracle,qvh_roracle,step2}_runner.py`,
`supportgain/flags/{fullvideo_acc,qvh_fullvideo,build_next_rnorm,build_qvh_rnorm_v4,build_v4_priorfree_workset}.py`
and `qvhme_build/qvh_shuf_*.py` — still read the original run's directory layout
(`results/...`, `ROOT=results_v*`) instead of `outputs/` and `flags/`. Everything
they produced ships here, so nothing under [Reproduction](#reproduction) calls
them; adapt their paths if you want to rebuild those inputs from scratch.

---

## What the paper measures

A grounded Video-QA model answers a question and points at a time window
$\hat{E}$ it claims the answer rests on. Overlap metrics (IoU, IoP) score that
window against a human annotation. They do not ask whether the window carries
the information the model's own answer depends on.

**Support Gain** does. For each question we re-query the same model three ways
and compare each answer with the model's own full-video answer $a_\text{full}$:

| quantity | input shown to the model |
|---|---|
| $\mathrm{Ret(none)}$ | no frames (blind) |
| $\mathrm{Ret(rand)}$ | a random window of the same length as $\hat{E}$, averaged over 5 draws |
| $\mathrm{Ret(pred)}$ | the predicted window $\hat{E}$ |

$$\mathrm{SG} \;=\; \mathrm{Ret(pred)} - \mathrm{Ret(rand)}$$

A positive control replaces $\hat{E}$ with the ground-truth window:
$\mathrm{SG_{gt}} = \mathrm{Ret(gt)} - \mathrm{Ret(rand\_gt)}$. A model
whose positive control is not positive is marked $\dagger$ and its SG is not
interpreted.

**Measurability sets.** Not every question can carry the measurement, so three
nested filters are applied, using three reference VLMs (Qwen2.5-VL,
InternVL2.5-8B, VideoLLaMA2-7B). With $b$ = mean blind accuracy, $u$ = mean
full-video accuracy, $r$ = mean random-window accuracy:

| set | condition | NExT-GQA | QVH-ME |
|---|---|---|---|
| All | — | 2,268 | 763 |
| $\mathcal{C}_1$ prior-exclusion | $b < 1$ | 1,526 | 560 |
| $\mathcal{C}_2$ answerability | $u - b \ge 1/3$ | 1,116 | 427 |
| $\mathcal{C}_3$ evidence-specificity | $\rho = \frac{r-b}{u-b} \le 0.4$ | 248 | 241 |

Both thresholds are applied with a `1e-9` tolerance — `u - b >= 1/3 - 1e-9`
and `rho <= 0.4 + 1e-9` in `paper/build_paper_numbers.py` — which is what keeps
the questions whose $\rho$ is exactly .4 in exact arithmetic (48 on NExT-GQA,
30 on QVH-ME) inside $\mathcal{C}_3$. Twelve of those on NExT-GQA and four on
QVH-ME land just above .4 in floating point, so implementing the condition
without the tolerance changes $|\mathcal{C}_3|$ to 236 and 237.

> **Note for readers of earlier artifacts.** The internal code used to call
> these sets by flipped names (`C2` for prior-exclusion, `C1` for
> answerability). **This repository uses the paper's names everywhere**, in the
> code, in `paper/figdata_*.json` and in every CSV.

---

## Install

CPU only, for the tables and figures:

```bash
pip install numpy pandas matplotlib
```

That is the complete dependency list for everything under
[Reproduction](#reproduction) — three packages, no torch, no decord.

To re-run inference (`supportgain/`) you additionally need, per model:
`torch`, `torchvision`, `transformers`, `decord`, `pillow`, `peft`,
`qwen_vl_utils`, `easydict`, `clip`, plus each upstream repository on
`PYTHONPATH` (see [Models](#models)). Those environments are mutually
incompatible — we used one conda env per model family, exactly as each
upstream repo specifies.

To rebuild the QVH-ME annotations (`qvhme_build/`): `google-genai`,
`huggingface_hub`, and a `GEMINI_API_KEY` in the environment.

---

## Reproduction

From a clean checkout, in this order:

```bash
# 0. rebuild figdata from the stored per-question outputs  (~40 s per benchmark)
python paper/build_paper_numbers.py --bench nextgqa
python paper/build_paper_numbers.py --bench qvhme

# 1. Tables 2-8 + supplementary S1-S4  ->  paper/tables.tex
python paper/gen_tables.py
#    gate: Table 5, QVH-ME C3, Qwen2.5-VL  ==  .303 / .293 / .747 / +.454
#          funnels 2,268 / 1,526 / 1,116 / 248  and  763 / 560 / 427 / 241
#          Table 6 mean SG  .023 / .026 / .032 / .124  and  .125 / .144 / .161 / .212

# 2. Table S3 overlap diagnostics, including the retention-by-IoU block
python analysis/c4_iou_high.py
#    gate: IoP bins .836/.761/.805/.783, .861/.809/.803/.764, .813/.683/.779/.765
#          IoU >= .9 answer-reproduction failure 11.6% (NExT-GQA C2) / 14.3% (QVH-ME C2)

# 3. Table 8 / S4 — grounded-correct answers by SG sign, three decimals
python analysis/c1_gc_sign.py
#    gate: all 21 published N_gc exact; QVH-ME proportions identical at 3 dp

# 4. Table S2 — components, SG CI and the SG_gt CI for all 24 cells
python analysis/c2_sggt_ci.py
#    gate: every SG_gt point estimate and both CI bounds equal figdata to 0.00e+00

# 5. Supplementary scalars build_paper_numbers does not emit: the
#    measurability-restricted analyses, Table S16, the Table S1 ratio column
#    and the paired SG difference of Sec. 6.1
python paper/gen_supp_numbers.py
#    gate: Spearman(IoU,SG) .091/.166 and .283/.344 over all grounders,
#          .099/.193 and .281/.338 over the measurable ones; residual random
#          support 19.0/18.0% and 11.6/12.0%; Table S16 prediction-matched
#          9.98/10.11% and 3.18/3.28%; the nine SG/SG_gt ratios; and
#          +.030 [-.015, +.075] for VideoChat-R1 - Qwen2.5-VL on QVH-ME C2

# 6. Table S9 and supp. A.1 — leave-one-reference-model-out
python analysis/a4_loo_reference.py
#    gate: the recomputed C1-C3 are set-identical to the released flags, all 32
#          Table S9 cells, and Qwen2.5-VL C3 SG .187 / .412 off its own reference
```

Each of these scripts **refuses to write anything if its gate fails**
(`SystemExit` before the first output file). `make verify` runs all of them
plus a static check that no adapter reads the answer text during grounding.

### Three things the released outputs do not reproduce exactly

1. **The annotation-matched random control is not the `r_oracle` field in
   `step3.json`.** The paper's SG_gt uses a control sampled with
   `forbidden = [E*]` only, shared by every model, which is what the released
   `outputs/{bench}/{model}/rand_gt_keep*.json` carries (identical segments
   across the three reference models; `gen_supp_numbers.py` asserts this).
   `supportgain/protocol/step2_segments.py` instead builds `r_oracle` with
   `forbidden = [E_hat, E*]`, which makes it model-specific, and the
   `r_oracle` field stored in `step3.json` is that one. Rebuilding the
   annotation-matched control from `step2_segments.py` or from `step3.json`
   therefore gives different windows than the paper used. Nothing reported in
   the paper reads that field.

   A related consequence: Table S16's NExT-GQA annotation-matched row (1.26% /
   1.31%) is taken over C1 (1,526 questions). The released `rand_gt_keep`
   covers the 1,507 that have a full-video reference output; the other 19 have
   `|E*|/T` between .42 and .50, so no length-matched window can avoid `E*` and
   all five of their draws fall back — they carry 95 of the 96 fallback draws.
   Over the released questions the rate is .01% / .07%, which is what
   `gen_supp_numbers.py` gates.

   A second one: the shared windows themselves ship inside the `segment` field
   of `rand_gt_keep*.json`, but the builder that drew the QVH-ME set
   (`qvh_r_oracle_shared.json`) is not in this repository. The NExT-GQA windows
   re-derive exactly from the seed rule in `supportgain/core/intervals.py`
   (`md5(vqid + '_oracle')`, one `default_rng(base + k)` per draw — 1,507 of
   1,507 match); the QVH-ME windows do not, because the rejection loop consumes
   a variable number of draws. Every number in the paper reads the stored
   windows, so none of them depends on re-deriving these.

2. **QVH-ME VideoLLaMA2 has two full-video runs that disagree.**
   `flags/qvhme/fullvideo_videollama2_shard*.json` and
   `outputs/qvhme/videollama2/step1.json` differ on 24 of the 565 questions
   both cover (4.2%; accuracy .492 vs .508). `u` and C1-C3 are defined from the
   `flags/` run. The only argument that differs between the two paths is the
   clip end time handed to VideoLLaMA2's video processor — the dataset
   `duration` column on one path, the decoder's own duration on the other —
   which sets the model's fixed 8-frame uniform grid; the prompt and the
   decoding settings are identical. The other four reference-model pairs agree
   on 100% of questions.

3. **19 NExT-GQA C1 questions never reach C2.** `u` needs a full-video output
   from all three reference models; 1,507 of the 1,526 C1 questions have one,
   so the remaining 19 are excluded from C2 by missing data rather than by the
   `u - b >= 1/3` criterion.

Figures:

```bash
python paper/make_fig4.py            # Fig 4  -> paper/figures/fig4.pdf  (+ .png)
python paper/make_fig5.py            # Fig 5  -> paper/figures/fig5.pdf  (+ .png)
                                     #          (prints the comparison against the
                                     #           published values)
python analysis/a2_threshold_sensitivity.py   # Fig S1 -> analysis/A2_threshold_sensitivity/
                                              #           fig_s1_rho_sweep.pdf (+ .png)
python analysis/a3_draw_variance.py           # Fig S2 -> analysis/A3_draw_variance/
                                              #           fig_s2_draw_subsampling.pdf (+ .png)
```

`make figures` runs all four. These are the only four images in the
repository: the paper's Figures 4, 5, S1 and S2, and nothing else.

Supplementary analyses (each self-gating, each writing a `REPORT.md`):

```bash
python analysis/a1_correctness_conditioned.py   # SG split by answer correctness
python analysis/a2_threshold_sensitivity.py     # tau_u and tau_rho sweeps
python analysis/a3_draw_variance.py             # variance across the 5 random draws
python analysis/b1_analyze.py                   # CROP vs MASK vs MARGIN (cropping control)
python analysis/a4_loo_reference.py             # leave-one-reference-model-out (Table S9)
```

`analysis/b1_requery.py` is the one script here that needs a GPU; its stored
responses (`analysis/B1_masking/responses_{mask,margin}.jsonl`) ship with the
repository, so `b1_analyze.py` runs without it.

---

## Layout

```
supportgain/            the evaluation protocol as an importable package
  paths.py              every filesystem root, resolved from __file__ or $TEF_* env vars
  config/               shared prompts (prompts.py) and constants (settings.py)
  core/                 intervals, interventions, metrics, answer parser, video decoding
  protocol/             step1 (answer + ground) -> step2 (random windows) -> step3
                        (intervention re-query) -> step4 (metrics), plus the
                        positive-control runners
  models/               base.py + one adapter per evaluated model
flags/                  inputs to the C1-C3 filters: the three reference models' blind
  nextgqa/ qvhme/       and full-video answers, and the stored rho
outputs/                stored per-question model outputs — the reproduction data
  nextgqa/<model>/      step1.json (answer + predicted window + IoU/IoP)
  qvhme/<model>/        step3.json (blind / keep / remove / 5 random draws)
                        gt_ceiling_shard*.json (positive control, Ret(gt))
                        rand_gt_keep*.json — only under the three reference models
                        (qwen25vl, internvl, videollama2): Ret(rand_gt) is reused
                        from their stored runs, while every other model takes it
                        from its own gt_ceiling record
paper/                  table and figure generation
  build_paper_numbers.py   the single builder for every paper number
  gen_tables.py            -> paper/tables.tex
  gen_aux_numbers.py       the Sec 6.3 scalars that figdata does not carry
  gen_supp_numbers.py      the supplementary scalars figdata does not carry:
                           measurability-restricted analyses, Table S16,
                           the Table S1 ratio column, the Sec 6.1 paired CI
  make_fig4.py make_fig5.py      -> paper/figures/fig4.pdf, fig5.pdf (+ .png)
  figdata_{nextgqa,qvhme}.json   regenerable; shipped so the tables build immediately
analysis/               camera-ready supplementary analyses, each with its own REPORT.md
  A2_threshold_sensitivity/    + Fig S1, fig_s1_rho_sweep.pdf (+ .png)
  A3_draw_variance/            + Fig S2, fig_s2_draw_subsampling.pdf (+ .png)
qvhme_build/            how QVH-ME was generated and screened (needs the API key)
scripts/                shard merging and video download (see scripts/README.md)
```

### Legacy key names

The keys inside the stored files predate the paper's notation and are left
untouched, so that the shipped JSON and CSV stay readable by the code that
wrote them. The mapping:

| key in `outputs/`, `flags/`, `paper/figdata_*.json`, `paper/tables/*.csv` | paper |
|---|---|
| `Kbl`, `K_blind` | $\mathrm{Ret(none)}$ |
| `KR`, `K_R` | $\mathrm{Ret(rand)}$ |
| `KE`, `K_E` | $\mathrm{Ret(pred)}$ |
| `KGT`, `K_GT`, `kgt` | $\mathrm{Ret(gt)}$ |
| `KRG`, `K_R_gt`, `krgt` | $\mathrm{Ret(rand\_gt)}$ |
| `SG`, `SG_pred` | SG |
| `SG_GT` | $\mathrm{SG_{gt}}$ |
| `ESR` | evidence support ratio |
| `N_grounded_correct` | $N_{gc}$ |
| `SGA` | share of grounded-correct answers with $\mathrm{SG} > 0$ |
| `UGA_A_eq0`, `UGA_B_lt0`, `UGA_total` | $\textsc{uga}_{=0}$, $\textsc{uga}_{<0}$, and their sum ($\mathrm{SG} \le 0$) |
| `predicted_iv`, `oracle_iv` | $\hat{E}$, $E^*$ |
| `ehl`, `gtl` | $\lvert\hat{E}\rvert$, $\lvert E^*\rvert$ |
| `r_pred`, `r_oracle` | the prediction-matched and annotation-matched random windows (for `r_oracle` see caveat 1 above) |
| `rnorm`, `r_norm` | $\rho$ |
| `blind`, `predicted_keep`, `r_pred_keep_<k>` (step3 `preds` labels) | the three re-queries behind $\mathrm{Ret(none)}$, $\mathrm{Ret(pred)}$ and $\mathrm{Ret(rand)}$ |
| `mean_of_means`, `pooled` | unweighted mean over models, mean over (question, model) rows |

`supportgain/core/metrics.py` and `supportgain/protocol/step4_metrics.py` hold
a separate, older metric suite (`Self-CTEF`, `GT-CTEF@Correct`, `CGA`,
`OP-Rescue`, …) reached only through `run.py --steps 4`. No paper number comes
from it, and its keys are not in the table above.

---

## Data

**QVH-ME**, the benchmark introduced in this paper (763 multiple-choice
questions over QVHighlights clips, two-stage generation, fully hand-verified),
is released separately:

> https://huggingface.co/datasets/kyunggyu-lee/QVH-ME — CC BY-NC-SA 4.0

NExT-GQA and the source videos are not redistributed here. See
[DATA_LICENSE.md](DATA_LICENSE.md) for where to obtain them and under which
terms. Nothing in [Reproduction](#reproduction) needs them: the stored outputs
under `outputs/` are sufficient.

---

## Models

Seven grounded Video-QA models are evaluated, and three general VLMs act as
reference models for the measurability filters (Qwen2.5-VL is both).

| | model | upstream |
|---|---|---|
| evaluated | Qwen2.5-VL-7B-Instruct | https://github.com/QwenLM/Qwen2.5-VL |
| evaluated | VideoMind-2B | https://github.com/yeliudev/VideoMind |
| evaluated | Grounded-VideoLLM (Phi-3.5) | https://github.com/WHB139426/Grounded-Video-LLM |
| evaluated | GroundVTS-Q | https://github.com/Florence365/GroundVTS |
| evaluated | VTimeLLM (Vicuna-7B-v1.5) | https://github.com/huangb23/VTimeLLM |
| evaluated | TimeChat (LLaMA-2-7B) | https://github.com/RenShuhuai-Andy/TimeChat |
| evaluated | VideoChat-R1-7B *(QVH-ME only)* | https://github.com/OpenGVLab/VideoChat-R1 |
| reference | InternVL2.5-8B | https://github.com/OpenGVLab/InternVL |
| reference | VideoLLaMA2-7B | https://github.com/DAMO-NLP-SG/VideoLLaMA2 |

Point the adapters at your clones with
`TEF_VIDEOMIND_ROOT`, `TEF_GROUNDVTS_ROOT`, `TEF_GVLLM_ROOT`,
`TEF_VTIMELLM_ROOT`, `TEF_TIMECHAT_ROOT` (defaults: `~/<RepoName>`), and at
checkpoints with `TEF_VIDEOMIND_CKPT`, `TEF_GROUNDVTS_CKPT`, `VCR1_MODEL`,
`TEF_INTERNVL_CKPT`, `TEF_VIDEOLLAMA2_CKPT`. (`TEF` is this project's working
name; the environment variables keep it so existing setups keep working.)

---

## How the protocol is applied per model

Everything below is read off `supportgain/`. The task text is shared; what
differs is each model's native input and output format. **Bold = this model
differs from the others in that column.**

The question and the five options are normalized once in
`supportgain/protocol/step1_full.py` (`normalize_question`: capitalize and
ensure a trailing `?`; `normalize_option`: capitalize), and every model decodes
greedily (`do_sample=False`, no beam search).

### QA

| model | QA prompt | decoder prime | max_new_tokens | letter parsing |
|---|---|---|---|---|
| Qwen2.5-VL | shared `build_qa_prompt`: `Question: …\nOptions:\n(A)…(E)…\nAnswer with the letter only.` | — | 16 | shared `parse_answer` |
| VideoChat-R1 | shared | — | **256** | shared, after **stripping `<think>…</think>` and taking the `<answer>` inner text** |
| GroundVTS-Q | shared | — | 16 | shared |
| Grounded-VideoLLM | shared, prefixed with the Phi-3.5 `<image>` token | — | 16 | shared |
| VideoMind | **own template**: `question\nOptions:\n(A)…\nPlease only give the best option.` (no `Question:` prefix) | **`Best Option: (`** | 32 | shared |
| VTimeLLM | **own copy of the shared wording, prefixed `<video>\n`** | — | 32 | shared |
| TimeChat | shared, under a **system message** | **`The answer is (`** | 32 | shared |
| InternVL2.5-8B *(ref)* | shared | — | 32 | shared |
| VideoLLaMA2-7B *(ref)* | shared | — | 32 | shared |

### Frames

| model | sampling | count | extra |
|---|---|---|---|
| Qwen2.5-VL | `decode_at_fps(fps=2.0)`, clipped to [4, 128] | variable | **effective fps `n/duration` passed to the processor** so `second_per_grid_ts` encodes real seconds |
| VideoChat-R1 | identical to Qwen (control parity) | variable | same eff-fps fix |
| GroundVTS-Q | identical to Qwen | variable | same eff-fps fix; **pixel budget 36–64 × 28²** |
| Grounded-VideoLLM | `read_frames_decord(..., 'middle')` | **fixed 96 @224 + 12 spatial segments @336** | two pixel streams |
| VideoMind | native file ingest, fps 2.0 | **answerer 32 frames (128–256 × 28²); grounder 100 frames (36–64 × 28²)** | the only model whose answerer and grounder see different budgets |
| VTimeLLM | uniform `linspace` over the clip | **fixed 100** | frames → **CLIP ViT-L/14 features**, not pixels |
| TimeChat | internal (`upload_video_without_audio`) | **fixed 96** | timestamp-aware encoder |
| InternVL2.5-8B *(ref)* | segment midpoints | **fixed 8 @448** | — |
| VideoLLaMA2-7B *(ref)* | upstream video processor | upstream default | — |

### How the Keep / Remove intervention reaches the model

| model | mechanism | blind |
|---|---|---|
| Qwen2.5-VL, GroundVTS-Q, VideoChat-R1 | **frame subset** of the already-decoded stack (`core/interventions.py`, phase 2 = select) | all-zero frames |
| Grounded-VideoLLM | `read_frames_decord(clip=(s,e))` for keep; **ffmpeg** trim+concat complement for remove | zeroed pixel tensors |
| VideoMind | native `video_start`/`video_end` crop for keep; **ffmpeg** complement for remove | **a 0.2 s sliver of the clip, not black** |
| VTimeLLM | **ffmpeg** crop, then re-sample 100 frames from the crop | zeros `100×224×224` |
| TimeChat | **ffmpeg** crop, then re-sample 96 frames | **a synthetic black mp4** (`lavfi color=black`, 1–8 s) |

Three models keep the original frame grid and subset it; four re-encode the
window and re-sample a *fixed* budget from it, so the latter group sees the
evidence window at a higher frame density. `analysis/b1_analyze.py` bounds the
size of that effect for Qwen2.5-VL on QVH-ME $\mathcal{C}_3$.

### Grounding: output format, conversion to seconds, parse failure

All models receive the same `GROUNDING_CORE` task text, except VideoMind.

| model | directive appended to `GROUNDING_CORE` | native output | → seconds | parse-fail path |
|---|---|---|---|---|
| Qwen2.5-VL | duration line + `Format only: start=X.X, end=Y.Y` | free text | `parse_grounding`, `start=/end=` regex | **free-form `X to Y` fallback**, then clamp, then `parse_fail` |
| VideoChat-R1 | duration line + `… within <answer></answer> tags in the format 'start to end'` | tagged text | strip `<think>`, take `<answer>`, then `parse_grounding` | as Qwen |
| GroundVTS-Q | duration line + `Respond in the format 'X to Y seconds'.` | free text | own regex | **falls back to the first two numbers anywhere in the response** — the loosest fallback of any model |
| Grounded-VideoLLM | `<timestamp_grounding>` token + `Please return the start and end timestamps.` | **discrete tokens `<0>…<299>`** | `duration × idx / 300` | needs ≥ 2 tokens, else no span |
| VideoMind | **does not use `GROUNDING_CORE`** — uses VideoMind's own `GROUNDER_PROMPT` | **regression register tokens** | `reg[0][:, :2] × duration`, top span | empty `reg` → no span |
| VTimeLLM | `During which frames does this evidence appear?` | **two integers 0–99** | **`× duration / 100`** | `parse_fail`; **manual clamp, `finalize_interval` not applied** |
| TimeChat | `… 'start - end seconds'` | two floats | already absolute seconds | `parse_fail`; **manual clamp, `finalize_interval` not applied** |

Grounding `max_new_tokens` is 64 everywhere except VideoMind and VideoChat-R1
(256). A failed parse yields `predicted_iv = None`, which step 3 records as
`pred = -1`; that never equals `a_full`, so the question scores as not
retained.

### Grounding never sees the answer

Grounding is question-conditioned by design: the predicted answer must not leak
into the window the model proposes. `step1_full.py` therefore passes `None` as
`predicted_answer`, filling it only for the answer-conditioned ablation
(`GVLLM_ANSCOND=1`). Every occurrence of `predicted_answer` inside a
`predict_grounding` body, across all nine adapters:

| adapter | occurrences in the body | what they are |
|---|---|---|
| `qwen25vl`, `groundvts`, `vtimellm`, `timechat`, `videochat_r1` | 1 each | a comment |
| `videomind`, `internvl`, `videollama2` | 0 | — |
| `grounded_videollm` | **1 real read** | `q = (predicted_answer if (self.anscond and predicted_answer) else question)`, reached only under `GVLLM_ANSCOND=1` |

`make verify` re-checks this statically on every run.

---

## License

Code: MIT, see [LICENSE](LICENSE).
QVH-ME annotations: CC BY-NC-SA 4.0, on the Hugging Face dataset page.
Source videos and NExT-GQA annotations keep their original licenses — see
[DATA_LICENSE.md](DATA_LICENSE.md).

## Citation

```bibtex
@inproceedings{lee2027grounding,
  title     = {Grounding is Not Evidence: Measuring Evidence Support in Grounded
               Video Question Answering},
  author    = {Lee, Kyunggyu and Jung, Jiyoung},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of
               Computer Vision (WACV)},
  year      = {2027}
}
```
