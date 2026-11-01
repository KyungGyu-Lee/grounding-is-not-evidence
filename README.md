# Grounding is Not Evidence: Measuring Evidence Support in Grounded Video Question Answering

**WACV 2027** · Kyunggyu Lee, Jiyoung Jung · University of Seoul

This repository reproduces every number, table and figure in the paper **from
stored per-question model outputs**. No model is re-run, no GPU is needed, and
no video is downloaded. The four commands in [Reproduction](#reproduction)
regenerate `paper/tables.tex` byte-for-byte and Figures 4, 5, S1 and S2
pixel-for-pixel.

The code to *re-run* the models is here too (`supportgain/`), for anyone who
wants to add a model or re-derive the stored outputs; that path does need GPUs,
the videos, and the upstream model repositories.

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
$\mathrm{SG_{gt}} = \mathrm{Ret(gt)} - \mathrm{Ret(rand\text{-}gt)}$. A model
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
```

Each of these four scripts **refuses to write anything if its gate fails**
(`SystemExit` before the first output file). `make verify` runs all of them
plus a static check that no adapter reads the answer text during grounding.

Figures:

```bash
python paper/make_fig4.py            # Fig 4  -> paper/figures/
python paper/make_fig5.py            # Fig 5  -> paper/figures/   (prints the comparison
                                     #            against the published values)
python analysis/a2_threshold_sensitivity.py   # Fig S1
python analysis/a3_draw_variance.py           # Fig S2
python analysis/c3_figures.py                 # Fig S1/S2 with the final legend placement
```

Supplementary analyses (each self-gating, each writing a `REPORT.md`):

```bash
python analysis/a1_correctness_conditioned.py   # SG split by answer correctness
python analysis/a2_threshold_sensitivity.py     # tau_u and tau_rho sweeps
python analysis/a3_draw_variance.py             # variance across the 5 random draws
python analysis/b1_analyze.py                   # CROP vs MASK vs MARGIN (cropping control)
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
                        (qwen25vl, internvl, videollama2): Ret(rand-gt) is reused
                        from their stored runs, while every other model takes it
                        from its own gt_ceiling record
paper/                  table and figure generation
  build_paper_numbers.py   the single builder for every paper number
  gen_tables.py            -> paper/tables.tex
  gen_aux_numbers.py       the Sec 6.3 scalars that figdata does not carry
  make_fig4.py make_fig5.py
  figdata_{nextgqa,qvhme}.json   regenerable; shipped so the tables build immediately
analysis/               camera-ready supplementary analyses, each with its own REPORT.md
qvhme_build/            how QVH-ME was generated and screened (needs the API key)
scripts/                shard merging and video download (see scripts/README.md)
```

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
`TEF_INTERNVL_CKPT`, `TEF_VIDEOLLAMA2_CKPT`.

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
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```
