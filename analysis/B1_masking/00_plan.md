# B1-0 — plan for the masking / margin re-query (Qwen2.5-VL, QVH-ME C3)

Prepared 2026-10-31. **No model was run for this stage.** Everything below is
read from stored results plus CPU-only video decoding.

Reviewer concern being answered: showing only the predicted window changes the
temporal context, the frame density and the input distribution, so a changed
answer cannot be attributed to the window's content rather than to the cropping
itself.

---

## 1. Where the re-query code lives and where the preprocessing hooks in

### 1.1 The existing step-3 path (Qwen2.5-VL)

| Stage | File | What it does |
|---|---|---|
| Driver | [run.py](run.py) `--steps 3 --phase phase2` | picks the adapter, forwards `phase` |
| Protocol | [pipeline/step3_intervention.py:138-210](pipeline/step3_intervention.py#L138-L210) | per sample: `adapter.decode_video(path)` **once**, then one `apply_intervention(...)` + `adapter.predict_qa(...)` per label |
| Transform | [utils/interventions.py:135](utils/interventions.py#L135) `apply_intervention(phase, frames, timestamps, interval, kind)` | the only place the window touches the pixels |
| Decoding | [models/qwen25vl/adapter.py:67-73](models/qwen25vl/adapter.py#L67-L73) `decode_video` → [utils/video_io.py:42](utils/video_io.py#L42) `decode_at_fps(fps=2.0, max_frames=128, min_frames=4)` | uniform `linspace(0, total-1, N)` grid |
| Generation | [models/qwen25vl/adapter.py:77-115](models/qwen25vl/adapter.py#L77-L115) `_generate` / `predict_qa` | greedy (`do_sample=False`), `max_new_tokens=16`, prompt from [configs/prompts.py](configs/prompts.py) `build_qa_prompt`, parsed by [utils/parser.py](utils/parser.py) `parse_answer` |

The window therefore enters the model at exactly **one** point —
`apply_intervention` — and nothing else in the path knows which variant is
running. That is the hook.

### 1.2 The three conditions, expressed in the pipeline's own functions

| Condition | Call | Effect on the input |
|---|---|---|
| **CROP** (paper, stored, never re-run) | `apply_intervention('phase2', frames, ts, Ê, 'keep')` → [`phase2_select_frames`](utils/interventions.py#L54) | keeps only the frames with `ts ∈ [s, e)`; frame **count shrinks** |
| **MASK** (new) | `apply_intervention('phase1', frames, ts, Ê, 'keep')` → [`phase1_replace_black`](utils/interventions.py#L27) | `out[~in_ev] = 0`; frame count, grid positions and timestamps **unchanged**, pixel content outside the window set to RGB(0,0,0) |
| **MARGIN ±5 s** (new) | `apply_intervention('phase2', frames, ts, widen(Ê, 5 s), 'keep')` | crop, but the window is grown by 5 s per side and clipped to `[0, T]` |

`phase1_replace_black` is already in the repo and already does precisely what
MASK requires; MARGIN needs only the interval arithmetic, which lives in
`b1_build_inputs.py:widen` and is baked into `interval_applied` in
`inputs.jsonl`. **No existing file is modified**: the new script
[analysis/camera_ready/b1_requery.py](analysis/camera_ready/b1_requery.py)
imports `apply_intervention` and `Adapter.predict_qa` unchanged, and maps
`variant → phase` through one dict:

```python
PHASE_OF = {'mask': 'phase1', 'margin': 'phase2'}
```

so the prompt, the greedy decode and the A–E parsing are byte-identical to the
stored CROP run.

### 1.3 Blacking out the frames

Done after decoding, on the `uint8 [N, H, W, 3]` numpy array, by
`out[~in_ev] = 0` where `in_ev = (timestamps >= s) & (timestamps < e)` — the
same mask expression CROP uses to select. The frames are converted to PIL and
handed to the processor afterwards, so the model sees a genuine black frame,
not a padding token. Verified on the preview dumps (§4): the number of
non-black frames in a MASK input equals the number of frames inside the window,
exactly.

### 1.4 Does `fps_eff` survive MASK?

Yes, and this is the reason MASK is the clean control.

`Adapter._eff_fps` is set **inside `decode_video`** as `len(frames) / duration`
and is passed to the processor as `fps=[eff_fps]` so Qwen2.5-VL's
`second_per_grid_ts = temporal_patch_size / fps` encodes real seconds
([adapter.py:39-47, 93-99](models/qwen25vl/adapter.py#L39-L47)). It is computed
once per video, before any intervention, and the interventions never touch it.

* **MASK** keeps all N frames at their original grid positions, so
  `eff_fps = N/T` is not merely reused but *still correct*: the input is the
  full-video input with some pixels zeroed. Frame count, frame positions,
  sampling rate and the absolute-time encoding are identical to the
  full-video query that produced `a_full`.
* **CROP / MARGIN** keep a contiguous sub-grid of the same uniform decode, so
  the inter-frame spacing is unchanged and `eff_fps` remains the true rate
  (this is the invariance the adapter comment already relies on); what changes
  is the number of frames, hence the visual-token count.

For QVH-ME every clip is ~150 s and `max_frames=128` binds, so
`eff_fps ≈ 0.853` for all 241 questions.

### 1.5 How large is the cropping perturbation we are controlling for?

Frame counts over the 241 × 6 = 1446 windows (uniform 128-frame grid):

| Condition | min | median | mean | max |
|---|---|---|---|---|
| CROP | 0 | 14 | 20.2 | 127 |
| MARGIN ±5 s | 4 | 22 | 28.1 | 127 |
| MASK | 128 | 128 | 128 | 128 |

Median window length 16.4 s out of ~150 s. MASK therefore carries ≈ **9.1×**
the visual tokens of CROP at the median — that is the input-distribution shift
the reviewer is pointing at.

One concrete consequence already visible in the stored CROP results:
**9 of the 1446 windows select zero frames**, so step 3 recorded
`pred = -1` with flag `empty_or_no_interval`, which scores as *not retained*
(`KE = int(pred == a_full)`,
[build_paper_numbers.py:173](analysis/build_paper_numbers.py#L173)). MASK can
never hit this (always 128 frames) and MARGIN's minimum is 4 frames, so neither
variant has a degenerate case. B1-2 keeps the same `-1 ≠ a_full` scoring rule so
the three conditions stay comparable.

---

## 2. The re-query input list

Built by [analysis/camera_ready/b1_build_inputs.py](analysis/camera_ready/b1_build_inputs.py)
(no GPU) from:

* `set_masks('QVH')` on the v5 tree → **C3 = 241** questions (paper C3 = code `C3`);
* `results_v5/qvh/qwen25vl/phase2/step3.json` → per question `a_full`,
  `answer_id`, `duration`, `predicted_iv` = Ê, and `r_pred[0..4]` = the five
  stored random windows, **read as stored, never resampled**;
* `results_v5/qvh/qwen25vl/step1.json` → the normalized question string step 3
  itself fed the model;
* the QVH-ME CSV → the five options, normalized with the same
  `normalize_option` step 3 uses.

Result: `outputs/camera_ready/B1_masking/inputs.jsonl`,
**241 × (1 pred + 5 rand) × 2 variants = 2892 lines**, 0 skipped, verified
6 windows per question per variant. Each line also carries `crop_parsed`, the
stored CROP prediction for that exact (question, window), so the CROP↔MASK↔MARGIN
pairing is question-wise **and** window-wise by construction.

MARGIN interval check: 0 violations of
`src ⊆ applied`, `0 ≤ applied ≤ T`, growth ≤ 5 s per side; 119 windows clipped at
0 s and 131 at the clip end (these start/end at a clip boundary, so they grow on
one side only).

**Blind is not re-queried.** It sees no frames, so `Ret(none)` is shared by all
three conditions — this is also why the count is 6 and not 7 windows.

---

## 3. The re-query script (written, not executed)

[analysis/camera_ready/b1_requery.py](analysis/camera_ready/b1_requery.py)

* imports `models.qwen25vl.adapter.Adapter` and calls `predict_qa`, so prompt,
  greedy decoding and A–E parsing are the pipeline's, unchanged;
* imports `utils.interventions.apply_intervention`, unchanged;
* decodes each video **once** per question and reuses the frame stack for all
  12 queries (6 MASK + 6 MARGIN), MASK then MARGIN in order within the shard;
* writes `responses_{mask,margin}[_tag]_shard{i}.jsonl`, one JSON object per
  query with `vqid, variant, window_type ∈ {pred, rand_0..4}, response, parsed`
  plus `interval_src, interval_applied, n_frames_in, n_frames_full, a_full,
  answer_id, flag`;
* **resumable**: on start it reads the output file and skips every
  `(vqid, variant, window_type)` already present, appending afterwards;
* sharding is `index % nshard == shard` over the sorted question list, so
  shards are balanced and a `--limit` smoke test stays a prefix.

Writes nothing outside `outputs/camera_ready/B1_masking/`. The stored CROP tree
is opened read-only and never written.

---

## 4. Preview dumps (verification by eye)

`python analysis/camera_ready/b1_requery.py --preview 3` — no model, no GPU.
Output in `outputs/camera_ready/B1_masking/preview/<vqid>/`:
`full.png`, `mask_pred.png`, `margin_pred.png` (16-column contact sheets, one
cell per decoded frame, in time order) plus `preview_meta.json`.

| vqid | Ê (s) | MASK frames | non-black | MARGIN interval | MARGIN frames | margin added (s) |
|---|---|---|---|---|---|---|
| `-4Mlqc7PbZY_60.0_210.0_4884` | 64.8–92.3 | 128 | 24 | 59.8–97.3 | 32 | 5.0 / 5.0 |
| `-_s0sXOfS3w_210.0_360.0_2435` | 0.0–26.8 | 128 | 23 | 0.0–31.8 | 27 | 0.0 / 5.0 |
| `-_s0sXOfS3w_510.0_660.0_8312` | 67.4–89.2 | 128 | 18 | 62.4–94.2 | 27 | 5.0 / 5.0 |

All three: `MASK frames = 128 = full-video frame count`, and
`non-black = frames inside the window`, so the black-out is exact. The second
row starts at 0 s, which is why only the right side grows by 5 s.
`mask_pred.png` shows one contiguous band of real frames in an otherwise black
grid; `full.png` is the same grid with no black cells.

---

## 5. Cost estimate for B1-1

Measured on this machine, CPU-only: **decode = 1.76 s median** per 150 s clip
(128 frames, 300×534).

Generation cost is taken from the stored logs of this exact model and data:

* `logs/qwen25vl_phase2_step1_gpu0.log`: 12.0 s per sample = 1 decode + one
  128-frame QA + one 128-frame grounding call → a **128-frame query ≈ 5.1 s**;
* `logs/qwen25vl_phase2_step3_gpu0.log`: 37.0 s per sample = 1 decode + 13
  CROP-sized queries → a **~20-frame query ≈ 2.7 s**. MARGIN's median is 22
  frames, so ≈ 3.0 s.

| | queries | s/query | total (1 GPU) |
|---|---|---|---|
| MASK (128 frames) | 1446 | ≈ 5.1 | ≈ 2.0 h |
| MARGIN (median 22 frames) | 1446 | ≈ 3.0 | ≈ 1.2 h |
| decode (once per question) | 241 | 1.76 | ≈ 7 min |
| **total** | **2892** | **≈ 4.2** | **≈ 3.4 h** |

**On 4 GPUs (4 shards of ~60 questions): ≈ 50–55 min.**
The 10-question smoke test (60 queries, 1 GPU) should take **≈ 5 min**.

**GPU memory.** Qwen2.5-VL-7B-Instruct in bf16 with FlashAttention-2 ≈ 16.5 GB
of weights. The heaviest B1 input is a MASK query, which is byte-for-byte the
same shape as the full-video query already run in step 1 (128 frames,
300×534 → ≈ 13 k visual tokens after 2×2 spatial + 2-frame temporal merging),
so peak usage should land near the step-1 figure, **≈ 20 GB estimated**, well
inside one A6000. All four A6000s are currently idle (49140 MiB each, 11–24 MiB
used), so one process per GPU is fine. The smoke test will report the measured
peak before the full run.

---

## 6. What B1-1 will run (for reference, not started)

```bash
# smoke: 10 questions, 60 queries, GPU 0
cd <repo root>
CUDA_VISIBLE_DEVICES=0 <qwen env>/bin/python \
  analysis/camera_ready/b1_requery.py --shard 0 --nshard 1 --limit 10 --tag smoke

# full: 4 shards, one per GPU
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g <qwen env>/bin/python \
    analysis/camera_ready/b1_requery.py --shard $g --nshard 4 \
    > logs/b1_requery_g$g.log 2>&1 &
done; wait
```

Env: `Qwen2.5_Correction` (torch 2.6.0+cu124, transformers 4.51.2, decord 0.6.0)
— the same environment the stored QVH-ME Qwen2.5-VL results were produced in.
