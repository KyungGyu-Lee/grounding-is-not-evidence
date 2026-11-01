# scripts/

Only two helpers ship here; neither is needed to reproduce the paper.

| file | what it does |
|---|---|
| `merge_shards.py` | merge sharded `step1` / `step3` outputs into one JSON |
| `download_videos.py` | fetch the NExT-QA videos a sample CSV refers to (set `TEF_NEXTQA_VIDEO_DIR`) |

The shell runners used for the paper's own GPU runs are **not** included: they
hard-coded one machine's conda environments and the internal result-tree
layout (`results_tsfix/`, `results_v5/`, `BP_CWD` wrappers), none of which
exists here. The equivalent invocation in this repository is:

```bash
export TEF_NEXTQA_VIDEO_DIR=/path/to/NExTVideo
export TEF_NEXTGQA_DIR=/path/to/nextgqa          # gsub_val.json, map_vid_vidorID.json
export VAL_SAMPLED_CSV=/path/to/val_full_strict.csv
export RESULTS_DIR=$PWD/outputs/nextgqa

# step 1 — full-video answer + predicted window, per model
python supportgain/run.py --model qwen25vl --step 1

# step 2 — the five random windows per question (deterministic, vqid-seeded)
python -m supportgain.protocol.step2_segments --model qwen25vl

# step 3 — the intervention re-queries (blind / keep / remove / 5 random draws)
python supportgain/run.py --model qwen25vl --step 3 --phase phase2
```

Run each model inside the environment its upstream repository specifies; they
are mutually incompatible. Shard a model across GPUs with `--start/--end` and
merge with `merge_shards.py`.
