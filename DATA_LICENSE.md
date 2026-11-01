# Data licensing and provenance

This repository ships **model outputs only** — one JSON record per question per
model. It redistributes no video and no third-party annotation.

## QVH-ME (ours)

763 five-choice questions over QVHighlights validation clips, generated in two
stages and hand-verified. Annotations:

> https://huggingface.co/datasets/kyunggyu-lee/QVH-ME — **CC BY-NC-SA 4.0**

The underlying clips come from **QVHighlights** (Lei et al., NeurIPS 2021),
which is released under CC BY-NC-SA 4.0 and distributes YouTube video ids, not
video files. Obtain the clips from the QVHighlights project and respect
YouTube's terms:

> https://github.com/jayleicn/moment_detr

`qvhme_build/` contains the generation and screening code. Re-running it calls
the Gemini API and needs `GEMINI_API_KEY` in the environment; no key is stored
in this repository.

## NExT-GQA

Questions, answers and temporal annotations come from **NExT-GQA** (Xiao et
al., CVPR 2023), built on **NExT-QA** (Xiao et al., CVPR 2021) over VidOR
videos. Obtain them from the authors and follow their terms:

> https://github.com/doc-doc/NExT-GQA
> https://github.com/doc-doc/NExT-QA

Set `TEF_NEXTGQA_DIR` and `TEF_NEXTQA_VIDEO_DIR` to your copies if you want to
re-run inference. Reproducing the tables and figures does not need them.

## Model weights

No checkpoint is included. Each evaluated model keeps its own license; see the
upstream repositories listed in the README.

## Code

MIT, see `LICENSE`.
