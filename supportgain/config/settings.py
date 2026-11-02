"""Project-wide constants. Paths, defaults, etc.

Anything model-specific lives in `models/<name>/` instead.
"""
import os

from supportgain.paths import ROOT as TEF_ROOT, DATA_DIR as _DATA
from supportgain import paths as _p

# Data
DATA_DIR = _DATA
# Active sample set. Default is the 500-sample STRICT set; override per-run with
#   export VAL_SAMPLED_CSV=$PWD/data/val_sampled.csv
# without editing this file. Subprocesses (run_parallel.sh) inherit the env var.
VAL_SAMPLED_CSV = os.environ.get(
    'VAL_SAMPLED_CSV', os.path.join(DATA_DIR, 'val_sampled_500.csv'))

# NExT-GQA / NExT-QA paths (env-overridable so other benchmarks e.g. QVH-GQA
# can reuse the same pipeline without touching defaults)
NEXTGQA_DIR = _p.NEXTGQA_DIR
GT_GROUND_JSON = os.environ.get('GT_GROUND_JSON', os.path.join(NEXTGQA_DIR, 'gsub_val.json'))
VID_MAP_JSON = os.environ.get('VID_MAP_JSON', os.path.join(NEXTGQA_DIR, 'map_vid_vidorID.json'))
VIDEO_DIR = _p.NEXTQA_VIDEO_DIR

# Results root (env-overridable so QVH-GQA writes to results/qvh, no collision)
RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join(TEF_ROOT, 'outputs'))

# Protocol defaults
N_RANDOM = 5                 # number of random seeds for R_pred and R_oracle
DEFAULT_PHASE = 'phase1'     # 'phase1' (frame replace black) | 'phase2' (frame select)

# Random-segment overlap control (PDF §2).
# A random control span is rejected if its IoP-overlap (|R∩E| / |R|, i.e. the
# fraction of the random span that falls inside a forbidden evidence region)
# with E_hat OR E_star is >= OVERLAP_THR. Rejection-sampled up to MAX_TRIES;
# on exhaustion the lowest-overlap candidate is kept.
OVERLAP_THR = 0.2
MAX_TRIES = 100

# Grounding output unification: every model's grounding is reduced to a SINGLE
# absolute-second interval via this rule, then clamped to [0, duration].
# 'longest' = pick the longest candidate span (model-agnostic, no confidence
# needed). See supportgain.core.intervals.finalize_interval.
GROUNDING_REDUCE = 'longest'

# Random seeds for K = 0..N-1 are derived from the vqid hash;
# see supportgain/core/intervals.py:random_intervals.

# Number of labels (NExT-QA is 5-choice)
NUM_CHOICES = 5
LABELS = ['A', 'B', 'C', 'D', 'E']

# Save granularity (lower = more frequent disk writes; balance vs IO cost).
# Override via SAVE_EVERY env var, e.g. SAVE_EVERY=1 for save-after-each.
SAVE_EVERY = int(os.environ.get('SAVE_EVERY', 10))
