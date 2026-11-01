"""Every filesystem root the package needs, resolved from this file's location
and overridable by environment variables. No absolute path is hard-coded."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name, default):
    return os.environ.get(name, default)


# ---- repository data (shipped) ---------------------------------------------
OUTPUTS_DIR = os.path.join(ROOT, 'outputs')
FLAGS_DIR = os.path.join(ROOT, 'flags')
PAPER_DIR = os.path.join(ROOT, 'paper')
ANALYSIS_DIR = os.path.join(ROOT, 'analysis')

# ---- source datasets (NOT shipped — download separately, see README) --------
DATA_DIR = _env('TEF_DATA_ROOT', os.path.join(ROOT, 'data'))
NEXTGQA_DIR = _env('TEF_NEXTGQA_DIR', os.path.join(DATA_DIR, 'nextgqa'))
NEXTQA_VIDEO_DIR = _env('TEF_NEXTQA_VIDEO_DIR', os.path.join(DATA_DIR, 'nextqa_videos'))
QVHME_DIR = _env('TEF_QVHME_DIR', os.path.join(DATA_DIR, 'qvhme'))
QVHME_VIDEO_DIR = _env('TEF_QVHME_VIDEO_DIR', os.path.join(QVHME_DIR, 'videos'))

# ---- upstream model repositories (clone these to re-run inference) ----------
VIDEOMIND_ROOT = _env('TEF_VIDEOMIND_ROOT', os.path.expanduser('~/VideoMind'))
GROUNDVTS_ROOT = _env('TEF_GROUNDVTS_ROOT', os.path.expanduser('~/GroundVTS'))
GVLLM_ROOT = _env('TEF_GVLLM_ROOT', os.path.expanduser('~/Grounded-Video-LLM'))
VTIMELLM_ROOT = _env('TEF_VTIMELLM_ROOT', os.path.expanduser('~/VTimeLLM'))
TIMECHAT_ROOT = _env('TEF_TIMECHAT_ROOT', os.path.expanduser('~/TimeChat'))
