"""Phase 1 / 2 / 3 interventions on a decoded frame stack.

All three operate on numpy frames so the model adapter doesn't need to
know which protocol is running. The adapter just passes the result to
its native inference pipeline.

Phase 1 — frame replace with black (artifact-prone but simple)
Phase 2 — frame select / drop (preserves no artifact but shifts length)
Phase 3 — natural-filler replace (preserves length AND avoids black artifact)
          We implement the "frame-grid" variant: keep the sampled grid
          and the timestamps; only swap the pixel content of frames in
          (or out of) the evidence interval with frames taken from the
          complement of that interval (within the same video).
"""
import numpy as np


def _in_evidence_mask(timestamps, interval):
    s, e = interval
    return (timestamps >= s) & (timestamps < e)


# ---------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------

def phase1_replace_black(frames, timestamps, interval, kind):
    """Replace selected frames with RGB(0,0,0). Frame count unchanged.

    kind: 'full', 'blind', 'keep' (non-evidence→black), 'remove' (evidence→black)
    """
    out = frames.copy()
    if kind == 'full':
        return out
    if kind == 'blind':
        out[:] = 0
        return out
    if interval is None:
        return None
    in_ev = _in_evidence_mask(timestamps, interval)
    if kind == 'keep':
        out[~in_ev] = 0
    elif kind == 'remove':
        out[in_ev] = 0
    else:
        raise ValueError(f'unknown kind: {kind}')
    return out


# ---------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------

def phase2_select_frames(frames, timestamps, interval, kind):
    """Keep / drop frames at the timestamp level. Frame count CHANGES.

    Returns None if the selection is empty.
    """
    if kind == 'full':
        return frames.copy()
    if kind == 'blind':
        return np.zeros_like(frames)
    if interval is None:
        return None
    in_ev = _in_evidence_mask(timestamps, interval)
    if kind == 'keep':
        sel = frames[in_ev]
    elif kind == 'remove':
        sel = frames[~in_ev]
    else:
        raise ValueError(f'unknown kind: {kind}')
    return sel if sel.shape[0] > 0 else None


# ---------------------------------------------------------------------
# Phase 3 (frame-grid variant)
# ---------------------------------------------------------------------

def phase3_replace_filler(frames, timestamps, interval, kind, rng=None):
    """Timeline-preserving counterfactual replay (frame-grid variant).

    The frame count and the sampling grid stay identical. Frames inside
    the evidence (Remove) or outside it (Keep) are swapped with frames
    taken — with replacement — from the complementary side of the same
    video. If the complement is empty, we fall back to black (degrades
    to Phase 1 for that sample).
    """
    out = frames.copy()
    if kind == 'full':
        return out
    if kind == 'blind':
        # 'blind' for Phase 3 is also natural-content swap of everything;
        # the most consistent definition is to shuffle frames in place
        # so timestamps stay the same but content order is destroyed.
        rng = rng if rng is not None else np.random.default_rng(0)
        idx = rng.permutation(out.shape[0])
        return out[idx]
    if interval is None:
        return None

    in_ev = _in_evidence_mask(timestamps, interval)
    if kind == 'remove':
        # swap evidence-side frames with frames from outside the evidence
        donor_idx = np.where(~in_ev)[0]
        target_idx = np.where(in_ev)[0]
    elif kind == 'keep':
        # swap non-evidence frames with frames from inside the evidence
        donor_idx = np.where(in_ev)[0]
        target_idx = np.where(~in_ev)[0]
    else:
        raise ValueError(f'unknown kind: {kind}')

    if len(donor_idx) == 0:
        # no donor available → fall back to black
        out[target_idx] = 0
        return out

    rng = rng if rng is not None else np.random.default_rng(0)
    sampled = rng.choice(donor_idx, size=len(target_idx), replace=True)
    out[target_idx] = frames[sampled]
    return out


# ---------------------------------------------------------------------
# Dispatcher (lets pipeline code stay phase-agnostic)
# ---------------------------------------------------------------------

PHASE_FN = {
    'phase1': phase1_replace_black,
    'phase2': phase2_select_frames,
    'phase3': phase3_replace_filler,
}


def apply_intervention(phase, frames, timestamps, interval, kind, **kwargs):
    if phase not in PHASE_FN:
        raise ValueError(f'unknown phase: {phase}')
    return PHASE_FN[phase](frames, timestamps, interval, kind, **kwargs) \
        if phase == 'phase3' else PHASE_FN[phase](frames, timestamps, interval, kind)
