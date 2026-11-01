"""Interval generation utilities.

  - oracle_interval(gt, qid)            ← longest GT location (E_star)
  - parse_grounding(text, duration)     ← parse free-form model output
  - random_intervals(vqid, length, dur, N)  ← N deterministic random spans
                                              (per-vqid hashed seeds)
"""
import hashlib
import re
import numpy as np


def oracle_interval(gt_video_entry, qid):
    """Longest GT location. Returns (start, end) in seconds or None."""
    locs = gt_video_entry.get('location', {}).get(str(qid))
    if not locs:
        return None
    best = max(locs, key=lambda l: l[1] - l[0])
    return (float(best[0]), float(best[1]))


def parse_grounding(text, duration):
    """Parse a free-form grounding output. Returns (start, end, flag).

    Returns (None, None, flag) when parsing fails or the interval is
    invalid even after clamping.
    """
    flags = []
    s = e = None

    m = re.search(r'start\s*=\s*([\d.]+)[^\d]+end\s*=\s*([\d.]+)',
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        s, e = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(
            r'(?:from\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*(?:to|-|–|—)\s*(\d+(?:\.\d+)?)',
            text, re.IGNORECASE,
        )
        if m:
            s, e = float(m.group(1)), float(m.group(2))
            flags.append('fallback_freeform')
        else:
            return None, None, 'parse_fail'

    if s > e:
        s, e = e, s
        flags.append('reversed')

    if s < 0 or s > duration or e < 0 or e > duration:
        s = max(0.0, min(s, float(duration)))
        e = max(0.0, min(e, float(duration)))
        flags.append('clamped')

    if e <= s:
        return None, None, '+'.join(flags + ['invalid_after_clamp'])

    if e - s < 0.5:
        flags.append('too_short')
    if (e - s) / duration > 0.95:
        flags.append('near_full')
    return s, e, '+'.join(flags) if flags else None


def finalize_interval(spans, duration, reduce='longest'):
    """Unify ANY model's grounding output into one absolute-second interval.

    Every adapter, after decoding its native grounding output (free-text /
    regression token / quantized temporal token) into a list of (start, end)
    second candidates, should pass them through this function so the final
    contract is identical across models:
      - reduce multiple candidates to ONE ('longest' = max span length;
        'first' = keep the model's top-ranked / first candidate)
      - swap reversed bounds, clamp to [0, duration], require start < end

    Returns (start, end, flag) or (None, None, flag) when nothing is valid.
    """
    flags = []
    cands = []
    for span in spans or []:
        if span is None:
            continue
        s, e = span
        if s is None or e is None:
            continue
        s, e = float(s), float(e)
        if s > e:
            s, e = e, s
            if 'reversed' not in flags:
                flags.append('reversed')
        s = max(0.0, min(s, float(duration)))
        e = max(0.0, min(e, float(duration)))
        if e > s:
            cands.append((s, e))
    if not cands:
        return None, None, '+'.join(flags + ['no_valid_span'])
    if len(cands) > 1:
        flags.append(f'reduced_{len(cands)}_to_1')
        if reduce == 'longest':
            s, e = max(cands, key=lambda x: x[1] - x[0])
        else:  # 'first' / top-ranked
            s, e = cands[0]
    else:
        s, e = cands[0]
    if e - s < 0.5:
        flags.append('too_short')
    if float(duration) > 0 and (e - s) / float(duration) > 0.95:
        flags.append('near_full')
    return s, e, '+'.join(flags) if flags else None


def overlap_coverage(rand, ref):
    """IoP-style overlap: |rand ∩ ref| / |rand|.

    Fraction of the random span that falls inside the (forbidden) evidence
    region `ref`. Returns 0.0 if either is None or rand has zero length.
    """
    if rand is None or ref is None:
        return 0.0
    s = max(rand[0], ref[0])
    e = min(rand[1], ref[1])
    inter = max(0.0, e - s)
    rlen = rand[1] - rand[0]
    return inter / rlen if rlen > 0 else 0.0


def random_intervals(vqid, length, duration, N=3, forbidden=None,
                     overlap_thr=0.2, max_tries=100, return_meta=False):
    """Return N random intervals of given `length` with deterministic
    per-vqid seeds. Same (vqid, k) always yields the same interval.

    forbidden : list of (s, e) evidence regions to avoid (e.g. [E_hat, E_star]);
                None entries are ignored. A candidate is rejected when its
                overlap_coverage (|R∩E| / |R|) with ANY forbidden region is
                >= overlap_thr. Rejection-sampled up to `max_tries`; on
                exhaustion the lowest-max-overlap candidate seen is kept so a
                valid-length span is always returned. When forbidden is empty
                the first draw always passes (identical to the unconstrained
                behaviour and to previously generated results).

    return_meta : if True, also return a parallel list of per-seed dicts
                  {'overlap': final IoP-overlap vs forbidden (None if no valid
                  span), 'fallback': True when the threshold could not be met
                  (max_tries exhausted, or the span spans the whole video)}.

    A non-positive `length` yields `[None] * N` (NOT a full-video span): a
    zero/invalid evidence length has no meaningful same-length random control.
    """
    if length <= 0:
        ivs = [None] * N
        if return_meta:
            return ivs, [{'overlap': None, 'fallback': False} for _ in range(N)]
        return ivs

    forb = [f for f in (forbidden or []) if f is not None]
    base_seed = int(hashlib.md5(str(vqid).encode()).hexdigest()[:8], 16) % (2**31)
    max_start = max(0.0, float(duration) - length)
    out, meta = [], []
    for k in range(N):
        rng = np.random.default_rng(base_seed + k)
        if max_start == 0:
            # Span covers (≈) the whole video — overlap is unavoidable.
            cand = (0.0, float(duration))
            ov = max((overlap_coverage(cand, f) for f in forb), default=0.0)
            out.append(cand)
            meta.append({'overlap': ov, 'fallback': True})
            continue
        best, best_ov, ok = None, None, False
        for _ in range(max_tries):
            s = float(rng.uniform(0.0, max_start))
            e = min(s + length, float(duration))
            cand = (s, e)
            ov = max((overlap_coverage(cand, f) for f in forb), default=0.0)
            if best_ov is None or ov < best_ov:
                best, best_ov = cand, ov
            if ov < overlap_thr:
                ok = True
                break
        out.append(best)
        meta.append({'overlap': best_ov, 'fallback': not ok})
    if return_meta:
        return out, meta
    return out
