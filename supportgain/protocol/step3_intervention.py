"""Step 3: Intervention inference.

Per sample run keep / remove for each segment family:
    E_hat, E_star, R_pred[0..N-1], R_oracle[0..N-1]
plus Blind.

Phase is selected via `phase` argument: 'phase1', 'phase2', or 'phase3'.
Saves to results/<model>/<phase>/step3.json.

Per-sample inference count:
    Blind                       : 1
    Keep + Remove × E_hat       : 2
    Keep + Remove × E_star      : 2
    Keep + Remove × R_pred[k]   : 2N
    Keep + Remove × R_oracle[k] : 2N
  --------------------------------------
    total                       : 5 + 4N    (e.g. N=3 → 17 per sample)

Combined with step 1 (full QA + grounding = 2) gives 7 + 4N per sample
(= 19 when N=3).
"""
import json
import os
import time
import traceback

from supportgain.config.settings import RESULTS_DIR, N_RANDOM, SAVE_EVERY
from supportgain.core.video_io import find_video_path
from supportgain.core.interventions import apply_intervention
from supportgain.config.prompts import normalize_option


def _video_path_for(vqid_meta, vid_map, video_dir):
    vid = vqid_meta['video_id']
    return find_video_path(vid, video_dir, vid_map)


def self_labels(n_random):
    """step3 labels consumed by Self-CTEF (E_hat + R_pred + blind diagnostic)."""
    out = ['blind', 'predicted_keep', 'predicted_remove']
    for k in range(n_random):
        out += [f'r_pred_keep_{k}', f'r_pred_remove_{k}']
    return out


def gt_labels(n_random):
    """step3 labels additionally consumed by GT-CTEF (E_star + R_oracle)."""
    out = ['oracle_keep', 'oracle_remove']
    for k in range(n_random):
        out += [f'r_oracle_keep_{k}', f'r_oracle_remove_{k}']
    return out


def present_families(preds, n_random):
    """Which metric families are fully present in a preds dict."""
    fams = []
    if all(lbl in preds for lbl in self_labels(n_random)):
        fams.append('self')
    if all(lbl in preds for lbl in gt_labels(n_random)):
        fams.append('gt')
    return fams


def _build_runs(r, families):
    """Yield (label, kind, interval) tuples for the requested metric families.

    families : iterable subset of {'self', 'gt'}.
      'self' = blind + E_hat (predicted) keep/remove + R_pred keep/remove
      'gt'   = E_star (oracle) keep/remove + R_oracle keep/remove
    """
    fam = set(families)
    if 'self' in fam:
        yield ('blind', 'blind', None)
        yield ('predicted_keep',   'keep',   r['predicted_iv'])
        yield ('predicted_remove', 'remove', r['predicted_iv'])
        for k, iv in enumerate(r['r_pred']):
            yield (f'r_pred_keep_{k}',   'keep',   iv)
            yield (f'r_pred_remove_{k}', 'remove', iv)
    if 'gt' in fam:
        yield ('oracle_keep',      'keep',   r['oracle_iv'])
        yield ('oracle_remove',    'remove', r['oracle_iv'])
        for k, iv in enumerate(r['r_oracle']):
            yield (f'r_oracle_keep_{k}',   'keep',   iv)
            yield (f'r_oracle_remove_{k}', 'remove', iv)


def run(adapter, model_name: str, phase: str = 'phase1',
        start: int = 0, end: int = -1, save_every: int = SAVE_EVERY,
        out_path: str = None, families='all'):
    from supportgain.config.settings import GT_GROUND_JSON, VID_MAP_JSON, VIDEO_DIR
    import pandas as pd
    from supportgain.config.settings import VAL_SAMPLED_CSV

    # Joint-protocol adapters (e.g. videochat_r1_joint, protocol C) must switch
    # their intervention-stage QA to the PLAIN prompt here: A and C may differ
    # ONLY in how E_hat is produced, never in the step3 prompt, or an SG gap
    # would have two causes. hasattr-guarded → no effect on the other adapters.
    if hasattr(adapter, 'begin_step3'):
        adapter.begin_step3()

    # Which metric families to run this pass. 'all' = self + gt (default).
    # Splitting lets you run Self-CTEF (self) first and top-up GT-CTEF (gt)
    # later without recomputing the self inferences — see label-level merge.
    if families in (None, 'all'):
        fam_set = {'self', 'gt'}
    elif isinstance(families, str):
        fam_set = set(families.split(','))
    else:
        fam_set = set(families)
    bad = fam_set - {'self', 'gt'}
    if bad:
        raise ValueError(f'unknown families: {bad} (allowed: self, gt, all)')

    step1_path = os.path.join(RESULTS_DIR, model_name, 'step1.json')
    step2_path = os.path.join(RESULTS_DIR, model_name, 'step2.json')
    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, model_name, phase, 'step3.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    step1 = json.load(open(step1_path))
    step2 = json.load(open(step2_path))
    vid_map = json.load(open(VID_MAP_JSON))
    df = pd.read_csv(VAL_SAMPLED_CSV)

    if end == -1:
        end = len(df)
    end = min(end, len(df))   # clamp so an over-specified shard range is harmless

    results = {}
    if os.path.exists(out_path):
        try:
            results = json.load(open(out_path))
        except Exception:
            results = {}

    t0 = time.time()
    processed = 0
    for idx in range(start, end):
        row = df.iloc[idx]
        vqid = f"{row['video_id']}_{row['qid']}"
        if vqid not in step1 or 'error' in step1[vqid]:
            continue
        s1 = step1[vqid]
        s2 = step2.get(vqid, {'r_pred': [None] * N_RANDOM,
                              'r_oracle': [None] * N_RANDOM})
        n_r = len(s2.get('r_pred', [])) or N_RANDOM

        # Plan the labels this pass needs, then drop ones already on disk so a
        # later --families gt pass only computes the missing E_star/R_oracle
        # inferences and merges them into the existing self record (top-up).
        planned = list(_build_runs(
            {'predicted_iv': s1['predicted_iv'],
             'oracle_iv':    s1['oracle_iv'],
             'r_pred':       s2['r_pred'],
             'r_oracle':     s2['r_oracle']}, fam_set))
        existing = results.get(vqid)
        merge = bool(existing) and 'error' not in (existing or {})
        if merge:
            have = existing.get('preds', {})
            todo = [t for t in planned if t[0] not in have]
            if not todo:
                continue  # every requested label already present
        else:
            todo = planned

        question = s1['question']  # already normalized in step1
        choices = [normalize_option(row[f'a{i}']) for i in range(5)]
        video_path = find_video_path(s1['video_id'], VIDEO_DIR, vid_map)
        if not video_path or not os.path.exists(video_path):
            continue

        try:
            frames, timestamps, duration = adapter.decode_video(video_path)

            preds, raws, flags = {}, {}, {}
            video_mode = adapter.supports_video_path_intervention()
            for label, kind, iv in todo:

                if video_mode:
                    if kind != 'blind' and iv is None:
                        preds[label] = -1
                        raws[label] = ''
                        flags[label] = 'empty_or_no_interval'
                        continue
                    spec = {
                        'phase': phase,
                        'kind': kind,
                        'interval': tuple(iv) if iv else None,
                        'total_duration': s1['duration'],
                    }
                    p, r = adapter.predict_qa_video(
                        video_path, spec, question, choices,
                    )
                else:
                    mod = apply_intervention(phase, frames, timestamps,
                                             tuple(iv) if iv else None, kind)
                    if mod is None:
                        preds[label] = -1
                        raws[label] = ''
                        flags[label] = 'empty_or_no_interval'
                        continue
                    if kind == 'blind' and not adapter.supports_blind_via_zero_frames() \
                            and phase == 'phase1':
                        # Phase 1 blind = all black frames (still supported); fall
                        # through to the standard path.
                        pass
                    p, r = adapter.predict_qa(mod, question, choices)
                preds[label] = int(p)
                raws[label] = r

            if merge:
                rec = existing
                rec.setdefault('preds', {}).update(preds)
                rec.setdefault('raws', {}).update(raws)
                rec.setdefault('flags', {}).update(flags)
            else:
                rec = {
                    'vqid': vqid,
                    'type': s1['type'],
                    'answer_id': s1['answer_id'],
                    'a_full': s1['a_full'],
                    'duration': s1['duration'],
                    'predicted_iv': s1['predicted_iv'],
                    'oracle_iv': s1['oracle_iv'],
                    'r_pred': s2['r_pred'],
                    # r_oracle here is step2's model-specific version
                    # (forbidden = [E_hat, E_star]); the annotation-matched
                    # control behind SG_gt is the shared forbidden = [E_star]
                    # one in rand_gt_keep*.json. See step2_segments.py.
                    'r_oracle': s2['r_oracle'],
                    'phase': phase,
                    'preds': preds,
                    'raws': raws,
                    'flags': flags,
                }
            rec['families_present'] = present_families(rec['preds'], n_r)
            results[vqid] = rec

            processed += 1
            elapsed = time.time() - t0
            print(f'[{idx:>4}] {vqid:<22} phase={phase} fam={",".join(sorted(fam_set))} '
                  f'a_full={s1["a_full"]} ans={s1["answer_id"]} '
                  f'+{len(preds)} preds (now {len(rec["preds"])})  ({elapsed:.0f}s)',
                  flush=True)
        except Exception as ex:
            traceback.print_exc()
            if vqid not in results:
                results[vqid] = {'vqid': vqid, 'error': str(ex)}
            continue

        if processed % save_every == 0:
            json.dump(results, open(out_path, 'w'))

    json.dump(results, open(out_path, 'w'))
    print(f'Saved {len(results)} -> {out_path}', flush=True)
    return results
