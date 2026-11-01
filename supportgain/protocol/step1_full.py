"""Step 1: Full inference.

Per sample, run:
  - decode the video natively (adapter.decode_video)
  - predict_qa on full frames  →  a_full
  - predict_grounding          →  E_hat
  - load E_star (oracle) from GT json
  - compute IoU / IoP / lengths

Saves to results/<model>/step1.json
"""
import json
import os
import time
import traceback
from typing import Dict, Any

import pandas as pd

from supportgain.config.settings import (
    DATA_DIR, VAL_SAMPLED_CSV, GT_GROUND_JSON, VID_MAP_JSON, VIDEO_DIR,
    RESULTS_DIR, SAVE_EVERY,
)
from supportgain.core.video_io import find_video_path
from supportgain.core.intervals import oracle_interval
from supportgain.config.prompts import normalize_question, normalize_option


def _iou_iop(pred, gt):
    if pred is None or gt is None:
        return None, None
    ps, pe = pred
    gs, ge = gt
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    iou = inter / union if union > 0 else 0.0
    pred_len = pe - ps
    iop = inter / pred_len if pred_len > 0 else 0.0
    return iou, iop


def run(adapter, model_name: str, start: int = 0, end: int = -1,
        out_path: str = None, save_every: int = SAVE_EVERY) -> Dict[str, Any]:
    df = pd.read_csv(VAL_SAMPLED_CSV)
    gt_ground = json.load(open(GT_GROUND_JSON))
    vid_map = json.load(open(VID_MAP_JSON))

    if end == -1:
        end = len(df)
    end = min(end, len(df))   # clamp so an over-specified shard range is harmless
    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, model_name, 'step1.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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
        vid_id, qid = str(row['video_id']), str(row['qid'])
        vqid = f'{vid_id}_{qid}'
        if vqid in results:
            continue
        if vid_id not in gt_ground:
            continue
        if qid not in gt_ground[vid_id].get('location', {}):
            continue

        # Normalize question + options + answer identically for all models
        # (capitalize first letter, add '?' to the question). Options and the
        # answer use the SAME rule so choices.index(answer) still matches.
        question = normalize_question(row['question'])
        choices = [normalize_option(row[f'a{i}']) for i in range(5)]
        answer_norm = normalize_option(row['answer'])
        if answer_norm not in choices:
            continue
        answer_id = choices.index(answer_norm)
        type_ = str(row['type'])

        video_path = find_video_path(vid_id, VIDEO_DIR, vid_map)
        if not video_path or not os.path.exists(video_path):
            continue

        try:
            frames, timestamps, duration = adapter.decode_video(video_path)

            # Full QA — dispatch to video-file path if the adapter opts in.
            if adapter.supports_video_path_intervention():
                a_full, full_raw = adapter.predict_qa_video(
                    video_path, None, question, choices,
                )
            else:
                a_full, full_raw = adapter.predict_qa(frames, question, choices)

            # Question-conditioned grounding: the predicted answer is NOT used
            # for grounding (RQ1 — evidence is predicted from the question only).
            # pred_text is still passed for adapter-contract compatibility but
            # adapters must ignore it (see configs/prompts.build_grounding_prompt).
            # RELEASE_PLAN 6.5 — the grounding call is question-conditioned,
            # so the answer text is NOT handed to the adapter by default.
            # It is filled only for the answer-conditioned ablation.
            pred_text = None
            if os.environ.get('GVLLM_ANSCOND') == '1':
                pred_text = choices[a_full] if 0 <= a_full < 5 else answer_norm
            iv, grd_raw, grd_flag = adapter.predict_grounding(
                frames, question, pred_text, duration,
                video_path=video_path,
            )
            predicted_iv = list(iv) if iv else None

            oracle_iv = oracle_interval(gt_ground[vid_id], qid)
            iou, iop = _iou_iop(iv, oracle_iv)

            results[vqid] = {
                'vqid': vqid,
                'type': type_,
                'video_id': vid_id,
                'qid': qid,
                'question': question,
                'answer_id': answer_id,
                'duration': float(duration),
                'a_full': int(a_full),
                'full_raw': full_raw,
                'predicted_iv': predicted_iv,
                'predicted_flag': grd_flag,
                'grounding_raw': grd_raw,
                'oracle_iv': list(oracle_iv) if oracle_iv else None,
                'iou': iou, 'iop': iop,
                'n_frames': int(frames.shape[0]),
            }
            processed += 1
            elapsed = time.time() - t0
            print(f'[{idx:>4}] {vqid:<22} type={type_:<3} '
                  f'dur={duration:>5.1f}s a_full={a_full} ans={answer_id} '
                  f'pred={predicted_iv}  ({elapsed:.0f}s)', flush=True)
        except Exception as ex:
            traceback.print_exc()
            results[vqid] = {'vqid': vqid, 'error': str(ex)}
            continue

        if processed % save_every == 0:
            json.dump(results, open(out_path, 'w'))

    json.dump(results, open(out_path, 'w'))
    print(f'Saved {len(results)} samples -> {out_path}', flush=True)
    return results
