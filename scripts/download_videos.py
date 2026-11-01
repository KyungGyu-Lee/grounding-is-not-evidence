"""Download the unique NExT-QA videos needed by a sampled CSV.

Pulls them from HuggingFace ``VLM2Vec/nextqa-rawvideo`` and places them
under ``$TEF_NEXTQA_VIDEO_DIR/<sub>/<vid>.mp4`` so the path
layout matches ``map_vid_vidorID.json``. Already-present videos are skipped
(``cached``), so running it for the 500 set only fetches the missing ones.

Usage:
    python scripts/download_videos.py                         # 250 set (default)
    python scripts/download_videos.py --csv data/val_sampled_500.csv
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEF_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, TEF_ROOT)

REPO = 'VLM2Vec/nextqa-rawvideo'
DEST_ROOT = os.environ.get('TEF_NEXTQA_VIDEO_DIR',
                           os.path.join(TEF_ROOT, 'data', 'nextqa_videos'))
DEFAULT_CSV = os.path.join(TEF_ROOT, 'data', 'val_sampled.csv')
VID_MAP_JSON = os.path.join(TEF_ROOT, 'data', '_nextgqa',
                             'map_vid_vidorID.json')

os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')


def _need_video_ids(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return sorted(df['video_id'].astype(str).unique())


def _download_one(vid, vid_map):
    from huggingface_hub import hf_hub_download
    rel = vid_map.get(vid)
    if rel is None:
        return vid, 'no_vidmap'
    dst_dir = os.path.join(DEST_ROOT, os.path.dirname(rel))
    dst_path = os.path.join(DEST_ROOT, rel + '.mp4')
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return vid, 'cached'
    os.makedirs(dst_dir, exist_ok=True)
    try:
        src = hf_hub_download(
            repo_id=REPO, repo_type='dataset', filename=f'{vid}.mp4',
        )
        shutil.copyfile(src, dst_path)
        return vid, 'ok'
    except Exception as ex:
        return vid, f'err:{type(ex).__name__}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV,
                    help='Sampled CSV whose video_id column drives the download.')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    vid_map = json.load(open(VID_MAP_JSON))
    need = _need_video_ids(args.csv)
    print(f'CSV: {args.csv}')
    print(f'Need {len(need)} videos -> {DEST_ROOT}')
    os.makedirs(DEST_ROOT, exist_ok=True)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_download_one, v, vid_map): v for v in need}
        for i, fut in enumerate(as_completed(futs), 1):
            vid, status = fut.result()
            if status.startswith('err') or status == 'no_vidmap':
                fail += 1
                print(f'  [{i}/{len(need)}] {vid} FAIL {status}', flush=True)
            else:
                ok += 1
                if i % 20 == 0 or i == len(need):
                    print(f'  [{i}/{len(need)}] last={vid} -> {status}', flush=True)

    print(f'\nDone. ok={ok}  fail={fail}')


if __name__ == '__main__':
    main()
