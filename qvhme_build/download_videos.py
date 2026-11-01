"""Download the QVHighlights clips referenced by prompts.jsonl (790 vids).

The `vid` field encodes {youtube_id}_{clip_start}_{clip_end}, e.g.
`NUsG9BgSes0_60.0_210.0`. The GT spans in the annotations are RELATIVE to the
150s clip (not the raw YouTube video).

Two sources:
  --source tarball  (DEFAULT, recommended): the official pre-cropped clip
      package `qvhilights_videos.tar.gz` — no YouTube needed, clips already at
      {vid}.mp4. Downloaded once, then only the 790 needed members are extracted.
  --source youtube  (fallback): yt-dlp the source YouTube video, then ffmpeg-crop
      [clip_start, clip_end] into {vid}.mp4. More fragile (removed/geo-blocked).

Saves to data/qvh_gqa/videos/{vid}.mp4. Skips already-present files (resumable).
Failures are appended to data/qvh_gqa/failed_downloads.log.

NOTE: script only — run manually:
    python data/qvh_gqa/download_videos.py                 # tarball (default)
    python data/qvh_gqa/download_videos.py --source youtube # fallback
"""
import os
import sys
import json
import argparse
import subprocess
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, 'prompts.jsonl')
VIDEO_DIR = os.path.join(HERE, 'videos')
FAILED_LOG = os.path.join(HERE, 'failed_downloads.log')
TARBALL_URL = 'https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz'
TARBALL_PATH = os.path.join(HERE, 'qvhilights_videos.tar.gz')


def needed_vids():
    vids = []
    seen = set()
    for line in open(PROMPTS):
        v = json.loads(line)['video_id']
        if v not in seen:
            seen.add(v)
            vids.append(v)
    return vids


def log_fail(vid, reason):
    with open(FAILED_LOG, 'a') as f:
        f.write(f'{vid}\t{reason}\n')


def already(vid):
    p = os.path.join(VIDEO_DIR, vid + '.mp4')
    return os.path.exists(p) and os.path.getsize(p) > 0


# ----------------------------------------------------------------------
def from_tarball(vids):
    if not (os.path.exists(TARBALL_PATH) and os.path.getsize(TARBALL_PATH) > 0):
        print(f'downloading tarball -> {TARBALL_PATH} (several GB) ...', flush=True)
        urllib.request.urlretrieve(TARBALL_URL, TARBALL_PATH)
    want = {v for v in vids if not already(v)}
    print(f'extracting {len(want)} needed clips from tarball ...', flush=True)
    found = 0
    with tarfile.open(TARBALL_PATH, 'r:gz') as tar:
        for m in tar:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            vid = base[:-4] if base.endswith('.mp4') else None
            if vid in want:
                src = tar.extractfile(m)
                with open(os.path.join(VIDEO_DIR, vid + '.mp4'), 'wb') as out:
                    out.write(src.read())
                found += 1
                if found % 50 == 0:
                    print(f'  extracted {found}/{len(want)}', flush=True)
    missing = [v for v in vids if not already(v)]
    for v in missing:
        log_fail(v, 'not_in_tarball')
    print(f'tarball done. present={len(vids) - len(missing)} missing={len(missing)}')


# ----------------------------------------------------------------------
def from_youtube(vids):
    for i, vid in enumerate(vids, 1):
        if already(vid):
            continue
        try:
            yt_id, s, e = vid.rsplit('_', 2)
            s, e = float(s), float(e)
        except ValueError:
            log_fail(vid, 'bad_vid_format')
            continue
        tmp = os.path.join(VIDEO_DIR, f'.tmp_{yt_id}.mp4')
        dst = os.path.join(VIDEO_DIR, vid + '.mp4')
        try:
            if not (os.path.exists(tmp) and os.path.getsize(tmp) > 0):
                subprocess.run(
                    ['yt-dlp', '-f', 'mp4', '-o', tmp,
                     f'https://www.youtube.com/watch?v={yt_id}'],
                    check=True, capture_output=True, timeout=600)
            # crop [s, e] -> {vid}.mp4 (even dims for libx264)
            subprocess.run(
                ['ffmpeg', '-y', '-ss', str(s), '-to', str(e), '-i', tmp,
                 '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                 '-c:v', 'libx264', '-c:a', 'aac', dst],
                check=True, capture_output=True, timeout=600)
        except Exception as ex:
            log_fail(vid, f'{type(ex).__name__}')
        if i % 20 == 0:
            print(f'  {i}/{len(vids)}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['tarball', 'youtube'], default='tarball')
    args = ap.parse_args()
    os.makedirs(VIDEO_DIR, exist_ok=True)
    vids = needed_vids()
    print(f'target unique vids: {len(vids)}   source={args.source}', flush=True)
    if args.source == 'tarball':
        from_tarball(vids)
    else:
        from_youtube(vids)
    present = sum(1 for v in vids if already(v))
    print(f'DONE. present={present}/{len(vids)}  (failures -> {FAILED_LOG})')


if __name__ == '__main__':
    main()