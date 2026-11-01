"""Video decoding helpers used by model adapters.

Each model adapter is free to override or ignore these. We provide the
two common patterns used in our project:
  - ``decode_uniform(path, N)``  ← fixed N frames (e.g. HawkEye uses 12)
  - ``decode_at_fps(path, fps)`` ← variable count at a target fps
                                  (e.g. Qwen2.5-VL uses fps=2.0)
"""
import glob
import os
import numpy as np


def find_video_path(vid, video_dir, vid_map):
    """Resolve a NExT-QA video id to its mp4 path."""
    if vid in vid_map:
        return os.path.join(video_dir, vid_map[vid] + '.mp4')
    hits = glob.glob(os.path.join(video_dir, '**', f'{vid}.mp4'), recursive=True)
    return hits[0] if hits else None


def decode_uniform(video_path, n_frames):
    """Decode `n_frames` uniformly spaced frames.

    Returns:
        frames: np.ndarray [n_frames, H, W, 3] uint8
        timestamps: np.ndarray [n_frames] seconds
        duration: float
    """
    import decord
    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(video_path)
    orig_fps = float(vr.get_avg_fps())
    total = len(vr)
    duration = total / orig_fps
    indices = np.linspace(0, total - 1, n_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = indices.astype(np.float64) / orig_fps
    return frames, timestamps, duration


def decode_at_fps(video_path, fps=2.0, max_frames=128, min_frames=4):
    """Decode at `fps` frames per second, clipped to [min_frames, max_frames].

    Returns:
        frames: np.ndarray [N, H, W, 3] uint8
        timestamps: np.ndarray [N] seconds
        duration: float
    """
    import decord
    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(video_path)
    orig_fps = float(vr.get_avg_fps())
    total = len(vr)
    duration = total / orig_fps
    target_n = max(min_frames, min(int(round(fps * duration)), max_frames))
    indices = np.linspace(0, total - 1, target_n).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = indices.astype(np.float64) / orig_fps
    return frames, timestamps, duration
