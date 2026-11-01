"""Merge sharded step1 / step3 outputs into the un-sharded file.

Example:
    python scripts/merge_shards.py qwen25vl step1
    python scripts/merge_shards.py qwen25vl phase1/step3

The script looks for ``$RESULTS_DIR/<model>/<rel>_<shard>.json`` files
(one per GPU) and writes their union to ``$RESULTS_DIR/<model>/<rel>.json``.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supportgain.config.settings import RESULTS_DIR


def merge(model: str, rel: str) -> None:
    root, ext = os.path.splitext(rel)
    base = os.path.join(RESULTS_DIR, model, root)
    target = base + ext or os.path.join(RESULTS_DIR, model, rel + '.json')
    if not ext:
        ext = '.json'
        target = base + ext
    pattern = f'{base}_*{ext}'
    shards = sorted(glob.glob(pattern))
    if not shards:
        print(f'No shards found matching {pattern}', file=sys.stderr)
        return

    merged = {}
    for sp in shards:
        d = json.load(open(sp))
        overlap = set(merged) & set(d)
        if overlap:
            print(f'  WARN: {len(overlap)} overlapping keys in {sp}')
        merged.update(d)
        print(f'  + {sp}: {len(d)} samples')

    json.dump(merged, open(target, 'w'))
    print(f'Merged {len(shards)} shards -> {target}  (total {len(merged)})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('rel', help='e.g. "step1" or "phase1/step3"')
    args = p.parse_args()
    merge(args.model, args.rel)
