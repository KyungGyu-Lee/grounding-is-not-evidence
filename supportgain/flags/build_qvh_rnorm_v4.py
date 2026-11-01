"""Build QVH r_norm for a v4 tree: r_norm[vqid] = (r - b)/(u - b) on the
measurable set C1 (mean3(u-b) > 0.2). Mirrors build_paper_numbers set_masks.

  ROOT=results_v4 python analysis/build_qvh_rnorm_v4.py
Reads   {ROOT}/blind/qvh_{qwen25vl,internvl2_5_8b,videollama2_7b}.json  (b)
        {ROOT}/blind/qvh_full_{qwen25vl,internvl,videollama2}_shard*    (u)
        {ROOT}/roracle_keep/qvh_{qwen25vl,internvl,videollama2}_shard*  (r)
Writes  {ROOT}/shared/qvh_r_norm.json  {'r_norm': {vqid: val}}
"""
import os, json, glob
import numpy as np

ROOT = os.environ.get('ROOT', 'results_v4')


def _ls(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        d.update(json.load(open(f)))
    return d


def main():
    bl = {k: json.load(open(f'{ROOT}/blind/qvh_{f}.json'))
          for k, f in [('q', 'qwen25vl'), ('i', 'internvl2_5_8b'), ('v', 'videollama2_7b')]}
    full = {m: _ls(f'{ROOT}/blind/qvh_full_{m}_shard[0-9]*.json')
            for m in ['qwen25vl', 'internvl', 'videollama2']}
    rora = {m: _ls(f'{ROOT}/roracle_keep/qvh_{m}_shard[0-9]*.json')
            for m in ['qwen25vl', 'internvl', 'videollama2']}
    ALL = set(bl['q']) & set(bl['i']) & set(bl['v'])
    basis = ALL & set(full['qwen25vl']) & set(full['internvl']) & set(full['videollama2']) \
        & set(rora['qwen25vl']) & set(rora['internvl']) & set(rora['videollama2'])

    def bmean(k):
        return (bl['q'][k]['correct'] + bl['i'][k]['correct'] + bl['v'][k]['correct']) / 3

    def umean(k):
        return (full['qwen25vl'][k]['correct'] + full['internvl'][k]['correct'] + full['videollama2'][k]['correct']) / 3

    def rmean(k):
        vals = []
        for m in ['qwen25vl', 'internvl', 'videollama2']:
            trials = [t['correct'] for t in rora[m][k] if not t.get('skip')]
            if trials:
                vals.append(np.mean(trials))
        return float(np.mean(vals)) if vals else np.nan

    rn = {}
    for k in basis:
        b, u = bmean(k), umean(k)
        den = u - b
        if den >= 1/3 - 1e-9:               # C1 measurable: u-b >= 1/3 (== old >0.2; u-b is a multiple of 1/3)
            r = rmean(k)
            if not np.isnan(r):
                rn[k] = (r - b) / den
    os.makedirs(f'{ROOT}/shared', exist_ok=True)
    json.dump({'r_norm': rn}, open(f'{ROOT}/shared/qvh_r_norm.json', 'w'))
    arr = np.array(list(rn.values()))
    print(f'r_norm built: basis={len(basis)} C1={len(rn)} median={np.median(arr):.3f} '
          f'-> {ROOT}/shared/qvh_r_norm.json', flush=True)


if __name__ == '__main__':
    main()
