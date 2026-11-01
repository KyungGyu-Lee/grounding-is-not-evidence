"""Build NExT rho: rho[vqid] = (r - b)/(u - b) on the answerability set C2
(mean3(u - b) >= 1/3). Mirrors build_qvh_rnorm_v4.py (identical formula) with
the NExT input layout used by build_paper_numbers.set_masks('NExT').

PROVENANCE: $ROOT must be the timestamp-scale-corrected run that produced the
shipped `flags/nextgqa/r_norm.json`. An uncorrected run gives different Qwen2.5-VL
and GroundVTS components, which moves C2 from 1116 to 1115 and changes 17 values.
Reproducing the paper does NOT call this script; it reads the stored rho.

  ROOT=<original run tree> python supportgain/flags/build_next_rnorm.py
Reads   {ROOT}/blind/{qwen25vl,internvl2_5_8b,videollama2_7b}.json          (b)
        {ROOT}/qwen25vl/step1.json  (a_full==answer_id)                     (u, qwen)
        {ROOT}/roracle_keep/{internvl,videollama2}_fullvideo_shard*.json    (u, i/v)
        {ROOT}/roracle_keep/{qwen25vl,internvl,videollama2}_full_shard*.json (r, 5 draws)
Writes  {ROOT}/shared/next_r_norm.json  {'r_norm': {vqid: val}}

b/u/r are each averaged over the 3 baselines (Qwen2.5-VL, InternVL2.5-8B,
VideoLLaMA2-7B). r = per-model mean over its non-skip random-keep trials, then
mean across models (ratio_of_means: numerator/denominator each model-averaged,
then one division). No clipping.
"""
import os, json, glob
import numpy as np

ROOT = os.environ.get('ROOT', 'results_v3')   # NExT 정본 트리 (ts-fix). NOT 'results'.


def _ls(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        d.update(json.load(open(f)))
    return d


def main():
    bl = {k: json.load(open(f'{ROOT}/blind/{f}.json'))
          for k, f in [('q', 'qwen25vl'), ('i', 'internvl2_5_8b'), ('v', 'videollama2_7b')]}
    s1q = json.load(open(f'{ROOT}/qwen25vl/step1.json'))
    full_q = {k: int(v['a_full'] == v['answer_id']) for k, v in s1q.items()}
    full_i = _ls(f'{ROOT}/roracle_keep/internvl_fullvideo_shard[0-9]*.json')
    full_v = _ls(f'{ROOT}/roracle_keep/videollama2_fullvideo_shard[0-9]*.json')
    rora = {m: _ls(f'{ROOT}/roracle_keep/{m}_full_shard[0-9]*.json')
            for m in ['qwen25vl', 'internvl', 'videollama2']}

    ALL = set(bl['q']) & set(bl['i']) & set(bl['v'])
    basis = ALL & set(full_q) & set(full_i) & set(full_v) \
        & set(rora['qwen25vl']) & set(rora['internvl']) & set(rora['videollama2'])

    def bmean(k):
        return (bl['q'][k]['correct'] + bl['i'][k]['correct'] + bl['v'][k]['correct']) / 3

    def umean(k):
        return (full_q[k] + full_i[k]['correct'] + full_v[k]['correct']) / 3

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
        if den >= 1/3 - 1e-9:               # C2 answerability: u-b >= 1/3 (u-b is a multiple of 1/3)
            r = rmean(k)
            if not np.isnan(r):
                rn[k] = (r - b) / den
    os.makedirs(f'{ROOT}/shared', exist_ok=True)
    json.dump({'r_norm': rn}, open(f'{ROOT}/shared/next_r_norm.json', 'w'))
    arr = np.array(list(rn.values()))
    print(f'r_norm built: basis={len(basis)} C1={len(rn)} median={np.median(arr):.3f} '
          f'-> {ROOT}/shared/next_r_norm.json', flush=True)


if __name__ == '__main__':
    main()
