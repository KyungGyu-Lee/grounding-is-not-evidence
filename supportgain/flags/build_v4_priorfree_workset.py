"""Build v4 prior_free_subset + roracle_workset from v4 blind (rule identical to
the 788 build: prior_solvable = all 3 blind baselines correct; prior_free = rest;
roracle work = all keys)."""
import os, json

ROOT = os.environ.get('ROOT', 'results_v4')
bl = {k: json.load(open(f'{ROOT}/blind/qvh_{f}.json'))
      for k, f in [('q', 'qwen25vl'), ('i', 'internvl2_5_8b'), ('v', 'videollama2_7b')]}
keys = sorted(set(bl['q']) & set(bl['i']) & set(bl['v']))
solv = [k for k in keys if bl['q'][k]['correct'] and bl['i'][k]['correct'] and bl['v'][k]['correct']]
free = [k for k in keys if k not in set(solv)]
json.dump({'keys_all': keys, 'prior_solvable': solv, 'prior_free': free,
           'scope': f'QVH-v4-{len(keys)}', 'rule': 'all 3 blind baselines correct (text-only)'},
          open(f'{ROOT}/blind/qvh_prior_free_subset.json', 'w'))
os.makedirs(f'{ROOT}/shared', exist_ok=True)
json.dump({'work': keys, 'non_measurable': []},
          open(f'{ROOT}/shared/qvh_roracle_workset.json', 'w'))
print(f'v4 prior_free: keys={len(keys)} prior_solvable={len(solv)} prior_free={len(free)} | '
      f'workset work={len(keys)}', flush=True)
