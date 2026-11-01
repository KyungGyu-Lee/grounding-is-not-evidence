"""A3 probe — does every kept question have exactly 5 draws, and does the
positive control (Ret(rand_gt)) retain per-draw responses? Read-only."""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{ROOT}/paper')
import build_paper_numbers as bpn  # noqa: E402  (chdir()s to $BP_CWD)

bench = sys.argv[1]
G = bpn.GROUNDERS[bench]
for disp, dr in G:
    s3 = json.load(open(bpn._s3(bench, dr)))
    gt = bpn._gt(bench, dr)
    if disp == 'Qwen':
        krg_src = (json.load(open('results/roracle_keep/qwen25vl_full.json'))
                   if bench == 'NExT'
                   else bpn._ls('results/roracle_keep/qvh_qwen25vl_shard*.json'))
    else:
        krg_src = None
    nd, ngt, kept, gt_rows = Counter(), Counter(), 0, 0
    for v, rec in s3.items():
        if 'error' in rec:
            continue
        p = rec.get('preds', {})
        af = rec.get('a_full')
        if af is None or af < 0 or 'predicted_keep' not in p or 'blind' not in p:
            continue
        rk = [p[f'r_pred_keep_{k}'] for k in range(5) if f'r_pred_keep_{k}' in p]
        if not rk:
            continue
        kept += 1
        nd[len(rk)] += 1
        if v in gt and 'kgt' in gt[v]:
            if krg_src is not None and v in krg_src:
                ps = [t['pred'] for t in krg_src[v]]
            elif 'krgt' in gt[v]:
                ps = [t['pred'] for t in gt[v]['krgt'] if not t.get('skip')]
            else:
                ps = None
            if ps:
                gt_rows += 1
                ngt[len(ps)] += 1
    print(f'{bench:5} {disp:13} kept={kept:5} draws/q={dict(sorted(nd.items()))} '
          f'| gt rows={gt_rows:5} gt-draws/q={dict(sorted(ngt.items()))}')
