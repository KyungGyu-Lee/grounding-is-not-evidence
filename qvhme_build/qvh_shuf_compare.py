"""Position-bias verdict: compare remap-predicted a_full vs actual re-queried
a_full on the shuffled QVH-ME. Pre-fixed criterion: agreement >=0.97 -> remap;
<0.97 -> full re-run."""
import json, csv
import numpy as np
from collections import Counter

LET = 'ABCDE'
smap = {str(m['qid']): m for m in json.load(open('data/qvh_gqa_shuffled/shuffle_map.json'))}
rows = {f"{r['video_id']}_{r['qid']}": r for r in csv.DictReader(open('data/qvh_gqa_shuffled/val_qvh_gqa_screened_shuf.csv'))}

# original a_full sources (pre-shuffle predicted index)
def load_orig(model):
    if model == 'qwen25vl':
        d = json.load(open('results_tsfix/qvh/qwen25vl/step1.json'))
        return {k: v.get('a_full') for k, v in d.items()}
    if model == 'vtimellm':
        d = json.load(open('results/qvh/vtimellm/step1.json'))
        return {k: v.get('a_full') for k, v in d.items()}
    if model == 'internvl':
        import glob
        d = {}
        for f in glob.glob('results/blind/qvh_full_internvl_shard*.json'):
            d.update(json.load(open(f)))
        for f in glob.glob('results/blind/qvh_full_internvl_solv_shard*.json'):
            d.update(json.load(open(f)))
        return {k: v.get('pred') for k, v in d.items()}


for model in ['qwen25vl', 'internvl', 'vtimellm']:
    try:
        req = json.load(open(f'results/qvh_shuf/requery_{model}.json'))
    except FileNotFoundError:
        print(f'[{model}] requery missing'); continue
    orig = load_orig(model)
    agree = 0; n = 0; dis = []
    post_pos = Counter(); pre_correct = 0; post_correct = 0; nc = 0
    for vqid, rr in req.items():
        qid = vqid.rsplit('_', 1)[1]
        m = smap.get(qid); oa = orig.get(vqid)
        if m is None or oa is None or oa < 0:
            continue
        perm = m['perm_newslot_to_origidx']
        remap_pred = perm.index(oa)          # where the orig-picked content now sits
        actual = rr['pred']
        if actual < 0:
            continue
        n += 1
        agree += (remap_pred == actual)
        post_pos[actual] += 1
        if remap_pred != actual:
            dis.append((vqid, oa, remap_pred, actual))
        # accuracy pre vs post
        nc += 1
        pre_correct += int(oa == m['orig_answer_id'])
        post_correct += int(actual == m['new_answer_id'])
    ar = agree / n if n else 0
    print(f"\n===== [{model}] n={n} =====")
    print(f"  1. remap vs 실제 일치율 = {ar:.4f}  ->  {'REMAP (>=0.97)' if ar>=0.97 else 'RE-RUN (<0.97)'}")
    print(f"  3. 재질의 답 위치분포: " + " ".join(f"{LET[k]} {post_pos.get(k,0)}({100*post_pos.get(k,0)/n:.0f}%)" for k in range(5)))
    print(f"  4. 정답률 pre(remap셋)={pre_correct/nc:.3f} -> post(재질의)={post_correct/nc:.3f}")
    print(f"  2. 불일치 {len(dis)}건, 예시(원idx/remap위치/실제): {[(d[1],d[2],d[3]) for d in dis[:8]]}")
