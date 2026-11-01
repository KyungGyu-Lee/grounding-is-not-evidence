"""STEP 2: fill baseline full/random on prior-SOLVABLE ids only.
Same settings as prior-free runs (greedy adapters, shared QA prompt, N=5 R_oracle
GT-length forbidden=GT). Saves to *_solv_shard files (pf files untouched).

  BENCH=next|qvh  MODEL=qwen25vl|internvl|videollama2  TASK=full|random
  SHARD NSHARD CUDA_VISIBLE_DEVICES  <env-py> analysis/step2_runner.py
  LIMIT=20  -> consistency probe (prints frames/greedy/forbidden, no full save)
"""
import sys, os, json, importlib
_R = os.path.dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.insert(0, _R)
import pandas as pd

BENCH = os.environ['BENCH']; MODEL = os.environ['MODEL']; TASK = os.environ['TASK']
SHARD = int(os.environ.get('SHARD', '0')); NSHARD = int(os.environ.get('NSHARD', '1'))
LIMIT = int(os.environ.get('LIMIT', '0'))
ROOT = os.environ.get('TSFIX_OUT', 'results')
if BENCH == 'qvh':
    OUT = f'{ROOT}/blind/qvh_full_{MODEL}_solv_shard{SHARD}.json' if TASK == 'full' \
          else f'{ROOT}/roracle_keep/qvh_{MODEL}_solv_shard{SHARD}.json'
else:
    OUT = f'{ROOT}/roracle_keep/{MODEL}_fullvideo_solv_shard{SHARD}.json' if TASK == 'full' \
          else f'{ROOT}/roracle_keep/{MODEL}_solv_shard{SHARD}.json'
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def qvh_vp(vid):
    _vd = os.environ.get('TEF_QVHME_VIDEO_DIR',
                         os.path.join(_R, 'data', 'qvhme', 'videos'))
    return os.path.join(_vd, f'{vid}.mp4')


def main():
    if BENCH == 'qvh':
        solv = json.load(open('results/blind/qvh_prior_solvable_ids.json'))
        df = pd.read_csv(os.environ.get('VAL_SAMPLED_CSV','data/qvh_gqa/val_qvh_gqa_screened.csv'))
        df['vqid'] = df['video_id'].astype(str) + '_' + df['qid'].astype(str)
        dfm = {r['vqid']: r for _, r in df.iterrows()}
        sh = json.load(open('results/shared/qvh_r_oracle_shared.json')) if TASK == 'random' else {}
        def vp(vqid): return qvh_vp(dfm[vqid]['video_id'])
        def meta(vqid):
            r = dfm[vqid]; return r['question'], [str(r[f'a{k}']) for k in range(5)], int(r['answer_id']), float(r['duration'])
    else:
        from supportgain.config.settings import VID_MAP_JSON, VIDEO_DIR
        from supportgain.core.video_io import find_video_path
        solv = json.load(open('results/blind/next_prior_solvable_ids.json'))
        s1 = json.load(open('results/qwen25vl/step1.json'))
        vid_map = json.load(open(VID_MAP_JSON))
        df = pd.read_csv('data/val_full_strict.csv'); df['vqid'] = df['video_id'].astype(str) + '_' + df['qid'].astype(str)
        dfm = {r['vqid']: r for _, r in df.iterrows()}
        sh = json.load(open('results/shared/next_solv_r_oracle_shared.json')) if TASK == 'random' else {}
        def vp(vqid): return find_video_path(str(dfm[vqid]['video_id']), VIDEO_DIR, vid_map)
        def meta(vqid):
            r = dfm[vqid]; one = s1.get(vqid, {})
            return r['question'], [str(r[f'a{k}']) for k in range(5)], int(one['answer_id']), float(one['duration'])

    ids = [v for i, v in enumerate(sorted(solv)) if i % NSHARD == SHARD]
    if LIMIT:
        ids = ids[:LIMIT]
    done = json.load(open(OUT)) if os.path.exists(OUT) and not LIMIT else {}

    ad = importlib.import_module(f'supportgain.models.{MODEL}.adapter').Adapter()
    ad.load('cuda:0')
    has_keep = hasattr(ad, 'predict_keep')
    if not has_keep and TASK == 'random':
        from supportgain.core.interventions import apply_intervention
    cache = {'vp': None}

    def keep(v, s, e, q, ch):
        if has_keep:
            return ad.predict_keep(v, s, e, q, ch)
        if cache['vp'] != v:
            fr, ts, _ = ad.decode_video(v); cache.update(vp=v, fr=fr, ts=ts)
        mf = apply_intervention('phase2', cache['fr'], cache['ts'], (s, e), 'keep')
        return (-1, '') if mf is None else ad.predict_qa(mf, q, ch)

    def full(v, q, ch, dur):
        if has_keep:
            return ad.predict_keep(v, 0.0, dur, q, ch)
        fr, _, _ = ad.decode_video(v); return ad.predict_qa(fr, q, ch)

    n = 0
    for vqid in ids:
        if vqid in done:
            continue
        p = vp(vqid)
        if not p or not os.path.exists(p):
            continue
        q, ch, gt, dur = meta(vqid)
        if TASK == 'full':
            a, raw = full(p, q, ch, dur)
            done[vqid] = {'pred': int(a), 'correct': int(a == gt and a >= 0)}
        else:
            seg = sh.get(vqid)
            if not seg:
                continue
            trials = []
            for k, iv in enumerate(seg['r_oracle']):
                fb = bool(seg['r_oracle_fallback'][k])
                if not iv:
                    trials.append({'trial_idx': k, 'pred': -1, 'correct': 0, 'skip': 1, 'fallback_flag': fb}); continue
                a, raw = keep(p, iv[0], iv[1], q, ch)
                trials.append({'trial_idx': k, 'pred': int(a), 'correct': int(a == gt and a >= 0), 'fallback_flag': fb})
            done[vqid] = trials
        n += 1
        if LIMIT and n <= 3:
            print(f'  PROBE {vqid}: task={TASK} pred_sample ok', flush=True)
        if not LIMIT and len(done) % 20 == 0:
            json.dump(done, open(OUT, 'w'))
    if not LIMIT:
        json.dump(done, open(OUT, 'w'))
    print(f'DONE {BENCH} {MODEL} {TASK} shard{SHARD}: n_new={n} total={len(done)}', flush=True)


if __name__ == '__main__':
    main()
