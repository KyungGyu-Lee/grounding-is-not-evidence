"""QVH R_oracle-keep: baseline keeps shared random GT-length segment, GT-judged.
Per-trial storage + resume. Video = data/qvh_gqa/videos/{video_id}.mp4.

  MODEL=videollama2 SHARD=0 NSHARD=4 CUDA_VISIBLE_DEVICES=0 <env-py> \
    analysis/qvh_roracle_runner.py
"""
import sys, os, json, importlib
_R = os.path.dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.insert(0, _R)
import pandas as pd

MODEL = os.environ['MODEL']
SHARD = int(os.environ.get('SHARD', '0')); NSHARD = int(os.environ.get('NSHARD', '1'))
VIDDIR = os.environ.get('TEF_QVHME_VIDEO_DIR',
                        os.path.join(_R, 'data', 'qvhme', 'videos'))
ROOT = os.environ.get('TSFIX_OUT', 'results')
OUT = f'{ROOT}/roracle_keep/qvh_{MODEL}_shard{SHARD}.json'


def main():
    os.makedirs(f'{ROOT}/roracle_keep', exist_ok=True)
    work = json.load(open(os.environ.get('QVH_WORKSET','results/shared/qvh_roracle_workset.json')))['work']
    work = [v for i, v in enumerate(work) if i % NSHARD == SHARD]
    sh = json.load(open('results/shared/qvh_r_oracle_shared.json'))
    df = pd.read_csv(os.environ.get('VAL_SAMPLED_CSV','data/qvh_gqa/val_qvh_gqa_screened.csv'))
    df['vqid'] = df['video_id'].astype(str) + '_' + df['qid'].astype(str)
    dfm = {r['vqid']: r for _, r in df.iterrows()}
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}

    ad = importlib.import_module(f'supportgain.models.{MODEL}.adapter').Adapter()
    ad.load('cuda:0')
    has_keep = hasattr(ad, 'predict_keep')
    if not has_keep:
        from supportgain.core.interventions import apply_intervention
    cache = {'vp': None}

    def predict(vp, s, e, q, ch):
        if has_keep:
            return ad.predict_keep(vp, s, e, q, ch)
        if cache['vp'] != vp:
            fr, ts, _ = ad.decode_video(vp); cache.update(vp=vp, fr=fr, ts=ts)
        modf = apply_intervention('phase2', cache['fr'], cache['ts'], (s, e), 'keep')
        return (-1, '') if modf is None else ad.predict_qa(modf, q, ch)

    for vqid in work:
        if vqid in done:
            continue
        r = dfm.get(vqid); seg = sh.get(vqid)
        if r is None or seg is None:
            continue
        vp = f"{VIDDIR}/{r['video_id']}.mp4"
        if not os.path.exists(vp):
            continue
        q = str(r['question']); ch = [str(r[f'a{k}']) for k in range(5)]
        gt = int(r['answer_id'])
        trials = []
        for k, iv in enumerate(seg['r_oracle']):
            fb = bool(seg['r_oracle_fallback'][k])
            if not iv:
                trials.append({'trial_idx': k, 'pred': -1, 'correct': 0, 'skip': 1, 'fallback_flag': fb})
                continue
            a, raw = predict(vp, iv[0], iv[1], q, ch)
            trials.append({'trial_idx': k, 'segment': [round(iv[0], 1), round(iv[1], 1)],
                           'pred': int(a), 'correct': int(a == gt and a >= 0),
                           'raw_output': str(raw)[:120], 'fallback_flag': fb})
        done[vqid] = trials
        if len(done) % 10 == 0:
            json.dump(done, open(OUT, 'w'))
    json.dump(done, open(OUT, 'w'))
    tr = [t for v in done.values() for t in v if not t.get('skip')]
    acc = sum(t['correct'] for t in tr) / len(tr) if tr else 0
    print(f'DONE {MODEL} qvh-roracle shard{SHARD}: samples={len(done)} '
          f'acc={acc:.3f} parsefail={sum(1 for t in tr if t["pred"]<0)}', flush=True)


if __name__ == '__main__':
    main()