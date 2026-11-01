"""QVH-ME full-video GT-accuracy for a reference model (whole video).
Reuses shared QA prompt (matches blind). Sharded + resume.

  MODEL=videollama2 SHARD=0 NSHARD=1 CUDA_VISIBLE_DEVICES=0 <env-py> \
    supportgain/flags/qvh_fullvideo.py
"""
import sys, os, json, importlib
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _R)
import pandas as pd

MODEL = os.environ['MODEL']
SHARD = int(os.environ.get('SHARD', '0')); NSHARD = int(os.environ.get('NSHARD', '1'))
VIDDIR = os.environ.get('TEF_QVHME_VIDEO_DIR',
                        os.path.join(_R, 'data', 'qvhme', 'videos'))
ROOT = os.environ.get('TSFIX_OUT', 'results')
OUT = f'{ROOT}/blind/qvh_full_{MODEL}_shard{SHARD}.json'
os.makedirs(f'{ROOT}/blind', exist_ok=True)


def main():
    pf = set(json.load(open(os.environ.get('QVH_PRIORFREE','results/blind/qvh_prior_free_subset.json')))['prior_free'])
    df = pd.read_csv(os.environ.get('VAL_SAMPLED_CSV','data/qvh_gqa/val_qvh_gqa_screened.csv'))
    df['vqid'] = df['video_id'].astype(str) + '_' + df['qid'].astype(str)
    rows = [r for _, r in df.iterrows() if r['vqid'] in pf]
    rows = [r for i, r in enumerate(rows) if i % NSHARD == SHARD]
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}

    ad = importlib.import_module(f'supportgain.models.{MODEL}.adapter').Adapter()
    ad.load('cuda:0')
    has_keep = hasattr(ad, 'predict_keep')

    for r in rows:
        vqid = r['vqid']
        if vqid in done:
            continue
        vp = f"{VIDDIR}/{r['video_id']}.mp4"
        if not os.path.exists(vp):
            continue
        q = str(r['question']); ch = [str(r[f'a{k}']) for k in range(5)]
        gt = int(r['answer_id']); dur = float(r['duration'])
        if has_keep:                                  # videollama2 / internvl
            a, raw = ad.predict_keep(vp, 0.0, dur, q, ch)
        else:                                         # qwen: decode full + predict_qa
            frames, _, _ = ad.decode_video(vp)
            a, raw = ad.predict_qa(frames, q, ch)
        done[vqid] = {'pred': int(a), 'correct': int(a == gt and a >= 0)}
        if len(done) % 20 == 0:
            json.dump(done, open(OUT, 'w'))
    json.dump(done, open(OUT, 'w'))
    acc = sum(v['correct'] for v in done.values()) / len(done)
    print(f'DONE {MODEL} qvh-full shard{SHARD}: N={len(done)} acc={acc:.3f}', flush=True)


if __name__ == '__main__':
    main()