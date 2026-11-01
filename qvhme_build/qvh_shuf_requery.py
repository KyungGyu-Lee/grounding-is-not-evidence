"""Position-bias probe: re-query a_full ONLY (no intervention, 1 call/question)
on the SHUFFLED QVH-ME 788, for one model. Full-video QA on the shuffled option
order. Writes results/qvh_shuf/requery_{MODEL}.json {vqid:{pred,...}}.

  MODEL=qwen25vl|internvl|vtimellm  CUDA_VISIBLE_DEVICES=k  <env-py> qvhme_build/qvh_shuf_requery.py
"""
import os, sys, json, csv, importlib
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
from supportgain.config.prompts import normalize_question, normalize_option

MODEL = os.environ['MODEL']
CSV = 'data/qvh_gqa_shuffled/val_qvh_gqa_screened_shuf.csv'
VID = os.environ.get('TEF_QVHME_VIDEO_DIR',
                     os.path.join(_R, 'data', 'qvhme', 'videos'))
OUT = f'results/qvh_shuf/requery_{MODEL}.json'
os.makedirs('results/qvh_shuf', exist_ok=True)


def main():
    ad = importlib.import_module(f'supportgain.models.{MODEL}.adapter').Adapter()
    ad.load('cuda:0')
    rows = list(csv.DictReader(open(CSV)))
    res = {}
    for i, r in enumerate(rows):
        vqid = f"{r['video_id']}_{r['qid']}"
        vp = f"{VID}/{r['video_id']}.mp4"
        if not os.path.exists(vp):
            continue
        q = normalize_question(str(r['question']))
        ch = [normalize_option(str(r[f'a{k}'])) for k in range(5)]
        dur = float(r['duration'])
        try:
            if MODEL == 'qwen25vl':
                fr, _, _ = ad.decode_video(vp)
                pred, raw = ad.predict_qa(fr, q, ch)
            elif MODEL == 'internvl':
                pred, raw = ad.predict_keep(vp, 0.0, dur, q, ch)   # full video
            else:  # vtimellm (video-path)
                pred, raw = ad.predict_qa_video(vp, None, q, ch)
        except Exception as ex:
            pred, raw = -1, f'ERR:{str(ex)[:80]}'
        res[vqid] = {'pred': int(pred), 'new_answer_id': int(r['answer_id']),
                     'correct': int(pred == int(r['answer_id']) and pred >= 0),
                     'raw': str(raw)[:40]}
        if (i + 1) % 100 == 0:
            json.dump(res, open(OUT, 'w'))
            print(f'  {i+1}/{len(rows)}', flush=True)
    json.dump(res, open(OUT, 'w'))
    acc = sum(v['correct'] for v in res.values()) / max(len(res), 1)
    print(f'DONE_{MODEL} n={len(res)} acc(shuffled)={acc:.4f} -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
