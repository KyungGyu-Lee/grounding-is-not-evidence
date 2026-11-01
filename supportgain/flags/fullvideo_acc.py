"""Full-video GT-accuracy for a baseline (1 inference/question, whole video).
Reference point for the blind < random < full sanity check. Sharded + resume."""
import sys, os, json, importlib
_R = os.path.dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.insert(0, _R)
import pandas as pd
from supportgain.config.settings import VID_MAP_JSON, VIDEO_DIR
from supportgain.core.video_io import find_video_path

MODEL = os.environ['MODEL']
SHARD = int(os.environ.get('SHARD', '0')); NSHARD = int(os.environ.get('NSHARD', '1'))
OUT = f'results/roracle_keep/{MODEL}_fullvideo_shard{SHARD}.json'


def main():
    work = json.load(open('results/shared/roracle_workset.json'))['work_1507']
    work = [v for i, v in enumerate(work) if i % NSHARD == SHARD]
    s1 = json.load(open('results/qwen25vl/step1.json'))
    vid_map = json.load(open(VID_MAP_JSON))
    df = pd.read_csv('data/val_full_strict.csv')
    df['vqid'] = df['video_id'].astype(str) + '_' + df['qid'].astype(str)
    dfm = {r['vqid']: r for _, r in df.iterrows()}
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}

    ad = importlib.import_module(f'supportgain.models.{MODEL}.adapter').Adapter()
    ad.load('cuda:0')
    for vqid in work:
        if vqid in done:
            continue
        r = dfm.get(vqid); one = s1.get(vqid)
        if r is None or one is None or 'error' in one:
            continue
        vp = find_video_path(str(r['video_id']), VIDEO_DIR, vid_map)
        if not vp or not os.path.exists(vp):
            continue
        q = str(r['question']); ch = [str(r[f'a{k}']) for k in range(5)]
        gt = int(one['answer_id']); dur = float(one['duration'])
        a, raw = ad.predict_keep(vp, 0.0, dur, q, ch)
        done[vqid] = {'pred': int(a), 'correct': int(a == gt and a >= 0)}
        if len(done) % 20 == 0:
            json.dump(done, open(OUT, 'w'))
    json.dump(done, open(OUT, 'w'))
    acc = sum(v['correct'] for v in done.values()) / len(done)
    print(f'DONE {MODEL} fullvideo shard{SHARD}: N={len(done)} acc={acc:.3f}', flush=True)


if __name__ == '__main__':
    main()
