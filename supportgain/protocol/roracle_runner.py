"""R_oracle-keep runner: baseline model keeps a shared random GT-length segment,
judged on GT-correctness. Per-question x per-model x per-trial storage + resume.

  MODEL=videollama2 TAG=pilot N_LIMIT=100 SHARD=0 NSHARD=1 \
    CUDA_VISIBLE_DEVICES=0 <env-python> supportgain/protocol/roracle_runner.py

Segments come ONLY from results/shared/r_oracle_shared.json (never regenerated).
Blind is NOT touched. Decoding is greedy (adapters enforce do_sample=False).
"""
import sys, os, json, time, random, importlib
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _R)
import pandas as pd
from supportgain.config.settings import VID_MAP_JSON, VIDEO_DIR
from supportgain.core.video_io import find_video_path

MODEL = os.environ['MODEL']
TAG = os.environ.get('TAG', 'full')
N_LIMIT = int(os.environ.get('N_LIMIT', '0'))     # 0 = full 1507, else pilot
SHARD = int(os.environ.get('SHARD', '0'))
NSHARD = int(os.environ.get('NSHARD', '1'))
OUTDIR = f"{os.environ.get('TSFIX_OUT', 'results')}/roracle_keep"
OUT = f'{OUTDIR}/{MODEL}_{TAG}_shard{SHARD}.json'


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    work = json.load(open('results/shared/roracle_workset.json'))['work_1507']
    if N_LIMIT:
        rng = random.Random(1234); rng.shuffle(work); work = work[:N_LIMIT]
    work = [v for i, v in enumerate(work) if i % NSHARD == SHARD]

    sh = json.load(open('results/shared/r_oracle_shared.json'))
    s1 = json.load(open('results/qwen25vl/step1.json'))   # answer_id/duration (model-indep)
    vid_map = json.load(open(VID_MAP_JSON))
    df = pd.read_csv('data/val_full_strict.csv')
    df['vqid'] = df['video_id'].astype(str) + '_' + df['qid'].astype(str)
    dfm = {r['vqid']: r for _, r in df.iterrows()}

    done = json.load(open(OUT)) if os.path.exists(OUT) else {}

    mod = importlib.import_module(f'supportgain.models.{MODEL}.adapter')
    adapter = mod.Adapter()
    adapter.load('cuda:0')
    has_keep = hasattr(adapter, 'predict_keep')
    video_path_mode = (not has_keep and
                       getattr(adapter, 'supports_video_path_intervention', lambda: False)())
    if not has_keep and not video_path_mode:
        from supportgain.core.interventions import apply_intervention   # frame-subset models

    _cache = {'vp': None, 'frames': None, 'ts': None}

    def predict(vp, s, e, q, ch, dur):
        if has_keep:                                   # videollama2 / internvl
            return adapter.predict_keep(vp, s, e, q, ch)
        if video_path_mode:                            # video-path models
            spec = {'phase': 'phase2', 'kind': 'keep', 'interval': (s, e),
                    'total_duration': dur}
            return adapter.predict_qa_video(vp, spec, q, ch)
        # frame-subset models (Qwen): decode once per video, subset per trial
        if _cache['vp'] != vp:
            fr, tsv, _ = adapter.decode_video(vp)
            _cache.update(vp=vp, frames=fr, ts=tsv)
        modf = apply_intervention('phase2', _cache['frames'], _cache['ts'],
                                  (s, e), 'keep')
        if modf is None:
            return -1, ''
        return adapter.predict_qa(modf, q, ch)

    t0 = time.time(); n_inf = 0
    for vqid in work:
        if vqid in done:
            continue
        r = dfm.get(vqid); seg = sh.get(vqid); one = s1.get(vqid)
        if r is None or seg is None or one is None or 'error' in one:
            continue
        vp = find_video_path(str(r['video_id']), VIDEO_DIR, vid_map)
        if not vp or not os.path.exists(vp):
            continue
        q = str(r['question']); ch = [str(r[f'a{k}']) for k in range(5)]
        gt = int(one['answer_id']); dur = one['duration']
        trials = []
        for k in range(len(seg['r_oracle'])):
            iv = seg['r_oracle'][k]; fb = bool(seg['r_oracle_fallback'][k])
            if not iv:
                trials.append({'vqid': vqid, 'model': MODEL, 'trial_idx': k,
                               'segment': None, 'pred': -1, 'correct': 0,
                               'raw_output': '', 'fallback_flag': fb, 'skip': 1})
                continue
            a, raw = predict(vp, iv[0], iv[1], q, ch, dur)
            trials.append({'vqid': vqid, 'model': MODEL, 'trial_idx': k,
                           'segment': [round(iv[0], 2), round(iv[1], 2)],
                           'pred': int(a), 'correct': int(a == gt and a >= 0),
                           'raw_output': str(raw)[:150], 'fallback_flag': fb})
            n_inf += 1
        done[vqid] = trials
        if len(done) % 10 == 0:
            json.dump(done, open(OUT, 'w'))
            print(f'  [{MODEL} s{SHARD}] {len(done)}/{len(work)} '
                  f'({n_inf} inf, {time.time()-t0:.0f}s)', flush=True)
    json.dump(done, open(OUT, 'w'))

    all_tr = [t for v in done.values() for t in v if not t.get('skip')]
    n = len(all_tr)
    pf = sum(1 for t in all_tr if t['pred'] < 0)
    acc = sum(t['correct'] for t in all_tr) / n if n else 0
    dt = time.time() - t0
    print(f'DONE {MODEL} shard{SHARD}: samples={len(done)} trials={n} '
          f'parse_fail={pf}({100*pf/max(n,1):.1f}%) GTacc={acc:.3f} '
          f'elapsed={dt:.0f}s per_inf={dt/max(n_inf,1):.2f}s', flush=True)


if __name__ == '__main__':
    main()
