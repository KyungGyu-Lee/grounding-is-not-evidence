"""GT-ceiling positive control: K_GT (keep GT interval) + K_R_gt (keep 5 shared
GT-length random segments) on the answerability subset C2.

Mirrors supportgain/protocol/roracle_runner.py EXACTLY (raw CSV
question/choices, greedy adapters, same predict() dispatch) so the new
K_GT/K_R_gt sit on the identical measurement basis as the already-stored
Qwen / reference-model K_R_gt that we reuse.

Segments come ONLY from the shared clean (forbidden=GT-only) files, which cover
a superset of C2 (NExT-GQA 1507 >= 1116; QVH-ME 788 >= 427).
oracle_iv (the GT interval the segments were forbidden against) is taken from the
same shared record so K_GT and the segments share one GT reference.

  BENCH=next|qvh  MODEL=<module=dir>  TASK=both|kgt
  SHARD NSHARD CUDA_VISIBLE_DEVICES  <env-python> \
    supportgain/protocol/gt_ceiling_runner.py
  LIMIT=20 -> probe: prints n_frames / greedy / forbidden(oracle overlap) and,
              for reuse models, checks K_R_gt determinism vs stored file. No save.

TASK=both : grounders with no stored K_R_gt (VideoMind/GroundVTS/GVLLM/VTimeLLM,
            +TimeChat on QVH) -> compute K_GT and K_R_gt.
TASK=kgt  : Qwen + the 2 reference models (InternVL/VideoLLaMA2) -> K_GT only
            (their K_R_gt is reused from the stored rand_gt_keep files).
"""
import sys, os, json, time, importlib
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _R)
import pandas as pd

BENCH = os.environ['BENCH']; MODEL = os.environ['MODEL']
TASK = os.environ.get('TASK', 'both')          # both | kgt
SHARD = int(os.environ.get('SHARD', '0')); NSHARD = int(os.environ.get('NSHARD', '1'))
LIMIT = int(os.environ.get('LIMIT', '0'))
OUTDIR = f"{os.environ.get('TSFIX_OUT', 'results')}/gt_ceiling"
OUT = f'{OUTDIR}/{BENCH}_{MODEL}_shard{SHARD}.json'


def load_bench():
    """Return (C2_ids_sorted, shared_segments, vp(vqid), meta(vqid))."""
    if BENCH == 'qvh':
        meas = list(json.load(open(os.environ.get('QVH_RNORM','results/shared/qvh_r_norm.json')))['r_norm'])
        sh = json.load(open('results/shared/qvh_r_oracle_shared.json'))
        df = pd.read_csv(os.environ.get('VAL_SAMPLED_CSV','data/qvh_gqa/val_qvh_gqa_screened.csv'))
        df['vqid'] = df['video_id'].astype(str) + '_' + df['qid'].astype(str)
        dfm = {r['vqid']: r for _, r in df.iterrows()}

        _vd = os.environ.get('TEF_QVHME_VIDEO_DIR',
                             os.path.join(_R, 'data', 'qvhme', 'videos'))

        def vp(v):
            return os.path.join(_vd, f"{dfm[v]['video_id']}.mp4")

        def meta(v):
            r = dfm[v]
            return (str(r['question']), [str(r[f'a{k}']) for k in range(5)],
                    int(r['answer_id']), float(r['duration']))
    else:
        from supportgain.config.settings import VID_MAP_JSON, VIDEO_DIR
        from supportgain.core.video_io import find_video_path
        meas = list(json.load(open('results/shared/next_r_norm.json'))['r_norm'])
        sh = json.load(open('results/shared/r_oracle_shared.json'))
        s1 = json.load(open('results/qwen25vl/step1.json'))   # answer_id/duration (model-indep)
        vid_map = json.load(open(VID_MAP_JSON))
        df = pd.read_csv('data/val_full_strict.csv')
        df['vqid'] = df['video_id'].astype(str) + '_' + df['qid'].astype(str)
        dfm = {r['vqid']: r for _, r in df.iterrows()}

        def vp(v):
            return find_video_path(str(dfm[v]['video_id']), VIDEO_DIR, vid_map)

        def meta(v):
            r = dfm[v]; one = s1[v]
            return (str(r['question']), [str(r[f'a{k}']) for k in range(5)],
                    int(one['answer_id']), float(one['duration']))
    return meas, sh, vp, meta


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    meas, sh, vp, meta = load_bench()
    ids = [v for i, v in enumerate(sorted(meas)) if i % NSHARD == SHARD]
    if LIMIT:
        ids = ids[:LIMIT]
    done = json.load(open(OUT)) if (os.path.exists(OUT) and not LIMIT) else {}

    ad = importlib.import_module(f'supportgain.models.{MODEL}.adapter').Adapter()
    ad.load('cuda:0')
    has_keep = hasattr(ad, 'predict_keep')
    video_path_mode = (not has_keep and
                       getattr(ad, 'supports_video_path_intervention', lambda: False)())
    if not has_keep and not video_path_mode:
        from supportgain.core.interventions import apply_intervention
    cache = {'vp': None}

    def predict(p, s, e, q, ch, dur):
        if has_keep:
            return ad.predict_keep(p, s, e, q, ch)
        if video_path_mode:
            spec = {'phase': 'phase2', 'kind': 'keep', 'interval': (s, e),
                    'total_duration': dur}
            return ad.predict_qa_video(p, spec, q, ch)
        if cache['vp'] != p:
            fr, ts, _ = ad.decode_video(p); cache.update(vp=p, fr=fr, ts=ts)
        mf = apply_intervention('phase2', cache['fr'], cache['ts'], (s, e), 'keep')
        return (-1, '') if mf is None else ad.predict_qa(mf, q, ch)

    # ---- probe mode: report consistency, no save ----
    if LIMIT:
        mode = ('predict_keep' if has_keep else
                'video_path' if video_path_mode else 'frame_subset')
        print(f'PROBE {BENCH}/{MODEL} dispatch={mode} '
              f'greedy_enforced_by_adapter=do_sample=False', flush=True)
        stored = _load_stored_krgt()
        for v in ids[:20]:
            p = vp(v); seg = sh.get(v)
            if not p or not os.path.exists(p) or seg is None:
                print(f'  {v}: SKIP (video/seg missing)'); continue
            q, ch, gt, dur = meta(v)
            gt_iv = seg['oracle_iv']
            # frame count via decode (frame-subset path) or note video mode
            nf = ''
            if not has_keep and not video_path_mode:
                if cache['vp'] != p:
                    fr, ts, _ = ad.decode_video(p); cache.update(vp=p, fr=fr, ts=ts)
                nf = f'n_frames={len(cache["fr"])}'
            a_gt, _ = predict(p, gt_iv[0], gt_iv[1], q, ch, dur)
            # forbidden check: r_oracle vs GT overlap
            def ov(x, y):
                return max(0, min(x[1], y[1]) - max(x[0], y[0]))
            ovs = [round(ov(iv, gt_iv), 2) for iv in seg['r_oracle'] if iv]
            line = (f'  {v}: {nf} oracle_iv={[round(x,1) for x in gt_iv]} '
                    f'K_GT_pred={a_gt}(gt={gt}) r_oracle_GT_overlap={ovs} '
                    f'fallback={seg["r_oracle_fallback"]}')
            if stored and v in stored:
                sp = [t.get('pred') for t in stored[v] if not t.get('skip')]
                line += f'  STORED_krgt_pred={sp}'
            print(line, flush=True)
        print('PROBE_DONE (no save)', flush=True)
        return

    # ---- full run ----
    t0 = time.time(); n_new = 0
    for v in ids:
        if v in done:
            continue
        p = vp(v); seg = sh.get(v)
        if not p or not os.path.exists(p) or seg is None:
            continue
        q, ch, gt, dur = meta(v)
        try:
            rec = {}
            # K_GT: keep the GT interval
            gt_iv = seg['oracle_iv']
            a, raw = predict(p, gt_iv[0], gt_iv[1], q, ch, dur)
            rec['kgt'] = {'segment': [round(gt_iv[0], 2), round(gt_iv[1], 2)],
                          'pred': int(a), 'correct': int(a == gt and a >= 0),
                          'raw_output': str(raw)[:150]}
            # K_R_gt: keep each shared R_oracle segment (only when not reused)
            if TASK == 'both':
                trials = []
                for k, iv in enumerate(seg['r_oracle']):
                    fb = bool(seg['r_oracle_fallback'][k])
                    if not iv:
                        trials.append({'trial_idx': k, 'segment': None, 'pred': -1,
                                       'correct': 0, 'fallback_flag': fb, 'skip': 1})
                        continue
                    a, raw = predict(p, iv[0], iv[1], q, ch, dur)
                    trials.append({'trial_idx': k,
                                   'segment': [round(iv[0], 2), round(iv[1], 2)],
                                   'pred': int(a), 'correct': int(a == gt and a >= 0),
                                   'raw_output': str(raw)[:150], 'fallback_flag': fb})
                rec['krgt'] = trials
        except Exception as ex:
            cache['vp'] = None   # drop possibly-bad decode cache
            rec = {'error': str(ex)[:200]}
            print(f'  ERR {v}: {str(ex)[:120]}', flush=True)
        done[v] = rec
        n_new += 1
        if len(done) % 10 == 0:
            json.dump(done, open(OUT, 'w'))
            print(f'  [{BENCH}/{MODEL} s{SHARD}] {len(done)}/{len(ids)} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    json.dump(done, open(OUT, 'w'))
    kacc = sum(r['kgt']['correct'] for r in done.values()) / max(len(done), 1)
    print(f'DONE {BENCH}/{MODEL} shard{SHARD}: n={len(done)} new={n_new} '
          f'K_GT_GTacc={kacc:.3f} elapsed={time.time()-t0:.0f}s', flush=True)


def _load_stored_krgt():
    """For probe: stored K_R_gt of reuse models to check determinism."""
    import glob
    if BENCH == 'qvh':
        pats = {'qwen25vl': 'results/roracle_keep/qvh_qwen25vl_shard*.json',
                'internvl': 'results/roracle_keep/qvh_internvl_shard*.json',
                'videollama2': 'results/roracle_keep/qvh_videollama2_shard*.json'}
    else:
        pats = {'qwen25vl': 'results/roracle_keep/qwen25vl_full.json',
                'internvl': 'results/roracle_keep/internvl_full.json',
                'videollama2': 'results/roracle_keep/videollama2_full.json'}
    pat = pats.get(MODEL)
    if not pat:
        return None
    d = {}
    for f in glob.glob(pat):
        d.update(json.load(open(f)))
    return d


if __name__ == '__main__':
    main()
