"""B1-1 verification — merge the shard outputs and check them against the
stored CROP run. Read-only on everything but the merged files.

Checks
  1. parse failures (parsed == -1) per variant
  2. exactly 6 responses per question (pred + rand_0..4), 241 questions
  3. 1:1 correspondence with CROP on (vqid, window_type), and that the window
     each variant was given is the stored one (MARGIN: the stored one grown by
     <= 5 s per side)
  4. no duplicate (vqid, variant, window_type); no missing one
  5. the 9 zero-frame CROP windows and, for MASK, the all-black inputs
"""
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f'{ROOT}/analysis/B1_masking'
CROP_S3 = f'{ROOT}/outputs/qvhme/qwen25vl/step3.json'
VARIANTS = ('mask', 'margin')
WTYPES = ['pred'] + [f'rand_{k}' for k in range(5)]
MARGIN_K = 5.0


def crop_label(wtype):
    return 'predicted_keep' if wtype == 'pred' else f'r_pred_keep_{wtype.split("_")[1]}'


def main():
    s3 = json.load(open(CROP_S3))
    inputs = [json.loads(l) for l in open(f'{OUT}/inputs.jsonl')]
    want = {(r['vqid'], r['variant'], r['window_type']) for r in inputs}
    questions = sorted({r['vqid'] for r in inputs})
    ok = True

    merged = {}
    for v in VARIANTS:
        rows = []
        for p in sorted(glob.glob(f'{OUT}/responses_{v}_shard[0-9].jsonl')):
            rows += [json.loads(l) for l in open(p)]
        keys = [(r['vqid'], r['window_type']) for r in rows]
        dup = [k for k, n in collections.Counter(keys).items() if n > 1]
        perq = collections.Counter(r['vqid'] for r in rows)
        fails = [r for r in rows if r['parsed'] < 0]
        print(f'--- {v}')
        print(f'    rows            : {len(rows)}  (expect 1446)')
        print(f'    questions       : {len(perq)}  (expect 241)')
        print(f'    responses/q     : {sorted(set(perq.values()))}  (expect [6])')
        print(f'    duplicates      : {len(dup)}')
        print(f'    parse fail (-1) : {len(fails)}')
        ok &= (len(rows) == 1446 and len(perq) == 241
               and set(perq.values()) == {6} and not dup)
        got = {(r['vqid'], v, r['window_type']) for r in rows}
        miss = {k for k in want if k[1] == v} - got
        extra = got - want
        print(f'    missing         : {len(miss)}   extra: {len(extra)}')
        ok &= not miss and not extra
        merged[v] = {(r['vqid'], r['window_type']): r for r in rows}

        # frame-count sanity
        nf = sorted(r['n_frames_in'] for r in rows)
        print(f'    n_frames_in     : min {nf[0]} med {nf[len(nf)//2]} max {nf[-1]}')
        if v == 'mask':
            bad = [r for r in rows if r['n_frames_in'] != r['n_frames_full']]
            print(f'    frame count != full-video : {len(bad)}  (expect 0)')
            ok &= not bad

    # ---- 3. pairing with the stored CROP run
    print('--- pairing with stored CROP')
    mism_win, mism_meta, n_pairs = [], [], 0
    for r in inputs:
        v, key = r['variant'], (r['vqid'], r['window_type'])
        row = merged[v][key]
        rec = s3[r['vqid']]
        n_pairs += 1
        src = rec['predicted_iv'] if r['window_type'] == 'pred' \
            else rec['r_pred'][int(r['window_type'].split('_')[1])]
        if [round(x, 6) for x in row['interval_src']] != [round(x, 6) for x in src]:
            mism_win.append(key)
        app = row['interval_applied']
        if v == 'mask':
            if app != row['interval_src']:
                mism_win.append(key)
        else:
            if not (app[0] <= src[0] and app[1] >= src[1]
                    and src[0] - app[0] <= MARGIN_K + 1e-9
                    and app[1] - src[1] <= MARGIN_K + 1e-9):
                mism_win.append(key)
        if (row['a_full'] != rec['a_full'] or row['answer_id'] != rec['answer_id']):
            mism_meta.append(key)
    print(f'    pairs checked   : {n_pairs}  (expect 2892)')
    print(f'    window mismatch : {len(mism_win)}')
    print(f'    a_full/answer_id mismatch : {len(mism_meta)}')
    ok &= (n_pairs == 2892 and not mism_win and not mism_meta)

    # every CROP keep label the analysis will use is present
    missing_crop = [(q, w) for q in questions for w in WTYPES
                    if crop_label(w) not in s3[q].get('preds', {})]
    print(f'    CROP labels missing : {len(missing_crop)}')
    ok &= not missing_crop

    # ---- 5. zero-frame / all-black bookkeeping
    zf = json.load(open(f'{OUT}/zero_frame_windows.json'))
    zset = {(w['vqid'], w['window_type']) for w in zf['windows']}
    ab = {k for k, r in merged['mask'].items() if r.get('all_black')}
    print('--- zero-frame windows')
    print(f'    stored CROP empty_or_no_interval : {len(zset)}')
    print(f'    MASK all_black                   : {len(ab)}')
    print(f'    identical sets                   : {ab == zset}')
    ok &= (ab == zset)
    mg0 = [k for k, r in merged['margin'].items() if r['n_frames_in'] == 0]
    print(f'    MARGIN empty inputs              : {len(mg0)}  (expect 0)')
    ok &= not mg0
    print(f'    affected questions               : {len(zf["questions"])}')

    # ---- write the merged files
    for v in VARIANTS:
        p = f'{OUT}/responses_{v}.jsonl'
        with open(p, 'w') as f:
            for q in questions:
                for w in WTYPES:
                    f.write(json.dumps(merged[v][(q, w)]) + '\n')
        print(f'merged -> {p}  ({len(merged[v])} rows)')

    print('\nB1-1 VERIFICATION: ' + ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
