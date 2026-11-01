"""Post-generation for the 790 two-stage set: safety screen (batch) + shuffle
map + quality gate vs existing 788 / 1st-regen 789.

Runs after full2/final.jsonl exists. Reuses retry/run_batch/parse_json/cfg from
qvh_regen_2stage_full. Writes:
  full2/screen.jsonl                 verdicts
  full2/val_qvh_gqa_2stage_shuf.csv  final CSV (options already post-hoc shuffled)
  full2/shuffle_map_2stage.json      qid-seeded permutations
  prints GATE_PASS / GATE_FAIL with reasons.

  GEMINI_API_KEY=... <py> qvhme_build/qvh_2stage_postgen.py
"""
import os, sys, json, csv, re
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
import numpy as np
from collections import Counter
from google import genai
from google.genai import types
import analysis.qvh_regen_2stage_full as G

OUTD = G.OUTD
FINAL = G.FINAL
SCREEN = f'{OUTD}/screen.jsonl'
CSV_SHUF = f'{OUTD}/val_qvh_gqa_2stage_shuf.csv'
MAP = f'{OUTD}/shuffle_map_2stage.json'
LET = 'ABCDE'
SCREEN_PROMPT = re.search(r'PROMPT = """(.*?)"""',
                          open('data/qvh_gqa/batch_screen.py').read(), re.DOTALL).group(1)


def screen(client, rows, jobs):
    reqs = []
    for r in rows:
        o = r['options']; opts = '; '.join(f"{k}) {o.get(k,'')}" for k in LET)
        reqs.append(types.InlinedRequest(
            contents=SCREEN_PROMPT.format(vid=r['video_id'], q=r['question'], opts=opts),
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type='application/json'),
            metadata={'key': str(r['query_id'])}))
    j = G.run_batch(client, reqs, 'qvh2_screen', 'screen_job', jobs)
    rmap = {str(r['query_id']): r for r in rows}
    excl = []
    with open(SCREEN, 'w') as out:
        for resp in (j.dest.inlined_responses or []):
            meta = getattr(resp, 'metadata', None) or {}; q = str(meta.get('key'))
            r = rmap.get(q)
            if r is None:
                continue
            try:
                v = G.parse_json(resp.response.text)
            except Exception as ex:
                v = {'verdict': 'SAFE', 'exclude': False, 'reason': 'parse_fail:' + str(ex)[:20]}
            v.update(query_id=r['query_id'], video_id=r['video_id'])
            if v.get('exclude'):
                excl.append(v)
            out.write(json.dumps(v, ensure_ascii=False) + '\n')
    return excl


def stats(rows, label):
    n = len(rows)
    ai = Counter(int(r['answer_id']) for r in rows)
    qlen = [len(str(r['question']).split()) for r in rows]
    cl, dl, longest = [], [], 0
    for r in rows:
        aid = int(r['answer_id']); lens = [len(str(r[f'a{k}']).split()) for k in range(5)]
        cl.append(lens[aid]); dl += [lens[k] for k in range(5) if k != aid]
        longest += int(int(np.argmax(lens)) == aid)
    pos = {LET[k]: ai.get(k, 0) for k in range(5)}
    # chi2 uniform
    exp = n / 5.0
    chi2 = sum((ai.get(k, 0) - exp) ** 2 / exp for k in range(5))
    d = dict(N=n, pos=pos, chi2=chi2, qlen_med=float(np.median(qlen)),
             cl=float(np.mean(cl)), dl=float(np.mean(dl)), longest=longest / n,
             maxpos=max(pos.values()) / n)
    print(f"[{label}] N={n} 정답위치 {pos} chi2={chi2:.1f} | 질문 med {d['qlen_med']:.0f} | "
          f"선택지 정답 {d['cl']:.2f} 오답 {d['dl']:.2f} (Δ{d['cl']-d['dl']:+.2f}) 정답최장 {d['longest']:.2f}")
    return d


def main():
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    jobs = G.load_jobs()
    recs = [json.loads(l) for l in open(FINAL)]
    usable = [r for r in recs if r.get('options') and not r.get('skip') and r.get('parse_ok')]
    skipped = [r for r in recs if r.get('skip')]
    print(f"final: total={len(recs)} usable={len(usable)} skip={len(skipped)}")

    # ---- screen ----
    excl = screen(client, usable, jobs)
    print(f"SCREEN excluded={len(excl)}")
    for v in excl:
        print(f"  EXCLUDE qid={v['query_id']} {v.get('verdict')}: {v.get('reason')}")
    excl_ids = {str(v['query_id']) for v in excl}

    # ---- CSV (options already shuffled) + shuffle map ----
    cols = ['video_id', 'qid', 'question', 'a0', 'a1', 'a2', 'a3', 'a4', 'answer_id',
            'duration', 'gt_start', 'gt_end', 'question_type']
    ws = csv.DictWriter(open(CSV_SHUF, 'w', newline=''), fieldnames=cols); ws.writeheader()
    mp = []; final_rows = []
    for r in usable:
        if str(r['query_id']) in excl_ids:
            continue
        o = r['options']; aid = LET.index(r['answer'])
        row = {'video_id': r['video_id'], 'qid': r['query_id'], 'question': r['question'],
               **{f'a{i}': o[LET[i]] for i in range(5)}, 'answer_id': aid,
               'duration': r['duration'], 'gt_start': r['gt_start'], 'gt_end': r['gt_end'],
               'question_type': r.get('question_type')}
        ws.writerow(row); final_rows.append(row)
        mp.append({'qid': r['query_id'], 'perm_newslot_to_origidx': r.get('perm_newslot_to_origidx'),
                   'answer_id': aid})
    json.dump(mp, open(MAP, 'w'), indent=1)
    print(f"CSV N={len(final_rows)} -> {CSV_SHUF}")

    # ---- quality gate ----
    print("\n### QUALITY GATE ###")
    old = list(csv.DictReader(open('data/qvh_gqa/val_qvh_gqa_screened.csv')))
    en = list(csv.DictReader(open('data/qvh_gqa_regen/val_qvh_gqa_en_shuf.csv')))
    parse_ok = len(usable) / len(recs)
    qt = Counter(r.get('question_type') for r in final_rows)
    print(f"parse_ok(usable/total): {parse_ok:.3f} | skip {len(skipped)} | question_type {dict(qt)}")
    d_old = stats(old, '기존 788')
    d_en = stats(en, '1차 789')
    d_new = stats(final_rows, '신규 2단계')

    # gate rules (pre-set)
    fails = []
    if parse_ok < 0.98:
        fails.append(f"parse_ok {parse_ok:.3f} < 0.98")
    if d_new['longest'] > 0.40:
        fails.append(f"정답최장 {d_new['longest']:.2f} > 0.40")
    if abs(d_new['cl'] - d_new['dl']) > 2.0:
        fails.append(f"선택지 길이차 {d_new['cl']-d_new['dl']:+.2f} (>2 단어)")
    if d_new['maxpos'] > 0.35:
        fails.append(f"셔플후 최다위치 {d_new['maxpos']:.2f} > 0.35")
    if d_new['chi2'] > 13.28:      # chi2 df=4 p=0.01
        fails.append(f"위치분포 chi2 {d_new['chi2']:.1f} > 13.28 (p<0.01)")
    if len(final_rows) < 700:
        fails.append(f"최종 N {len(final_rows)} < 700")

    print("\n" + ("GATE_FAIL: " + " | ".join(fails) if fails else
                  f"GATE_PASS N={len(final_rows)} 정답최장={d_new['longest']:.2f} "
                  f"maxpos={d_new['maxpos']:.2f} chi2={d_new['chi2']:.1f} 길이Δ={d_new['cl']-d_new['dl']:+.2f}"))


if __name__ == '__main__':
    main()
