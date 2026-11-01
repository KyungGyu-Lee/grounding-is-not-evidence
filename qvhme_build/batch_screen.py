"""Inappropriate-content screening for QVH-GQA via Gemini Batch API.

Judges each QA (question + 5 options + video_id) for sexual / violence / hate
content. One batch pass over all 790. Output: results/qvh_screen.jsonl with
{query_id, verdict(SAFE/SEXUAL/VIOLENCE/HATE), exclude(bool), reason}.

  export GEMINI_API_KEY=...
  python data/qvh_gqa/batch_screen.py
"""
import os, sys, json, time, re
from google import genai
from google.genai import types

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, 'generated_qa.jsonl')
OUT = os.path.join(HERE, 'screen.jsonl')
MODEL = 'gemini-3.1-pro-preview'

PROMPT = """You are screening a video-QA item for a research benchmark. Decide \
if the item's SUBJECT MATTER is inappropriate (would need exclusion) based on \
sexual, violent, or hateful content that is EXPLICIT or GRATUITOUS.

Be calibrated: ordinary content is SAFE. A bikini at a beach, cooking, boxing \
as sport, a news report mentioning a place named after "Blood", casual words \
like "butter"/"striped"/"stabilize" are SAFE. Flag ONLY genuinely explicit \
sexual content (nudity, sexual acts, sexualized body focus), graphic violence \
(gore, real killing/assault), or hate speech.

Item:
video_id: {vid}
question: {q}
options: {opts}

Return JSON: {{"verdict": "SAFE"|"SEXUAL"|"VIOLENCE"|"HATE", "exclude": true|false, "reason": "<8 words"}}"""


def parse(t):
    t = t.strip()
    if t.startswith('```'):
        t = t[t.find('\n')+1:].rsplit('```', 1)[0]
    s = t.find('{')
    return json.loads(t[s:t.rfind('}')+1])


def main():
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        sys.exit('set GEMINI_API_KEY')
    client = genai.Client(api_key=key)
    rows = [json.loads(l) for l in open(GEN) if json.loads(l).get('parse_ok')]
    reqs = []
    for r in rows:
        o = r['options']
        opts = '; '.join(f"{k}) {o.get(k,'')}" for k in 'ABCDE')
        p = PROMPT.format(vid=r['video_id'], q=r['question'], opts=opts)
        reqs.append(types.InlinedRequest(
            contents=p, config=types.GenerateContentConfig(
                temperature=0.0, response_mime_type='application/json'),
            metadata={'key': str(r['query_id'])}))
    job = None
    for a in range(8):
        try:
            job = client.batches.create(model=MODEL, src=reqs,
                    config=types.CreateBatchJobConfig(display_name='qvh_screen'))
            break
        except Exception as ex:
            print('retry', a, type(ex).__name__, flush=True); time.sleep(min(30, 3*(a+1)))
    print('submitted', job.name, flush=True)
    while True:
        j = client.batches.get(name=job.name)
        st = str(j.state); print(' ', st, flush=True)
        if 'SUCCEEDED' in st: break
        if any(x in st for x in ['FAILED','CANCELLED','EXPIRED']): sys.exit('batch failed '+st)
        time.sleep(30)
    rmap = {str(r['query_id']): r for r in rows}
    out = open(OUT, 'w'); ok = 0
    for resp in (j.dest.inlined_responses or []):
        meta = getattr(resp, 'metadata', None) or {}
        k = meta.get('key'); r = rmap.get(str(k))
        if r is None: continue
        try:
            v = parse(resp.response.text)
        except Exception as ex:
            v = {'verdict':'SAFE','exclude':False,'reason':'parse_fail:'+str(ex)[:40]}
        v.update(query_id=r['query_id'], video_id=r['video_id'], question=r['question'])
        out.write(json.dumps(v, ensure_ascii=False)+'\n'); ok += 1
    out.close()
    print(f'DONE screen n={ok} -> {OUT}', flush=True)


if __name__ == '__main__':
    main()