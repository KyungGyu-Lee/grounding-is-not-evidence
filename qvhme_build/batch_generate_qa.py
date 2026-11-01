"""Batch-generate QVH Grounded-VideoQA items via the Gemini Batch API.

Uses gemini-3.1-pro-preview (pinned, = what gemini-pro-latest resolves to) in
BATCH mode: no per-minute RPM limits, ~50% cheaper, async. Submits all
not-yet-generated prompts as one inlined batch, polls, then merges parsed
results into generated_qa.jsonl (skipping ones already present).

    export GEMINI_API_KEY=...
    pip install google-genai
    python data/qvh_gqa/batch_generate_qa.py            # submit + poll + merge
    python data/qvh_gqa/batch_generate_qa.py --poll <batch_name>   # resume polling
"""
import os
import re
import sys
import json
import time
import argparse

from google import genai
from google.genai import types

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, 'prompts.jsonl')
OUT = os.path.join(HERE, 'generated_qa.jsonl')
MODEL = 'gemini-3.1-pro-preview'   # pinned (= gemini-pro-latest currently)


def _client():
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        sys.exit('ERROR: set GEMINI_API_KEY')
    return genai.Client(api_key=key)


def load_done():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                if r.get('parse_ok'):
                    done.add(str(r['query_id']))
            except Exception:
                pass
    return done


def parse_json(text):
    """Robust JSON extraction. Handles ```-fences, trailing 'Extra data', and
    the common 'Expecting , delimiter' caused by an unescaped double-quote or a
    raw newline inside a string value."""
    t = text.strip()
    if t.startswith('```'):
        t = t[t.find('\n') + 1:].rsplit('```', 1)[0]
    t = t.strip()
    # 1) first well-formed object (fixes "Extra data: ... trailing")
    start = t.find('{')
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(t[start:])
            return obj
        except Exception:
            pass
    # 2) light repair: escape stray newlines inside strings, then retry
    repaired = re.sub(r'(?<!\\)\n', ' ', t[start:] if start != -1 else t)
    return json.loads(repaired)


def get_job(client, name):
    for _ in range(6):
        try:
            return client.batches.get(name=name)
        except Exception:
            time.sleep(5)
    return None


def merge_results(client, job):
    """Parse a SUCCEEDED job's inlined responses into generated_qa.jsonl."""
    prompts = {str(json.loads(l)['query_id']): json.loads(l)
               for l in open(PROMPTS)}
    dest = job.dest
    resps = getattr(dest, 'inlined_responses', None) or []
    ok = fail = 0
    out = open(OUT, 'a')
    for r in resps:
        meta = getattr(r, 'metadata', None) or {}
        key = meta.get('key') if isinstance(meta, dict) else None
        p = prompts.get(str(key))
        if p is None:
            continue
        rec = {k: p[k] for k in ('video_id', 'query_id', 'duration',
                                 'gt_start', 'gt_end')}
        resp = getattr(r, 'response', None)
        err = getattr(r, 'error', None)
        try:
            if err or resp is None:
                raise RuntimeError(str(err))
            qa = parse_json(resp.text)
            rec.update({'question': qa.get('question'), 'options': qa.get('options'),
                        'answer': qa.get('answer'),
                        'question_type': qa.get('question_type'),
                        'model': MODEL,
                        'model_version': getattr(resp, 'model_version', None),
                        'parse_ok': True})
            ok += 1
        except Exception as ex:
            raw = getattr(resp, 'text', None) if resp is not None else None
            rec.update({'parse_ok': False, 'error': str(ex)[:200], 'raw': raw})
            fail += 1
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    out.close()
    print(f'merged: ok={ok} fail={fail}', flush=True)


def poll_until_done(client, name):
    while True:
        job = get_job(client, name)
        if job is None:
            time.sleep(15)
            continue
        st = str(job.state)
        print(f'  {st}', flush=True)
        if 'SUCCEEDED' in st:
            merge_results(client, job)
            return
        if 'FAILED' in st or 'CANCELLED' in st or 'EXPIRED' in st:
            print(f'batch ended without success: {st}', flush=True)
            return
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--poll', help='resume polling an existing batch name')
    args = ap.parse_args()
    client = _client()

    if args.poll:
        poll_until_done(client, args.poll)
        return

    prompts = [json.loads(l) for l in open(PROMPTS)]
    done = load_done()
    todo = [p for p in prompts if str(p['query_id']) not in done]
    print(f'total={len(prompts)} done={len(done)} todo={len(todo)}', flush=True)
    if not todo:
        print('nothing to do')
        return

    reqs = [types.InlinedRequest(
                contents=p['prompt'],
                config=types.GenerateContentConfig(
                    temperature=0.7, response_mime_type='application/json'),
                metadata={'key': str(p['query_id'])})
            for p in todo]
    job = None
    for attempt in range(8):
        try:
            job = client.batches.create(
                model=MODEL, src=reqs,
                config=types.CreateBatchJobConfig(display_name='qvh_gqa'))
            break
        except Exception as ex:
            print(f'  create retry {attempt}: {type(ex).__name__}', flush=True)
            time.sleep(min(30, 3 * (attempt + 1)))
    if job is None:
        sys.exit('ERROR: batch create failed after retries')
    print(f'submitted batch: {job.name}  state={job.state}', flush=True)
    poll_until_done(client, job.name)


if __name__ == '__main__':
    main()