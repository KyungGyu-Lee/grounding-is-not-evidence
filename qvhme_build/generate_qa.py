"""Batch-generate Grounded-VideoQA items from QVHighlights prompts via Gemini.

Reads prompts.jsonl (one prompt per candidate segment), calls gemini-2.5-pro,
parses the JSON answer, and appends to generated_qa.jsonl. Resumable: already
generated query_ids are skipped. Rate-limit safe: retry with backoff + sleep.

Usage:
    export GEMINI_API_KEY=...
    pip install google-genai
    python data/qvh_gqa/generate_qa.py [--sleep 2.0] [--max-retries 5] [--limit 0]

Note: uses the NEW `google-genai` SDK. Model `gemini-pro-latest` is the latest GA
Gemini Pro (= gemini-2.5-pro); dated names like `gemini-2.5-pro` are 404 for new
API users, so the `-latest` alias is used.
"""
import os
import sys
import json
import time
import argparse

from google import genai

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, 'prompts.jsonl')
OUT = os.path.join(HERE, 'generated_qa.jsonl')
MODEL = 'gemini-pro-latest'   # latest GA Gemini Pro (2.5-pro)


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)['query_id'])
            except Exception:
                pass
    return done


def parse_json(text):
    """Robustly extract a JSON object from the model output."""
    t = text.strip()
    if t.startswith('```'):                       # strip ```json ... ``` fences
        t = t[t.find('\n') + 1:]
        t = t.rsplit('```', 1)[0]
    return json.loads(t.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sleep', type=float, default=2.0,
                    help='seconds to sleep between successful calls (rate limit)')
    ap.add_argument('--max-retries', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0, help='0 = all remaining')
    args = ap.parse_args()

    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        sys.exit('ERROR: set GEMINI_API_KEY environment variable.')
    client = genai.Client(api_key=key)
    gen_cfg = {'temperature': 0.7, 'response_mime_type': 'application/json'}

    prompts = [json.loads(l) for l in open(PROMPTS)]
    done = load_done(OUT)
    todo = [p for p in prompts if p['query_id'] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f'total={len(prompts)}  done={len(done)}  todo={len(todo)}', flush=True)

    out = open(OUT, 'a')
    ok = fail = 0
    for i, p in enumerate(todo, 1):
        rec = {k: p[k] for k in ('video_id', 'query_id', 'duration',
                                 'gt_start', 'gt_end')}
        last_err, raw = None, None
        for attempt in range(args.max_retries):
            try:
                resp = client.models.generate_content(
                    model=MODEL, contents=p['prompt'], config=gen_cfg)
                raw = resp.text
                qa = parse_json(raw)
                rec.update({
                    'question': qa.get('question'),
                    'options': qa.get('options'),
                    'answer': qa.get('answer'),
                    'question_type': qa.get('question_type'),
                    'parse_ok': True,
                })
                ok += 1
                break
            except Exception as ex:                # rate limit / transient / parse
                last_err = f'{type(ex).__name__}: {ex}'
                time.sleep(min(60.0, args.sleep * (2 ** attempt)))
        else:
            rec.update({'parse_ok': False, 'error': last_err, 'raw': raw})
            fail += 1
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        out.flush()
        if i % 20 == 0:
            print(f'  {i}/{len(todo)}  ok={ok} fail={fail}', flush=True)
        time.sleep(args.sleep)
    out.close()
    print(f'DONE  ok={ok}  fail={fail}  -> {OUT}', flush=True)


if __name__ == '__main__':
    main()