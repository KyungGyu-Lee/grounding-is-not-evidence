"""QVH-ME 790 two-stage FULL generation via Batch API.

Stage 1 (text batch)  : caption+dur+seg -> question + answer_text + type
Stage 2 (video batch) : Files API + video_metadata(fps=2.0) -> 4 distractors
Answer = caption-derived (stage1). Distractors = video-grounded (stage2).
Same prompt as the 100-pilot (style-symmetry instruction included), temp 0.3.

Robustness:
  * every network call (upload / batch.create / batch.get / response read)
    wrapped in retry() with exponential backoff; max wait logged.
  * batch job names written to full2/jobs.json BEFORE polling starts, so a
    poll drop never loses the job.
  * resume: reuses pilot 76, stage1.jsonl, uploads.json, per-chunk stage2 —
    completed query_ids are skipped.
  * ALL raw responses saved under full2/raw/.

  GEMINI_API_KEY=... <py> qvhme_build/qvh_regen_2stage_full.py
"""
import os, sys, json, time, re, random
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
from google import genai
from google.genai import types

HERE = os.environ.get('TEF_QVHME_DIR', os.path.join(_R, 'data', 'qvhme'))
VID = os.environ.get('TEF_QVHME_VIDEO_DIR', f'{HERE}/videos')
OUTD = f'{HERE}/regen/full2'
RAW = f'{OUTD}/raw'
os.makedirs(RAW, exist_ok=True)
MODEL = 'gemini-3.1-pro-preview'; FPS = 2.0
LET = list('ABCDE')
PILOT = f'{HERE}/regen/pilot2_2stage.jsonl'
JOBS = f'{OUTD}/jobs.json'
S1_OUT = f'{OUTD}/stage1.jsonl'
UP_OUT = f'{OUTD}/uploads.json'
S2_OUT = f'{OUTD}/stage2_distractors.jsonl'
FINAL = f'{OUTD}/final.jsonl'
CHUNK = 200
MAXWAIT = {'v': 0.0}

P1 = '''Video duration: {dur}s
Target segment: [{s}, {e}] ({L}s)
Target caption: "{caption}"

For this target segment:
1. Generate ONE question that can ONLY be answered by watching this scene. The question must be written in English.
2. Question type: choose exactly one of causal (why/how) or temporal (when / what happens first).
3. Provide the single correct answer to the question. Derive it from the caption. Written in English, matching the style of a multiple-choice option.
Response format (JSON only, no other text):
{{
  "question": "...",
  "answer_text": "...",
  "question_type": "causal" or "temporal"
}}'''

P2 = '''Target caption: "{caption}"
Question: {question}
Correct answer: {answer}

The video segment provided is the target segment.

Generate four distractor options for this question.

Requirements:
- Each distractor must be something that could plausibly happen in THIS setting, with the same people and objects.
- Each distractor must NOT actually occur anywhere in this video segment. You can see the segment; verify this.
- Do not use actions or objects from a different scene or setting. A viewer who sees a random other part of this video, but not this segment, must NOT be able to eliminate any distractor.
- The distractors must not overlap with one another or with the correct answer.
- Match the correct answer in length, specificity, and level of detail. If the correct answer is short and general, the distractors must also be short and general. Do not make distractors more specific than the correct answer.
- If you cannot produce four distractors meeting these conditions, output {{"skip": true, "reason": "..."}}.

Output (JSON only, no other text):
{{"distractors": ["...", "...", "...", "..."]}}'''


def retry(fn, tag, tries=12, base=6, cap=180):
    for a in range(tries):
        try:
            return fn()
        except Exception as ex:
            w = min(cap, base * (2 ** a))
            MAXWAIT['v'] = max(MAXWAIT['v'], w)
            msg = str(ex)[:80]
            is429 = '429' in msg or 'RESOURCE_EXHAUSTED' in msg
            print(f'  {tag} retry {a} ({type(ex).__name__}{" 429" if is429 else ""}) wait {w}s', flush=True)
            if a == tries - 1:
                raise
            time.sleep(w)


def parse_json(text):
    t = text.strip()
    if t.startswith('```'):
        t = t[t.find('\n') + 1:].rsplit('```', 1)[0]
    t = t.strip(); j = t.find('{')
    if j != -1:
        try:
            return json.JSONDecoder().raw_decode(t[j:])[0]
        except Exception:
            pass
    return json.loads(re.sub(r'(?<!\\)\n', ' ', t[j:] if j != -1 else t))


def load_jobs():
    return json.load(open(JOBS)) if os.path.exists(JOBS) else {}


def save_jobs(j):
    json.dump(j, open(JOBS, 'w'), indent=1)


def cfg():
    return types.GenerateContentConfig(temperature=0.3, response_mime_type='application/json')


def wait_active(client, f, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        g = retry(lambda: client.files.get(name=f.name), 'files.get')
        st = str(g.state)
        if 'ACTIVE' in st:
            return g
        if 'FAILED' in st:
            return None
        time.sleep(4)
    return None


def run_batch(client, reqs, tag, jobkey, jobs):
    """Create (or resume) a batch, persist job name BEFORE polling, return job."""
    name = jobs.get(jobkey)
    if not name:
        job = retry(lambda: client.batches.create(model=MODEL, src=reqs,
                    config=types.CreateBatchJobConfig(display_name=tag)), f'{tag}.create')
        name = job.name
        jobs[jobkey] = name; save_jobs(jobs)          # persist BEFORE poll
        print(f'  submitted {tag}: {name}', flush=True)
    else:
        print(f'  resume {tag}: {name}', flush=True)
    t0 = time.time()
    while True:
        j = retry(lambda: client.batches.get(name=name), f'{tag}.get')
        st = str(j.state)
        print(f'  [{tag}] {st} ({time.time()-t0:.0f}s)', flush=True)
        if 'SUCCEEDED' in st:
            return j
        if any(x in st for x in ['FAILED', 'CANCELLED', 'EXPIRED']):
            raise RuntimeError(f'{tag} terminal {st}: {getattr(j,"error",None)}')
        time.sleep(25)


def shuffle5(qid, texts):
    perm = list(range(5)); random.Random(str(qid)).shuffle(perm)
    new = [texts[perm[i]] for i in range(5)]
    return new, LET[perm.index(0)], perm


def main():
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    prompts = [json.loads(l) for l in open(f'{HERE}/prompts.jsonl')]
    cap_re = re.compile(r'Target caption:\s*"(.*?)"')
    pm = {str(p['query_id']): p for p in prompts}
    cap_of = {str(p['query_id']): cap_re.search(p['prompt']).group(1) for p in prompts}
    jobs = load_jobs()

    # reuse pilot 76 (fully done: question+answer+distractors+options)
    reuse = {}
    if os.path.exists(PILOT):
        for l in open(PILOT):
            r = json.loads(l)
            if r.get('parse_ok') and r.get('options') and not r.get('skip'):
                reuse[str(r['query_id'])] = r
    print(f'REUSE pilot: {len(reuse)}', flush=True)

    # ---------- STAGE 1 ----------
    s1 = {}
    if os.path.exists(S1_OUT):
        for l in open(S1_OUT):
            r = json.loads(l); s1[str(r['query_id'])] = r
    need1 = [str(p['query_id']) for p in prompts
             if str(p['query_id']) not in reuse and str(p['query_id']) not in s1]
    print(f'STAGE1 need={len(need1)} have={len(s1)}', flush=True)
    if need1:
        reqs = [types.InlinedRequest(
            contents=P1.format(dur=pm[q]['duration'], s=pm[q]['gt_start'], e=pm[q]['gt_end'],
                               L=pm[q]['gt_end'] - pm[q]['gt_start'], caption=cap_of[q]),
            config=cfg(), metadata={'key': q}) for q in need1]
        j = run_batch(client, reqs, 'qvh2_s1', 'stage1_job', jobs)
        rawf = open(f'{RAW}/stage1_responses.jsonl', 'a')
        with open(S1_OUT, 'a') as out:
            for r in (j.dest.inlined_responses or []):
                meta = getattr(r, 'metadata', None) or {}; q = str(meta.get('key'))
                if q not in pm:
                    continue
                txt = getattr(getattr(r, 'response', None), 'text', None)
                rawf.write(json.dumps({'qid': q, 'text': txt, 'error': str(getattr(r, 'error', None))}, ensure_ascii=False) + '\n')
                rec = {'query_id': pm[q]['query_id'], 'video_id': pm[q]['video_id']}
                try:
                    qa = parse_json(txt)
                    rec.update(question=qa.get('question'), answer_text=qa.get('answer_text'),
                               question_type=qa.get('question_type'), s1_ok=True)
                except Exception as ex:
                    rec.update(s1_ok=False, error=str(ex)[:120])
                out.write(json.dumps(rec, ensure_ascii=False) + '\n'); s1[q] = rec
        rawf.close()
    s1ok = {q: r for q, r in s1.items() if r.get('s1_ok') and r.get('answer_text')}
    print(f'STAGE1 done: ok={len(s1ok)} fail={len(s1)-len(s1ok)}', flush=True)

    # ---------- uploads (dedup by video_id) ----------
    uploads = json.load(open(UP_OUT)) if os.path.exists(UP_OUT) else {}
    vids_needed = sorted({pm[q]['video_id'] for q in s1ok})
    todo = [v for v in vids_needed if v not in uploads]
    print(f'UPLOAD need={len(todo)} have={len(uploads)} (unique videos, 8-way parallel)', flush=True)
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock(); done_ct = {'n': 0}

    def _up(vid):
        f = retry(lambda: client.files.upload(file=f'{VID}/{vid}.mp4'), f'upload[{vid}]')
        fo = wait_active(client, f)
        return vid, (fo.uri if fo else None)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_up, v) for v in todo]
        for fut in as_completed(futs):
            vid, uri = fut.result()
            with lock:
                if uri:
                    uploads[vid] = uri
                done_ct['n'] += 1
                if done_ct['n'] % 20 == 0:
                    json.dump(uploads, open(UP_OUT, 'w'))
                    print(f'  uploaded {done_ct["n"]}/{len(todo)}', flush=True)
    json.dump(uploads, open(UP_OUT, 'w'))
    print(f'UPLOAD done: {len(uploads)} videos', flush=True)

    # ---------- STAGE 2 (chunked video batch) ----------
    s2 = {}
    if os.path.exists(S2_OUT):
        for l in open(S2_OUT):
            r = json.loads(l); s2[str(r['query_id'])] = r
    need2 = [q for q in s1ok if q not in s2 and pm[q]['video_id'] in uploads]
    print(f'STAGE2 need={len(need2)} have={len(s2)}', flush=True)
    chunks = [need2[i:i + CHUNK] for i in range(0, len(need2), CHUNK)]
    rawf = open(f'{RAW}/stage2_responses.jsonl', 'a')
    for ci, chunk in enumerate(chunks):
        reqs = []
        for q in chunk:
            p = pm[q]; s, e = p['gt_start'], p['gt_end']
            part = types.Part(file_data=types.FileData(file_uri=uploads[p['video_id']], mime_type='video/mp4'),
                              video_metadata=types.VideoMetadata(start_offset=f'{s}s', end_offset=f'{e}s', fps=FPS))
            prompt2 = P2.format(caption=cap_of[q], question=s1ok[q]['question'], answer=s1ok[q]['answer_text'])
            reqs.append(types.InlinedRequest(contents=types.Content(parts=[part, types.Part(text=prompt2)]),
                                             config=cfg(), metadata={'key': q}))
        j = run_batch(client, reqs, f'qvh2_s2_c{ci}', f'stage2_job_c{ci}', jobs)
        with open(S2_OUT, 'a') as out:
            for r in (j.dest.inlined_responses or []):
                meta = getattr(r, 'metadata', None) or {}; q = str(meta.get('key'))
                if q not in pm:
                    continue
                txt = getattr(getattr(r, 'response', None), 'text', None)
                rawf.write(json.dumps({'qid': q, 'text': txt, 'error': str(getattr(r, 'error', None))}, ensure_ascii=False) + '\n')
                rec = {'query_id': pm[q]['query_id']}
                try:
                    d = parse_json(txt)
                    if d.get('skip'):
                        rec.update(skip=True, skip_reason=d.get('reason'), s2_ok=True)
                    else:
                        dis = d.get('distractors') or []
                        rec.update(distractors=dis, skip=False, s2_ok=(len(dis) == 4))
                except Exception as ex:
                    rec.update(s2_ok=False, error=str(ex)[:120])
                out.write(json.dumps(rec, ensure_ascii=False) + '\n'); s2[q] = rec
        print(f'  chunk {ci+1}/{len(chunks)} parsed', flush=True)
    rawf.close()

    # ---------- assemble FINAL (reuse 76 + new) ----------
    n_ok = n_skip = n_fail = 0
    with open(FINAL, 'w') as out:
        for p in prompts:
            q = str(p['query_id'])
            if q in reuse:
                out.write(json.dumps(reuse[q], ensure_ascii=False) + '\n'); n_ok += 1; continue
            a = s1ok.get(q); b = s2.get(q)
            if not a or not b:
                n_fail += 1; continue
            rec = {'video_id': p['video_id'], 'query_id': p['query_id'], 'duration': p['duration'],
                   'gt_start': p['gt_start'], 'gt_end': p['gt_end'], 'caption': cap_of[q],
                   'question': a['question'], 'answer_text': a['answer_text'],
                   'question_type': a.get('question_type'), 'model': MODEL}
            if b.get('skip'):
                rec.update(skip=True, skip_reason=b.get('skip_reason'), parse_ok=True)
                out.write(json.dumps(rec, ensure_ascii=False) + '\n'); n_skip += 1; continue
            if not b.get('s2_ok'):
                n_fail += 1; continue
            texts = [a['answer_text']] + [str(x) for x in b['distractors']]
            new, ans, perm = shuffle5(q, texts)
            rec.update(distractors=b['distractors'], options={LET[i]: new[i] for i in range(5)},
                       answer=ans, perm_newslot_to_origidx=perm, skip=False, parse_ok=True)
            out.write(json.dumps(rec, ensure_ascii=False) + '\n'); n_ok += 1
    print(f'FINAL: ok={n_ok} skip={n_skip} fail={n_fail} -> {FINAL}', flush=True)
    print(f'MAX_RETRY_WAIT={MAXWAIT["v"]:.0f}s', flush=True)
    print('GEN_DONE', flush=True)


if __name__ == '__main__':
    main()
