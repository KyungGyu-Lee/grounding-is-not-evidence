"""§3.4 blind-proof — VideoLLaMA2-7B (Mistral) text-only (no video) QA.

mm_infer(None, prompt, ..., modal='text') -> pure-text answer. Same
normalization / QA prompt / parser as the other blind baselines.
"""
import sys, os, json
_R = os.path.dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.insert(0, _R)
import pandas as pd
from videollama2 import model_init, mm_infer
from supportgain.config.prompts import normalize_question, normalize_option, build_qa_prompt
from supportgain.core.parser import parse_answer

MODEL_PATH = os.environ.get('TEF_VIDEOLLAMA2_CKPT', 'DAMO-NLP-SG/VideoLLaMA2-7B')
CSV = os.environ.get('BLIND_CSV', os.path.join(_R, 'data', 'val_full_strict.csv'))
OUT = os.environ.get('BLIND_OUT',
                     os.path.join(_R, 'flags', 'nextgqa', 'blind_videollama2_7b.json'))

model, processor, tokenizer = model_init(MODEL_PATH)

df = pd.read_csv(CSV)
res = {}
for _, r in df.iterrows():
    vqid = f"{r['video_id']}_{r['qid']}"
    q = normalize_question(str(r['question']))
    choices = [normalize_option(str(r[f'a{i}'])) for i in range(5)]
    ans_id = choices.index(normalize_option(str(r['answer'])))
    prompt = build_qa_prompt(q, choices)
    raw = mm_infer(None, prompt, model=model, tokenizer=tokenizer,
                   modal='text', do_sample=False)
    pred = parse_answer(str(raw), choices)
    res[vqid] = {'pred': pred, 'ans_id': int(ans_id),
                 'correct': int(pred == ans_id), 'raw': str(raw)[:40]}
    if len(res) % 50 == 0:
        print(f'  {len(res)}/{len(df)}', flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, 'w'), indent=1)
acc = sum(v['correct'] for v in res.values()) / len(res)
print(f'DONE_VIDEOLLAMA2_BLIND  blind_acc={acc:.4f}  n={len(res)}  -> {OUT}', flush=True)