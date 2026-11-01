"""§3.4 blind-proof — InternVL2.5-8B text-only (no video) QA.

pixel_values=None -> pure-text chat. Same normalization / QA prompt / parser
as the TEF pipeline for consistency with the other blind baselines.
"""
import sys, os, json
_R = os.path.dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.insert(0, _R)
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from supportgain.config.prompts import normalize_question, normalize_option, build_qa_prompt
from supportgain.core.parser import parse_answer

MODEL_PATH = os.environ.get('TEF_INTERNVL_CKPT', 'OpenGVLab/InternVL2_5-8B')
CSV = os.environ.get('BLIND_CSV', os.path.join(_R, 'data', 'val_full_strict.csv'))
OUT = os.environ.get('BLIND_OUT',
                     os.path.join(_R, 'flags', 'nextgqa', 'blind_internvl2_5_8b.json'))

model = AutoModel.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
gen_cfg = dict(max_new_tokens=16, do_sample=False)
print('DBG type(tok)=', type(tok), ' type(model)=', type(model), flush=True)

df = pd.read_csv(CSV)
res = {}
for _, r in df.iterrows():
    vqid = f"{r['video_id']}_{r['qid']}"
    q = normalize_question(str(r['question']))
    choices = [normalize_option(str(r[f'a{i}'])) for i in range(5)]
    ans_id = choices.index(normalize_option(str(r['answer'])))
    prompt = build_qa_prompt(q, choices)
    with torch.inference_mode():
        raw = model.chat(tokenizer=tok, pixel_values=None, question=prompt,
                         generation_config=gen_cfg, history=None, return_history=False)
    pred = parse_answer(str(raw), choices)
    res[vqid] = {'pred': pred, 'ans_id': int(ans_id),
                 'correct': int(pred == ans_id), 'raw': str(raw)[:40]}
    if len(res) % 50 == 0:
        print(f'  {len(res)}/{len(df)}', flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, 'w'), indent=1)
acc = sum(v['correct'] for v in res.values()) / len(res)
print(f'DONE_INTERNVL_BLIND  blind_acc={acc:.4f}  n={len(res)}  -> {OUT}', flush=True)