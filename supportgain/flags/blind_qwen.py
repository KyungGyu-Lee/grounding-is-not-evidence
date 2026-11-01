"""§3.4 blind-proof — Qwen2.5-VL text-only (no video) QA.

Feeds only the (normalized) question + 5 options and records the predicted
letter. A correct answer here = the sample is solvable by language prior alone.
Uses the SAME normalization / QA prompt / letter parser as the TEF pipeline.
"""
import sys, os, json
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _R)
import torch
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from supportgain.config.prompts import normalize_question, normalize_option, build_qa_prompt
from supportgain.core.parser import parse_answer

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
CSV = os.environ.get('BLIND_CSV', os.path.join(_R, 'data', 'val_full_strict.csv'))
OUT = os.environ.get('BLIND_OUT',
                     os.path.join(_R, 'flags', 'nextgqa', 'blind_qwen25vl.json'))

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2', device_map={'': 'cuda:0'}).eval()
proc = AutoProcessor.from_pretrained(MODEL_ID)

df = pd.read_csv(CSV)
res = {}
for _, r in df.iterrows():
    vqid = f"{r['video_id']}_{r['qid']}"
    q = normalize_question(str(r['question']))
    choices = [normalize_option(str(r[f'a{i}'])) for i in range(5)]
    ans_id = choices.index(normalize_option(str(r['answer'])))
    prompt = build_qa_prompt(q, choices)
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], return_tensors='pt').to('cuda:0')
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    raw = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    pred = parse_answer(raw, choices)
    res[vqid] = {'pred': pred, 'ans_id': int(ans_id),
                 'correct': int(pred == ans_id), 'raw': raw[:40]}
    if len(res) % 50 == 0:
        print(f'  {len(res)}/{len(df)}', flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, 'w'), indent=1)
acc = sum(v['correct'] for v in res.values()) / len(res)
print(f'DONE_QWEN_BLIND  blind_acc={acc:.4f}  n={len(res)}  -> {OUT}', flush=True)