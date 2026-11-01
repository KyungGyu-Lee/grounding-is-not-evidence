#!/usr/bin/env python
"""Fail if any adapter reads `predicted_answer` inside predict_grounding,
other than Grounded-VideoLLM's documented GVLLM_ANSCOND branch.

Grounding is question-conditioned by design: step1 passes None, but an adapter
that read the argument anyway would silently answer-condition its own windows.
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'supportgain', 'models')
ALLOWED = {('grounded_videollm', 'self.anscond')}

bad = []
checked = 0
for d in sorted(os.listdir(MODELS)):
    p = os.path.join(MODELS, d, 'adapter.py')
    if not os.path.isfile(p):
        continue
    checked += 1
    src = open(p, encoding='utf-8').read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == 'predict_grounding'):
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id == 'predicted_answer' and \
                    isinstance(n.ctx, ast.Load):
                line = src.split('\n')[n.lineno - 1].strip()
                if d == 'grounded_videollm' and 'self.anscond' in line:
                    continue
                bad.append(f'{d}/adapter.py:{n.lineno}: {line}')

if bad:
    print('ANSWER-LEAK CHECK FAILED — predict_grounding reads the answer text:')
    print('\n'.join('  ' + b for b in bad))
    sys.exit(1)
print(f'answer-leak check OK ({checked} adapters; the only read is '
      "grounded_videollm's GVLLM_ANSCOND branch)")
