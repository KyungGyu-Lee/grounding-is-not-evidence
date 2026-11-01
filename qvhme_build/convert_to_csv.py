"""generated_qa.jsonl -> val_qvh_gqa.csv  (NExT-GQA / TEF CSV format).

Columns match data/val_sampled_500.csv so existing TEF eval code runs unchanged.

Answer convention: TEF's `answer` column is the TEXT of the correct option
(the pipeline does choices.index(normalize_option(answer))), NOT an integer.
So `answer` = options[letter]. The integer index you asked for is added as an
extra `answer_id` column (A=0..E=4). `gt_start`/`gt_end` are kept as extra
columns to build a QVH GT-grounding file later (spans are clip-relative).

Skips records with parse_ok=False (and malformed options / answer).

    python data/qvh_gqa/convert_to_csv.py
"""
import os
import csv
import json

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, 'generated_qa.jsonl')
OUT = os.path.join(HERE, 'val_qvh_gqa.csv')

LETTERS = ['A', 'B', 'C', 'D', 'E']
COLS = ['video_id', 'frame_count', 'width', 'height', 'question', 'answer',
        'qid', 'type', 'a0', 'a1', 'a2', 'a3', 'a4', 'duration', 'n_span',
        'evidence_cov', 'longest_span',
        # extra (beyond val_sampled_500): integer answer + clip-relative GT span
        'answer_id', 'gt_start', 'gt_end']


def main():
    rows = []
    n_total = n_skip = 0
    for line in open(GEN):
        n_total += 1
        r = json.loads(line)
        if not r.get('parse_ok'):
            n_skip += 1
            continue
        opts = r.get('options') or {}
        ans = r.get('answer')
        q = r.get('question')
        # validate: all 5 letters present, answer is a valid letter, question exists
        if (not q or ans not in LETTERS or
                any(L not in opts or not str(opts[L]).strip() for L in LETTERS)):
            n_skip += 1
            continue
        a = [str(opts[L]).strip() for L in LETTERS]
        answer_text = a[LETTERS.index(ans)]
        dur = float(r['duration'])
        span_len = float(r['gt_end']) - float(r['gt_start'])
        rows.append({
            'video_id': r['video_id'],
            'frame_count': 0,          # filled after video download (metadata only)
            'width': 0,
            'height': 0,
            'question': q,
            'answer': answer_text,     # TEXT (TEF convention)
            'qid': r['query_id'],
            'type': r.get('question_type', ''),   # 'causal' / 'temporal'
            'a0': a[0], 'a1': a[1], 'a2': a[2], 'a3': a[3], 'a4': a[4],
            'duration': dur,
            'n_span': 1,
            'evidence_cov': round(span_len / dur, 4) if dur else 0.0,
            'longest_span': span_len,
            'answer_id': LETTERS.index(ans),      # integer (A=0..E=4)
            'gt_start': r['gt_start'],
            'gt_end': r['gt_end'],
        })

    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f'total={n_total}  skipped(parse/malformed)={n_skip}  written={len(rows)}')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()