"""Quality gate: new regenerated (shuffled) QVH-ME vs the existing 788. Prints
the 7 gate metrics side by side + 20 questions for eyeballing. Read-only."""
import csv, json
import numpy as np
from collections import Counter

OLD = 'data/qvh_gqa/val_qvh_gqa_screened.csv'
NEW_SHUF = 'data/qvh_gqa_regen/val_qvh_gqa_en_shuf.csv'
NEW_GEN = 'data/qvh_gqa_regen/generated_qa_en.jsonl'
LET = 'ABCDE'


def stats(rows, label):
    n = len(rows)
    ai = Counter(int(r['answer_id']) for r in rows)
    qlen = [len(r['question'].split()) for r in rows]
    cl, dl, longest = [], [], 0
    for r in rows:
        aid = int(r['answer_id']); lens = [len(str(r[f'a{k}']).split()) for k in range(5)]
        cl.append(lens[aid]); dl += [lens[k] for k in range(5) if k != aid]
        longest += int(int(np.argmax(lens)) == aid)
    print(f"\n[{label}] N={n}")
    print(f"  정답위치: " + " ".join(f"{LET[k]} {ai.get(k,0)}({100*ai.get(k,0)/n:.0f}%)" for k in range(5)))
    print(f"  질문길이 단어: median {np.median(qlen):.0f} mean {np.mean(qlen):.1f}")
    print(f"  선택지길이: 정답 {np.mean(cl):.2f} 오답 {np.mean(dl):.2f} (정답-오답 {np.mean(cl)-np.mean(dl):+.2f}) | 정답최장 {longest/n:.2f}")


def main():
    old = list(csv.DictReader(open(OLD)))
    new = list(csv.DictReader(open(NEW_SHUF)))
    gen = [json.loads(l) for l in open(NEW_GEN)]
    ok = sum(g.get('parse_ok') for g in gen)
    print(f"### 파싱: 신규 parse_ok {ok}/{len(gen)} ({100*ok/len(gen):.1f}%) | 기존 100%")
    qt = Counter(g.get('question_type') for g in gen if g.get('parse_ok'))
    print(f"### question_type 신규 {dict(qt)} | 기존 causal 430 / temporal 360")
    stats(old, '기존 788')
    stats(new, '신규(셔플後)')
    # 20 side-by-side
    oldm = {r['qid']: r for r in old}
    print("\n### 질문 20개 나란히 (기존 vs 신규) ###")
    shown = 0
    for r in new:
        if r['qid'] in oldm and shown < 20:
            print(f"[qid {r['qid']}]")
            print(f"  기존: {oldm[r['qid']]['question']}")
            print(f"  신규: {r['question']}")
            shown += 1


if __name__ == '__main__':
    main()
