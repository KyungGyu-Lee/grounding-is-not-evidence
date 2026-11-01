"""5-choice MCQ output parser."""
import re

LABELS = ['A', 'B', 'C', 'D', 'E']


def parse_answer(text, choices):
    """Return letter index 0-4 or -1 on parse fail."""
    if not text:
        return -1
    t = text.strip()
    for i, lbl in enumerate(LABELS):
        if f'({lbl})' in t or f'({lbl.lower()})' in t:
            return i
    m = re.match(r'\s*([ABCDE])\b', t, re.IGNORECASE)
    if m:
        return LABELS.index(m.group(1).upper())
    m = re.search(r'(?:answer|option|choice)\s*(?:is|:|=)\s*\(?([ABCDE])\)?',
                  t, re.IGNORECASE)
    if m:
        return LABELS.index(m.group(1).upper())
    m = re.search(r'\b([ABCDE])\b', t)
    if m:
        return LABELS.index(m.group(1).upper())
    tl = t.lower()
    for i, c in enumerate(choices):
        if c and c.strip().lower() in tl:
            return i
    return -1
