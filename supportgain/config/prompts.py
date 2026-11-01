"""Default prompt templates.

Models that need different formatting can override these inside their own
adapter (see models/base.py). The defaults follow the format used in our
prior Qwen2.5-VL experiments.
"""

QA_PROMPT = (
    "Question: {question}\n"
    "Options:\n"
    "(A) {a0}\n"
    "(B) {a1}\n"
    "(C) {a2}\n"
    "(D) {a3}\n"
    "(E) {a4}\n"
    "Answer with the letter only."
)

# Shared grounding CORE — identical for every model (the part being compared).
# Each adapter appends only its own native output directive / trigger token, so
# the task + question stays byte-identical across models while each model emits
# its span in its native form (free-text / reg token / quantized time token).
GROUNDING_CORE = (
    "Identify the single continuous time segment in the video that contains "
    "the visual evidence needed to answer the question.\n"
    "Question: {question}"
)


def normalize_question(q):
    """Capitalize the first letter and ensure a trailing '?'.

    NExT-GQA raw questions are all-lowercase with no '?'. We normalize so every
    model sees the same well-formed question (matches VideoMind's convention).
    """
    q = str(q).strip()
    if not q:
        return q
    q = q[0].upper() + q[1:]
    if not q.endswith('?'):
        q = q + '?'
    return q


def normalize_option(t):
    """Capitalize the first letter of an option / answer text.

    Options AND the answer must be normalized with the SAME rule so that the
    answer-to-option index lookup (choices.index(answer)) still matches.
    """
    t = str(t).strip()
    return (t[0].upper() + t[1:]) if t else t


def build_qa_prompt(question, choices):
    a0, a1, a2, a3, a4 = choices
    return QA_PROMPT.format(question=question, a0=a0, a1=a1, a2=a2, a3=a3, a4=a4)


def build_grounding_prompt(question, duration=None):
    """Question-conditioned grounding prompt for free-text models (e.g. Qwen).

    Built as: shared GROUNDING_CORE  +  optional duration anchor  +  the
    free-text output directive. Other models reuse GROUNDING_CORE directly and
    append their own native directive/trigger instead of the format line.

    The predicted answer is intentionally NOT used (no answer leakage; the same
    question is grounded by every model, aligned with the NExT-GQA GT span).
    Passing `duration` anchors the valid range and reduces out-of-range/clamping.
    """
    core = GROUNDING_CORE.format(question=question.strip())
    dur = f"\nThe video is {duration:.1f} seconds long." if duration else ""
    return (core + dur +
            "\nReturn exactly one segment in absolute seconds.\n"
            "Format only: start=X.X, end=Y.Y")
