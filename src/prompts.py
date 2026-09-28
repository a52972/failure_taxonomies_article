"""Prompt templates. Domain-neutral; task-typed by `task_type` only."""
from __future__ import annotations

from config import ABSTAIN_TOKEN
from src.datasets import Passage, Question


def format_passages(passages, max_chars: int = 500) -> str:
    if not passages:
        return "(no context available)"
    lines = []
    for i, item in enumerate(passages, start=1):
        p = item[0] if isinstance(item, tuple) else item
        title = (p.title or "").strip() or "Untitled"
        text = (p.text or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        lines.append(f"[{i}] {title}: {text}")
    return "\n".join(lines)


_SYS_QA_SHORT = (
    "You are a question-answering assistant. Answer using ONLY the provided "
    "context. Give the SHORTEST possible answer — a single entity or a few "
    "words. Do not repeat the question or explain. If the context does not "
    "contain the answer, output your best guess in one or two words.")
_SYS_QA_MULTIHOP = (
    "You are a multi-hop question-answering assistant. The question requires "
    "combining information from multiple passages. Reason silently, then "
    "output only a concise final answer (a short phrase). Do not explain.")
_SYS_TRUE_FALSE = (
    "You are a fact-checking assistant. Given a claim and supporting context, "
    "decide whether the claim is TRUE or FALSE. Output exactly one word: TRUE or FALSE.")
_SYS_MCQ = (
    "You are a multiple-choice assistant. Read the question, the options, and "
    "the context. Output ONLY the letter of the correct option (A, B, C, or D).")

_ABSTAIN_SUFFIX = (
    f" If the context does not contain enough information to answer, output "
    f"exactly {ABSTAIN_TOKEN}.")


def build_generator_messages(question: Question, passages, *, allow_abstain: bool = False) -> list[dict]:
    ctx = format_passages(passages)
    tt = question.task_type
    if tt == "short_form":
        sys = _SYS_QA_SHORT
        user = f"Context:\n{ctx}\n\nQuestion: {question.question}\n\nAnswer:"
    elif tt == "multi_hop":
        sys = _SYS_QA_MULTIHOP
        user = f"Context:\n{ctx}\n\nQuestion: {question.question}\n\nAnswer:"
    elif tt == "true_false":
        sys = _SYS_TRUE_FALSE
        user = f"Context:\n{ctx}\n\nClaim: {question.question}\n\nIs this claim TRUE or FALSE? Answer:"
    elif tt == "mcq":
        sys = _SYS_MCQ
        user = (f"Context:\n{ctx}\n\nQuestion: {question.question}\n\n"
                f"Options:\n" + "\n".join(question.choices or []) + "\n\nAnswer:")
    else:
        raise ValueError(tt)
    if allow_abstain and tt in ("short_form", "multi_hop"):
        sys = sys.replace(
            " If the context does not contain the answer, output your best guess in one or two words.", "")
        sys += _ABSTAIN_SUFFIX
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def build_no_retrieval_messages(question: Question) -> list[dict]:
    tt = question.task_type
    if tt == "short_form":
        sys = _SYS_QA_SHORT.replace("using ONLY the provided context", "using your own knowledge")
        user = f"Question: {question.question}\n\nAnswer:"
    elif tt == "multi_hop":
        sys = _SYS_QA_MULTIHOP.replace(
            "The question requires combining information from multiple passages. ", "")
        user = f"Question: {question.question}\n\nAnswer:"
    elif tt == "true_false":
        sys = _SYS_TRUE_FALSE.replace("and supporting context, ", "")
        user = f"Claim: {question.question}\n\nAnswer:"
    elif tt == "mcq":
        sys = _SYS_MCQ.replace("and the context. ", "")
        user = (f"Question: {question.question}\n\nOptions:\n"
                + "\n".join(question.choices or []) + "\n\nAnswer:")
    else:
        raise ValueError(tt)
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ── diagnosis (3-way) ─────────────────────────────────────────────────

DIAGNOSIS_PROMPT = """\
You are a retrieval-failure classifier. A question was searched and the \
passages below came back, but they do not add up to a confident answer. \
Decide WHY, using two checks in order.

Question: {question}
Search terms used: {reformulated_query}

Top {n} passages (score in [0,1]; ~0.5 = noise floor):
{passages_summary}

CHECK 1 — Is the fact the question asks for actually PRESENT in the passages?
  If NO (passages are about the right topic or entity but the specific fact, \
date, name or number asked for is not there, OR they are about other \
entities/events) → answer INSUFFICIENT.

CHECK 2 — If YES, why is it still not a confident answer?
  If the question mentions two or more entities, a relation between them, a \
"which/who ... of the ... that ..." chain, a comparison, or a bridge \
("the director of the film that ..."), and the passages each cover one link \
of that chain → answer COMPLEX.
  If the question is short or under-specified and could refer to different \
things (several people, works, places or time periods with the same name; \
missing year/edition/version), and the passages cover DIFFERENT readings → \
answer VAGUE.

Examples:
Q: "Who is the spouse of the performer of the album Green?" — passages: one \
about the album Green (performer Steve Hillage), one about Steve Hillage's \
career, one about the band Gong → COMPLEX
Q: "Where is the TV show The Ranch located?" — passages: one about the show's \
filming studio, one about the fictional town, one about a different show → VAGUE
Q: "In what year did Gregorio Selser move to Mexico?" — passages: Selser's \
biography without the move, two other journalists → INSUFFICIENT
Q: "Which country is larger, Canada or Brazil?" — passages: Canada's area, \
Brazil's population, a list of countries by GDP → COMPLEX

Respond with ONLY one word: INSUFFICIENT, COMPLEX, or VAGUE."""


def summarise_pool(pool, n_chars: int = 200) -> str:
    lines = []
    for i, (p, s) in enumerate(pool, start=1):
        title = (p.title or "Untitled").strip()
        snippet = (p.text or "").strip().replace("\n", " ")[:n_chars]
        lines.append(f"{i}. [score={s:.2f}] {title}: {snippet}")
    return "\n".join(lines)


def build_diagnosis_messages(question: str, reformulated_query: str, pool) -> list[dict]:
    return [{"role": "user", "content": DIAGNOSIS_PROMPT.format(
        question=question, reformulated_query=reformulated_query,
        n=len(pool), passages_summary=summarise_pool(pool))}]


# ── strategies ────────────────────────────────────────────────────────

_REWRITE = """\
You are a query reformulator for a document-retrieval system.
Rewrite the question so a retriever is more likely to find passages containing \
the answer. Preserve the information need; expand vague references; add likely \
search keywords (entity names, category words). Do NOT invent facts.

Question: {question}

Respond in EXACTLY this format (one line):
QUERY: <rewritten query>"""

_DECOMPOSE = """\
You are a query decomposer for a document-retrieval system.
The question is complex: it needs several facts or multi-step reasoning. \
Decompose it into 2 or 3 simpler, self-contained sub-queries that together \
cover the full information need. Each sub-query must be answerable on its own. \
No boolean operators. Do not add sub-queries not implied by the question.

Question: {question}

Respond in EXACTLY this format:
SUBQUERY_1: <first>
SUBQUERY_2: <second>
SUBQUERY_3: <third, optional>"""

_DISAMBIGUATE = """\
You are a query disambiguator for a document-retrieval system.
The question is ambiguous; initial retrieval returned passages about different \
readings. Using the anchor passages, commit to the MOST LIKELY intended reading \
and reformulate the question to be specific to it. Add distinguishing keywords \
or entity names from the anchors. Do NOT hedge with "or". Do NOT copy passage text.

Question: {question}

Anchor passages (top from initial retrieval):
{anchors}

Respond in EXACTLY this format (one line):
QUERY: <disambiguated query>"""

_ANCHOR_REWRITE = """\
You are a query reformulator for a document-retrieval system.
The question is specific but the initial retrieval missed the needed fact. \
Use the anchor passages only as vocabulary hints (related names, terms, \
aliases) and rewrite the question with alternative phrasings and search \
keywords so a retriever can find the missing passage. Preserve the information \
need. Do NOT invent facts.

Question: {question}

Anchor passages (top from initial retrieval):
{anchors}

Respond in EXACTLY this format (one line):
QUERY: <rewritten query>"""



_DECOMPOSE_ITER = """\
QUESTION: {question}

Decompose the QUESTION above into TWO sub-questions that must be answered in \
order. Both sub-questions must be about the entities named in the QUESTION — \
copy those names exactly; never introduce other topics. Q1 must be answerable \
on its own. Q2 may use the placeholder [A1] for the answer to Q1. For a \
comparison between two entities, Q1 asks about the first and Q2 about the \
second (no placeholder). Do not invent facts.

Format (nothing else):
Q1: <sub-question about the QUESTION>
Q2: <sub-question about the QUESTION>"""


def build_decompose_iter_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": _DECOMPOSE_ITER.format(question=question)}]


_STOP = set("the a an of in on at to for by with and or is was were who what when where which how did "
            "does do that this from as its his her their be been are".split())


def _content_tokens(s: str) -> set[str]:
    import re
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOP and len(w) > 2}


def on_topic(question: str, sub: str, min_overlap: float = 0.2) -> bool:
    """Guard against derailed reformulations: a sub-query must share at least
    `min_overlap` of its content tokens with the question (or contain [A1])."""
    if "[A1]" in (sub or ""):
        return True
    b = _content_tokens(sub)
    if not b:
        return False
    return len(_content_tokens(question) & b) / len(b) >= min_overlap


def parse_q1_q2(text: str, fallback: str) -> tuple[str, str]:
    q1 = q2 = ""
    for line in (text or "").splitlines():
        s = line.strip()
        if s.upper().startswith("Q1:"):
            q1 = s[3:].strip()
        elif s.upper().startswith("Q2:"):
            q2 = s[3:].strip()
    if not on_topic(fallback, q1):
        q1 = fallback
    if not on_topic(fallback, q2):
        q2 = fallback
    return (q1 or fallback), (q2 or fallback)


def build_subanswer_messages(sub_question: str, passages) -> list[dict]:
    ctx = format_passages(passages)
    return [{"role": "system", "content": _SYS_QA_SHORT},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {sub_question}\n\nAnswer:"}]


def _anchors(pool, k: int = 3, n_chars: int = 220) -> str:
    lines = []
    for i, (p, _s) in enumerate(pool[:k], start=1):
        title = (p.title or "Untitled").strip()
        snippet = (p.text or "").strip().replace("\n", " ")[:n_chars]
        lines.append(f"[{i}] {title}: {snippet}")
    return "\n".join(lines) or "(no anchors)"


def build_strategy_messages(strategy: str, question: str, pool) -> list[dict]:
    if strategy == "rewrite":
        c = _REWRITE.format(question=question)
    elif strategy == "decompose":
        c = _DECOMPOSE.format(question=question)
    elif strategy == "disambiguate":
        c = _DISAMBIGUATE.format(question=question, anchors=_anchors(pool))
    elif strategy == "anchor_rewrite":
        c = _ANCHOR_REWRITE.format(question=question, anchors=_anchors(pool))
    else:
        raise ValueError(strategy)
    return [{"role": "user", "content": c}]


def parse_single_query(text: str, fallback: str) -> str:
    out = ""
    for line in (text or "").splitlines():
        line = line.strip()
        if line.upper().startswith("QUERY:"):
            out = line.split(":", 1)[1].strip()
            break
    if not out:
        t = (text or "").strip().splitlines()
        out = t[0].strip() if t else ""
    return out if (out and on_topic(fallback, out)) else fallback


def parse_subqueries(text: str, fallback: str) -> list[str]:
    subs = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line.upper().startswith("SUBQUERY"):
            if ":" in line:
                rhs = line.split(":", 1)[1].strip()
                if rhs and on_topic(fallback, rhs):
                    subs.append(rhs)
    return subs[:3] or [fallback]
