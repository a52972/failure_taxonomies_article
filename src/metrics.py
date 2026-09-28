"""Evaluation metrics for RAG QA.

The metric choices follow the CRAG (Yan et al. 2024) and RQ-RAG (Chan et
al. 2024) papers so that our numbers are directly comparable with the
literature:

- Short-form (PopQA, TriviaQA, NQ): **Accuracy(Contains)** and F1.
  CRAG's Table 1 uses Accuracy where "contains any gold string" counts
  as correct (see Self-RAG's original eval script). We report F1 too.
- Multi-hop (HotpotQA, 2Wiki): **Exact Match** and F1 following Yang et
  al. 2018 (the HotpotQA paper's standard).
- True/False (PubHealth): **Accuracy(Exact)** — the answer must equal
  "true" or "false" (case-insensitive).
- MCQ (ARC-Challenge): **Accuracy(Exact)** — the answer must contain the
  correct label letter (A / B / C / D).

For retrieval, we optionally report MRR and Recall@K over the *rerank
input pool*, but since our per-question pool is fixed and small, these
are more diagnostic than headline metrics.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable


# ── Normalisation (SQuAD-style, matches Self-RAG's eval) ──────────────────

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalise_answer(s: str) -> str:
    """Lowercase, strip punctuation, drop articles, collapse whitespace."""
    if s is None:
        return ""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    return s


# ── QA metrics ────────────────────────────────────────────────────────────

def exact_match(prediction: str, golds: list[str]) -> float:
    if not golds:
        return 0.0
    p = normalise_answer(prediction)
    return float(any(p == normalise_answer(g) for g in golds))


def accuracy_contains(prediction: str, golds: list[str]) -> float:
    """CRAG's short-form 'accuracy' — gold appears anywhere in prediction."""
    if not golds:
        return 0.0
    p = normalise_answer(prediction)
    if not p:
        return 0.0
    for g in golds:
        gn = normalise_answer(g)
        if gn and gn in p:
            return 1.0
    return 0.0


def f1_score(prediction: str, golds: list[str]) -> float:
    """Best token-level F1 across all gold answers (SQuAD convention)."""
    if not golds:
        return 0.0
    p_tokens = normalise_answer(prediction).split()
    if not p_tokens:
        return 0.0
    best = 0.0
    for g in golds:
        g_tokens = normalise_answer(g).split()
        if not g_tokens:
            continue
        common = Counter(p_tokens) & Counter(g_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(p_tokens)
        recall = num_same / len(g_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best


def accuracy_exact(prediction: str, golds: list[str]) -> float:
    """Alias of exact_match — kept as separate name to reflect the paper's
    protocol distinction between short-form 'contains' and closed-set
    'exact' accuracy."""
    return exact_match(prediction, golds)


# ── MCQ answer extraction ────────────────────────────────────────────────

_MCQ_LABEL_RE = re.compile(r"\b([A-Da-d])\b")


def mcq_extract_label(prediction: str) -> str:
    """Extract the first A/B/C/D letter from a free-form prediction."""
    m = _MCQ_LABEL_RE.search(prediction or "")
    return m.group(1).upper() if m else ""


def accuracy_mcq(prediction: str, gold_label: str) -> float:
    return float(mcq_extract_label(prediction) == gold_label.upper())


# ── True/False answer extraction ─────────────────────────────────────────

def true_false_extract(prediction: str) -> str:
    p = normalise_answer(prediction)
    if not p:
        return ""
    if p.startswith(("true", "yes", "supports", "supported")):
        return "true"
    if p.startswith(("false", "no", "refutes", "refuted")):
        return "false"
    for token in p.split():
        if token in ("true", "yes", "supports"):
            return "true"
        if token in ("false", "no", "refutes"):
            return "false"
    return ""


def accuracy_true_false(prediction: str, golds: list[str]) -> float:
    extracted = true_false_extract(prediction)
    if not extracted:
        return 0.0
    return float(any(extracted == normalise_answer(g) for g in golds))


# ── Retrieval metrics ────────────────────────────────────────────────────

def mrr_at_k(
    ranked_pids: Iterable[str],
    relevant_pids: set[str],
    k: int = 10,
) -> float:
    """Mean Reciprocal Rank at K for a single query."""
    for rank, pid in enumerate(list(ranked_pids)[:k], start=1):
        if pid in relevant_pids:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    ranked_pids: Iterable[str],
    relevant_pids: set[str],
    k: int = 10,
) -> float:
    if not relevant_pids:
        return 0.0
    top_k = set(list(ranked_pids)[:k])
    return len(top_k & relevant_pids) / len(relevant_pids)


# ── Dispatch by task ─────────────────────────────────────────────────────

def score_prediction(
    task_type: str,
    prediction: str,
    golds: list[str],
) -> dict[str, float]:
    """Return a dict of every applicable metric for this task type."""
    out: dict[str, float] = {}
    if task_type in ("short_form", "multi_hop"):
        out["exact_match"] = exact_match(prediction, golds)
        out["accuracy_contains"] = accuracy_contains(prediction, golds)
        out["f1"] = f1_score(prediction, golds)
    elif task_type == "true_false":
        out["accuracy_exact"] = accuracy_true_false(prediction, golds)
    elif task_type == "mcq":
        # golds is a list containing a single label like "C"
        gold_label = golds[0] if golds else ""
        out["accuracy_exact"] = accuracy_mcq(prediction, gold_label)
    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")
    return out
