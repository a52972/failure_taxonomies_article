"""Retrieval state — what the first retrieval actually found.

The failure types (VAGUE / COMPLEX / INSUFFICIENT) describe the question; the
retrieval state describes how much of the required evidence the first retrieval
put in front of the generator. It is an analysis variable only (docs/20): it is
recorded for every question at run time and used for stratified analyses, the
oracle-state ceiling and the predictability labels, all from this one function.

    NONE_FOUND  no required evidence among the passages shown
    PARTIAL     some, but not all, of a multi-passage chain
    COMPLETE    all required evidence shown

Rule:
  * questions with annotated gold passages → coverage of those passages in the
    shown top-k (0 → NONE_FOUND, 1 → COMPLETE, otherwise PARTIAL);
  * questions without gold passages but with gold answers (single-hop pools
    such as PopQA/TriviaQA, constructed unanswerables) → COMPLETE iff a
    normalised gold answer string (≥ 2 characters) occurs in a shown passage,
    else NONE_FOUND. Such questions can never be PARTIAL.
"""
from __future__ import annotations

from config import RERANK_TOPK
from src.metrics import normalise_answer

STATES = ["NONE_FOUND", "PARTIAL", "COMPLETE"]


def retrieval_state(question, pool, k: int = RERANK_TOPK) -> str:
    from src.loop import gold_keys
    shown = pool[:k]
    n_gold = len(set(question.gold_pids))
    if n_gold:
        hit = len(gold_keys(question) & {p.pid for p, _ in shown})
        cov = min(1.0, hit / n_gold)
        return "NONE_FOUND" if cov == 0 else ("COMPLETE" if cov >= 1 else "PARTIAL")
    golds = [g for g in (normalise_answer(a) for a in (question.golden_answers or [])) if len(g) >= 2]
    if golds:
        text = " ".join(normalise_answer(f"{p.title} {p.text}") for p, _ in shown)
        if any(f" {g} " in f" {text} " for g in golds):
            return "COMPLETE"
    return "NONE_FOUND"
