"""Dispatch policies (label → strategy) and strategy execution.

Dispatch policies:
    typed              config.STRATEGY_MAP (VAGUE→disambiguate, COMPLEX→decompose,
                       INSUFFICIENT→anchor_rewrite, NONE→rewrite)
    always:<strategy>  ignore the label
    none               no correction (handled in the loop)

Strategy execution returns either a single query (str) or a list of
sub-queries (decompose); the loop reranks each and RRF-fuses lists.
"""
from __future__ import annotations

from config import REFORMULATION_MAX_TOKENS, STRATEGY_MAP
from src.prompts import build_strategy_messages, parse_single_query, parse_subqueries


STRATEGY_MAP_ITER = {**STRATEGY_MAP, "COMPLEX": "decompose_iter"}


def dispatch(policy: str, label: str) -> str:
    if policy == "typed":
        return STRATEGY_MAP.get(label, "rewrite")
    if policy == "typed_iter":          # COMPLEX → answer-informed sequential decomposition
        return STRATEGY_MAP_ITER.get(label, "rewrite")
    if policy.startswith("always:"):
        return policy.split(":", 1)[1]
    raise ValueError(f"unknown dispatch policy {policy!r}")


def reformulate(strategy: str, question: str, pool, ctx) -> str | list[str]:
    msgs = build_strategy_messages(strategy, question, pool)
    raw = ctx.call(msgs, max_tokens=REFORMULATION_MAX_TOKENS, step="reformulate")
    if strategy == "decompose":
        return parse_subqueries(raw, question)
    return parse_single_query(raw, question)
