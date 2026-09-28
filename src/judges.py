"""Diagnosis judges — every judge maps (question, query, pool) → label.

    oracle      construction label (config.ORACLE_LABEL[question.cls])
    random      uniform over LABELS (seeded)
    constant:X  always X
    llm         3-way prompt to the generator (or another Ollama model)
"""
from __future__ import annotations

import random
from typing import Protocol

from config import DIAGNOSIS_MAX_TOKENS, LABELS, ORACLE_LABEL
from src.datasets import Question
from src.prompts import build_diagnosis_messages


class Judge(Protocol):
    name: str
    def label(self, question: Question, query: str, pool, ctx) -> str: ...


def parse_label(raw: str) -> str:
    up = (raw or "").upper()
    for tok in ("INSUFFICIENT", "COMPLEX", "VAGUE"):
        if tok in up:
            return tok
    return "VAGUE"


class OracleJudge:
    name = "oracle"

    def label(self, question, query, pool, ctx) -> str:
        return ORACLE_LABEL.get(question.cls or "", "NONE")


class RandomJudge:
    name = "random"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def label(self, question, query, pool, ctx) -> str:
        # deterministic per question so policies are comparable across runs
        r = random.Random(f"{self.rng.random()}|{question.qid}")
        return r.choice(LABELS)


class ConstantJudge:
    def __init__(self, label: str):
        self._label = label
        self.name = f"constant:{label}"

    def label(self, question, query, pool, ctx) -> str:
        return self._label


class LLMJudge:
    """`ctx` is the loop's call-hook: ctx.call(messages, max_tokens, step, model)."""
    name = "llm"

    def __init__(self, model: str | None = None):
        self.model = model
        if model:
            self.name = f"llm:{model}"

    def label(self, question, query, pool, ctx) -> str:
        raw = ctx.call(build_diagnosis_messages(question.question, query, pool[:5]),
                       max_tokens=DIAGNOSIS_MAX_TOKENS, step="diagnose",
                       model=self.model)
        return parse_label(raw)


def make_judge(spec: str, seed: int = 42) -> Judge:
    if spec == "oracle":
        return OracleJudge()
    if spec == "random":
        return RandomJudge(seed)
    if spec.startswith("constant:"):
        return ConstantJudge(spec.split(":", 1)[1])
    if spec == "llm":
        return LLMJudge()
    if spec.startswith("llm:"):
        return LLMJudge(spec.split(":", 1)[1])
    raise ValueError(f"unknown judge {spec!r}")
