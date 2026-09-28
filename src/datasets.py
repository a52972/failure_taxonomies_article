"""Common schema for every question in the FTC suite + JSONL (de)serialisation.

A `Question` carries its own passage pool (pool-based protocol) plus the
construction metadata that makes the failure class known by construction:
    cls          : failure class (config.CLASSES) or None before construction
    gold_pids    : pids of passages that carry the gold evidence (if known)
    hops         : decomposition hops for multi-hop sources
    construction : how the record was built (drops, pads, source split …)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True, slots=True)
class Passage:
    pid: str
    title: str
    text: str
    source_score: float | None = None


@dataclass(frozen=True, slots=True)
class Question:
    qid: str
    question: str
    golden_answers: list[str]
    task_type: str                       # short_form | multi_hop | true_false | mcq
    passages: list[Passage]
    choices: list[str] | None = None
    source: str = ""
    language: str = "en"
    cls: str | None = None
    gold_pids: list[str] = field(default_factory=list)
    hops: list[dict] = field(default_factory=list)
    construction: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def with_(self, **changes) -> "Question":
        d = asdict(self)
        d["passages"] = list(self.passages)
        d.update(changes)
        return Question(**d)


def question_to_dict(q: Question) -> dict:
    d = asdict(q)
    d["passages"] = [asdict(p) for p in q.passages]
    return d


def question_from_dict(d: dict) -> Question:
    d = dict(d)
    d["passages"] = [Passage(**p) for p in d.get("passages", [])]
    return Question(**d)


def write_jsonl(path: Path, questions: Iterable[Question]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(question_to_dict(q), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[Question]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield question_from_dict(json.loads(line))
