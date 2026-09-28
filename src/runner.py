"""Run one system over one question set; append per-question rows (resumable).

Output: results/raw/<exp>/<system_tag>__<set>.<split>__<generator>.jsonl
Each row: qid, set, cls, source, task_type, system, generator, answer, abstained,
metrics (exact_match / accuracy_contains / accuracy_exact / f1 / in_acc),
retrieval state at t=0, label/strategy, gold coverage before and after
correction, cost, latency, and the full trace.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from config import ABSTAIN_TOKEN, RAW_RESULTS_DIR, SUITE_DIR
from src.datasets import Question, read_jsonl
from src.loop import CorrectiveLoop
from src.metrics import score_prediction

_log = logging.getLogger(__name__)


def system_tag(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9=+:_.-]+", "_", name).replace("|", "_")


def score(question: Question, answer: str, abstained: bool) -> dict:
    if abstained or not question.golden_answers:
        m = {"exact_match": 0.0, "accuracy_contains": 0.0, "f1": 0.0, "accuracy_exact": 0.0}
    else:
        m = {"exact_match": 0.0, "accuracy_contains": 0.0, "f1": 0.0, "accuracy_exact": 0.0}
        m.update(score_prediction(question.task_type, answer, question.golden_answers))
    if question.task_type in ("short_form", "multi_hop"):
        m["in_acc"] = m["accuracy_contains"]
    else:
        m["in_acc"] = m["accuracy_exact"]
    # hallucination on unanswerable: produced a non-abstain answer
    m["answered"] = 0.0 if abstained else 1.0
    return m


def run_file(loop: CorrectiveLoop, set_name: str, split: str, *, exp: str, tag: str = "s1",
             limit: int | None = None, generator: str | None = None) -> Path:
    path = SUITE_DIR / tag / f"{set_name}.{split}.jsonl"
    questions = list(read_jsonl(path))
    if limit:
        questions = questions[:limit]
    gen = generator or loop.generator
    out = RAW_RESULTS_DIR / exp / f"{system_tag(loop.name)}__{set_name}.{split}__{gen}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("error"):          # rows with a technical error are re-run; the
                done.add(r["qid"])          # later row replaces the earlier one when loaded
    todo = [q for q in questions if q.qid not in done]
    _log.info("%s | %s.%s | %d done, %d todo", loop.name, set_name, split, len(done), len(todo))
    t0 = time.perf_counter()
    with out.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, start=1):
            try:
                r = loop.run(q)
                err = ""
            except Exception as exc:  # keep going; record the error
                _log.exception("error on %s", q.qid)
                from src.loop import RunResult
                r, err = RunResult(answer=""), f"{type(exc).__name__}: {exc}"
            it0 = (r.trace.get("iterations") or [{}])[0]
            row = {
                "qid": q.qid, "set": set_name, "cls": q.cls, "split": split, "source": q.source,
                "language": q.language, "task_type": q.task_type,
                "system": loop.name, "generator": gen,
                "answer": r.answer, "abstained": r.abstained,
                **score(q, r.answer, r.abstained),
                "state_t0": it0.get("state"), "label_t0": it0.get("label"),
                "strategy_t0": it0.get("strategy"),
                "n_iterations": r.trace.get("n_iterations", 0),
                "gold_rank_t0": it0.get("gold_rank"),
                "gold_cov_t0": it0.get("gold_cov"),
                "final_gold_rank": r.trace.get("final_gold_rank"),
                "final_gold_cov": r.trace.get("final_gold_cov"),
                "n_llm_calls": r.n_llm_calls, "n_llm_calls_uncached": r.n_llm_calls_uncached,
                "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                "latency_ms": round(r.latency_ms, 1), "step_ms": {k: round(v, 1) for k, v in r.step_ms.items()},
                "error": err, "trace": r.trace,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                el = time.perf_counter() - t0
                _log.info("  %d/%d  (%.1fs/q)", i, len(todo), el / i)
    return out
