"""Metric audit (docs/20): a second model judges whether each answer is correct, so the
conclusions can be re-checked with a semantic metric besides substring accuracy, EM and F1.

The judge is Gemma-2 9B for Llama-3.1 8B answers (the judge is never the generator
being graded). Only distinct (question, answer) pairs of answerable questions are
judged; abstentions are skipped. Resumable.

    python scripts/judge_audit.py --split test
    → data/judge_audit/s1_<split>_verdicts.jsonl  {key, qid, answer, raw, verdict}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import ABSTAIN_TOKEN, DATA_DIR, RAW_RESULTS_DIR, SUITE_DIR  # noqa: E402

JUDGE = "gemma2-9b"
GENERATOR = "llama3-8b"
PROMPT = """You are grading a question-answering system.

Question: {q}
Reference answer(s): {gold}
System answer: {ans}

Does the system answer give the same answer as one of the references? Minor wording,
formatting or extra correct detail is fine; a different entity, date or value is not.
Reply with exactly one word: CORRECT or INCORRECT."""


def key(qid: str, answer: str) -> str:
    return hashlib.sha1(f"{qid}\n{answer.strip()}".encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--limit-pairs", type=int, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from run_study import ARMS, SYSTEMS, system_name
    from src.datasets import read_jsonl
    from src.llm_client import get_client
    from src.runner import system_tag

    pairs = {}
    for set_name in ARMS:
        Q = {q.qid: q for q in read_jsonl(SUITE_DIR / "s1" / f"{set_name}.{a.split}.jsonl")}
        for arm in ARMS[set_name]:
            f = RAW_RESULTS_DIR / f"s1_{a.split}" / f"{system_tag(system_name(SYSTEMS[arm]))}__{set_name}.{a.split}__{GENERATOR}.jsonl"
            if not f.exists():
                continue
            for r in map(json.loads, f.open(encoding="utf-8")):
                q = Q[r["qid"]]
                ans = (r["answer"] or "").strip()
                if r.get("error") or r["abstained"] or q.cls.startswith("INSUFFICIENT") or not ans \
                        or ABSTAIN_TOKEN in ans.upper():
                    continue
                pairs.setdefault(key(q.qid, ans), (q.qid, q.question, q.golden_answers, ans))
    out = DATA_DIR / "judge_audit" / f"s1_{a.split}_verdicts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in out.open()} if out.exists() else set()
    todo = [(k, v) for k, v in pairs.items() if k not in done][: a.limit_pairs]
    logging.info("distinct pairs %d; judged %d; to judge %d", len(pairs), len(done), len(todo))
    client = get_client()
    t0 = time.perf_counter()
    with out.open("a", encoding="utf-8") as f:
        for i, (k, (qid, q, gold, ans)) in enumerate(todo, 1):
            msg = [{"role": "user", "content": PROMPT.format(q=q, gold=" | ".join(gold[:6]), ans=ans[:400])}]
            txt = client.chat(msg, model=JUDGE, temperature=0.0, max_tokens=4).text.strip().upper()
            verdict = 1 if txt.startswith("CORRECT") else (0 if txt.startswith("INCORRECT") else None)
            f.write(json.dumps({"key": k, "qid": qid, "answer": ans, "raw": txt, "verdict": verdict},
                               ensure_ascii=False) + "\n")
            if i % 200 == 0:
                f.flush()
                el = time.perf_counter() - t0
                logging.info("  %d/%d (%.2fs/pair, ETA %.0f min)", i, len(todo), el / i, (len(todo) - i) * el / i / 60)
    logging.info("done")


if __name__ == "__main__":
    main()
