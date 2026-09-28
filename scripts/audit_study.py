"""Construction audit of the study sets (docs/20), run before any generation.

    python scripts/audit_study.py      → data/suite/s1/audit.json

Checks, per set and class (no LLM calls):
  A1  evidence in the corpus: COMPLEX — every gold passage of the chain is a corpus
      passage; ANSWERABLE and VAGUE — at least one answer-bearing corpus passage;
      VAGUE — evidence for at least two distinct answers.
  A2  unanswerable constructions: after exclusion and scrubbing, no passage among
      the top-100 hybrid candidates contains a gold answer, when querying with the
      question and with each gold answer string itself.
  A3  no base question id is shared between sets or with the pilot.
  A4  short answers (≤ 4 characters), which can match by chance.
  A5  COMPLEX: the answer string occurs in a corpus passage outside the annotated chain
      (whole corpus: dominated by common strings such as yes/no, years, countries).
  A5p COMPLEX: the same within the question's own passages (the pilot's measure).

    python scripts/audit_study.py --pool-only   → adds A5p to an existing audit.json
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CORPUS_DIR, DATA_DIR, SUITE_DIR  # noqa: E402
from src.corpus import FAMILY_FOR_CLASS, get_corpus, text_hash  # noqa: E402
from src.datasets import read_jsonl  # noqa: E402
from src.metrics import normalise_answer  # noqa: E402

SETS = [("MH", "test"), ("MIXED", "test"), ("MH", "dev"), ("MIXED", "dev"), ("STATE_TRAIN", "train")]


def golds(q, n=2):
    return [g for g in (normalise_answer(a) for a in q.golden_answers or []) if len(g) >= n]


def pool_checks() -> dict:
    """A5p per set and source (no corpus needed)."""
    out: dict = {}
    for name, split in SETS:
        tot, hit, yn = Counter(), Counter(), Counter()
        for q in read_jsonl(SUITE_DIR / "s1" / f"{name}.{split}.jsonl"):
            if q.cls != "COMPLEX":
                continue
            gs = golds(q)
            tot[q.source] += 1
            yn[q.source] += bool(set(gs) & {"yes", "no"})
            gold = set(q.gold_pids)
            hit[q.source] += any(f" {g} " in f" {normalise_answer(p.title + ' ' + p.text)} "
                                 for g in gs for p in q.passages if p.pid not in gold)
        if tot:
            out[f"{name}.{split}"] = {"n": sum(tot.values()), "A5p_answer_outside_chain_in_pool": sum(hit.values()),
                                      "by_source": {s: {"n": tot[s], "A5p": hit[s], "yes_no_answer": yn[s]} for s in tot}}
    return out


def main():
    used = set(json.load(open(DATA_DIR / "pilot_used_qids.json"))["qids"])
    qs_by_fam: dict[str, list] = defaultdict(list)
    seen: dict[str, str] = {}
    report: dict = {"A3_overlap": {"with_pilot": 0, "between_sets": 0}}
    for name, split in SETS:
        for q in read_jsonl(SUITE_DIR / "s1" / f"{name}.{split}.jsonl"):
            b = q.qid.split("#")[0]
            report["A3_overlap"]["with_pilot"] += b in used
            if b in seen and seen[b] != name + split:
                report["A3_overlap"]["between_sets"] += 1
            seen[b] = name + split
            qs_by_fam[FAMILY_FOR_CLASS[q.cls]].append((f"{name}.{split}", q))

    stats: dict = defaultdict(lambda: defaultdict(Counter))
    for fam, items in qs_by_fam.items():
        logging.info("[%s] %d questions: indexing answer strings …", fam, len(items))
        rows = [json.loads(l) for l in (CORPUS_DIR / fam / "passages.jsonl").open(encoding="utf-8")]
        hashes = {r["pid"] for r in rows}
        texts = [normalise_answer(f"{r['title']} {r['text']}") for r in rows]
        corpus = get_corpus(fam)
        # inverted index of answer strings → corpus rows, only for the answers we need
        # (token n-gram lookup: an answer "occurs" when it matches whole tokens)
        needed = {g for _, q in items for g in golds(q)}
        max_len = min(12, max((len(g.split()) for g in needed), default=1))
        where: dict[str, set[int]] = defaultdict(set)
        for i, t in enumerate(texts):
            toks = t.split()
            for n in range(1, max_len + 1):
                for j in range(len(toks) - n + 1):
                    g = " ".join(toks[j:j + n])
                    if g in needed:
                        where[g].add(i)
        logging.info("[%s] checking questions …", fam)
        for i, (set_name, q) in enumerate(items, 1):
            if i % 250 == 0:
                logging.info("[%s] %d/%d", fam, i, len(items))
            s = stats[set_name][q.cls]
            s["n"] += 1
            gs = golds(q)
            s["A4_short_answer"] += any(len(g) <= 4 for g in golds(q, 1))
            if q.cls == "COMPLEX":
                gp = [p for p in q.passages if p.pid in set(q.gold_pids)]
                s["A1_chain_in_corpus"] += all(text_hash(p.title, p.text) in hashes for p in gp)
                chain = {text_hash(p.title, p.text) for p in gp}
                outside = {rows[i]["pid"] for g in gs for i in where[g]} - chain
                s["A5_answer_outside_chain"] += bool(outside)
            elif q.cls in ("ANSWERABLE", "VAGUE"):
                s["A1_answer_in_corpus"] += any(where[g] for g in gs)
                if q.cls == "VAGUE":
                    s["A1_two_answers_in_corpus"] += sum(bool(where[g]) for g in set(golds(q, 3))) >= 2
            else:   # unanswerable
                cons = q.construction
                ex = set(cons.get("exclude_hashes") or [])
                leak = False
                for query in [q.question, *gs[:3]]:
                    for p, _ in corpus.search(query, k=100, cand=100, exclude_hashes=ex, scrub_golds=q.golden_answers):
                        t = f" {normalise_answer(p.title + ' ' + p.text)} "
                        if any(f" {g} " in t for g in golds(q)):
                            leak = True
                            break
                    if leak:
                        break
                s["A2_answer_retrievable"] += leak
    report["sets"] = {k: {c: dict(v) for c, v in d.items()} for k, d in stats.items()}
    report["A5p"] = pool_checks()
    (SUITE_DIR / "s1" / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if "--pool-only" in sys.argv:
        path = SUITE_DIR / "s1" / "audit.json"
        rep = json.loads(path.read_text())
        rep["A5p"] = pool_checks()
        path.write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep["A5p"], indent=2))
    else:
        main()
