"""Build the question sets and family corpora of the study (docs/20).

    python scripts/build_study.py select      # choose questions → data/suite/s1/*.jsonl (no corpus needed)
    python scripts/build_study.py corpus      # data/corpus_s1/<family>: pilot corpus ∪ study pools
    python scripts/build_study.py finalise    # corpus-wide answer exclusion for unanswerables + checks
    python scripts/build_study.py all

Every question is new: base qids listed in data/pilot_used_qids.json are excluded.
Roles are filled from one deterministic shuffle per source, in a fixed order, so
the sets are disjoint by construction and a question and its unanswerable twin
can never be split across roles (twins are built from distinct parents anyway).

Sets written to data/suite/s1/:
    MH.test.jsonl        COMPLEX, 500 per multi-hop source                 (H1, H3)
    MIXED.test.jsonl     ANSWERABLE 90 + INSUFFICIENT_SINGLE 70 per single-hop source,
                         COMPLEX 90 + INSUFFICIENT_HOP 70 per multi-hop source,
                         VAGUE 150                                         (H2)
    MH.dev.jsonl         COMPLEX, 100 per multi-hop source
    MIXED.dev.jsonl      same classes as MIXED.test at about one third of the size
    STATE_TRAIN.train.jsonl  rows for the secondary predictability analysis (no generation)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CORPUS_DIR, DATA_DIR, SEED, SUITE_DIR  # noqa: E402
from src.corpus import FamilyCorpus, text_hash  # noqa: E402
from src.datasets import Passage, read_jsonl, write_jsonl  # noqa: E402
from src.ftc import loaders  # noqa: E402
from src.ftc.construct import _vague_ok, passages_with_gold_string  # noqa: E402
from src.metrics import normalise_answer  # noqa: E402

TAG = "s1"
OUT = SUITE_DIR / TAG
PILOT_CORPUS = DATA_DIR / "corpus"
SALT = "study-s1"
BAD_ANS = {"yes", "no", "true", "false"}

MH_SRC = ["musique_train", "hotpotqa", "2wiki"]
SH_SRC = ["popqa", "triviaqa", "squad"]
AMB_SRC = "asqa_train"

# (set, class, n per source) in the order in which roles are filled
MH_ROLES = [("MH.test", "COMPLEX", 500), ("MIXED.test", "COMPLEX", 90), ("MIXED.test", "INSUFFICIENT_HOP", 70),
            ("MH.dev", "COMPLEX", 100), ("MIXED.dev", "COMPLEX", 30), ("MIXED.dev", "INSUFFICIENT_HOP", 25),
            ("STATE_TRAIN.train", "COMPLEX", 600), ("STATE_TRAIN.train", "INSUFFICIENT_HOP", 60)]
SH_ROLES = [("MIXED.test", "ANSWERABLE", 90), ("MIXED.test", "INSUFFICIENT_SINGLE", 70),
            ("MIXED.dev", "ANSWERABLE", 30), ("MIXED.dev", "INSUFFICIENT_SINGLE", 25),
            ("STATE_TRAIN.train", "ANSWERABLE", 300), ("STATE_TRAIN.train", "INSUFFICIENT_SINGLE", 50)]
AMB_ROLES = [("MIXED.test", "VAGUE", 150), ("MIXED.dev", "VAGUE", 50), ("STATE_TRAIN.train", "VAGUE", 100)]
SETS = ["MH.test", "MIXED.test", "MH.dev", "MIXED.dev", "STATE_TRAIN.train"]
CLASS_ORDER = ["ANSWERABLE", "INSUFFICIENT_SINGLE", "COMPLEX", "INSUFFICIENT_HOP", "VAGUE"]
FAMILY = {"ANSWERABLE": "single", "INSUFFICIENT_SINGLE": "single", "COMPLEX": "multihop",
          "INSUFFICIENT_HOP": "multihop", "VAGUE": "ambig"}

_log = logging.getLogger("build_study")


def u01(s: str) -> float:
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) / 16 ** 12


def golds3(q) -> list[str]:
    return [g for g in (normalise_answer(a) for a in q.golden_answers or []) if len(g) >= 3]


def base(qid: str) -> str:
    return qid.split("#")[0]


# ── eligibility per class ─────────────────────────────────────────────
def ok_complex(q) -> bool:
    g = set(q.gold_pids)
    return len(g) >= 2 and bool(q.golden_answers) and g <= {p.pid for p in q.passages}


def ok_ihop(q) -> bool:
    return ok_complex(q) and bool(golds3(q)) and not set(golds3(q)) & BAD_ANS


def ok_answerable(q) -> bool:
    return bool(q.passages) and bool(golds3(q)) and not set(golds3(q)) & BAD_ANS \
        and bool(passages_with_gold_string(q))


def ok_isingle(q) -> bool:
    return bool(golds3(q)) and not set(golds3(q)) & BAD_ANS


ELIGIBLE = {"COMPLEX": ok_complex, "INSUFFICIENT_HOP": ok_ihop, "ANSWERABLE": ok_answerable,
            "INSUFFICIENT_SINGLE": ok_isingle, "VAGUE": _vague_ok}


def fill(qs, roles, src, out, man):
    """Assign shuffled questions to roles in order; a question takes the first
    role that still needs questions and whose eligibility it meets."""
    need = {(s, c): n for s, c, n in roles}
    order = [(s, c) for s, c, _ in roles]
    for q in qs:
        for key in order:
            if need[key] > 0 and ELIGIBLE[key[1]](q):
                out[key].append((src, q))
                need[key] -= 1
                break
        if not any(need.values()):
            break
    man[src] = {f"{s}/{c}": n - need[(s, c)] for s, c, n in roles}
    short = {f"{s}/{c}": need[(s, c)] for s, c, _ in roles if need[(s, c)] > 0}
    if short:
        man[src]["short"] = short
        _log.warning("[%s] not enough eligible questions: %s", src, short)


def tag_question(q, set_name, cls):
    split = set_name.split(".")[0]
    cons = {"set": split, "study": TAG}
    if cls == "INSUFFICIENT_HOP":
        # the gold chain is annotated but its answer-bearing passages are
        # excluded corpus-wide at finalise; no gold pids remain for the state rule
        return q.with_(qid=q.qid + "#s1_drop", cls=cls, gold_pids=[],
                       construction={**cons, "parent": q.qid, "scrub_golds": True, "parent_gold_pids": q.gold_pids})
    if cls == "INSUFFICIENT_SINGLE":
        return q.with_(qid=q.qid + "#s1_drop", cls=cls, gold_pids=[],
                       construction={**cons, "parent": q.qid, "scrub_golds": True})
    if cls == "ANSWERABLE":
        return q.with_(cls=cls, gold_pids=[], construction=cons)
    return q.with_(cls=cls, construction=cons)


def select() -> None:
    used = set(json.load(open(DATA_DIR / "pilot_used_qids.json"))["qids"])
    out: dict[tuple, list] = defaultdict(list)
    man: dict = {"excluded_pilot_qids": len(used), "salt": SALT}
    for srcs, roles in ((MH_SRC, MH_ROLES), (SH_SRC, SH_ROLES), ([AMB_SRC], AMB_ROLES)):
        for src in srcs:
            _log.info("loading %s …", src)
            qs = [q for q in loaders.LOADERS[src]() if base(q.qid) not in used]
            qs.sort(key=lambda q: u01(f"{SALT}|{q.qid}"))
            man.setdefault("available", {})[src] = len(qs)
            fill(qs, roles, src, out, man.setdefault("filled", {}))
    OUT.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for set_name in SETS:
        rows = []
        for cls in CLASS_ORDER:            # grouped by corpus family → one corpus resident at a time
            by_src: dict[str, list] = defaultdict(list)
            for (s, c), items in out.items():
                if s == set_name and c == cls:
                    for src, q in items:
                        by_src[src].append(tag_question(q, set_name, cls))
            for i in range(max((len(v) for v in by_src.values()), default=0)):   # interleave sources
                for src in sorted(by_src):
                    if i < len(by_src[src]):
                        rows.append(by_src[src][i])
        for q in rows:
            b = base(q.qid)
            assert b not in seen and b not in used, f"overlap {q.qid}"
            seen.add(b)
        name, split = set_name.split(".")
        write_jsonl(OUT / f"{name}.{split}.jsonl", rows)
        man.setdefault("sets", {})[set_name] = {c: dict(Counter(q.source for q in rows if q.cls == c))
                                                for c in CLASS_ORDER if any(q.cls == c for q in rows)}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man["sets"], indent=2))


# ── corpora ───────────────────────────────────────────────────────────
def build_corpus(family: str, batch: int = 32) -> None:
    """Study corpus = pilot family corpus ∪ every passage of every study question
    of that family. Vectors of pilot passages are copied; only new passages
    are embedded."""
    old_dir = PILOT_CORPUS / family
    old_rows = [json.loads(l) for l in (old_dir / "passages.jsonl").open(encoding="utf-8")]
    old_emb = np.load(old_dir / "emb.npy", mmap_mode="r")
    assert old_emb.shape[0] == len(old_rows)
    passages = [Passage(pid=r["pid"], title=r["title"], text=r["text"]) for r in old_rows]
    seen = {r["pid"] for r in old_rows}
    n_new = 0
    for set_name in SETS:
        name, split = set_name.split(".")
        for q in read_jsonl(OUT / f"{name}.{split}.jsonl"):
            if FAMILY[q.cls] != family:
                continue
            for p in q.passages:
                h = text_hash(p.title, p.text)
                if h in seen or not (p.text or "").strip():
                    continue
                seen.add(h)
                passages.append(Passage(pid=h, title=p.title, text=p.text))
                n_new += 1
    _log.info("[%s] pilot %d + new %d = %d passages", family, len(old_rows), n_new, len(passages))
    corp = FamilyCorpus(family)
    corp.dir.mkdir(parents=True, exist_ok=True)
    with (corp.dir / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps({"pid": p.pid, "title": p.title, "text": p.text}, ensure_ascii=False) + "\n")
    corp.passages, corp.hashes = passages, [p.pid for p in passages]
    # seed the dense matrix with the pilot vectors, then resume embedding after them
    path, prog, done = corp.dir / "emb.npy", corp.dir / "emb.progress", corp.dir / "emb.done"
    if not (path.exists() and done.exists()):
        if not (path.exists() and prog.exists() and int(prog.read_text()) >= len(old_rows)):
            mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(passages), old_emb.shape[1]))
            n_old = len(old_rows)
            for s in range(0, n_old, 50_000):
                e = min(s + 50_000, n_old)
                mm[s:e] = old_emb[s:e]
            mm.flush()
            prog.write_text(str(len(old_rows)))
            del mm
    corp._build_bm25()
    corp._build_dense(batch)


# ── finalise: corpus-wide exclusion for unanswerables ─────────────────
def finalise() -> None:
    fams = {"single": ["INSUFFICIENT_SINGLE"], "multihop": ["INSUFFICIENT_HOP"]}
    report: dict = {}
    for set_name in SETS:
        name, split = set_name.split(".")
        path = OUT / f"{name}.{split}.jsonl"
        rows = list(read_jsonl(path))
        for fam, classes in fams.items():
            targets = {q.qid: golds3(q) for q in rows if q.cls in classes}
            if not targets:
                continue
            ex: dict[str, set] = {k: set() for k in targets}
            for line in (CORPUS_DIR / fam / "passages.jsonl").open(encoding="utf-8"):
                r = json.loads(line)
                t = normalise_answer(f"{r['title']} {r['text']}")
                for k, gs in targets.items():
                    if any(g in t for g in gs):
                        ex[k].add(r["pid"])
            rows = [q.with_(construction={**q.construction, "exclude_hashes": sorted(ex[q.qid])})
                    if q.qid in ex else q for q in rows]
        write_jsonl(path, rows)
        report[set_name] = dict(Counter(q.cls for q in rows))
    (OUT / "finalise.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "corpus", "finalise", "all"])
    ap.add_argument("--families", nargs="+", default=["multihop", "single", "ambig"])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if a.step in ("select", "all"):
        select()
    if a.step in ("corpus", "all"):
        for fam in a.families:
            build_corpus(fam)
    if a.step in ("finalise", "all"):
        finalise()
