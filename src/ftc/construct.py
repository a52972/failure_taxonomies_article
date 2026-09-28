"""Class assignment and controlled constructions for the FTC suite.

Class definitions are relative to the pool (that is what makes them
constructible):

    ANSWERABLE          single-hop; gold evidence in pool (string- or pid-level)
    VAGUE               >=2 annotated interpretations; evidence for >=1 in pool
    COMPLEX             multi-hop; every gold hop paragraph in pool
    INSUFFICIENT_HOP    COMPLEX with one gold hop paragraph deleted
    INSUFFICIENT_SINGLE ANSWERABLE with every gold-bearing passage deleted
    INSUFFICIENT_NAT    NoMIRACL non-relevant subset (human-judged)
    PARAMETRIC          closed-form MCQ / claim verification

Every constructed record keeps `construction` notes and the parent qid, so
paired analyses (same question, gold present vs. absent) are possible.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path

from config import CLASSES, DEV_FRACTION, PAD_MODE, POOL_SIZE, SEED, SUITE_DIR
from src.datasets import Passage, Question, write_jsonl
from src.metrics import normalise_answer
from src.ftc import loaders

_log = logging.getLogger(__name__)


# ── evidence checks ───────────────────────────────────────────────────

def passages_with_gold_string(q: Question) -> list[str]:
    """pids whose text contains any normalised gold answer (>= 2 chars)."""
    golds = [normalise_answer(g) for g in q.golden_answers]
    golds = [g for g in golds if len(g) >= 2]
    hits = []
    for p in q.passages:
        t = normalise_answer(f"{p.title} {p.text}")
        if any(g in t for g in golds):
            hits.append(p.pid)
    return hits


def gold_in_pool(q: Question) -> bool:
    if q.gold_pids:
        pids = {p.pid for p in q.passages}
        return all(g in pids for g in q.gold_pids)
    return bool(passages_with_gold_string(q))


# ── constructions ─────────────────────────────────────────────────────

def _pad_pool(passages: list[Passage], bank: list[Passage], rng: random.Random,
              size: int, exclude: set[str]) -> list[Passage]:
    have = {p.pid for p in passages}
    while len(passages) < size and bank:
        p = bank[rng.randrange(len(bank))]
        if p.pid in have or p.pid in exclude:
            continue
        passages.append(p)
        have.add(p.pid)
    return passages


def drop_one_hop(q: Question, rng: random.Random, bank: list[Passage]) -> Question | None:
    """Break the reasoning chain: delete the final hop paragraph AND every
    other passage that still states the answer.

    Deleting only the hop paragraph is not enough — measured on suite v4,
    32 % of such questions still showed the answer in the top-3, because the
    answer entity is often named in the first-hop passage, in a distractor,
    or in the question's own comparison targets. The class is defined as
    "the chain is broken and the answer is not recoverable from the pool",
    so the answer-bearing passages must go too. What still distinguishes this
    class from INSUFFICIENT_SINGLE is that the *first* hop remains: partial
    evidence, broken chain, versus no evidence at all."""
    if not q.gold_pids:
        return None
    final_pid = None
    fa = normalise_answer(q.golden_answers[0]) if q.golden_answers else ""
    for h in q.hops:
        if fa and normalise_answer(h.get("answer", "")) == fa:
            final_pid = h.get("pid")
    drop = final_pid if final_pid in q.gold_pids else rng.choice(q.gold_pids)
    bad = {drop} | set(passages_with_gold_string(q))
    if bad >= {p.pid for p in q.passages}:          # nothing would remain
        return None
    dropped = [p for p in q.passages if p.pid in bad]
    keep = [p for p in q.passages if p.pid not in bad]
    keep = _pad_pool(keep, bank, rng, len(q.passages), exclude=bad)
    tmp = q.with_(passages=keep)
    leftover = set(passages_with_gold_string(tmp))   # padding may re-introduce it
    if leftover:
        keep = [p for p in keep if p.pid not in leftover]
    from src.corpus import text_hash
    return q.with_(
        # v2 of the construction (answer-scrubbed). The suffix is bumped so
        # results computed against the older, contaminated variant can never be
        # silently matched to these questions by qid.
        qid=q.qid + "#drop_hop2", passages=keep, cls="INSUFFICIENT_HOP",
        gold_pids=[g for g in q.gold_pids if g not in bad],
        construction={**q.construction, "dropped_pid": drop, "parent": q.qid,
                      "dropped_is_final_hop": drop == final_pid,
                      "scrub_golds": True, "dropped_pids": sorted(bad),
                      "exclude_hashes": [text_hash(p.title, p.text) for p in dropped]})


def drop_gold_single(q: Question, rng: random.Random, bank: list[Passage]) -> Question | None:
    """Delete every passage that carries the gold (pid-level and string-level)."""
    bad = set(q.gold_pids) | set(passages_with_gold_string(q))
    if not bad:
        return None
    keep = [p for p in q.passages if p.pid not in bad]
    keep = _pad_pool(keep, bank, rng, len(q.passages), exclude=bad)
    # padding may re-introduce gold strings from the bank → re-check
    tmp = q.with_(passages=keep)
    if passages_with_gold_string(tmp):
        keep = [p for p in keep if p.pid not in set(passages_with_gold_string(tmp))]
    from src.corpus import text_hash
    return q.with_(
        qid=q.qid + "#drop_gold", passages=keep, cls="INSUFFICIENT_SINGLE",
        gold_pids=[], construction={**q.construction, "dropped_pids": sorted(bad),
                                    "parent": q.qid, "scrub_golds": True,
                                    "exclude_hashes": [text_hash(p.title, p.text)
                                                       for p in q.passages if p.pid in bad]})


class BankIndex:
    """Distractor bank with a BM25 index built once (hard-negative padding)."""

    def __init__(self, passages: list[Passage]):
        from rank_bm25 import BM25Okapi
        self.passages = passages
        self._bm25 = BM25Okapi([f"{p.title} {p.text}".lower().split() for p in passages]) if passages else None

    def random(self, rng: random.Random, k: int) -> list[Passage]:
        return [self.passages[rng.randrange(len(self.passages))] for _ in range(k)] if self.passages else []

    def nearest(self, query: str, k: int, exclude: set[str]) -> list[Passage]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        order = sorted(range(len(self.passages)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            p = self.passages[i]
            if p.pid not in exclude:
                out.append(p)
                if len(out) >= k:
                    break
        return out


def fix_pool(q: Question, rng: random.Random, bank: BankIndex, size: int = POOL_SIZE,
             pad_mode: str = PAD_MODE) -> Question:
    """Uniform pool size across sources.
    - larger pools: keep every gold passage, sample the rest;
    - smaller pools: pad with distractors from `bank` (random or BM25 hard
      negatives), never re-introducing gold strings for gold-free classes."""
    gold = set(q.gold_pids)
    if q.cls == "ANSWERABLE":
        gold |= set(passages_with_gold_string(q))
    # Classes built by removing the answer must stay answer-free even though
    # they still carry gold_pids (INSUFFICIENT_HOP keeps its first hop), so the
    # padding must be scrubbed whenever the construction says so.
    must_scrub = bool((q.construction or {}).get("scrub_golds"))
    orig = len(q.passages)
    if orig > size:
        keep = [p for p in q.passages if p.pid in gold]
        rest = [p for p in q.passages if p.pid not in gold]
        rng.shuffle(rest)
        pool = (keep + rest)[:max(size, len(keep))]
    else:
        pool = list(q.passages)
        have = {p.pid for p in pool}
        need = size - len(pool)
        if need > 0:
            cands = (bank.nearest(q.question, need * 3, have) if pad_mode == "bm25"
                     else bank.random(rng, need * 6))
            golds = [normalise_answer(g) for g in q.golden_answers if len(normalise_answer(g)) >= 2]
            for p in cands:
                if len(pool) >= size:
                    break
                if p.pid in have:
                    continue
                if (must_scrub or not gold) and golds:   # keep the class answer-free
                    t = normalise_answer(f"{p.title} {p.text}")
                    if any(g in t for g in golds):
                        continue
                pool.append(p)
                have.add(p.pid)
    rng.shuffle(pool)
    return q.with_(passages=pool, construction={**q.construction, "pool_orig": orig,
                                                "pool_pad_mode": pad_mode})


def _bank(questions: list[Question], rng: random.Random, cap: int = 5000) -> list[Passage]:
    ps = [p for q in questions for p in q.passages]
    rng.shuffle(ps)
    return ps[:cap]


# ── class pools ───────────────────────────────────────────────────────

def _resolve_condambig_gold_pids(q: Question) -> Question:
    """CondAmbigQA citations are a stringified list of {title, text}; titles
    carry a leading rank prefix ("4. Ouray, Colorado"). Map to pool pids."""
    import ast
    import re
    title2pid = {(p.title or "").strip().lower(): p.pid for p in q.passages}
    gold: list[str] = []
    for prop in q.metadata.get("properties", []):
        raw = prop.get("citations")
        cits = raw if isinstance(raw, list) else []
        if isinstance(raw, str):
            try:
                cits = ast.literal_eval(raw)
            except Exception:
                cits = []
        for c in cits or []:
            t = re.sub(r"^\s*\d+\.\s*", "", str(c.get("title", ""))).strip().lower()
            pid = title2pid.get(t)
            if pid and pid not in gold:
                gold.append(pid)
    return q.with_(gold_pids=gold)


def distinct_answers_in_pool(q: Question) -> int:
    """How many *different* gold answers are actually evidenced in the pool.
    This is what makes a question ambiguous *for the system*: two readings are
    both supported by what was retrieved."""
    seen = set()
    texts = [normalise_answer(f"{p.title} {p.text}") for p in q.passages]
    for g in q.golden_answers:
        n = normalise_answer(g)
        if len(n) >= 3 and any(n in t for t in texts):
            seen.add(n)
    return len(seen)


def _vague_ok(q: Question) -> bool:
    """Require (a) ≥2 annotated interpretations and (b) evidence for ≥2 of them
    in the pool. Without (b) the question is under-specified *and* unanswerable,
    which belongs to the INSUFFICIENT side, not to VAGUE. Measured on suite v4,
    half of the VAGUE questions had no gold answer in the pool at all."""
    if q.metadata.get("n_interp", 0) < 2:
        return False
    if distinct_answers_in_pool(q) < 2:
        return False
    if q.source == "condambigqa":
        return len(q.gold_pids) >= 2          # citations resolved to ≥2 passages
    return True


def build_class_pools(seed: int = SEED, sources: dict[str, list[Question]] | None = None,
                      ) -> dict[str, list[Question]]:
    rng = random.Random(seed)
    S = sources or {}

    def get(name):
        if name not in S:
            _log.info("loading %s …", name)
            S[name] = loaders.LOADERS[name]()
        return S[name]

    pools: dict[str, list[Question]] = {c: [] for c in CLASSES}

    # multi-hop → COMPLEX (+ hop drops)
    mh = [q for src in ("musique", "hotpotqa", "2wiki") for q in get(src)]
    mh_bank = _bank(mh, rng)
    complex_ok = [q.with_(cls="COMPLEX") for q in mh if q.gold_pids and gold_in_pool(q)]
    pools["COMPLEX"] = complex_ok
    for q in complex_ok:
        d = drop_one_hop(q, rng, mh_bank)
        if d is not None:
            pools["INSUFFICIENT_HOP"].append(d)

    # ambiguity → VAGUE
    vague = [_resolve_condambig_gold_pids(q) for q in get("condambigqa")] + list(get("asqa"))
    pools["VAGUE"] = [q.with_(cls="VAGUE") for q in vague if _vague_ok(q)]

    # single-hop → ANSWERABLE (+ gold drops)
    sh = [q for src in ("popqa", "triviaqa", "squad") for q in get(src)]
    sh_bank = _bank(sh, rng)
    ans = [q.with_(cls="ANSWERABLE") for q in sh if q.passages and gold_in_pool(q)]
    pools["ANSWERABLE"] = ans
    for q in ans:
        d = drop_gold_single(q, rng, sh_bank)
        if d is not None and len(d.passages) >= max(3, len(q.passages) // 2):
            pools["INSUFFICIENT_SINGLE"].append(d)

    # natural insufficiency
    for src in ("nomiracl-en", "nomiracl-es", "nomiracl-fr"):
        try:
            pools["INSUFFICIENT_NAT"].extend(q.with_(cls="INSUFFICIENT_NAT") for q in get(src))
        except Exception as exc:  # language file missing
            _log.warning("skip %s: %s", src, exc)

    # closed-form
    pools["PARAMETRIC"] = [q.with_(cls="PARAMETRIC") for src in ("arc_c", "pubhealth")
                           for q in get(src)]

    # padding banks (BM25 index built lazily in build_suite, only for sampled items)
    pools["_banks"] = {  # type: ignore[assignment]
        "mh": mh_bank, "sh": sh_bank, "all": _bank(mh + sh + vague, rng, cap=8000)}
    return pools


_BANK_FOR_CLASS = {"COMPLEX": "mh", "INSUFFICIENT_HOP": "mh",
                   "ANSWERABLE": "sh", "INSUFFICIENT_SINGLE": "sh"}


# ── sampling + splits ─────────────────────────────────────────────────

def _split(qid: str) -> str:
    h = int(hashlib.sha256(qid.split("#")[0].encode()).hexdigest(), 16) % 1000
    return "dev" if h < DEV_FRACTION * 1000 else "test"


def sample_balanced(items: list[Question], n: int, rng: random.Random) -> list[Question]:
    """Sample n items balanced across `source`."""
    by_src: dict[str, list[Question]] = {}
    for q in items:
        by_src.setdefault(q.source, []).append(q)
    for v in by_src.values():
        rng.shuffle(v)
    out, srcs = [], sorted(by_src)
    i = 0
    while len(out) < n and any(by_src.values()):
        s = srcs[i % len(srcs)]
        if by_src[s]:
            out.append(by_src[s].pop())
        i += 1
    return out


def build_suite(n_per_class: int, seed: int = SEED, out_dir: Path = SUITE_DIR,
                tag: str = "v1", extend_from: str | None = None,
                classes: list[str] | None = None,
                fresh: list[str] | None = None) -> dict:
    """Build (or extend) the suite.

    `extend_from`: keep every question already in that tag's files — same qids,
    same pools — and top up to `n_per_class` with fresh draws. This makes the
    larger suite a strict superset, so results already computed stay valid and
    their LLM-cache entries still hit."""
    from src.datasets import read_jsonl
    rng = random.Random(seed)
    pools = build_class_pools(seed)
    raw_banks = pools.pop("_banks")
    banks: dict[str, BankIndex] = {}
    manifest = {"tag": tag, "seed": seed, "n_per_class": n_per_class,
                "pool_size": POOL_SIZE, "pad_mode": PAD_MODE,
                "extend_from": extend_from, "classes": {}}
    target_classes = classes or list(CLASSES)

    fresh_set = set(fresh or ())          # rebuilt from scratch: ignore what exists

    def existing(cls: str, split: str) -> list[Question]:
        if not extend_from or cls in fresh_set:
            return []
        p = out_dir / extend_from / f"{cls}.{split}.jsonl"
        return list(read_jsonl(p)) if p.exists() else []

    # sample n for test + n*DEV/(1-DEV) for dev from the same pool, disjoint parents
    for cls in CLASSES:
        if cls not in target_classes:
            # carry the previous files over unchanged so the tag stays complete
            for split in ("test", "dev"):
                prev = existing(cls, split)
                if prev:
                    path = out_dir / tag / f"{cls}.{split}.jsonl"
                    n = write_jsonl(path, prev)
                    manifest["classes"].setdefault(cls, {})[split] = {
                        "n": n, "path": str(path), "carried_over": True,
                        "by_source": dict(Counter(q.source for q in prev)),
                        "pool_size_mean": round(sum(len(q.passages) for q in prev) / max(1, n), 2)}
            continue
        items = pools[cls]
        for q in items:
            q.construction.setdefault("split", _split(q.qid))
        by_split = {"test": [q for q in items if q.construction["split"] == "test"],
                    "dev": [q for q in items if q.construction["split"] == "dev"]}
        n_dev = max(20, int(n_per_class * DEV_FRACTION))
        bkey = _BANK_FOR_CLASS.get(cls, "all")
        if bkey not in banks:
            _log.info("building BM25 bank %s (%d passages) …", bkey, len(raw_banks[bkey]))
            banks[bkey] = BankIndex(raw_banks[bkey])
        for split, target in (("test", n_per_class), ("dev", n_dev)):
            keep = existing(cls, split)                       # already-run questions
            have = {q.qid for q in keep}
            # exclude their parents too, so a question and its construction twin
            # never land on opposite sides of a top-up
            parents = {q.qid.split("#")[0] for q in keep}
            fresh_pool = [q for q in by_split[split]
                          if q.qid not in have and q.qid.split("#")[0] not in parents]
            need = max(0, target - len(keep))
            fresh = [fix_pool(q, rng, banks[bkey]) for q in sample_balanced(fresh_pool, need, rng)]
            rows = keep + fresh
            path = out_dir / tag / f"{cls}.{split}.jsonl"
            n = write_jsonl(path, rows)
            manifest["classes"].setdefault(cls, {})[split] = {
                "n": n, "path": str(path), "kept": len(keep), "new": len(fresh),
                "by_source": dict(Counter(q.source for q in rows)),
                "pool_size_mean": round(sum(len(q.passages) for q in rows) / max(1, n), 2),
                "available": len(by_split[split]),
            }
    (out_dir / tag / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
