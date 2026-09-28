"""Family mini-corpora: hybrid (BM25 + dense) retrieval over the union of all
passages of a source family, so that query reformulation can actually change
what is retrieved (the per-question pool protocol leaves no headroom: a
cross-encoder re-ranks the same 20 passages whatever the query).

Layout: data/corpus/<family>/passages.jsonl  {pid, title, text, h}
                          /emb.npy           float16 [N, D], row i ↔ line i
                          /bm25/             bm25s index
Construction-time exclusions are applied at query time:
    exclude_hashes  — text hashes of dropped gold passages (INSUFFICIENT_HOP)
    scrub_golds     — drop any candidate containing a gold answer string
                      (INSUFFICIENT_SINGLE: "corpus with the answer scrubbed")
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from config import CORPUS_DIR, RRF_K
from src.datasets import Passage
from src.metrics import normalise_answer

_log = logging.getLogger(__name__)


def text_hash(title: str, text: str) -> str:
    return hashlib.sha1(f"{title.strip()}\n{text.strip()}".encode("utf-8")).hexdigest()


class FamilyCorpus:
    def __init__(self, family: str):
        self.family = family
        self.dir = CORPUS_DIR / family
        self.passages: list[Passage] = []
        self.hashes: list[str] = []
        self._emb: np.ndarray | None = None
        self._bm25 = None
        self._loaded = False

    # ── build ─────────────────────────────────────────────────────────
    def build(self, passages: list[Passage], *, embed_batch: int = 32) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        uniq: list[Passage] = []
        for p in passages:
            h = text_hash(p.title, p.text)
            if h in seen or not (p.text or "").strip():
                continue
            seen.add(h)
            uniq.append(Passage(pid=h, title=p.title, text=p.text))
        with (self.dir / "passages.jsonl").open("w", encoding="utf-8") as f:
            for p in uniq:
                f.write(json.dumps({"pid": p.pid, "title": p.title, "text": p.text}, ensure_ascii=False) + "\n")
        _log.info("[%s] %d unique passages (from %d)", self.family, len(uniq), len(passages))
        self.passages, self.hashes = uniq, [p.pid for p in uniq]
        self._build_bm25()
        self._build_dense(embed_batch)
        self._loaded = True

    def _build_bm25(self) -> None:
        import bm25s
        t0 = time.perf_counter()
        # tokenise in chunks to bound RAM (bm25s materialises token ids per doc)
        bm = bm25s.BM25()
        toks = bm25s.tokenize([f"{p.title} {p.text}"[:2000] for p in self.passages],
                              stopwords="en", show_progress=False, return_ids=True)
        bm.index(toks, show_progress=False)
        bm.save(str(self.dir / "bm25"))
        self._bm25 = bm
        del toks
        _log.info("[%s] BM25 indexed in %.1fs", self.family, time.perf_counter() - t0)

    def _build_dense(self, batch: int, *, pause_every: int = 50, pause_s: float = 0.5) -> None:
        """Resumable, throttled embedding. Progress is tracked in emb.progress so
        a crash resumes where it stopped; small batches + periodic pauses keep
        Ollama from starving other GPU users."""
        from src.retrieval import _embed_uncached
        n = len(self.passages)
        path = self.dir / "emb.npy"
        done_path = self.dir / "emb.done"
        prog_path = self.dir / "emb.progress"
        if path.exists() and done_path.exists():
            self._emb = np.load(path, mmap_mode="r")
            if self._emb.shape[0] == n:
                _log.info("[%s] dense matrix already built", self.family)
                return
        d = _embed_uncached([f"{self.passages[0].title}. {self.passages[0].text}"]).shape[1]
        start = 0
        if path.exists() and prog_path.exists():
            try:
                start = int(prog_path.read_text().strip())
                mm = np.lib.format.open_memmap(path, mode="r+")
                if mm.shape != (n, d):
                    start = 0
                    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, d))
            except Exception:
                start = 0
                mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, d))
        else:
            mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, d))
        _log.info("[%s] embedding %d passages from %d (batch %d)", self.family, n, start, batch)
        t0 = time.perf_counter()
        for bi, i in enumerate(range(start, n, batch)):
            texts = [f"{p.title}. {p.text}"[:1500] for p in self.passages[i:i + batch]]
            mm[i:i + len(texts)] = _embed_uncached(texts).astype(np.float16)
            if bi % 20 == 0:
                mm.flush()
                prog_path.write_text(str(i + len(texts)))
            if bi % pause_every == 0:
                time.sleep(pause_s)
            if bi % 200 == 0:
                el = time.perf_counter() - t0
                done = i + len(texts) - start
                rate = done / max(el, 1e-6)
                _log.info("[%s] embedded %d/%d (%.0f/s, ETA %.0f min)", self.family, i + len(texts), n, rate, (n - i - len(texts)) / max(rate, 1e-6) / 60)
        mm.flush()
        prog_path.write_text(str(n))
        done_path.write_text("ok")
        self._emb = np.load(path, mmap_mode="r")

    # ── load ──────────────────────────────────────────────────────────
    def load(self) -> "FamilyCorpus":
        if self._loaded:
            return self
        import bm25s
        self.passages = []
        for line in (self.dir / "passages.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            self.passages.append(Passage(pid=r["pid"], title=r["title"], text=r["text"]))
        self.hashes = [p.pid for p in self.passages]
        self._bm25 = bm25s.BM25.load(str(self.dir / "bm25"))
        self._emb = np.load(self.dir / "emb.npy", mmap_mode="r")
        assert self._emb.shape[0] == len(self.passages), "emb/passages mismatch"
        self._loaded = True
        _log.info("[%s] loaded %d passages", self.family, len(self.passages))
        return self

    # ── search ────────────────────────────────────────────────────────
    def _bm25_top(self, query: str, k: int) -> list[tuple[int, float]]:
        import bm25s
        q = bm25s.tokenize([query], stopwords="en", show_progress=False)
        idx, sc = self._bm25.retrieve(q, k=min(k, len(self.passages)), show_progress=False)
        return [(int(i), float(s)) for i, s in zip(idx[0], sc[0])]

    def _dense_top(self, query: str, k: int) -> list[tuple[int, float]]:
        from src.retrieval import _embed
        qv = _embed([query])[0].astype(np.float32)
        n = self._emb.shape[0]
        best_idx: list[int] = []
        best_sc: list[float] = []
        chunk = 100_000
        scores = np.empty(n, dtype=np.float32)
        for s in range(0, n, chunk):
            scores[s:s + chunk] = np.asarray(self._emb[s:s + chunk], dtype=np.float32) @ qv
        top = np.argpartition(-scores, min(k, n - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def search(self, query: str, k: int = 10, *, method: str = "hybrid",
               exclude_hashes: set[str] | None = None, scrub_golds: list[str] | None = None,
               cand: int = 100) -> list[tuple[Passage, float]]:
        ex = exclude_hashes or set()
        golds = [normalise_answer(g) for g in (scrub_golds or []) if len(normalise_answer(g)) >= 2]

        def ok(i: int) -> bool:
            p = self.passages[i]
            if p.pid in ex:
                return False
            if golds:
                t = normalise_answer(f"{p.title} {p.text}")
                if any(g in t for g in golds):
                    return False
            return True

        rankings: list[list[tuple[int, float]]] = []
        if method in ("bm25", "hybrid"):
            rankings.append([(i, s) for i, s in self._bm25_top(query, cand) if ok(i)])
        if method in ("dense", "hybrid"):
            rankings.append([(i, s) for i, s in self._dense_top(query, cand) if ok(i)])
        if len(rankings) == 1:
            out = rankings[0][:k]
        else:
            fused: dict[int, float] = {}
            for r in rankings:
                for rank, (i, _s) in enumerate(r):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            out = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.passages[i], float(s)) for i, s in out]


_CACHE: dict[str, FamilyCorpus] = {}
#: How many family corpora to keep resident. Each one holds a BM25 index and a
#: passage list in RAM (100-400 MB), so a sweep that walks several classes would
#: otherwise accumulate all of them. Only the active family is ever needed.
MAX_RESIDENT = 1


def release_corpus(family: str) -> None:
    c = _CACHE.pop(family, None)
    if c is not None:
        c._bm25 = None
        c._emb = None          # memory-mapped; closing the reference frees it
        c.passages = []
        c.hashes = []
        c._loaded = False


def get_corpus(family: str) -> FamilyCorpus:
    if family not in _CACHE:
        while len(_CACHE) >= MAX_RESIDENT:
            oldest = next(iter(_CACHE))
            _log.info("[corpus] a libertar %s para carregar %s", oldest, family)
            release_corpus(oldest)
            import gc
            gc.collect()
        _CACHE[family] = FamilyCorpus(family).load()
    return _CACHE[family]


FAMILY_FOR_CLASS = {
    "ANSWERABLE": "single", "INSUFFICIENT_SINGLE": "single",
    "COMPLEX": "multihop", "INSUFFICIENT_HOP": "multihop",
    "VAGUE": "ambig", "INSUFFICIENT_NAT": "nomiracl", "PARAMETRIC": "closed",
}
FAMILY_SOURCES = {
    "multihop": ["musique", "hotpotqa", "2wiki"],
    "single": ["popqa", "triviaqa", "squad"],
    "ambig": ["asqa", "condambigqa"],
    "nomiracl": ["nomiracl-en", "nomiracl-es", "nomiracl-fr"],
    "closed": ["arc_c", "pubhealth"],
}
# per-source cap on *questions* contributing passages (gold of suite questions
# always kept). Sized so each family corpus is ≤ ~60k passages: ~1 h of bge-m3
# embedding on the RTX 5060 Ti and ~120 MB fp16 per family.
SOURCE_QUESTION_CAP = {"2wiki": 1500, "hotpotqa": 2000, "musique": 1200,
                       "triviaqa": 1200, "popqa": 800, "squad": 3000,
                       "asqa": 500, "condambigqa": 1000,
                       "arc_c": 600, "pubhealth": 600}
