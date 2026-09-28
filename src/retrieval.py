"""Retrieval over the per-question passage pool.

Every question ships with a small pool (~10-25 passages) selected by a
prior retriever (Contriever or the distractor set). Our "retrieval" step
is therefore an *initial ranking* over that pool. We provide:

- BM25Okapi           (lexical, CPU, no model dependency)
- Dense (BGE-M3)      (semantic, either local sentence-transformers or
                       Ollama HTTP backend — controlled by config)
- RRF fusion          (combine BM25 + Dense rankings)
- Passthrough         (use the retriever's original score if present)

Every ranker returns:
    list[tuple[Passage, float]]  # sorted, high → low

The embedder backend is selected by config.EMBEDDER_BACKEND:
    "local"  → sentence-transformers in-process (needs CUDA or CPU)
    "ollama" → POST http://ollama:11434/api/embed  (uses Ollama's GPU)

Both return L2-normalised vectors, so downstream dot-product == cosine.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

import httpx
import numpy as np
from rank_bm25 import BM25Okapi

from src.datasets import Passage
from config import (
    EMBEDDER_BACKEND, EMBEDDER_MODEL_LOCAL,
    EMBEDDER_MODEL_OLLAMA, EMBEDDER_OLLAMA_URL,
    RRF_K, HTTP_MAX_RETRIES,
)

_log = logging.getLogger(__name__)


# ── Tokenisation ──────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


# ── BM25 ──────────────────────────────────────────────────────────────────

def bm25_rank(query: str, passages: list[Passage]) -> list[tuple[Passage, float]]:
    corpus_tokens = [_tokenise(f"{p.title} {p.text}") for p in passages]
    if not corpus_tokens:
        return []
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(_tokenise(query))
    ranked = sorted(
        zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [(p, float(s)) for p, s in ranked]


# ── Dense embedding — local backend ───────────────────────────────────────

@lru_cache(maxsize=1)
def _get_local_embedder():
    from sentence_transformers import SentenceTransformer
    _log.info("Loading local embedder %s (this may take a moment)…",
              EMBEDDER_MODEL_LOCAL)
    return SentenceTransformer(EMBEDDER_MODEL_LOCAL)


def _embed_local(texts: list[str]) -> np.ndarray:
    model = _get_local_embedder()
    return np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


# ── Dense embedding — Ollama backend ──────────────────────────────────────

@lru_cache(maxsize=1)
def _get_ollama_client() -> httpx.Client:
    _log.info("Ollama embedder → %s (model=%s)",
              EMBEDDER_OLLAMA_URL, EMBEDDER_MODEL_OLLAMA)
    return httpx.Client(base_url=EMBEDDER_OLLAMA_URL, timeout=120.0)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _embed_ollama(texts: list[str]) -> np.ndarray:
    """POST /api/embed with a batch; Ollama returns raw embeddings which
    we L2-normalise for cosine similarity via dot product."""
    client = _get_ollama_client()
    body = {"model": EMBEDDER_MODEL_OLLAMA, "input": texts}
    last_exc: Exception | None = None
    for attempt in range(HTTP_MAX_RETRIES):
        try:
            r = client.post("/api/embed", json=body)
            r.raise_for_status()
            data = r.json()
            embs = data.get("embeddings")
            if embs is None:
                raise RuntimeError(f"Ollama /api/embed returned no 'embeddings': {data}")
            arr = np.asarray(embs, dtype=np.float32)
            return _l2_normalize(arr)
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            last_exc = e
            _log.warning(
                "Ollama embed call failed (attempt %d/%d): %s",
                attempt + 1, HTTP_MAX_RETRIES, e)
    raise RuntimeError(
        f"Ollama embedder at {EMBEDDER_OLLAMA_URL} failed after "
        f"{HTTP_MAX_RETRIES} attempts: {last_exc}") from last_exc


def _embed_uncached(texts: list[str]) -> np.ndarray:
    if EMBEDDER_BACKEND == "local":
        return _embed_local(texts)
    if EMBEDDER_BACKEND == "ollama":
        return _embed_ollama(texts)
    raise ValueError(
        f"Unknown EMBEDDER_BACKEND {EMBEDDER_BACKEND!r}. "
        f"Options: 'local' | 'ollama'.")


# ── Embedding cache (memory + disk). Pool passages repeat across policies,
# iterations and runs, so this removes most embedder calls. ──────────────
import hashlib as _hashlib
from pathlib import Path as _Path
from config import EMB_CACHE_DIR as _EMB_CACHE_DIR

_EMB_DIR = _EMB_CACHE_DIR / (EMBEDDER_MODEL_OLLAMA if EMBEDDER_BACKEND == "ollama" else EMBEDDER_MODEL_LOCAL).replace("/", "_")
_EMB_DIR.mkdir(parents=True, exist_ok=True)
_EMB_MEM: dict[str, np.ndarray] = {}


def _emb_key(text: str) -> str:
    return _hashlib.sha1(text.encode("utf-8")).hexdigest()


def _embed(texts: list[str]) -> np.ndarray:
    keys = [_emb_key(t) for t in texts]
    out: list[np.ndarray | None] = [None] * len(texts)
    missing: list[int] = []
    for i, k in enumerate(keys):
        v = _EMB_MEM.get(k)
        if v is None:
            fp = _EMB_DIR / f"{k}.npy"
            if fp.exists():
                try:
                    v = np.load(fp)
                    _EMB_MEM[k] = v
                except Exception:
                    v = None
        if v is None:
            missing.append(i)
        else:
            out[i] = v
    if missing:
        fresh = _embed_uncached([texts[i] for i in missing])
        for j, i in enumerate(missing):
            v = np.asarray(fresh[j], dtype=np.float32)
            out[i] = v
            _EMB_MEM[keys[i]] = v
            try:
                np.save(_EMB_DIR / f"{keys[i]}.npy", v)
            except Exception:
                pass
    return np.stack(out)  # type: ignore[arg-type]


def dense_rank(query: str, passages: list[Passage]) -> list[tuple[Passage, float]]:
    if not passages:
        return []
    texts = [f"{p.title}. {p.text}" for p in passages]
    q_emb = _embed([query])[0]
    p_emb = _embed(texts)
    scores = (p_emb @ q_emb).tolist()
    ranked = sorted(
        zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [(p, float(s)) for p, s in ranked]


# ── Passthrough (trust incoming order / source_score) ─────────────────────

def passthrough_rank(passages: list[Passage]) -> list[tuple[Passage, float]]:
    if any(p.source_score is not None for p in passages):
        return sorted(
            [(p, p.source_score or 0.0) for p in passages],
            key=lambda x: x[1], reverse=True)
    return [(p, 1.0 / (i + 1)) for i, p in enumerate(passages)]


# ── RRF fusion ────────────────────────────────────────────────────────────

def rrf_fuse(
    rankings: list[list[tuple[Passage, float]]],
    k: int = RRF_K,
) -> list[tuple[Passage, float]]:
    """Reciprocal Rank Fusion (Cormack et al. 2009). Dedup by pid."""
    scores: dict[str, float] = {}
    lookup: dict[str, Passage] = {}
    for ranking in rankings:
        for rank, (p, _s) in enumerate(ranking):
            scores[p.pid] = scores.get(p.pid, 0.0) + 1.0 / (k + rank + 1)
            lookup[p.pid] = p
    return sorted(
        [(lookup[pid], sc) for pid, sc in scores.items()],
        key=lambda x: x[1], reverse=True,
    )


# ── Unified entry point ───────────────────────────────────────────────────

def initial_rank(
    query: str,
    passages: list[Passage],
    *,
    method: str = "hybrid",
) -> list[tuple[Passage, float]]:
    """method ∈ {"bm25", "dense", "hybrid", "passthrough"}"""
    if not passages:
        return []
    if method == "bm25":
        return bm25_rank(query, passages)
    if method == "dense":
        return dense_rank(query, passages)
    if method == "hybrid":
        return rrf_fuse([bm25_rank(query, passages),
                         dense_rank(query, passages)])
    if method == "passthrough":
        return passthrough_rank(passages)
    raise ValueError(f"Unknown ranking method: {method!r}")
