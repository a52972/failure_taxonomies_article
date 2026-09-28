"""Cross-encoder reranker with pluggable backend.

Two backends:
- "local"  → sentence-transformers CrossEncoder loaded in-process
             (BAAI/bge-reranker-v2-m3). Uses CUDA if available.
- "http"   → HTTP microservice speaking the VIMOS reranker API:
                 POST /rerank
                 { "query": str, "passages": list[str], "top_k": int }
                 → { "results": [{"index": int, "score": float}] }
             This is what `vimos_reranker` on port 8001 exposes.

Backend selection is in config.py (`RERANKER_BACKEND`). The local backend
is the default so that a fresh clone of this repo works standalone; the
HTTP backend is what we use inside the VIMOS devcontainer to route
reranking to the existing GPU-backed microservice.

Both backends return scores in [0, 1] (sigmoid-normalised for the local
backend; the microservice already returns sigmoid scores).
"""
from __future__ import annotations

import logging
from functools import lru_cache

import httpx
import numpy as np

from src.datasets import Passage
from config import (
    RERANKER_BACKEND, RERANKER_MODEL_LOCAL,
    RERANKER_HTTP_URL, HTTP_MAX_RETRIES,
)

_log = logging.getLogger(__name__)


# ── Local backend (sentence-transformers) ─────────────────────────────────

@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder
    _log.info("Loading reranker %s (this may take a moment)…",
              RERANKER_MODEL_LOCAL)
    return CrossEncoder(RERANKER_MODEL_LOCAL, max_length=512)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rerank_local(
    query: str, passages: list[Passage],
    top_k: int | None, sigmoid: bool,
) -> list[tuple[Passage, float]]:
    model = _get_reranker()
    pairs = [(query, f"{p.title}. {p.text}"[:2000]) for p in passages]
    raw = np.asarray(model.predict(pairs, show_progress_bar=False))
    scores = _sigmoid(raw) if sigmoid else raw
    ranked = sorted(
        zip(passages, scores.tolist()),
        key=lambda x: x[1], reverse=True,
    )
    if top_k is not None:
        ranked = ranked[:top_k]
    return [(p, float(s)) for p, s in ranked]


# ── HTTP backend (VIMOS reranker microservice) ────────────────────────────

@lru_cache(maxsize=1)
def _get_http_client() -> httpx.Client:
    _log.info("Reranker HTTP backend → %s", RERANKER_HTTP_URL)
    return httpx.Client(base_url=RERANKER_HTTP_URL, timeout=60.0)


def _rerank_http(
    query: str, passages: list[Passage],
    top_k: int | None,
) -> list[tuple[Passage, float]]:
    client = _get_http_client()
    body = {
        "query": query,
        "passages": [f"{p.title}. {p.text}"[:2000] for p in passages],
        "top_k": top_k if top_k is not None else len(passages),
    }
    last_exc: Exception | None = None
    for attempt in range(HTTP_MAX_RETRIES):
        try:
            r = client.post("/rerank", json=body)
            r.raise_for_status()
            data = r.json()
            results = data.get("results") or []
            # Convert to (Passage, score); results is [{index, score}, ...]
            out: list[tuple[Passage, float]] = []
            for item in results:
                i = int(item["index"])
                if 0 <= i < len(passages):
                    out.append((passages[i], float(item["score"])))
            return out
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            last_exc = e
            _log.warning(
                "Reranker HTTP call failed (attempt %d/%d): %s",
                attempt + 1, HTTP_MAX_RETRIES, e)
    raise RuntimeError(
        f"Reranker HTTP backend at {RERANKER_HTTP_URL} failed "
        f"after {HTTP_MAX_RETRIES} attempts: {last_exc}") from last_exc


# ── Public API ────────────────────────────────────────────────────────────

def rerank(
    query: str,
    passages: list[Passage],
    *,
    top_k: int | None = None,
    sigmoid: bool = True,
) -> list[tuple[Passage, float]]:
    """Rerank passages, returning [(Passage, score), ...] sorted high→low.

    top_k = None keeps all passages. `sigmoid` only applies to the local
    backend (the microservice already returns sigmoid-scaled scores).
    """
    if not passages:
        return []
    if RERANKER_BACKEND == "local":
        return _rerank_local(query, passages, top_k, sigmoid)
    if RERANKER_BACKEND == "http":
        return _rerank_http(query, passages, top_k)
    raise ValueError(
        f"Unknown RERANKER_BACKEND {RERANKER_BACKEND!r}. "
        f"Options: 'local' | 'http'.")
