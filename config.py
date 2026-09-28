"""Central configuration for the failures_taxonomies harness.

Everything experiment-level is a Python constant here so the project is
self-contained. Only secrets come from env vars (none needed for local
Ollama). Provider: local Ollama only (owner decision 2026-09-09: no cloud).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# Providers / generators
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str | None
    default_timeout_s: float
    supports_usage_field: bool
    notes: str


PROVIDERS: dict[str, ProviderConfig] = {
    "ollama_local": ProviderConfig(
        name="ollama_local",
        base_url=os.environ.get("OLLAMA_OPENAI_URL", "http://ollama:11434/v1"),
        api_key_env=None,
        default_timeout_s=600.0,
        supports_usage_field=True,
        notes="Local Ollama, OpenAI-compatible endpoint. logprobs supported (>=0.32).",
    ),
}
ACTIVE_PROVIDER: str = "ollama_local"

GENERATORS: dict[str, dict[str, str]] = {
    "ollama_local": {
        # Three families only (owner decision 2026-09-09: no Mistral, no
        # sub-7B models, no cloud). Each ~5–6 GB q4_K_M.
        "llama3-8b":   "llama3.1:8b-instruct-q4_K_M",
        "qwen2p5-7b":  "qwen2.5:7b-instruct-q4_K_M",
        "gemma2-9b":   "gemma2:9b-instruct-q4_K_M",
    },
}
GENERATOR_FAMILIES = ("llama3-8b", "qwen2p5-7b", "gemma2-9b")
DEFAULT_GENERATOR = "llama3-8b"

# ═══════════════════════════════════════════════════════════════════════
# Embedder + reranker backends
# ═══════════════════════════════════════════════════════════════════════

EMBEDDER_BACKEND: str = os.environ.get("EMBEDDER_BACKEND", "ollama")   # "local" | "ollama"
EMBEDDER_MODEL_LOCAL: str = "BAAI/bge-m3"
EMBEDDER_MODEL_OLLAMA: str = "bge-m3"
EMBEDDER_OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://ollama:11434")

RERANKER_BACKEND: str = os.environ.get("RERANKER_BACKEND", "http")     # "local" | "http"
RERANKER_MODEL_LOCAL: str = "BAAI/bge-reranker-v2-m3"
RERANKER_HTTP_URL: str = os.environ.get("RERANKER_URL", "http://reranker:8001")

# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
HF_CACHE_DIR = DATA_DIR / "cache" / "hf"
LLM_CACHE_DIR = Path(os.environ.get("FTC_LLM_CACHE", DATA_DIR / "cache" / "llm"))
EMB_CACHE_DIR = Path(os.environ.get("FTC_EMB_CACHE", DATA_DIR / "cache" / "emb"))
# Retrieval cache (docs/20): the first retrieval is identical for every arm, so its
# result is stored on disk. Deterministic retrieval → cannot change any answer.
# Off unless FTC_RETRIEVAL_CACHE=1 (verified identical on the dev set before use).
RETRIEVAL_CACHE_DIR = DATA_DIR / "cache" / "retrieval_s1"
RETRIEVAL_CACHE = os.environ.get("FTC_RETRIEVAL_CACHE", "0") == "1"
SUITE_DIR = DATA_DIR / "suite"
# Family corpora. "corpus" = pilot corpora; "corpus_s1" = study corpora (docs/20).
CORPUS_DIR = DATA_DIR / os.environ.get("FTC_CORPUS", "corpus_s1")
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
for _d in (HF_CACHE_DIR, LLM_CACHE_DIR, SUITE_DIR, RAW_RESULTS_DIR,
           TABLES_DIR, FIGURES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# HTTP behaviour
# ═══════════════════════════════════════════════════════════════════════

HTTP_MAX_RETRIES = 5
HTTP_RETRY_INITIAL_BACKOFF_S = 2.0
HTTP_RETRY_MAX_BACKOFF_S = 60.0

# ═══════════════════════════════════════════════════════════════════════
# Retrieval / rerank / loop thresholds
# ═══════════════════════════════════════════════════════════════════════

# Pool protocol. Env-overridable so suite variants (v1: 10/5, v2: 20/3) can be
# built and compared without editing code.
POOL_SIZE = int(os.environ.get("FTC_POOL_SIZE", "20"))     # canonical pool size
RETRIEVAL_TOPK = int(os.environ.get("FTC_RETRIEVAL_TOPK", "10"))  # kept after initial ranking
RERANK_TOPK = int(os.environ.get("FTC_RERANK_TOPK", "3"))  # shown to generator / gate
RRF_K = 60
PAD_MODE = os.environ.get("FTC_PAD_MODE", "bm25")  # "random" | "bm25" (hard negatives)



# ═══════════════════════════════════════════════════════════════════════
# LLM budgets
# ═══════════════════════════════════════════════════════════════════════

GEN_TEMPERATURE = 0.0
GEN_MAX_TOKENS_SHORT = 64
GEN_MAX_TOKENS_LONG = 128
DIAGNOSIS_MAX_TOKENS = 16
REFORMULATION_MAX_TOKENS = 160

ABSTAIN_TOKEN = "UNANSWERABLE"

# ═══════════════════════════════════════════════════════════════════════
# Failure-type taxonomy
# ═══════════════════════════════════════════════════════════════════════

LABELS = ("VAGUE", "COMPLEX", "INSUFFICIENT")
CLASSES = (
    "ANSWERABLE",          # single-hop, gold evidence in pool (control)
    "VAGUE",               # ambiguous question, evidence for >=2 readings in pool
    "COMPLEX",             # multi-hop, all gold hops in pool
    "INSUFFICIENT_HOP",    # multi-hop with one gold hop removed
    "INSUFFICIENT_SINGLE", # single-hop with gold passage(s) removed
    "INSUFFICIENT_NAT",    # NoMIRACL non-relevant (human-judged, no answer)
    "PARAMETRIC",          # closed-form; retrieval tends to hurt
)
# Oracle label per class (NONE = no retrieval-side failure to repair)
ORACLE_LABEL = {
    "ANSWERABLE": "NONE",
    "VAGUE": "VAGUE",
    "COMPLEX": "COMPLEX",
    "INSUFFICIENT_HOP": "INSUFFICIENT",
    "INSUFFICIENT_SINGLE": "INSUFFICIENT",
    "INSUFFICIENT_NAT": "INSUFFICIENT",
    "PARAMETRIC": "NONE",
}
STRATEGY_MAP = {
    "VAGUE": "disambiguate",
    "COMPLEX": "decompose",
    "INSUFFICIENT": "anchor_rewrite",
    "NONE": "rewrite",
}

SEED = 42
DEV_FRACTION = 0.2


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def get_provider(name: str | None = None) -> ProviderConfig:
    key = name or ACTIVE_PROVIDER
    if key not in PROVIDERS:
        raise ValueError(f"Unknown provider {key!r}; available {sorted(PROVIDERS)}")
    return PROVIDERS[key]


def resolve_model_id(alias_or_id: str, provider: str | None = None) -> str:
    key = provider or ACTIVE_PROVIDER
    return GENERATORS.get(key, {}).get(alias_or_id, alias_or_id)


def get_api_key(provider: str | None = None) -> str:
    cfg = get_provider(provider)
    if cfg.api_key_env is None:
        return ""
    return os.environ.get(cfg.api_key_env, "").strip()
