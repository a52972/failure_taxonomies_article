"""Chat client for the active provider (DeepInfra cloud or local Ollama).

Provider selection lives in config.py (`ACTIVE_PROVIDER`). Both providers
speak the OpenAI-compatible ``/chat/completions`` API, so a single client
class covers both. Behaviour differences that matter:

- **Auth**: DeepInfra needs a bearer token, Ollama needs none.
- **Timeouts**: local models take much longer on the first request
  (cold-load); we use a 10-min timeout for Ollama vs 2 min for DeepInfra.
- **Retries**: we retry only on transport errors, 5xx, and 429 —
  never on 4xx (bad request would just keep failing).
- **Preflight**: we ping the provider once per process to fail fast if
  it's unreachable (e.g. Ollama service isn't running).

Every request is cached to disk by
``sha256(model_id + messages + temperature + max_tokens)``. Cache hits
skip the network entirely. Because the key includes the resolved
model_id, cache buckets for DeepInfra and Ollama are naturally separate
even for the same alias — DeepInfra's ``meta-llama/Meta-Llama-3.1-8B-
Instruct`` and Ollama's ``llama3.1:8b-instruct-q4_K_M`` do not collide.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry, retry_if_exception, stop_after_attempt,
    wait_exponential_jitter, RetryError,
)

from config import (
    LLM_CACHE_DIR,
    HTTP_MAX_RETRIES,
    HTTP_RETRY_INITIAL_BACKOFF_S,
    HTTP_RETRY_MAX_BACKOFF_S,
    get_provider, get_api_key, resolve_model_id,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    latency_ms: float
    provider: str
    model_id: str


# ── Think-tag stripping ───────────────────────────────────────────────────
# Some models (Qwen2.5/3, DeepSeek-R1, and other reasoning-tuned variants)
# emit chain-of-thought inside <think>...</think> blocks even when asked
# for terse answers. Without stripping, EM/F1 metrics collapse to ~0 for
# those models. Stripping is applied to the .text field of every ChatResult,
# on both cache hit and cache miss — the raw response is preserved in the
# on-disk cache for auditability, but callers see clean text.

_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks. Handles truncated (unclosed) blocks."""
    if not text or "<think>" not in text.lower():
        return text
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


# ── Reasoning-model max_tokens boost ──────────────────────────────────────
# Qwen 2.5/3 and DeepSeek-R1 wrap every response in <think>...</think>
# blocks, which consume tokens BEFORE the model gets to the answer. If the
# caller asks for max_tokens=8 (e.g. Self-RAG's relevance check) or
# max_tokens=128 (short-form generator), the model runs out of budget
# inside the reasoning block → response is empty after strip_think_tags →
# metric goes to 0. Empirically the DeepInfra-hosted Qwen 2.5-7B was
# collapsing to 0% accuracy on short-form / true-false / MCQ tasks for
# this reason.
#
# Fix: silently boost max_tokens for reasoning-tuned model IDs. Applies to
# BOTH cache key and API request, so cache is consistent. Callers pass the
# same nominal max_tokens as before — no widespread edits needed.

_REASONING_MODEL_MARKERS = ("qwen", "deepseek-r1", "deepseek_r1")


def _is_reasoning_model(model_id: str) -> bool:
    m = (model_id or "").lower()
    return any(marker in m for marker in _REASONING_MODEL_MARKERS)


def _effective_max_tokens(model_id: str, requested: int) -> int:
    """Boost max_tokens for reasoning-tuned models so <think> + answer fits.

    Rule (empirically calibrated on DeepInfra-hosted Qwen 2.5-7B fp16):
    for reasoning models, ensure ≥ 1024 tokens (covers even small classifier
    calls whose <think> block routinely runs 200-800 tokens before emitting
    a 1-word label) and boost 4× otherwise, capped at 2048.

    Note on Ollama q4_K_M Qwen: quantisation appears to suppress the <think>
    behaviour, so any floor ≥ ~200 works. The 1024 floor is inclusive of
    both providers so a single formula covers all reasoning models.
    """
    if not _is_reasoning_model(model_id):
        return requested
    boosted = max(1024, requested * 4)
    return min(boosted, 2048)


# ── Retry predicate ───────────────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    """Retry only on transport errors, 5xx, and 429. Never on 4xx bad-request."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code >= 500 or code == 429
    if isinstance(exc, httpx.HTTPError):        # transport / connection errors
        return True
    return False


# ── Client ────────────────────────────────────────────────────────────────

class LLMClient:
    """OpenAI-compatible chat client. Provider is either explicit
    (constructor argument) or falls back to config.ACTIVE_PROVIDER.

    Multiple LLMClient instances (e.g. one for deepinfra, one for
    ollama_local) can coexist in the same process — the module-level
    `get_client(provider)` caches them per provider name.
    """

    def __init__(
        self,
        provider: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._provider = get_provider(provider)
        self._api_key = get_api_key(self._provider.name)
        self._cache_dir = cache_dir or LLM_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if self._provider.api_key_env is not None and not self._api_key:
            raise RuntimeError(
                f"Provider {self._provider.name!r} requires an API key. "
                f"Set the env var {self._provider.api_key_env} "
                f"(or OPENAI_API_KEY as fallback) before running.")

        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        elif self._provider.name == "ollama_local":
            # Ollama's OpenAI endpoint accepts any bearer, including empty.
            headers["Authorization"] = "Bearer ollama"

        self._client = httpx.Client(
            base_url=self._provider.base_url,
            headers=headers,
            timeout=self._provider.default_timeout_s,
        )
        self._preflight_done = False

    # ── Provider info (used by runner for logging + CSV row) ─────────────

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def resolve(self, alias_or_id: str) -> str:
        return resolve_model_id(alias_or_id, provider=self._provider.name)

    # ── Public API ────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        use_cache: bool = True,
    ) -> ChatResult:
        """Send a chat completion request.

        `model` may be an alias (`"llama3-8b"`) or a full provider-specific
        model ID (`"meta-llama/..."` or `"llama3.1:8b-instruct-q4_K_M"`).

        For reasoning-tuned models (Qwen, DeepSeek-R1) `max_tokens` is
        auto-boosted so the <think> block + answer fits (see
        `_effective_max_tokens`). Cache key uses the boosted value so
        different runs stay consistent.
        """
        model_id = self.resolve(model)
        max_tokens = _effective_max_tokens(model_id, max_tokens)
        key = self._cache_key(model_id, messages, temperature, max_tokens)

        if use_cache:
            hit = self._cache_get(key)
            if hit is not None:
                return ChatResult(
                    # Strip <think> tags on read so cached responses from
                    # reasoning models produce clean text without a re-fetch.
                    text=strip_think_tags(hit["text"]),
                    prompt_tokens=hit.get("prompt_tokens", 0),
                    completion_tokens=hit.get("completion_tokens", 0),
                    cached=True,
                    latency_ms=0.0,
                    provider=self._provider.name,
                    model_id=model_id,
                )

        if not self._preflight_done:
            self._preflight()
            self._preflight_done = True

        t0 = time.perf_counter()
        try:
            text, usage = self._call_api(
                model_id, messages, temperature, max_tokens)
        except RetryError as re:
            # Retry loop exhausted — surface the underlying exception.
            raise re.last_attempt.exception() from re

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not text:
            # Empty completions from a 200 OK almost always mean the model
            # was cut off or the provider returned an odd payload. Log
            # loudly but do not raise — downstream metrics will show 0.
            _log.warning(
                "Empty completion from %s/%s (prompt_tokens=%d)",
                self._provider.name, model_id, usage.get("prompt_tokens", 0))

        # Persist the RAW text (including any <think> block) so token
        # counts remain reproducible and future re-parsing is possible.
        self._cache_put(key, {
            "provider": self._provider.name,
            "model": model_id,
            "text": text,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
        })

        # Callers see clean text.
        return ChatResult(
            text=strip_think_tags(text),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cached=False,
            latency_ms=latency_ms,
            provider=self._provider.name,
            model_id=model_id,
        )

    # ── Raw-completion path (bypasses server-side chat template) ─────────
    #
    # Needed by baselines whose fine-tuned checkpoint expects a specific
    # inline format (e.g. Self-RAG's "### Instruction:\n{q}\n\n
    # ### Response:\n[Retrieval]<paragraph>...") that Ollama's
    # /v1/chat/completions endpoint would otherwise wrap in an
    # <|im_start|>/<|im_end|> chat template, corrupting the format.
    # Uses the legacy /v1/completions endpoint which passes the prompt
    # through unchanged.

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        use_cache: bool = True,
    ) -> ChatResult:
        """Legacy /v1/completions call. Same cache scheme as chat() —
        we key on a synthetic single-message wrapper so keys are
        distinct from chat() keys but still deterministic.
        """
        model_id = self.resolve(model)
        max_tokens = _effective_max_tokens(model_id, max_tokens)
        # Cache key: separate namespace via a role="__raw__" wrapper so
        # a chat() call with the same string as content won't collide.
        wrapper = [{"role": "__raw__", "content": prompt}]
        key = self._cache_key(model_id, wrapper, temperature, max_tokens)

        if use_cache:
            hit = self._cache_get(key)
            if hit is not None:
                return ChatResult(
                    text=strip_think_tags(hit["text"]),
                    prompt_tokens=hit.get("prompt_tokens", 0),
                    completion_tokens=hit.get("completion_tokens", 0),
                    cached=True,
                    latency_ms=0.0,
                    provider=self._provider.name,
                    model_id=model_id,
                )

        if not self._preflight_done:
            self._preflight()

        text, usage, latency_ms = self._call_completions(
            model_id, prompt, temperature, max_tokens)

        self._cache_put(key, {
            "text": text,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        })

        return ChatResult(
            text=strip_think_tags(text),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cached=False,
            latency_ms=latency_ms,
            provider=self._provider.name,
            model_id=model_id,
        )

    def _call_completions(
        self, model_id: str, prompt: str,
        temperature: float, max_tokens: int,
    ) -> tuple[str, dict[str, int], float]:
        """HTTP call to /v1/completions (legacy) — Ollama passes the
        prompt through without any chat-template wrapping."""

        @retry(
            stop=stop_after_attempt(HTTP_MAX_RETRIES),
            wait=wait_exponential_jitter(
                initial=HTTP_RETRY_INITIAL_BACKOFF_S,
                max=HTTP_RETRY_MAX_BACKOFF_S,
                jitter=2.0),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        def _do() -> tuple[str, dict[str, int], float]:
            payload = {
                "model": model_id,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            t0 = time.perf_counter()
            r = self._client.post("/completions", json=payload)
            if r.status_code == 429:
                retry_after = float(r.headers.get("retry-after", "5"))
                _log.warning("%s 429 (completions): sleeping %.1fs",
                             self._provider.name, retry_after)
                time.sleep(min(retry_after, 30.0))
                r.raise_for_status()
            if r.status_code >= 400:
                body = r.text[:400]
                _log.warning("%s %d (completions): %s",
                             self._provider.name, r.status_code, body)
                r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(
                    f"{self._provider.name} /completions response without "
                    f"choices: {json.dumps(data)[:400]}")
            text = choices[0].get("text") or ""
            usage = data.get("usage") or {}
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return text, {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }, latency_ms

        return _do()

    # ── HTTP call with retry ──────────────────────────────────────────────

    def _call_api(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, int]]:

        @retry(
            stop=stop_after_attempt(HTTP_MAX_RETRIES),
            wait=wait_exponential_jitter(
                initial=HTTP_RETRY_INITIAL_BACKOFF_S,
                max=HTTP_RETRY_MAX_BACKOFF_S,
                jitter=2.0),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        def _do() -> tuple[str, dict[str, int]]:
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            r = self._client.post("/chat/completions", json=payload)
            if r.status_code == 429:
                retry_after = float(r.headers.get("retry-after", "5"))
                _log.warning("%s 429: sleeping %.1fs",
                             self._provider.name, retry_after)
                time.sleep(min(retry_after, 30.0))
                r.raise_for_status()
            if r.status_code >= 400:
                # Log the body for diagnosis before raising.
                body = r.text[:400]
                _log.warning(
                    "%s %d %s: %s",
                    self._provider.name, r.status_code, r.reason_phrase, body)
                r.raise_for_status()

            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(
                    f"{self._provider.name} response without choices: "
                    f"{json.dumps(data)[:400]}")
            text = (choices[0].get("message") or {}).get("content") or ""
            usage = data.get("usage") or {}
            return text, {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }

        return _do()

    # ── Preflight ─────────────────────────────────────────────────────────

    def _preflight(self) -> None:
        """Health check with a few retries — fail fast if truly unreachable."""
        last_exc: BaseException | None = None
        for attempt in range(3):
            try:
                r = self._client.get("/models", timeout=15.0)
                r.raise_for_status()
                _log.info("Provider %s reachable at %s",
                          self._provider.name, self._provider.base_url)
                return
            except httpx.HTTPStatusError as e:
                # DeepInfra returns 404 on /models for the openai path —
                # that just means /chat/completions is the right endpoint.
                if (e.response.status_code == 404
                        and self._provider.name == "deepinfra"):
                    _log.info(
                        "Provider deepinfra reachable at %s "
                        "(no /models endpoint, OK)",
                        self._provider.base_url)
                    return
                # Auth error: no point retrying.
                if e.response.status_code in (401, 403):
                    raise RuntimeError(
                        f"Provider {self._provider.name!r} returned HTTP "
                        f"{e.response.status_code} (auth). "
                        f"Check {self._provider.api_key_env}.") from e
                last_exc = e
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_exc = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                _log.warning("Preflight retry %d/3 for %s",
                             attempt + 2, self._provider.name)
        raise RuntimeError(
            f"Provider {self._provider.name!r} at {self._provider.base_url!r} "
            f"is not reachable after 3 preflight attempts "
            f"({type(last_exc).__name__ if last_exc else '?'}: {last_exc}). "
            f"For ollama_local, is the ollama service running? "
            f"For deepinfra, check network + DEEPINFRA_API_KEY."
        ) from last_exc

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _cache_key(self, model_id: str, messages: list[dict],
                   temperature: float, max_tokens: int) -> str:
        canonical = json.dumps(
            {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        d = self._cache_dir / key[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json"

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            _log.warning("Corrupt cache entry %s (%s); re-fetching", p, e)
            return None

    def _cache_put(self, key: str, value: dict[str, Any]) -> None:
        p = self._cache_path(key)
        # Atomic-ish: write to a sibling tempfile then rename.
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ── Module-level cache keyed by provider name ─────────────────────────────

_clients: dict[str, LLMClient] = {}


def get_client(provider: str | None = None) -> LLMClient:
    """Return a cached client for the given provider (or ACTIVE_PROVIDER)."""
    key = provider or ""     # empty string means "whatever ACTIVE_PROVIDER is"
    if key not in _clients:
        _clients[key] = LLMClient(provider=provider or None)
    return _clients[key]


# Back-compat re-export (some baselines import DeepInfraClient by name).
DeepInfraClient = LLMClient
