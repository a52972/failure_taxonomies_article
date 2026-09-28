"""One-round corrective loop with injectable judge and dispatch (docs/20).

    pool_0 = rerank(retrieve(question))                      top-3 shown
    policy "none"   → answer from the top-k of the first retrieval
    otherwise       → label = judge(question, pool_0); strategy = dispatch(policy, label)
                      query_1 = reformulate(strategy, question, anchors = pool_0)
                      pool_1 = pool_0 + the best 2 passages of retrieve(query_1) not already shown
                      answer from pool_1

Every system shares the identical first retrieval and answers with the same
generator at temperature 0, so per-question comparisons are paired exactly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from config import (ABSTAIN_TOKEN, DEFAULT_GENERATOR, GEN_MAX_TOKENS_LONG,
                    GEN_MAX_TOKENS_SHORT, GEN_TEMPERATURE, RERANK_TOPK, RETRIEVAL_CACHE,
                    RETRIEVAL_CACHE_DIR, RETRIEVAL_TOPK)
from src.datasets import Question
from src.judges import make_judge
from src.llm_client import get_client
from src.prompts import build_generator_messages
from src.reranker import rerank
from src.retrieval import rrf_fuse
from src.strategies import dispatch, reformulate

_log = logging.getLogger(__name__)

#: new passages appended by a correction ("augment, do not replace")
AUGMENT_K = 2


@dataclass
class RunResult:
    answer: str = ""
    abstained: bool = False
    trace: dict = field(default_factory=dict)
    n_llm_calls: int = 0
    n_llm_calls_uncached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    step_ms: dict = field(default_factory=dict)

    def add_step(self, name: str, ms: float):
        self.step_ms[name] = self.step_ms.get(name, 0.0) + ms


class _Ctx:
    """Call hook handed to judges/strategies; accounts usage into RunResult."""

    def __init__(self, loop: "CorrectiveLoop", result: RunResult):
        self.loop, self.result = loop, result

    def call(self, messages, *, max_tokens: int, step: str, model: str | None = None) -> str:
        r = self.loop.client.chat(messages, model=model or self.loop.generator,
                                  temperature=GEN_TEMPERATURE, max_tokens=max_tokens)
        self.result.n_llm_calls += 1
        if not r.cached:
            self.result.n_llm_calls_uncached += 1
        self.result.prompt_tokens += r.prompt_tokens
        self.result.completion_tokens += r.completion_tokens
        self.result.add_step(step, r.latency_ms)
        return r.text


def _pids(pool):
    return [p.pid for p, _ in pool]


def gold_keys(question: Question) -> set[str]:
    """Gold identifiers valid in the family corpora: suite pids and the text-hash
    pids used by the corpora."""
    from src.corpus import text_hash
    gold = set(question.gold_pids)
    for p in question.passages:
        if p.pid in gold:
            gold.add(text_hash(p.title, p.text))
    return gold


def gold_rank(question: Question, pool, gold: set[str] | None = None) -> int | None:
    """1-based rank of the first gold passage in the pool, or None."""
    gold = gold if gold is not None else gold_keys(question)
    if not gold:
        return None
    for i, (p, _s) in enumerate(pool, start=1):
        if p.pid in gold:
            return i
    return None


def gold_coverage(question: Question, pool, k: int, gold: set[str] | None = None) -> float | None:
    """Fraction of gold passages inside the top-k (None if no gold)."""
    gold = gold if gold is not None else gold_keys(question)
    n_gold = len(set(question.gold_pids))
    if not n_gold:
        return None
    top = {p.pid for p, _ in pool[:k]}
    return min(1.0, len(gold & top) / n_gold)


class CorrectiveLoop:
    """policy: "none" (no correction), "typed" / "typed_iter" (label → strategy,
    config.STRATEGY_MAP), or "always:<strategy>" via a constant judge."""

    def __init__(self, *, generator: str = DEFAULT_GENERATOR, judge: str = "oracle",
                 policy: str = "typed", allow_abstain: bool = False,
                 initial_top_k: int = RETRIEVAL_TOPK, rerank_top_k: int = RERANK_TOPK, seed: int = 42):
        self.generator = generator
        self.client = get_client()
        self.judge = make_judge(judge, seed)
        self.policy = policy
        self.allow_abstain = allow_abstain       # the generator may answer UNANSWERABLE
        self.initial_top_k = initial_top_k
        self.rerank_top_k = rerank_top_k         # context size when no correction runs
        self._current_question: Question | None = None

    @property
    def name(self) -> str:
        ab = "+abstain" if self.allow_abstain else ""
        if self.policy == "none":
            return f"none|k={self.rerank_top_k}{ab}"
        return f"judge={self.judge.name}|policy={self.policy}{ab}"

    # ── retrieval over the family corpus ──────────────────────────────
    def _retrieve(self, query: str | list[str], result: RunResult, *, rescore_query: str,
                  top_k: int = RERANK_TOPK):
        """Retrieve and rerank one query, or several sub-queries (decompose): each
        sub-query is retrieved and reranked, the lists are RRF-fused, and the
        fused set is re-scored against `rescore_query` (the original question)."""
        t0 = time.perf_counter()
        key = self._retrieval_key(query, rescore_query, top_k) if RETRIEVAL_CACHE else None
        if key:
            hit = self._retrieval_cache_get(key)
            if hit is not None:
                result.add_step("retrieve", (time.perf_counter() - t0) * 1000.0)
                return hit
        queries = query if isinstance(query, list) else [query]
        pools = [rerank(q, [p for p, _ in self._corpus_candidates(q)], top_k=top_k, sigmoid=True)
                 for q in queries]
        if len(pools) == 1:
            pool = pools[0]
        else:
            fused = rrf_fuse(pools)[: top_k * 2]
            pool = rerank(rescore_query, [p for p, _ in fused], top_k=top_k, sigmoid=True)
        if key:
            self._retrieval_cache_put(key, pool)
        result.add_step("retrieve", (time.perf_counter() - t0) * 1000.0)
        return pool

    # ── retrieval cache (config.RETRIEVAL_CACHE) ──────────────────────
    def _retrieval_key(self, query, rescore_query: str, top_k: int) -> str:
        import hashlib
        import json
        from config import CORPUS_DIR
        from src.corpus import FAMILY_FOR_CLASS
        q = self._current_question
        cons = q.construction or {}
        spec = {"corpus": CORPUS_DIR.name, "family": FAMILY_FOR_CLASS[q.cls or "ANSWERABLE"],
                "query": query, "rescore": rescore_query, "top_k": top_k, "initial_top_k": self.initial_top_k,
                "exclude": sorted(cons.get("exclude_hashes") or []),
                "scrub": list(q.golden_answers) if cons.get("scrub_golds") else None}
        return hashlib.sha1(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _retrieval_cache_get(self, key: str):
        import json
        from src.corpus import FAMILY_FOR_CLASS, get_corpus
        p = RETRIEVAL_CACHE_DIR / key[:2] / f"{key}.json"
        if not p.exists():
            return None
        corpus = get_corpus(FAMILY_FOR_CLASS[self._current_question.cls or "ANSWERABLE"])
        if not hasattr(corpus, "_by_pid"):
            corpus._by_pid = {x.pid: x for x in corpus.passages}
        return [(corpus._by_pid[pid], s) for pid, s in json.loads(p.read_text())]

    def _retrieval_cache_put(self, key: str, pool) -> None:
        import json
        p = RETRIEVAL_CACHE_DIR / key[:2] / f"{key}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps([[x.pid, s] for x, s in pool]))
        tmp.replace(p)

    def _corpus_candidates(self, query: str):
        from src.corpus import FAMILY_FOR_CLASS, get_corpus
        q = self._current_question
        corpus = get_corpus(FAMILY_FOR_CLASS[q.cls or "ANSWERABLE"])
        cons = q.construction or {}
        return corpus.search(
            query, k=self.initial_top_k,
            exclude_hashes=set(cons.get("exclude_hashes") or []),
            scrub_golds=q.golden_answers if cons.get("scrub_golds") else None)

    def _decompose_iter(self, question: Question, ctx: "_Ctx", result: RunResult, rec: dict) -> list[str]:
        """Answer-informed sequential decomposition: Q1 → retrieve → answer A1 →
        substitute into Q2. Returns both sub-queries, which are retrieved and fused."""
        from config import REFORMULATION_MAX_TOKENS
        from src.prompts import build_decompose_iter_messages, build_subanswer_messages, parse_q1_q2
        raw = ctx.call(build_decompose_iter_messages(question.question),
                       max_tokens=REFORMULATION_MAX_TOKENS, step="reformulate")
        q1, q2 = parse_q1_q2(raw, question.question)
        pool1 = self._retrieve(q1, result, rescore_query=q1)
        a1 = ctx.call(build_subanswer_messages(q1, pool1), max_tokens=32, step="generate").strip()
        a1 = a1.split("\n")[0].strip().strip('"\'.')
        q2s = q2.replace("[A1]", a1) if "[A1]" in q2 else q2
        rec["decompose_iter"] = {"q1": q1, "a1": a1, "q2": q2s}
        return [q1, q2s]

    def _max_tokens(self, q: Question) -> int:
        return GEN_MAX_TOKENS_LONG if q.task_type == "multi_hop" else GEN_MAX_TOKENS_SHORT

    # ── main ──────────────────────────────────────────────────────────
    def run(self, question: Question) -> RunResult:
        from src.state import retrieval_state
        result = RunResult()
        ctx = _Ctx(self, result)
        self._current_question = question
        t_start = time.perf_counter()
        trace: dict = {"system": self.name, "iterations": []}

        pool0 = self._retrieve(question.question, result, rescore_query=question.question)
        rec = {"iteration": 0, "query": question.question,
               "pool_pids": _pids(pool0), "pool_scores": [round(s, 4) for _, s in pool0],
               "gold_rank": gold_rank(question, pool0),
               "gold_cov": gold_coverage(question, pool0, RERANK_TOPK),
               "state": retrieval_state(question, pool0)}
        trace["iterations"].append(rec)

        if self.policy == "none":
            final_pool = pool0 if self.rerank_top_k == RERANK_TOPK else \
                self._retrieve(question.question, result, rescore_query=question.question,
                               top_k=self.rerank_top_k)
        else:
            label = self.judge.label(question, question.question, pool0, ctx)
            strategy = dispatch(self.policy, label)
            rec["label"], rec["strategy"] = label, strategy
            if strategy == "decompose_iter":
                query = self._decompose_iter(question, ctx, result, rec)
            else:
                query = reformulate(strategy, question.question, pool0, ctx)
            cand = self._retrieve(query, result, rescore_query=question.question)
            have = set(_pids(pool0))
            new = [(p, s) for p, s in cand if p.pid not in have][:AUGMENT_K]
            final_pool = list(pool0) + new
            trace["iterations"].append({"iteration": 1, "query": query,
                                        "pool_pids": _pids(final_pool),
                                        "pool_scores": [round(s, 4) for _, s in final_pool],
                                        "gold_rank": gold_rank(question, final_pool),
                                        "gold_cov": gold_coverage(question, final_pool, len(final_pool))})

        result.answer = ctx.call(build_generator_messages(question, final_pool,
                                                          allow_abstain=self.allow_abstain),
                                 max_tokens=self._max_tokens(question), step="generate")
        if self.allow_abstain and ABSTAIN_TOKEN in (result.answer or "").upper():
            result.abstained = True
        result.trace = trace | {"n_iterations": len(trace["iterations"]),
                                "final_pool_pids": _pids(final_pool),
                                "final_gold_rank": gold_rank(question, final_pool),
                                "final_gold_cov": gold_coverage(question, final_pool, len(final_pool))}
        result.latency_ms = (time.perf_counter() - t_start) * 1000.0
        return result
