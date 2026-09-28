"""Source loaders for the Failure-Type-Controlled suite.

Every loader returns `list[Question]` with a native passage pool and, where
the source provides it, `gold_pids` and `hops`. Class assignment and
constructions (gold drops, padding, splits) live in `construct.py`.

Sources (all public, all ship their own pools):
    musique      dgslibisey/MuSiQue dev          20 paras, is_supporting, decomposition
    hotpotqa     hotpotqa/hotpot_qa distractor   10 paras, supporting_facts, type/level
    2wiki        FlashRAG 2wikimultihopqa/dev    10 paras, supporting_facts, type
    condambigqa  Apocalypse-AGI-DAO/CondAmbigQA  20 ctxs, properties (conditions)
    asqa         awinml Self-RAG asqa gtr top100 100 docs, qa_pairs
    popqa        awinml popqa_longtail_w_gs      ctxs (Contriever)
    triviaqa     awinml triviaqa_test_w_gs       ctxs (Contriever)
    squad        FlashRAG squad/dev              own paragraph (+ distractors added here)
    nomiracl     miracl/nomiracl {en,es,fr}      judged passages, no answers
    arc_c        awinml arc_challenge_processed  MCQ
    pubhealth    awinml health_claims_processed  true/false
"""
from __future__ import annotations

import gzip
import json
import logging
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

from config import HF_CACHE_DIR, POOL_SIZE, SEED
from src.datasets import Passage, Question

_log = logging.getLogger(__name__)


def _dl(repo: str, path: str) -> Path:
    return Path(hf_hub_download(repo, path, repo_type="dataset",
                                cache_dir=str(HF_CACHE_DIR)))


def _iter_jsonl(fp: Path):
    op = gzip.open if fp.suffix == ".gz" else open
    with op(fp, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _ctxs(prefix: str, ctxs: list[dict], limit: int | None = None) -> list[Passage]:
    out = []
    for i, c in enumerate(ctxs or []):
        if limit is not None and i >= limit:
            break
        pid = str(c.get("id") or c.get("docid") or f"{prefix}_p{i}")
        sc = c.get("score")
        try:
            sc = float(sc) if sc is not None else None
        except (TypeError, ValueError):
            sc = None
        out.append(Passage(pid=pid, title=str(c.get("title") or ""),
                           text=str(c.get("text") or c.get("paragraph_text") or ""),
                           source_score=sc))
    return out


# ── multi-hop ─────────────────────────────────────────────────────────

def load_musique(split: str = "dev") -> list[Question]:
    fp = _dl("dgslibisey/MuSiQue", f"musique_ans_v1.0_{split}.jsonl")
    out = []
    for r in _iter_jsonl(fp):
        qid = str(r["id"])
        passages = [Passage(pid=f"{qid}_p{p['idx']}", title=p["title"],
                            text=p["paragraph_text"]) for p in r["paragraphs"]]
        gold = [f"{qid}_p{p['idx']}" for p in r["paragraphs"] if p.get("is_supporting")]
        hops = [{"question": h["question"], "answer": h["answer"],
                 "pid": f"{qid}_p{h['paragraph_support_idx']}"}
                for h in r.get("question_decomposition", [])]
        out.append(Question(
            qid=f"musique:{qid}", question=r["question"],
            golden_answers=[r["answer"], *r.get("answer_aliases", [])],
            task_type="multi_hop", passages=passages, source="musique",
            gold_pids=gold, hops=hops,
            metadata={"n_hops": len(hops), "answerable": r.get("answerable", True)}))
    return out


def load_hotpotqa() -> list[Question]:
    import pandas as pd
    fp = _dl("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet")
    df = pd.read_parquet(fp)
    out = []
    for r in df.itertuples(index=False):
        qid = str(r.id)
        titles = list(r.context["title"])
        sents = list(r.context["sentences"])
        passages = [Passage(pid=f"{qid}_p{i}", title=str(t),
                            text=" ".join(list(s)) if not isinstance(s, str) else s)
                    for i, (t, s) in enumerate(zip(titles, sents))]
        sup = set(map(str, r.supporting_facts["title"]))
        gold = [f"{qid}_p{i}" for i, t in enumerate(titles) if str(t) in sup]
        out.append(Question(
            qid=f"hotpotqa:{qid}", question=str(r.question),
            golden_answers=[str(r.answer)], task_type="multi_hop",
            passages=passages, source="hotpotqa", gold_pids=gold,
            metadata={"type": str(r.type), "level": str(r.level), "n_hops": 2}))
    return out


def load_2wiki() -> list[Question]:
    fp = _dl("RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa/dev.jsonl")
    out = []
    for r in _iter_jsonl(fp):
        qid = str(r["id"])
        md = r.get("metadata") or {}
        ctx = md.get("context") or {}
        titles = ctx.get("title") or []
        contents = ctx.get("content") or []
        passages = [Passage(pid=f"{qid}_p{i}", title=str(t),
                            text=" ".join(c) if isinstance(c, list) else str(c))
                    for i, (t, c) in enumerate(zip(titles, contents))]
        sup = set(map(str, (md.get("supporting_facts") or {}).get("title") or []))
        gold = [f"{qid}_p{i}" for i, t in enumerate(titles) if str(t) in sup]
        out.append(Question(
            qid=f"2wiki:{qid}", question=r["question"],
            golden_answers=list(r.get("golden_answers") or []),
            task_type="multi_hop", passages=passages, source="2wiki",
            gold_pids=gold, metadata={"type": md.get("type"), "n_hops": len(gold)}))
    return out


# ── ambiguity ─────────────────────────────────────────────────────────

def load_condambigqa() -> list[Question]:
    fp = _dl("Apocalypse-AGI-DAO/CondAmbigQA-2K", "CondAmbigQA.json")
    data = json.load(open(fp, encoding="utf-8"))
    out = []
    for r in data:
        qid = str(r["id"])
        passages = _ctxs(qid, r.get("ctxs") or [], limit=20)
        props = r.get("properties") or []
        golds = [str(p.get("groundtruth") or "").strip() for p in props]
        golds = [g for g in golds if g]
        # citations: keep raw; resolved to pids by title match in construct
        out.append(Question(
            qid=f"condambigqa:{qid}", question=r["question"],
            golden_answers=golds, task_type="short_form", passages=passages,
            source="condambigqa",
            metadata={"n_interp": len(props),
                      "properties": [{"condition": p.get("condition"),
                                      "groundtruth": p.get("groundtruth"),
                                      "citations": p.get("citations")} for p in props]}))
    return out


def load_asqa(top_docs: int = 20) -> list[Question]:
    fp = _dl("awinml/crag_self_rag_datasets", "eval_data/asqa_eval_gtr_top100.json")
    data = json.load(open(fp, encoding="utf-8"))
    out = []
    for r in data:
        qid = str(r["sample_id"])
        passages = _ctxs(qid, r.get("docs") or [], limit=top_docs)
        pairs = r.get("qa_pairs") or []
        golds = []
        for p in pairs:
            golds.extend([str(a) for a in (p.get("short_answers") or [])])
        out.append(Question(
            qid=f"asqa:{qid}", question=r["question"], golden_answers=golds,
            task_type="short_form", passages=passages, source="asqa",
            metadata={"n_interp": len(pairs),
                      "sub_questions": [p.get("question") for p in pairs]}))
    return out


def load_asqa_train() -> list[Question]:
    """ASQA train split (din0s/asqa). The pilot used every eligible question of
    the dev split (the Self-RAG GTR file), so the study draws ambiguous
    questions from train. Train has no retrieved documents; the pool is the
    evidence annotated in ASQA itself: the Wikipedia context of each
    disambiguated question (qa_pairs[*].context) and the annotators'
    knowledge passages (annotations[*].knowledge)."""
    import pandas as pd
    fp = _dl("din0s/asqa", "data/train-00000-of-00001-87b7d64f7913b544.parquet")
    df = pd.read_parquet(fp)
    out = []
    for r in df.itertuples(index=False):
        qid = str(r.sample_id)
        texts: list[tuple[str, str]] = []
        for p in r.qa_pairs:
            c = str(p.get("context") or "").strip()
            if c and c != "No context provided":
                texts.append((str(p.get("wikipage") or ""), c))
        for a in r.annotations:
            for k in (a.get("knowledge") if a.get("knowledge") is not None else []):
                c = str(k.get("content") or "").strip()
                if c:
                    texts.append((str(k.get("wikipage") or ""), c))
        seen, passages = set(), []
        for i, (t, c) in enumerate(texts):
            if c in seen:
                continue
            seen.add(c)
            passages.append(Passage(pid=f"{qid}_e{i}", title=t, text=c))
        golds = [str(a) for p in r.qa_pairs for a in (p.get("short_answers") if p.get("short_answers") is not None else [])]
        out.append(Question(
            qid=f"asqa_train:{qid}", question=str(r.ambiguous_question), golden_answers=golds,
            task_type="short_form", passages=passages, source="asqa_train",
            metadata={"n_interp": len(r.qa_pairs),
                      "sub_questions": [p.get("question") for p in r.qa_pairs]}))
    return out


# ── single-hop answerable ─────────────────────────────────────────────

def _load_awinml_jsonl(name: str, task_type: str, source: str) -> list[Question]:
    fp = _dl("awinml/crag_self_rag_datasets", f"eval_data/{name}")
    out = []
    for r in _iter_jsonl(fp):
        qid = str(r.get("id") or r.get("question", "")[:60])
        out.append(Question(
            qid=f"{source}:{qid}", question=r.get("question") or r.get("claim") or "",
            golden_answers=[str(a) for a in (r.get("answers") or r.get("golden_answers") or [])],
            task_type=task_type, passages=_ctxs(qid, r.get("ctxs") or []),
            source=source, metadata={k: r.get(k) for k in ("prop", "pop") if k in r}))
    return out


def load_popqa() -> list[Question]:
    return _load_awinml_jsonl("popqa_longtail_w_gs.jsonl", "short_form", "popqa")


def load_triviaqa() -> list[Question]:
    return _load_awinml_jsonl("triviaqa_test_w_gs.jsonl", "short_form", "triviaqa")


def load_squad(seed: int = SEED, pool_size: int = POOL_SIZE) -> list[Question]:
    """SQuAD dev: own paragraph + (pool_size-1) distractor paragraphs from
    other articles. Distractors are drawn deterministically."""
    fp = _dl("RUC-NLPIR/FlashRAG_datasets", "squad/dev.jsonl")
    rows = list(_iter_jsonl(fp))
    # unique paragraphs keyed by (title, text)
    paras: dict[tuple[str, str], str] = {}
    for r in rows:
        md = r.get("metadata") or {}
        key = (str(md.get("title") or ""), str(md.get("text") or ""))
        paras.setdefault(key, f"squad_par{len(paras)}")
    keys = list(paras)
    rng = random.Random(seed)
    out = []
    for r in rows:
        qid = str(r["id"])
        md = r.get("metadata") or {}
        own = (str(md.get("title") or ""), str(md.get("text") or ""))
        own_pid = paras[own]
        pool = [Passage(pid=own_pid, title=own[0], text=own[1])]
        while len(pool) < pool_size:
            k = keys[rng.randrange(len(keys))]
            if k[0] == own[0]:
                continue
            pool.append(Passage(pid=paras[k], title=k[0], text=k[1]))
        rng.shuffle(pool)
        out.append(Question(
            qid=f"squad:{qid}", question=r["question"],
            golden_answers=list(r.get("golden_answers") or []),
            task_type="short_form", passages=pool, source="squad",
            gold_pids=[own_pid], construction={"distractors": "random_other_article"}))
    return out


# ── NoMIRACL (natural insufficiency, no answers) ──────────────────────

_NOMIRACL_LANG = {"en": "english", "es": "spanish", "fr": "french", "de": "german"}


def load_nomiracl(lang: str = "en", subset: str = "non_relevant", split: str = "test") -> list[Question]:
    L = _NOMIRACL_LANG[lang]
    topics = _dl("miracl/nomiracl", f"data/{L}/topics/{split}.{subset}.tsv")
    qrels = _dl("miracl/nomiracl", f"data/{L}/qrels/{split}.{subset}.tsv")
    corpus = _dl("miracl/nomiracl", f"data/{L}/corpus.jsonl.gz")
    docs = {r["docid"]: r for r in _iter_jsonl(corpus)}
    rel: dict[str, list[tuple[str, int]]] = {}
    for line in open(qrels, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            rel.setdefault(parts[0], []).append((parts[2], int(parts[3])))
    out = []
    for line in open(topics, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        qid, query = parts[0], parts[1]
        pool, gold = [], []
        for docid, r in rel.get(qid, []):
            d = docs.get(docid)
            if d is None:
                continue
            pool.append(Passage(pid=docid, title=d.get("title", ""), text=d.get("text", "")))
            if r > 0:
                gold.append(docid)
        if not pool:
            continue
        out.append(Question(
            qid=f"nomiracl-{lang}:{qid}", question=query, golden_answers=[],
            task_type="short_form", passages=pool, source=f"nomiracl-{lang}",
            language=lang, gold_pids=gold,
            metadata={"subset": subset, "has_relevant": bool(gold)}))
    return out


# ── closed-form ───────────────────────────────────────────────────────

def load_arc_c() -> list[Question]:
    fp = _dl("awinml/crag_self_rag_datasets", "eval_data/arc_challenge_processed.jsonl")
    out = []
    for r in _iter_jsonl(fp):
        qid = str(r["id"])
        ch = r.get("choices") or {}
        labels = ch.get("label") or ["A", "B", "C", "D"]
        texts = ch.get("text") or []
        out.append(Question(
            qid=f"arc_c:{qid}", question=r["question"],
            golden_answers=[str(r["answerKey"])], task_type="mcq",
            passages=_ctxs(qid, r.get("ctxs") or []),
            choices=[f"{l}. {t}" for l, t in zip(labels, texts)],
            source="arc_c", metadata={"labels": list(labels)}))
    return out


def load_pubhealth() -> list[Question]:
    fp = _dl("awinml/crag_self_rag_datasets", "eval_data/health_claims_processed.jsonl")
    out = []
    for r in _iter_jsonl(fp):
        q = r.get("question") or r.get("claim") or ""
        ans = [str(a).lower() for a in (r.get("answers") or [])]
        if not ans and r.get("label"):
            ans = ["true"] if str(r["label"]).upper() == "SUPPORTS" else ["false"]
        qid = str(r.get("id") or q[:60])
        out.append(Question(
            qid=f"pubhealth:{qid}", question=q, golden_answers=ans,
            task_type="true_false", passages=_ctxs(qid, r.get("ctxs") or []),
            source="pubhealth"))
    return out


LOADERS = {
    "musique": load_musique, "hotpotqa": load_hotpotqa, "2wiki": load_2wiki,
    "musique_train": lambda: [q.with_(source="musique_train") for q in load_musique("train")],
    "asqa_train": load_asqa_train,
    "condambigqa": load_condambigqa, "asqa": load_asqa,
    "popqa": load_popqa, "triviaqa": load_triviaqa, "squad": load_squad,
    "nomiracl-en": lambda: load_nomiracl("en"),
    "nomiracl-es": lambda: load_nomiracl("es"),
    "nomiracl-fr": lambda: load_nomiracl("fr"),
    "arc_c": load_arc_c, "pubhealth": load_pubhealth,
}
