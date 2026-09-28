"""Secondary analysis (docs/20): can the retrieval state be read from what is
observable after the first retrieval? Measurement only; no system routes on it.

    python scripts/predictability.py features --sets STATE_TRAIN.train MH.dev MIXED.dev
    python scripts/predictability.py train                     # on STATE_TRAIN only
    python scripts/predictability.py evaluate --sets MH.test MIXED.test

Predictors of the true state (NONE_FOUND / PARTIAL / COMPLETE of src/state.py):
    text       DistilBERT on the question and the top-3 passages (title, first 300
               characters, re-ranker score)
    question   the same model and training rows, question only
    scores     multinomial logistic regression on the three re-ranker scores
    majority   the most frequent training label
Training uses STATE_TRAIN only; hyper-parameters are fixed here and not tuned.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, RERANK_TOPK, SEED, SUITE_DIR  # noqa: E402

LABELS = ["NONE_FOUND", "PARTIAL", "COMPLETE"]
FEAT_DIR = DATA_DIR / "predictability"
MODEL_DIR = DATA_DIR / "models" / "state_s1"
HP = dict(base="distilbert-base-uncased", epochs=4, lr=3e-5, batch=16, max_len=512, holdout=0.10, seed=SEED)


def featurise(question: str, pool) -> str:
    parts = [f"question: {question}"]
    for i, (p, s) in enumerate(pool[:RERANK_TOPK], start=1):
        parts.append(f"[{i}] score={s:.2f} {p.title}: {(p.text or '')[:300]}")
    return "\n".join(parts)


def build_features(set_names: list[str]) -> None:
    from src.datasets import read_jsonl
    from src.loop import CorrectiveLoop, RunResult
    from src.state import retrieval_state
    loop = CorrectiveLoop(policy="none")
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    for sn in set_names:
        out = FEAT_DIR / f"{sn}.jsonl"
        done = {json.loads(l)["qid"] for l in out.open()} if out.exists() else set()
        qs = [q for q in read_jsonl(SUITE_DIR / "s1" / f"{sn}.jsonl") if q.qid not in done]
        logging.info("%s: %d to build", sn, len(qs))
        with out.open("a", encoding="utf-8") as f:
            for i, q in enumerate(qs, 1):
                loop._current_question = q
                pool = loop._retrieve(q.question, RunResult(), rescore_query=q.question)
                f.write(json.dumps({"qid": q.qid, "cls": q.cls, "source": q.source,
                                    "label": retrieval_state(q, pool),
                                    "scores": [round(s, 4) for _, s in pool[:RERANK_TOPK]],
                                    "text": featurise(q.question, pool)}, ensure_ascii=False) + "\n")
                if i % 200 == 0:
                    f.flush()
                    logging.info("  %d/%d", i, len(qs))


def rows(sn: str) -> list[dict]:
    return [json.loads(l) for l in (FEAT_DIR / f"{sn}.jsonl").open(encoding="utf-8")]


def train_text(name: str, question_only: bool) -> None:
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                              TrainingArguments)
    data = rows("STATE_TRAIN.train")
    if question_only:
        data = [dict(r, text=r["text"].split("\n", 1)[0]) for r in data]
    rng = random.Random(HP["seed"])
    parents = sorted({r["qid"].split("#")[0] for r in data})
    rng.shuffle(parents)
    held = set(parents[: int(len(parents) * HP["holdout"])])
    tr = [r for r in data if r["qid"].split("#")[0] not in held]
    va = [r for r in data if r["qid"].split("#")[0] in held]
    tok = AutoTokenizer.from_pretrained(HP["base"])

    def enc(b):
        e = tok(b["text"], truncation=True, max_length=HP["max_len"])
        e["labels"] = [LABELS.index(l) for l in b["label"]]
        return e
    cols = sorted(tr[0])
    ds_tr = Dataset.from_list(tr).map(enc, batched=True, remove_columns=cols)
    ds_va = Dataset.from_list(va).map(enc, batched=True, remove_columns=cols)
    model = AutoModelForSequenceClassification.from_pretrained(HP["base"], num_labels=len(LABELS))
    out = MODEL_DIR / name
    args = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=HP["epochs"],
                             learning_rate=HP["lr"], per_device_train_batch_size=HP["batch"],
                             per_device_eval_batch_size=32, eval_strategy="epoch", save_strategy="no",
                             logging_steps=50, report_to=[], fp16=torch.cuda.is_available(), seed=HP["seed"])
    t = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va, processing_class=tok,
                compute_metrics=lambda p: metrics(p.label_ids, p.predictions.argmax(-1)))
    t.train()
    rep = t.evaluate()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "meta.json").write_text(json.dumps({"question_only": question_only, "hp": HP, "n_train": len(tr),
                                               "n_val": len(va), "val": rep,
                                               "train_labels": dict(Counter(r["label"] for r in tr))}, indent=2))


def metrics(y, pred) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    f1 = {}
    for c, lab in enumerate(LABELS):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1[lab] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    present = [LABELS[c] for c in sorted(set(y.tolist()))]
    return {"accuracy": float((pred == y).mean()), "macro_f1": float(np.mean([f1[l] for l in present])),
            **{f"f1_{k}": v for k, v in f1.items()}}


def predict_text(name: str, data: list[dict]) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    d = MODEL_DIR / name
    q_only = json.loads((d / "meta.json").read_text())["question_only"]
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    out = []
    for i in range(0, len(data), 32):
        texts = [r["text"].split("\n", 1)[0] if q_only else r["text"] for r in data[i:i + 32]]
        enc = tok(texts, truncation=True, max_length=HP["max_len"], padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            out.extend(model(**enc).logits.argmax(-1).tolist())
    return np.array(out)


def evaluate(set_names: list[str]) -> None:
    from sklearn.linear_model import LogisticRegression
    train = rows("STATE_TRAIN.train")
    Xtr = np.array([r["scores"] + [0.0] * (RERANK_TOPK - len(r["scores"])) for r in train])
    ytr = np.array([LABELS.index(r["label"]) for r in train])
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    majority = Counter(ytr.tolist()).most_common(1)[0][0]
    report = {"hp": HP, "train_labels": dict(Counter(r["label"] for r in train))}
    for sn in set_names:
        data = rows(sn)
        y = np.array([LABELS.index(r["label"]) for r in data])
        X = np.array([r["scores"] + [0.0] * (RERANK_TOPK - len(r["scores"])) for r in data])
        report[sn] = {"n": len(data), "true_labels": dict(Counter(r["label"] for r in data)),
                      "text": metrics(y, predict_text("text", data)),
                      "question": metrics(y, predict_text("question", data)),
                      "scores": metrics(y, lr.predict(X)),
                      "majority": metrics(y, np.full(len(y), majority))}
    out = ROOT / "results" / "analysis" / "predictability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["features", "train", "evaluate"])
    ap.add_argument("--sets", nargs="+", default=["STATE_TRAIN.train"])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if a.step == "features":
        build_features(a.sets)
    elif a.step == "train":
        train_text("text", question_only=False)
        train_text("question", question_only=True)
    else:
        evaluate(a.sets)
