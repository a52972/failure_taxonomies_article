"""Rebuild the BM25 indexes and dense-embedding matrices of the study corpora from the
released passage texts (data/corpus_s1/<family>/passages.jsonl).

The exact matrices used in the study are published as a release asset
(corpus_s1_indexes.tar); use this script only if you cannot download it. Embeddings are
recomputed with BGE-M3 through Ollama and may differ from the originals in the last
floating-point digits, which can change a few retrieval ranks.

    python scripts/rebuild_indexes.py [--families multihop single ambig]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import FamilyCorpus  # noqa: E402
from src.datasets import Passage  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["multihop", "single", "ambig"])
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    for fam in a.families:
        c = FamilyCorpus(fam)
        rows = [json.loads(l) for l in (c.dir / "passages.jsonl").open(encoding="utf-8")]
        c.build([Passage(pid=r["pid"], title=r["title"], text=r["text"]) for r in rows], embed_batch=a.batch)
