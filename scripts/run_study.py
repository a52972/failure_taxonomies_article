"""Run the study systems (docs/20) over one or more question sets.

Systems are addressed by NAME, never by position.

    python scripts/run_study.py --generator llama3-8b --split dev --sets MH MIXED
    python scripts/run_study.py --list --split test --sets MH MIXED

Set → systems:
    MH     no correction (top-3, top-5), every constant strategy, and failure-type
           routing with the oracle, random and self-judge labels.
    MIXED  the same arms with abstention allowed (the generator may answer
           UNANSWERABLE), plus no correction without abstention.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.loop import CorrectiveLoop  # noqa: E402
from src.runner import run_file  # noqa: E402

SYSTEMS: dict[str, dict] = {
    # no correction
    "no_corr":        dict(policy="none"),
    "no_corr_k5":     dict(policy="none", rerank_top_k=5),     # same context size as a corrected answer
    # one strategy applied to every question
    "s_rewrite":      dict(judge="constant:NONE", policy="typed"),
    "s_decompose":    dict(judge="constant:COMPLEX", policy="typed"),
    "s_decomp_iter":  dict(judge="constant:COMPLEX", policy="typed_iter"),
    "s_disambiguate": dict(judge="constant:VAGUE", policy="typed"),
    "s_anchor":       dict(judge="constant:INSUFFICIENT", policy="typed"),
    # failure-type routing (config.STRATEGY_MAP) with three label sources
    "j_oracle":       dict(judge="oracle", policy="typed"),
    "j_random":       dict(judge="random", policy="typed"),
    "j_llm":          dict(judge="llm", policy="typed"),
}
MH_ARMS = list(SYSTEMS)
MIXED_ARMS = ["no_corr_k5"] + [f"{n}_ab" for n in SYSTEMS if n != "no_corr"]
for _n in list(SYSTEMS):
    SYSTEMS[f"{_n}_ab"] = dict(SYSTEMS[_n], allow_abstain=True)
ARMS = {"MH": MH_ARMS, "MIXED": MIXED_ARMS}


def system_name(cfg: dict) -> str:
    """The name CorrectiveLoop.name gives this configuration (used for file names)."""
    from config import RERANK_TOPK
    ab = "+abstain" if cfg.get("allow_abstain") else ""
    if cfg.get("policy") == "none":
        return f"none|k={cfg.get('rerank_top_k', RERANK_TOPK)}{ab}"
    return f"judge={cfg.get('judge', 'oracle')}|policy={cfg['policy']}{ab}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="llama3-8b")
    ap.add_argument("--sets", nargs="+", default=["MH", "MIXED"], choices=list(ARMS))
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--systems", nargs="+", default=None, help="subset of system names")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exp", default=None, help="default: s1_<split>")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    exp = a.exp or f"s1_{a.split}"
    for set_name in a.sets:
        arms = [n for n in ARMS[set_name] if not a.systems or n in a.systems]
        for name in arms:
            if a.list:
                print(f"{set_name:6s} {a.split:5s} {name}")
                continue
            loop = CorrectiveLoop(generator=a.generator, **SYSTEMS[name])
            assert loop.name == system_name(SYSTEMS[name]), (loop.name, system_name(SYSTEMS[name]))
            logging.info("── %s.%s / %s → %s", set_name, a.split, name, loop.name)
            run_file(loop, set_name, a.split, exp=exp, limit=a.limit, generator=a.generator)
