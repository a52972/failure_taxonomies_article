"""Verify that the retrieval cache (config.RETRIEVAL_CACHE) changes nothing.

Re-runs three arms on the first N questions of MH.dev and MIXED.dev with the cache ON
(first arm fills it, the others read it) and compares every row with the stored dev
rows, which were produced without the cache: answer, passages shown before and after
correction, and their scores must be identical.

    FTC_RETRIEVAL_CACHE=1 python scripts/check_retrieval_cache.py --generator llama3-8b --n 60
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import RAW_RESULTS_DIR, RETRIEVAL_CACHE  # noqa: E402
from run_study import SYSTEMS, system_name  # noqa: E402
from src.loop import CorrectiveLoop  # noqa: E402
from src.runner import run_file, system_tag  # noqa: E402

ARMS = {"MH": ["no_corr_k5", "s_disambiguate", "s_decompose"],
        "MIXED": ["no_corr_k5_ab", "s_disambiguate_ab", "s_decompose_ab"]}
FIELDS = ["answer", "abstained", "state_t0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="llama3-8b")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    assert RETRIEVAL_CACHE, "set FTC_RETRIEVAL_CACHE=1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    report, bad = {}, 0
    for pas in (1, 2):                    # pass 1 fills the cache, pass 2 reads everything from it
        for set_name, arms in ARMS.items():
            for arm in arms:
                exp = f"s1_cachecheck_p{pas}"
                loop = CorrectiveLoop(generator=a.generator, **SYSTEMS[arm])
                out = run_file(loop, set_name, "dev", exp=exp, limit=a.n, generator=a.generator)
                ref = RAW_RESULTS_DIR / "s1_dev" / f"{system_tag(system_name(SYSTEMS[arm]))}__{set_name}.dev__{a.generator}.jsonl"
                R = {r["qid"]: r for r in map(json.loads, ref.open(encoding="utf-8"))}
                n = diff = 0
                for r in map(json.loads, out.open(encoding="utf-8")):
                    o = R[r["qid"]]
                    same = all(r[f] == o[f] for f in FIELDS) and \
                        [(i["pool_pids"], i["pool_scores"]) for i in r["trace"]["iterations"]] == \
                        [(i["pool_pids"], i["pool_scores"]) for i in o["trace"]["iterations"]]
                    n += 1
                    diff += not same
                report[f"pass{pas}/{set_name}/{arm}"] = {"n": n, "different": diff}
                bad += diff
    report["identical"] = bad == 0
    print(json.dumps(report, indent=2))
    (RAW_RESULTS_DIR.parent / "analysis").mkdir(parents=True, exist_ok=True)
    (RAW_RESULTS_DIR.parent / "analysis" / "retrieval_cache_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
