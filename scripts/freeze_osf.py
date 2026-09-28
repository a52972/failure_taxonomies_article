"""Freeze the analysis plan and build the OSF registration folder (docs/20 §6, step 4).

    python scripts/freeze_osf.py        → osf_registration/

1. SHA-256 of every frozen file (code, question sets, corpora, classifier weights,
   pilot exclusion list, development results) → written into §10 of the plan together
   with the freeze date.
2. SHA-256 of the completed plan → frozen_files.sha256 lists everything, plan included.
3. osf_registration/ gets the plan, the hash list, the construction audit, the set
   manifest, the dev analysis, the question sets and the frozen code (zip), and the
   registration summary.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "osf_registration"
PLAN = ROOT / "docs" / "21_analysis_plan.md"

GROUPS = {
    "Analysis plan and design": ["docs/20_study_design.md"],
    "Analysis and run code": ["scripts/analyze_study.py", "scripts/run_study.py", "scripts/run_study.sh",
                              "scripts/run_latency.sh", "scripts/judge_audit.py", "scripts/predictability.py",
                              "scripts/check_retrieval_cache.py", "scripts/build_study.py", "scripts/audit_study.py",
                              "scripts/freeze_osf.py", "config.py"],
    "Pipeline code": sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")),
    "Question sets": sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data/suite/s1").glob("*")),
    "Pilot exclusion list": ["data/pilot_used_qids.json"],
    "Retrieval corpora": [f"data/corpus_s1/{f}/{x}" for f in ("multihop", "single", "ambig")
                          for x in ("passages.jsonl", "emb.npy")],
    "Predictability classifiers": [f"data/models/state_s1/{m}/{x}" for m in ("text", "question")
                                   for x in ("model.safetensors", "meta.json")],
    "Development results": ["results/analysis/s1_dev.json", "results/analysis/retrieval_cache_check.json"]
                           + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "results/raw/s1_dev").glob("*.jsonl")),
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    plan = PLAN.read_text(encoding="utf-8")
    if "{{HASHES}}" not in plan or "{{FREEZE_DATE}}" not in plan:
        sys.exit("plan already frozen (no placeholders left) — refusing to overwrite")
    hashes: dict[str, str] = {}
    for files in GROUPS.values():
        for f in files:
            p = ROOT / f
            assert p.exists(), f
            hashes[f] = sha(p)
    key = ["scripts/analyze_study.py", "scripts/run_study.py", "src/loop.py", "src/prompts.py", "src/state.py",
           "config.py", "data/suite/s1/MH.test.jsonl", "data/suite/s1/MIXED.test.jsonl",
           "data/suite/s1/MIXED.dev.jsonl", "data/suite/s1/MH.dev.jsonl"]
    sec = ["The SHA-256 of every frozen file (%d files: code, question sets, corpora," % len(hashes),
           "classifier weights, pilot exclusion list, development results) is listed in",
           "`frozen_files.sha256`, registered with this plan. The most important ones:", "",
           "| file | SHA-256 |", "|---|---|"]
    sec += [f"| `{f}` | `{hashes[f]}` |" for f in key]
    today = dt.date.today().isoformat()
    plan = plan.replace("{{HASHES}}", "\n".join(sec)).replace("{{FREEZE_DATE}}", today)
    PLAN.write_text(plan, encoding="utf-8")
    hashes = {"docs/21_analysis_plan.md": sha(PLAN), **hashes}

    OUT.mkdir(exist_ok=True)
    with (OUT / "frozen_files.sha256").open("w") as f:
        f.write(f"# Frozen {today}. Verify with: sha256sum -c frozen_files.sha256 (from the project root)\n")
        for group, files in [("Analysis plan", ["docs/21_analysis_plan.md"])] + list(GROUPS.items()):
            f.write(f"# {group}\n")
            for x in files:
                f.write(f"{hashes[x]}  {x}\n")
    shutil.copy(PLAN, OUT / "analysis_plan.md")
    shutil.copy(ROOT / "docs/20_study_design.md", OUT / "study_design.md")
    shutil.copy(ROOT / "data/suite/s1/audit.json", OUT / "construction_audit.json")
    shutil.copy(ROOT / "data/suite/s1/manifest.json", OUT / "question_sets_manifest.json")
    shutil.copy(ROOT / "results/analysis/s1_dev.md", OUT / "development_run_summary.md")
    with zipfile.ZipFile(OUT / "question_sets_s1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in GROUPS["Question sets"]:
            z.write(ROOT / f, f)
    with zipfile.ZipFile(OUT / "frozen_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in GROUPS["Analysis and run code"] + GROUPS["Pipeline code"] + ["docs/20_study_design.md",
                                                                                "docs/21_analysis_plan.md"]:
            z.write(ROOT / f, f)
    with (OUT / "package.sha256").open("w") as f:
        for p in sorted(OUT.iterdir()):
            if p.name != "package.sha256" and p.is_file():
                f.write(f"{sha(p)}  {p.name}\n")
    print(f"frozen {today}: {len(hashes)} files; plan sha256 {hashes['docs/21_analysis_plan.md']}")


if __name__ == "__main__":
    main()
