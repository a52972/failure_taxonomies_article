"""Analysis of the study (docs/20). One script computes every reported number.

    python scripts/analyze_study.py --split dev     # pipeline check + mapping selection (H2)
    python scripts/analyze_study.py --split test    # primary and secondary analyses

Input : results/raw/s1_<split>/<system>__<SET>.<split>__<generator>.jsonl
        (for --split test the dev runs are read too: the type-to-action mapping of
        H2 and the best constant policy are selected on MIXED.dev, per generator)
Output: results/analysis/s1_<split>.json and s1_<split>.md

Primary hypotheses (primary generator only; Holm over the three p values):
  H1  MH.test     accuracy of failure-type routing with the oracle label is
                  equivalent to no correction (top-5) within ±2 pp (TOST).
  H2  MIXED.test  utility of the oracle label with the best type-to-action mapping
                  is equivalent to the best constant policy within ±2 pp (TOST);
                  both selected on MIXED.dev.
  H3  MH.test     the gain of Always-Correct (disambiguate) over no correction
                  (top-5) is larger in the PARTIAL state than in the COMPLETE state.
Everything else is secondary or descriptive and reported without correction.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import DATA_DIR, RAW_RESULTS_DIR, RESULTS_DIR, SUITE_DIR  # noqa: E402
from run_study import ARMS, SYSTEMS, system_name  # noqa: E402

PRIMARY = "llama3-8b"
GENERATORS = ["llama3-8b", "qwen2p5-7b", "gemma2-9b"]
MARGIN = 0.02                   # equivalence margin (proportion)
N_BOOT = 5000
SEED = 42
ALPHA = 0.05

TYPE_OF_CLASS = {"ANSWERABLE": "NONE", "COMPLEX": "COMPLEX", "VAGUE": "VAGUE",
                 "INSUFFICIENT_SINGLE": "INSUFFICIENT", "INSUFFICIENT_HOP": "INSUFFICIENT"}
UNANSWERABLE = {"INSUFFICIENT_SINGLE", "INSUFFICIENT_HOP"}
# actions available to a type-to-action mapping on MIXED, in tie-break order
ACTIONS = {"answer": "no_corr_k5_ab", "rewrite": "s_rewrite_ab", "decompose": "s_decompose_ab",
           "decomp_iter": "s_decomp_iter_ab", "disambiguate": "s_disambiguate_ab", "anchor": "s_anchor_ab"}
STRATEGY_ARMS = ["s_rewrite", "s_decompose", "s_decomp_iter", "s_disambiguate", "s_anchor"]


# ═══════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════

def load_set(split: str, set_name: str, gen: str) -> dict[str, dict[str, dict]]:
    """arm → qid → row, for every arm of the set; pairing is asserted."""
    order = [json.loads(l)["qid"] for l in (SUITE_DIR / "s1" / f"{set_name}.{split}.jsonl").open(encoding="utf-8")]
    out: dict[str, dict[str, dict]] = {}
    for arm in ARMS[set_name]:
        from src.runner import system_tag
        p = RAW_RESULTS_DIR / f"s1_{split}" / f"{system_tag(system_name(SYSTEMS[arm]))}__{set_name}.{split}__{gen}.jsonl"
        if not p.exists():
            continue
        rows = {}
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            rows[r["qid"]] = r
        out[arm] = rows
    complete = {a: len(r) == len(order) for a, r in out.items()}
    return {"order": order, "arms": out, "complete": complete}


def arr(rows: dict[str, dict], order: list[str], key: str) -> np.ndarray:
    return np.array([float(rows[q][key]) for q in order])


def utility(rows: dict[str, dict], order: list[str]) -> np.ndarray:
    u = []
    for q in order:
        r = rows[q]
        if r["cls"] in UNANSWERABLE:
            u.append(1.0 if r["abstained"] else 0.0)
        else:
            u.append(0.0 if r["abstained"] else float(r["in_acc"]))
    return np.array(u)


# ═══════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════

def boot_ci(d: np.ndarray, level: float, rng: np.random.Generator) -> tuple[float, float]:
    n = len(d)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return float(np.quantile(means, lo)), float(np.quantile(means, hi))


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    w = int(((a == 1) & (b == 0)).sum())
    l_ = int(((a == 0) & (b == 1)).sum())
    p = stats.binomtest(w, w + l_, 0.5).pvalue if w + l_ else 1.0
    return w, l_, float(p)


def tost_p(d: np.ndarray, margin: float = MARGIN) -> float:
    """Two one-sided tests on the paired mean difference (normal approximation)."""
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float("inf")
    if se == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0
    p_low = 1 - stats.norm.cdf((d.mean() + margin) / se)
    p_up = 1 - stats.norm.cdf((margin - d.mean()) / se)
    return float(max(p_low, p_up))


def compare(a: np.ndarray, b: np.ndarray, rng, *, binary: bool = True) -> dict:
    d = a - b
    out = {"a": float(a.mean()), "b": float(b.mean()), "diff": float(d.mean()),
           "ci95": boot_ci(d, 0.95, rng), "ci90": boot_ci(d, 0.90, rng), "n": int(len(d)),
           "tost_p": tost_p(d)}
    if binary:
        w, l_, p = mcnemar(a, b)
        out.update(wins=w, losses=l_, p=p)
    else:
        nz = d[d != 0]
        out["p"] = float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0
    out["equivalent_90"] = bool(-MARGIN < out["ci90"][0] and out["ci90"][1] < MARGIN)
    return out


def strata_diff(g: np.ndarray, s: np.ndarray, hi: str, lo: str, rng) -> dict:
    gh, gl = g[s == hi], g[s == lo]
    diff = gh.mean() - gl.mean()
    se = math.sqrt(gh.var(ddof=1) / len(gh) + gl.var(ddof=1) / len(gl))
    p = float(2 * (1 - stats.norm.cdf(abs(diff) / se))) if se > 0 else 1.0
    bs = []
    for _ in range(N_BOOT):
        bs.append(rng.choice(gh, len(gh)).mean() - rng.choice(gl, len(gl)).mean())
    return {"gain_hi": float(gh.mean()), "gain_lo": float(gl.mean()), "n_hi": int(len(gh)), "n_lo": int(len(gl)),
            "diff": float(diff), "ci95": (float(np.quantile(bs, .025)), float(np.quantile(bs, .975))), "p": p}


def ols_hc3(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (e / np.clip(1 - h, 1e-12, None)) ** 2
    cov = XtX_inv @ (X.T * w) @ X @ XtX_inv
    return beta, np.sqrt(np.diag(cov))


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda x: x[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# ═══════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════

def question_meta(split: str, set_name: str) -> dict[str, dict]:
    meta = {}
    for line in (SUITE_DIR / "s1" / f"{set_name}.{split}.jsonl").open(encoding="utf-8"):
        q = json.loads(line)
        md = q.get("metadata") or {}
        qtype = md.get("type") or (f"{md.get('n_hops')}hop" if md.get("n_hops") else "na")
        meta[q["qid"]] = {"source": q["source"], "cls": q["cls"], "qtype": f"{q['source']}:{qtype}"}
    return meta


def states(rows: dict[str, dict], order: list[str]) -> np.ndarray:
    return np.array([rows[q]["state_t0"] for q in order])


def calls(rows, order) -> float:
    return float(np.mean([rows[q]["n_llm_calls"] for q in order]))


def tokens(rows, order) -> float:
    return float(np.mean([rows[q]["prompt_tokens"] + rows[q]["completion_tokens"] for q in order]))


def select_mapping(dev: dict) -> dict:
    """Best action per oracle type and best constant action on MIXED.dev.
    Ties → fewer mean LLM calls, then the fixed ACTIONS order."""
    order, arms = dev["order"], dev["arms"]
    types = np.array([TYPE_OF_CLASS[arms["no_corr_k5_ab"][q]["cls"]] for q in order])
    u = {a: utility(arms[arm], order) for a, arm in ACTIONS.items()}
    c = {a: calls(arms[arm], order) for a, arm in ACTIONS.items()}
    rank = list(ACTIONS)

    def best(mask):
        return max(rank, key=lambda a: (round(u[a][mask].mean(), 10), -c[a], -rank.index(a)))

    mapping = {t: best(types == t) for t in sorted(set(types))}
    const = best(np.ones(len(order), dtype=bool))
    table = {t: {a: float(u[a][types == t].mean()) for a in rank} for t in sorted(set(types))}
    return {"mapping": mapping, "constant": const, "dev_utility_by_type": table,
            "dev_utility_overall": {a: float(u[a].mean()) for a in rank}}


# ═══════════════════════════════════════════════════════════════════════
# Analyses per generator
# ═══════════════════════════════════════════════════════════════════════

def analyse_mh(split: str, gen: str, rng) -> dict:
    S = load_set(split, "MH", gen)
    order, A = S["order"], S["arms"]
    need = ["no_corr_k5", "j_oracle", "s_disambiguate"]
    if not all(S["complete"].get(a) for a in need):
        return {"incomplete": S["complete"]}
    meta = question_meta(split, "MH")
    acc = {a: arr(A[a], order, "in_acc") for a in A if S["complete"][a]}
    st = states(A["no_corr_k5"], order)
    res: dict = {"n": len(order), "complete": S["complete"]}
    res["accuracy"] = {a: {"acc": float(v.mean()), "calls": calls(A[a], order), "tokens": tokens(A[a], order),
                           "errors": int(sum(bool(A[a][q].get("error")) for q in order))} for a, v in acc.items()}
    # H1
    res["H1"] = compare(acc["j_oracle"], acc["no_corr_k5"], rng)
    # H3
    g = acc["s_disambiguate"] - acc["no_corr_k5"]
    res["H3"] = strata_diff(g, st, "PARTIAL", "COMPLETE", rng)
    # H3 adjusted: gain ~ PARTIAL + source×question-type + best re-ranker score (COMPLETE and PARTIAL only)
    keep = np.isin(st, ["PARTIAL", "COMPLETE"])
    cats = sorted({meta[q]["qtype"] for q in order})
    X = [[1.0, float(st[i] == "PARTIAL"), max(A["no_corr_k5"][q]["trace"]["iterations"][0]["pool_scores"] or [0])]
         + [float(meta[q]["qtype"] == c) for c in cats[1:]] for i, q in enumerate(order)]
    X = np.array(X)[keep]
    beta, se = ols_hc3(g[keep], X)
    res["H3_adjusted"] = {"coef_partial": float(beta[1]), "se_hc3": float(se[1]),
                          "p": float(2 * (1 - stats.norm.cdf(abs(beta[1] / se[1])))), "n": int(keep.sum()),
                          "covariates": ["best_reranker_score"] + [f"qtype={c}" for c in cats[1:]]}
    # descriptives
    res["state_distribution"] = dict(Counter(st.tolist()))
    res["state_by_source"] = {s: dict(Counter(st[[meta[q]["source"] == s for q in order]].tolist()))
                              for s in sorted({m["source"] for m in meta.values()})}
    res["uncorrected_acc_by_state"] = {s: float(acc["no_corr_k5"][st == s].mean()) for s in set(st.tolist())}
    res["gain_by_state"] = {a: {s: {"n": int((st == s).sum()), **compare(acc[a][st == s], acc["no_corr_k5"][st == s], rng)}
                                for s in ["NONE_FOUND", "PARTIAL", "COMPLETE"] if (st == s).sum() > 1}
                            for a in STRATEGY_ARMS + ["j_oracle", "j_llm", "j_random"] if a in acc}
    res["vs_no_corr_k5"] = {a: compare(acc[a], acc["no_corr_k5"], rng) for a in acc if a != "no_corr_k5"}
    res["by_source"] = {s: {a: compare(acc[a][m], acc["no_corr_k5"][m], rng) for a in ["s_disambiguate", "j_oracle"]}
                        for s in sorted({x["source"] for x in meta.values()})
                        for m in [np.array([meta[q]["source"] == s for q in order])]}
    # taxonomy value: oracle route minus the best label-free alternative (constants and no correction)
    alts = {a: acc[a].mean() for a in STRATEGY_ARMS + ["no_corr_k5"] if a in acc}
    best_alt = max(alts, key=alts.get)
    res["taxonomy_value"] = {"best_label_free": best_alt, **compare(acc["j_oracle"], acc[best_alt], rng)}
    if "j_llm" in A:
        res["judge_labels"] = dict(Counter(A["j_llm"][q]["label_t0"] for q in order))
        res["judge_loss"] = compare(acc["j_oracle"], acc["j_llm"], rng)
    # oracle-state ceiling (analysis only): PARTIAL → disambiguate, otherwise no correction
    ceil = np.where(st == "PARTIAL", acc["s_disambiguate"], acc["no_corr_k5"])
    c_calls = np.mean([A["s_disambiguate"][q]["n_llm_calls"] if st[i] == "PARTIAL" else 1 for i, q in enumerate(order)])
    res["oracle_state_ceiling"] = {"acc": float(ceil.mean()), "calls": float(c_calls),
                                   "vs_always_correct": compare(ceil, acc["s_disambiguate"], rng)}
    # coverage change after correction (descriptive, conditions on a post-treatment variable)
    cov = {}
    for a in ["s_disambiguate", "s_decompose"]:
        ch = []
        for q in order:
            r = A[a][q]
            c0 = r["gold_cov_t0"] or 0.0
            c1 = r["final_gold_cov"] or 0.0
            ch.append("up" if c1 > c0 + 1e-9 else ("down" if c1 < c0 - 1e-9 else "same"))
        ch = np.array(ch)
        cov[a] = {k: {"n": int((ch == k).sum()), "gain": float((acc[a] - acc["no_corr_k5"])[ch == k].mean())
                      if (ch == k).any() else None} for k in ["up", "same", "down"]}
    res["coverage_change"] = cov
    V = verdicts_for(split) if gen == PRIMARY else None
    if V:
        J = {a: judged(A[a], order, V) for a in need}
        res["H1_judge"] = compare(J["j_oracle"], J["no_corr_k5"], rng)
        res["H3_judge"] = strata_diff(J["s_disambiguate"] - J["no_corr_k5"], st, "PARTIAL", "COMPLETE", rng)
        sub = [(A[a][q]["in_acc"], V.get(__import__("judge_audit").key(q, A[a][q]["answer"])))
               for a in A for q in order if not A[a][q]["abstained"] and (A[a][q]["answer"] or "").strip()]
        sub = [(x, v) for x, v in sub if v is not None]
        if sub:
            x, v = np.array(sub).T
            po, pe = (x == v).mean(), x.mean() * v.mean() + (1 - x.mean()) * (1 - v.mean())
            res["judge_agreement_MH"] = {"n": len(sub), "agreement": float(po), "kappa": float((po - pe) / (1 - pe)),
                                         "substring_rate": float(x.mean()), "judge_rate": float(v.mean())}
    # other metrics for H1/H3
    for key in ["exact_match", "f1"]:
        m = {a: arr(A[a], order, key) for a in need}
        res[f"H1_{key}"] = compare(m["j_oracle"], m["no_corr_k5"], rng, binary=(key == "exact_match"))
        res[f"H3_{key}"] = strata_diff(m["s_disambiguate"] - m["no_corr_k5"], st, "PARTIAL", "COMPLETE", rng)
    return res


def analyse_mixed(split: str, gen: str, rng, dev_sel: dict | None) -> dict:
    S = load_set(split, "MIXED", gen)
    order, A = S["order"], S["arms"]
    if not all(S["complete"].get(a) for a in ACTIONS.values()):
        return {"incomplete": S["complete"]}
    cls = np.array([A["no_corr_k5_ab"][q]["cls"] for q in order])
    types = np.array([TYPE_OF_CLASS[c] for c in cls])
    st = states(A["no_corr_k5_ab"], order)
    U = {a: utility(A[a], order) for a in A if S["complete"][a]}
    res: dict = {"n": len(order), "composition": dict(Counter(cls.tolist())), "complete": S["complete"]}

    def components(rows):
        ans = [q for q in order if rows[q]["cls"] not in UNANSWERABLE]
        una = [q for q in order if rows[q]["cls"] in UNANSWERABLE]
        return {"acc_answerable": float(np.mean([0.0 if rows[q]["abstained"] else rows[q]["in_acc"] for q in ans])),
                "false_abstention": float(np.mean([rows[q]["abstained"] for q in ans])),
                "correct_abstention": float(np.mean([rows[q]["abstained"] for q in una]))}

    res["utility"] = {a: {"utility": float(v.mean()), "calls": calls(A[a], order), **components(A[a])}
                      for a, v in U.items()}
    res["utility_by_class"] = {a: {c: float(v[cls == c].mean()) for c in sorted(set(cls))} for a, v in U.items()}
    res["state_by_class"] = {c: dict(Counter(st[cls == c].tolist())) for c in sorted(set(cls))}
    if "no_corr_k5" in U:
        res["generator_abstention_effect"] = compare(U["no_corr_k5_ab"], U["no_corr_k5"], rng)
    if "j_llm_ab" in A:
        res["judge_labels_by_class"] = {c: dict(Counter(A["j_llm_ab"][q]["label_t0"] for q in np.array(order)[cls == c]))
                                        for c in sorted(set(cls))}
    if dev_sel is None:
        return res
    # H2: oracle type with the dev-selected mapping vs the dev-selected constant policy
    mp, const = dev_sel["mapping"], dev_sel["constant"]
    u_map = np.array([U[ACTIONS[mp.get(types[i], "answer")]][i] for i in range(len(order))])
    u_const = U[ACTIONS[const]]
    res["H2"] = {"mapping": mp, "constant": const, "mapping_is_constant": len(set(mp.values())) == 1 and
                 set(mp.values()) == {const}, **compare(u_map, u_const, rng)}
    V = verdicts_for(split) if gen == PRIMARY else None
    if V:
        UJ = {a: judged(A[a], order, V) for a in ACTIONS.values()}
        res["H2_judge"] = compare(np.array([UJ[ACTIONS[mp.get(types[i], "answer")]][i] for i in range(len(order))]),
                                  UJ[ACTIONS[const]], rng)
    res["H2_calls"] = {"mapping": float(np.mean([A[ACTIONS[mp.get(types[i], 'answer')]][q]["n_llm_calls"]
                                                   for i, q in enumerate(order)])),
                       "constant": calls(A[ACTIONS[const]], order)}
    # ceilings with the abstain action (descriptive)
    abst = (types == "INSUFFICIENT")
    una = np.isin(cls, list(UNANSWERABLE))
    u_map_abst = np.where(abst, 1.0, u_map)          # abstaining on an unanswerable question scores 1
    dis, ans_arm = U["s_disambiguate_ab"], U["no_corr_k5_ab"]
    u_state = np.where(st == "NONE_FOUND", una.astype(float), np.where(st == "PARTIAL", dis, ans_arm))
    u_state_noabst = np.where(st == "PARTIAL", dis, ans_arm)
    u_state_una_only = np.where(una, 1.0, np.where(st == "PARTIAL", dis, ans_arm))
    res["ceilings"] = {
        "oracle_type_mapping": float(u_map.mean()),
        "oracle_type_mapping_plus_abstain_on_insufficient": float(u_map_abst.mean()),
        "oracle_state_no_abstain": float(u_state_noabst.mean()),
        "oracle_state_with_abstain_on_none": float(u_state.mean()),
        "knowledge_of_unanswerability_only": float(u_state_una_only.mean()),
        "type_vs_state_no_abstain": compare(u_map, u_state_noabst, rng),
        "type_vs_state_with_abstain": compare(u_map_abst, u_state, rng),
    }
    return res


def verdicts_for(split: str) -> dict[str, float] | None:
    """LLM-judge verdicts (scripts/judge_audit.py), primary generator only."""
    p = DATA_DIR / "judge_audit" / f"s1_{split}_verdicts.jsonl"
    if not p.exists():
        return None
    return {r["key"]: r["verdict"] for r in map(json.loads, p.open(encoding="utf-8"))}


def judged(rows: dict[str, dict], order: list[str], V: dict) -> np.ndarray:
    """Correctness (answerable) or utility (mixed) with the judge's verdict in place of
    substring accuracy; a pair the judge could not parse falls back to substring accuracy."""
    from judge_audit import key
    out = []
    for q in order:
        r = rows[q]
        if r["cls"] in UNANSWERABLE:
            out.append(1.0 if r["abstained"] else 0.0)
        elif r["abstained"] or not (r["answer"] or "").strip():
            out.append(0.0)
        else:
            v = V.get(key(q, r["answer"]))
            out.append(float(r["in_acc"]) if v is None else float(v))
    return np.array(out)


# ═══════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════

def fmt(c: dict) -> str:
    return (f"{100*c['diff']:+.1f} pp [{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}]; "
            f"90% [{100*c['ci90'][0]:+.1f}, {100*c['ci90'][1]:+.1f}]; p={c.get('p', float('nan')):.3g}; "
            f"TOST p={c['tost_p']:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--generators", nargs="+", default=GENERATORS)
    a = ap.parse_args()
    out: dict = {"split": a.split, "margin": MARGIN, "primary": PRIMARY, "generators": {}}
    for gen in a.generators:
        rng = np.random.default_rng(SEED)
        dev = load_set("dev", "MIXED", gen)
        sel = select_mapping(dev) if all(dev["complete"].get(x) for x in ACTIONS.values()) else None
        g = {"mapping_selection_dev": sel,
             "MH": analyse_mh(a.split, gen, rng),
             "MIXED": analyse_mixed(a.split, gen, rng, sel)}
        out["generators"][gen] = g
    P = out["generators"].get(PRIMARY, {})
    ps = {}
    if "H1" in P.get("MH", {}):
        ps["H1"] = P["MH"]["H1"]["tost_p"]
        ps["H3"] = P["MH"]["H3"]["p"]
    if "H2" in P.get("MIXED", {}):
        ps["H2"] = P["MIXED"]["H2"]["tost_p"]
    if ps:
        adj = holm(ps)
        out["primary_tests"] = {
            k: {"p": ps[k], "p_holm": adj[k],
                "confirmed": bool(adj[k] < ALPHA and (k != "H3" or P["MH"]["H3"]["diff"] > 0))}
            for k in ps}
    lat_dir = RAW_RESULTS_DIR / "s1_latency"
    if a.split == "test" and lat_dir.exists():
        from src.runner import system_tag
        lat = {}
        for arm in ["no_corr_k5", "s_disambiguate", "s_decompose", "s_decomp_iter", "j_llm"]:
            f = lat_dir / f"{system_tag(system_name(SYSTEMS[arm]))}__MH.test__{PRIMARY}.jsonl"
            if f.exists():
                ms = np.array([json.loads(l)["latency_ms"] for l in f.open(encoding="utf-8")]) / 1000
                lat[arm] = {"n": int(len(ms)), "mean_s": float(ms.mean()), "p95_s": float(np.quantile(ms, .95))}
        out["latency_uncached"] = lat
    dest = RESULTS_DIR / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"s1_{a.split}.json").write_text(json.dumps(out, indent=2, default=str))

    L = [f"# Study s1 — {a.split} analysis", ""]
    for k, v in (out.get("primary_tests") or {}).items():
        L.append(f"- **{k}** p={v['p']:.3g}, Holm p={v['p_holm']:.3g} → {'confirmed' if v['confirmed'] else 'not confirmed'}")
    for gen, g in out["generators"].items():
        L += ["", f"## {gen}", ""]
        mh = g["MH"]
        if "H1" in mh:
            L.append(f"- MH n={mh['n']}: H1 oracle type − no-corr(k5): {fmt(mh['H1'])}")
            h3 = mh["H3"]
            L.append(f"- H3 gain PARTIAL {100*h3['gain_hi']:+.1f} (n={h3['n_hi']}) vs COMPLETE {100*h3['gain_lo']:+.1f} "
                     f"(n={h3['n_lo']}): diff {100*h3['diff']:+.1f} [{100*h3['ci95'][0]:+.1f}, {100*h3['ci95'][1]:+.1f}], p={h3['p']:.3g}")
            L.append(f"- H3 adjusted: {100*mh['H3_adjusted']['coef_partial']:+.1f} pp (SE {100*mh['H3_adjusted']['se_hc3']:.1f})")
            L.append(f"- states: {mh['state_distribution']}")
            L.append("")
            L.append("| arm | acc | calls | errors |")
            L.append("|---|---|---|---|")
            for arm, v in mh["accuracy"].items():
                L.append(f"| {arm} | {100*v['acc']:.1f} | {v['calls']:.2f} | {v['errors']} |")
        else:
            L.append(f"- MH incomplete: {mh.get('incomplete')}")
        mx = g["MIXED"]
        if "utility" in mx:
            L += ["", f"- MIXED n={mx['n']} {mx['composition']}"]
            if g["mapping_selection_dev"]:
                L.append(f"- dev mapping: {g['mapping_selection_dev']['mapping']}; constant: {g['mapping_selection_dev']['constant']}")
            if "H2" in mx:
                L.append(f"- H2 mapping − constant: {fmt(mx['H2'])}")
            L.append("")
            L.append("| arm | utility | acc ans. | false abst. | correct abst. | calls |")
            L.append("|---|---|---|---|---|---|")
            for arm, v in mx["utility"].items():
                L.append(f"| {arm} | {100*v['utility']:.1f} | {100*v['acc_answerable']:.1f} | {100*v['false_abstention']:.1f} | "
                         f"{100*v['correct_abstention']:.1f} | {v['calls']:.2f} |")
        else:
            L.append(f"- MIXED incomplete: {mx.get('incomplete')}")
    (dest / f"s1_{a.split}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
