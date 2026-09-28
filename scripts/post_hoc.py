"""Post hoc analyses (not in the registered plan, docs/21). They read only the released
per-question rows, question sets, corpora and, for item 11, the trained state classifiers;
no LLM is called and no registered output is changed.

    python scripts/post_hoc.py        → results/analysis/post_hoc.json

 1  headroom-normalised gain of Always-Correct by retrieval state
 2  dose–response: gain by coverage level of the evidence chain and by number of hops;
    the H3 regression with the number of hops as a further covariate
 3  alternative state for constructed unanswerable multi-hop questions: coverage of the
    parent evidence chain in the top 3 (instead of the answer-string rule), and the oracle
    ceilings of MIXED.test recomputed with it
 4  answer visibility: share of COMPLETE retrievals whose answer string lies inside the
    500 characters of a top-3 passage that the generator is shown
 5  state of the top 3 vs state of the top 5 shown to No-Correction (transition counts)
 6  fallback rule: how often each reformulation was replaced by the original question
 7  H1–H3 with whole-token matching instead of character-substring matching
 8  number of corpus passages excluded per unanswerable question
 9  utility under other shares of unanswerable questions and with a penalty for wrong answers
10  Holm correction within each replication generator
11  confusion matrices of the state classifiers; accuracy under the alternative state of item 3
12  bootstrap intervals for the uncached latency
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import analyze_study as A  # noqa: E402
from config import DATA_DIR, RAW_RESULTS_DIR, RESULTS_DIR, SUITE_DIR  # noqa: E402
from src.corpus import text_hash  # noqa: E402
from src.metrics import normalise_answer  # noqa: E402

GENS = A.GENERATORS
STRATS = ["s_rewrite", "s_decompose", "s_decomp_iter", "s_disambiguate", "s_anchor"]
UNANS = A.UNANSWERABLE


def suite(name):
    return {json.loads(l)["qid"]: json.loads(l) for l in (SUITE_DIR / "s1" / f"{name}.jsonl").open(encoding="utf-8")}


def boot_mean_ci(x, rng, n=A.N_BOOT):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def state_from_cov(c):
    return "NONE_FOUND" if c == 0 else ("COMPLETE" if c >= 1 else "PARTIAL")


def main():
    out: dict = {}
    rng = np.random.default_rng(A.SEED)
    MH = suite("MH.test")
    MX = suite("MIXED.test")

    # ── 1, 2, 5, 6, 7 on MH.test ──────────────────────────────────────
    passages = None
    for g in GENS:
        S = A.load_set("test", "MH", g)
        order, Ar = S["order"], S["arms"]
        acc = {a: A.arr(Ar[a], order, "in_acc") for a in Ar}
        st = A.states(Ar["no_corr_k5"], order)
        gain = acc["s_disambiguate"] - acc["no_corr_k5"]
        G = out.setdefault("generators", {}).setdefault(g, {})
        # 1 headroom
        G["headroom"] = {s: {"n": int((st == s).sum()), "base": float(acc["no_corr_k5"][st == s].mean()),
                             "gain": float(gain[st == s].mean()),
                             "normalised": float(gain[st == s].mean() / (1 - acc["no_corr_k5"][st == s].mean()))}
                         for s in ("NONE_FOUND", "PARTIAL", "COMPLETE")}
        # 2 coverage and hops
        cov = np.array([round(Ar["no_corr_k5"][q]["gold_cov_t0"] or 0.0, 2) for q in order])
        hops = np.array([len(set(MH[q]["gold_pids"])) for q in order])
        G["by_coverage"] = {f"{c:.2f}": {"n": int((cov == c).sum()), "base": float(acc["no_corr_k5"][cov == c].mean()),
                                         "gain": float(gain[cov == c].mean()),
                                         "ci95": boot_mean_ci(gain[cov == c], rng) if (cov == c).sum() > 1 else None}
                            for c in sorted(set(cov.tolist()))}
        G["by_hops"] = {int(h): {s: {"n": int(((hops == h) & (st == s)).sum()),
                                     "gain": float(gain[(hops == h) & (st == s)].mean()) if ((hops == h) & (st == s)).any() else None}
                                 for s in ("PARTIAL", "COMPLETE")} for h in sorted(set(hops.tolist()))}
        meta = A.question_meta("test", "MH")
        keep = np.isin(st, ["PARTIAL", "COMPLETE"])
        cats = sorted({meta[q]["qtype"] for q in order})
        hvals = sorted(set(hops.tolist()))
        X = np.array([[1.0, float(st[i] == "PARTIAL"), max(Ar["no_corr_k5"][q]["trace"]["iterations"][0]["pool_scores"] or [0])]
                      + [float(meta[q]["qtype"] == c) for c in cats[1:]] + [float(hops[i] == h) for h in hvals[1:]]
                      for i, q in enumerate(order)])[keep]
        beta, se = A.ols_hc3(gain[keep], X)
        G["h3_adjusted_with_hops"] = {"coef_partial": float(beta[1]), "se_hc3": float(se[1])}
        # 5 P3 → P5 transitions (retrieval is the same for every generator)
        if g == A.PRIMARY:
            tr = Counter()
            for i, q in enumerate(order):
                r = Ar["no_corr_k5"][q]
                gold = {text_hash(p["title"], p["text"]) for p in MH[q]["passages"] if p["pid"] in set(MH[q]["gold_pids"])}
                c5 = len(gold & set(r["trace"]["final_pool_pids"])) / len(gold)
                tr[f"{st[i]}->{state_from_cov(c5)}"] += 1
            out["top3_to_top5"] = dict(tr)
        # 6 fallback to the original question
        fb = {}
        for a in STRATS + ["j_llm"]:
            n = 0
            for q in order:
                its = Ar[a][q]["trace"]["iterations"]
                q1 = its[1]["query"] if len(its) > 1 else None
                qq = MH[q]["question"]
                n += (q1 == qq) or (isinstance(q1, list) and q1 == [qq])
            fb[a] = n / len(order)
        G["fallback_share"] = fb
        # 7 whole-token matching
        def tok_acc(rows):
            out_ = []
            for q in order:
                r = rows[q]
                p = f" {normalise_answer(r['answer'])} "
                golds = [normalise_answer(x) for x in MH[q]["golden_answers"]]
                out_.append(0.0 if r["abstained"] else float(any(x and f" {x} " in p for x in golds)))
            return np.array(out_)
        T = {a: tok_acc(Ar[a]) for a in ("no_corr_k5", "j_oracle", "s_disambiguate")}
        G["token_match"] = {"H1": A.compare(T["j_oracle"], T["no_corr_k5"], rng),
                            "H3": A.strata_diff(T["s_disambiguate"] - T["no_corr_k5"], st, "PARTIAL", "COMPLETE", rng),
                            "acc_no_corr_k5": float(T["no_corr_k5"].mean()),
                            "yes_no_share": float(np.mean([bool({normalise_answer(x) for x in MH[q]["golden_answers"]} & {"yes", "no"}) for q in order]))}

    # ── 3, 8, 9 on MIXED.test ─────────────────────────────────────────
    for g in GENS:
        S = A.load_set("test", "MIXED", g)
        order, Ar = S["order"], S["arms"]
        cls = np.array([Ar["no_corr_k5_ab"][q]["cls"] for q in order])
        una = np.isin(cls, list(UNANS))
        st = A.states(Ar["no_corr_k5_ab"], order)
        alt = st.copy()
        for i, q in enumerate(order):
            Q = MX[q]
            if Q["cls"] == "INSUFFICIENT_HOP":
                parent = set(Q["construction"]["parent_gold_pids"])
                chain = {text_hash(p["title"], p["text"]) for p in Q["passages"] if p["pid"] in parent}
                shown = set(Ar["no_corr_k5_ab"][q]["trace"]["iterations"][0]["pool_pids"])
                alt[i] = state_from_cov(len(chain & shown) / len(chain))
        U = {a: A.utility(Ar[a], order) for a in Ar if a.endswith("_ab")}
        dis, ans = U["s_disambiguate_ab"], U["no_corr_k5_ab"]
        G = out["generators"][g]
        G["mixed_alt_state"] = {
            "ihop_states_alt": dict(Counter(alt[cls == "INSUFFICIENT_HOP"].tolist())),
            "oracle_state_no_abstain": float(np.where(alt == "PARTIAL", dis, ans).mean()),
            "oracle_state_with_abstain_on_none": float(np.where(alt == "NONE_FOUND", una.astype(float),
                                                                np.where(alt == "PARTIAL", dis, ans)).mean()),
            "registered_with_abstain_on_none": float(np.where(st == "NONE_FOUND", una.astype(float),
                                                              np.where(st == "PARTIAL", dis, ans)).mean())}
        # 9 prevalence and penalty
        def util_parts(rows):
            a = [0.0 if rows[q]["abstained"] else float(rows[q]["in_acc"]) for q in np.array(order)[~una]]
            wrong_a = [float(not rows[q]["abstained"] and not rows[q]["in_acc"]) for q in np.array(order)[~una]]
            c = [float(rows[q]["abstained"]) for q in np.array(order)[una]]
            return np.mean(a), np.mean(wrong_a), np.mean(c)
        prev = {}
        for arm in ("no_corr_k5_ab", "s_disambiguate_ab", "s_decomp_iter_ab", "j_oracle_ab"):
            a, w, c = util_parts(Ar[arm])
            prev[arm] = {f"{p:.2f}": float((1 - p) * a + p * c) for p in (0.1, 0.2, 0.38, 0.5)}
            prev[arm]["penalty1_at_0.38"] = float(0.62 * (a - w) + 0.38 * (c - (1 - c)))
        G["prevalence"] = prev
        if g == A.PRIMARY:
            ex = {c: [len(MX[q]["construction"]["exclude_hashes"]) for q in order if MX[q]["cls"] == c] for c in UNANS}
            out["excluded_per_question"] = {c: {"median": float(np.median(v)), "q25": float(np.quantile(v, .25)),
                                                "q75": float(np.quantile(v, .75)), "max": int(max(v)),
                                                "share_over_100": float(np.mean(np.array(v) > 100))} for c, v in ex.items()}
            out["mixed_alt_labels"] = {q: alt[i] for i, q in enumerate(order)}

    # ── 4 answer visibility in the truncated window (MH.test, top 3) ──
    P = {}
    for line in (DATA_DIR / "corpus_s1" / "multihop" / "passages.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        P[r["pid"]] = r
    S = A.load_set("test", "MH", A.PRIMARY)
    vis = Counter()
    for q in S["order"]:
        r = S["arms"]["no_corr"][q]
        if r["state_t0"] != "COMPLETE":
            continue
        golds = [normalise_answer(x) for x in MH[q]["golden_answers"] if len(normalise_answer(x)) >= 2]
        shown = [P[p] for p in r["trace"]["iterations"][0]["pool_pids"]]
        full = " ".join(normalise_answer(f"{x['title']} {x['text']}") for x in shown)
        trunc = " ".join(normalise_answer(f"{x['title']} {x['text'][:500]}") for x in shown)
        vis["n"] += 1
        vis["answer_in_full"] += any(g_ in full for g_ in golds)
        vis["answer_in_window"] += any(g_ in trunc for g_ in golds)
    out["answer_visibility_complete"] = dict(vis)

    # ── 10 Holm within each generator ─────────────────────────────────
    test = json.loads((RESULTS_DIR / "analysis" / "s1_test.json").read_text())
    out["holm_by_generator"] = {}
    for g in GENS:
        T_ = test["generators"][g]
        ps = {"H1": T_["MH"]["H1"]["tost_p"], "H2": T_["MIXED"]["H2"]["tost_p"], "H3": T_["MH"]["H3"]["p"]}
        out["holm_by_generator"][g] = {k: {"p": v, "p_holm": A.holm(ps)[k]} for k, v in ps.items()}

    # ── 11 classifiers: confusion matrices, accuracy under the alternative state ─
    try:
        import predictability as PR
        LAB = PR.LABELS
        res = {}
        for sn in ("MH.test", "MIXED.test"):
            rows = PR.rows(sn)
            y = np.array([LAB.index(r["label"]) for r in rows])
            for m in ("text", "question"):
                pred = PR.predict_text(m, rows)
                cm = np.zeros((3, 3), dtype=int)
                for a_, b_ in zip(y, pred):
                    cm[a_, b_] += 1
                d = {"labels": LAB, "confusion": cm.tolist()}
                if sn == "MIXED.test":
                    ya = np.array([LAB.index(out["mixed_alt_labels"].get(r["qid"], r["label"])) for r in rows])
                    d["accuracy_alt_state"] = float((pred == ya).mean())
                    d["macro_f1_alt_state"] = PR.metrics(ya, pred)["macro_f1"]
                res[f"{sn}/{m}"] = d
        out["classifiers"] = res
    except Exception as exc:          # classifier weights are a release asset
        out["classifiers"] = {"skipped": f"{type(exc).__name__}: {exc}"}
    out.pop("mixed_alt_labels", None)

    # ── 12 latency intervals ──────────────────────────────────────────
    from run_study import SYSTEMS, system_name
    from src.runner import system_tag
    lat = {}
    for arm in ("no_corr_k5", "s_disambiguate", "s_decompose", "s_decomp_iter", "j_llm"):
        f = RAW_RESULTS_DIR / "s1_latency" / f"{system_tag(system_name(SYSTEMS[arm]))}__MH.test__{A.PRIMARY}.jsonl"
        if f.exists():
            s = np.array([json.loads(l)["latency_ms"] for l in f.open(encoding="utf-8")]) / 1000
            lat[arm] = {"mean": float(s.mean()), "ci95": boot_mean_ci(s, rng), "sd": float(s.std(ddof=1))}
    out["latency"] = lat

    dest = RESULTS_DIR / "analysis" / "post_hoc.json"
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
