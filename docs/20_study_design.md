# Study s1 — design

**When Does Corrective Retrieval Pay Off?** · Davi S. Eleuterio · 23 September 2026

This document describes the study on which the article rests. Everything run before
23 September 2026 (w1, w2, w3 and the revision runs) is an exploratory **pilot**: it
suggested the hypotheses below and is archived outside the repository
(`paper/WRITING_REPORT.md` §2). No pilot question is reused.

## 1. Questions

- **RQ1.** How much is a perfect failure-type label worth for choosing the correction?
- **RQ2.** Which observable variable conditions whether correction pays off?
- **RQ3.** Do the findings hold with unanswerable questions, across generators and under other metrics?

The study proposes no method. It compares routing by an oracle failure-type label with
label-free alternatives, and measures how the benefit of correction varies with the
retrieval state. The retrieval state is an analysis variable only.

## 2. Corrective loop (unchanged from the pilot)

One retrieval with the original question: BM25 (`bm25s`) and BGE-M3 dense, 100
candidates each, reciprocal rank fusion (k=60), top 10 re-scored by
`bge-reranker-v2-m3`; the generator sees the top 3. A system either answers from the
top k (k=3 or 5) of this retrieval, or asks the LLM for one reformulation with a
strategy, retrieves for it, and answers from the three initial passages plus the two
best new ones ("augment, do not replace"). Generators at temperature 0, 4-bit
quantised, served by Ollama. Code: `src/loop.py`; prompts: `src/prompts.py`.

**Strategies:** rewrite; decompose (2–3 sub-queries, fused and re-scored against the
question); iterative decomposition (Q1 → answer → Q2, three calls); disambiguate
(anchored on the top-3 passages; "Always-Correct"); anchored rewrite.

**Failure-type routing** (`config.STRATEGY_MAP`): VAGUE → disambiguate, COMPLEX →
decompose, INSUFFICIENT → anchored rewrite, NONE → rewrite. Label sources: the
construction label (oracle), a uniform random label, and the generator itself
prompted as a three-way judge.

**Retrieval state** (`src/state.py`) of the top-3 of the first retrieval: coverage of
the annotated evidence passages (0 → NONE_FOUND, 1 → COMPLETE, otherwise PARTIAL);
for questions without annotated passages, COMPLETE if a normalised gold answer (≥ 2
characters) occurs in a shown passage, else NONE_FOUND.

## 3. Data

Built by `scripts/build_study.py`; audited by `scripts/audit_study.py`.

**Sources.** Multi-hop: MuSiQue (**train** split — the pilot used every eligible
question of the dev split), HotpotQA (distractor, validation), 2WikiMultiHopQA (dev).
Single-hop: PopQA, TriviaQA, SQuAD (same files as the pilot). Ambiguous: ASQA
(**train** split, `din0s/asqa`; the pilot used every eligible dev question), with the
evidence annotated in ASQA itself (the Wikipedia context of each disambiguated
question and the annotators' knowledge passages) as the question's passages.
All 18,856 base question ids used anywhere in the pilot are excluded
(`data/pilot_used_qids.json`).

**Classes.** COMPLEX: ≥ 2 annotated supporting paragraphs, all in the corpus.
ANSWERABLE: single-hop with an answer-bearing passage. VAGUE: ≥ 2 annotated
interpretations with evidence for ≥ 2 distinct answers. INSUFFICIENT_HOP /
INSUFFICIENT_SINGLE: every corpus passage containing a gold answer (≥ 3 characters)
is excluded, and any retrieved candidate containing one is dropped at query time; the
multi-hop version keeps the rest of the chain (partial evidence, broken chain).
Unanswerable constructions exclude yes/no answers and answers under 3 characters.

**Sets** (`data/suite/s1/`), disjoint by base question id:

| set | content | use |
|---|---|---|
| MH.test | 1,500 COMPLEX (500 per source) | H1, H3 |
| MIXED.test | 270 ANSWERABLE, 210 INSUFFICIENT_SINGLE, 270 COMPLEX, 210 INSUFFICIENT_HOP, 150 VAGUE | H2 |
| MH.dev | 300 COMPLEX | pipeline check |
| MIXED.dev | 90 / 75 / 90 / 75 / 50 of the same classes | selection of the H2 mapping |
| STATE_TRAIN.train | 2,890 rows (no generation) | predictability analysis only |

**Corpora** (`data/corpus_s1/`): the pilot family corpora plus every passage of every
study question of the family, deduplicated by text hash. A question retrieves only from
the corpus of its family, with its construction's exclusions.

## 4. Systems

`scripts/run_study.py`. MH sets: no correction (k=3, k=5), the five constant
strategies, and failure-type routing with the oracle, random and self-judge labels.
MIXED sets: the same arms with abstention allowed (the generator may output
UNANSWERABLE), plus no correction (k=5) without abstention. Oracle-state routing and
type-to-action mappings are computed from these arms, never run separately: every
branch of such a router is an existing arm, call for call.

## 5. Hypotheses and analysis

`scripts/analyze_study.py`; the frozen analysis plan is `docs/21_analysis_plan.md`.
Primary generator Llama-3.1 8B; Qwen-2.5 7B and Gemma-2 9B are run on the same full
test sets as replications.

- **H1** (MH.test): routing with the oracle failure-type label is equivalent to no
  correction (top-5) in accuracy, margin ±2 pp (two one-sided tests).
- **H2** (MIXED.test): the oracle failure-type label with the best type-to-action
  mapping is equivalent in utility to the best constant policy, margin ±2 pp; both are
  selected on MIXED.dev.
- **H3** (MH.test): the gain of Always-Correct over no correction (top-5) is larger in
  the PARTIAL state than in the COMPLETE state.

Holm over H1–H3. Everything else (per-state gains of every strategy, adjusted
regression, judge collapse, ceilings, generator abstention, coverage change, per-source
results, EM/F1/LLM-judge versions, predictability of the state) is secondary or
descriptive and reported without correction.

## 6. Order of work

1. Build sets and corpora; audit (no LLM).
2. Development run on MH.dev and MIXED.dev with the three generators; run the analysis
   on dev (pipeline check, mapping selection).
3. Train the predictability classifiers on STATE_TRAIN.
4. Freeze `docs/21_analysis_plan.md`, compute SHA-256 of the plan, the analysis script,
   the run script, the set files and the dev mapping; the author stores them in the
   cloud and registers them on OSF (embargoed).
5. Test run; metric audit; analysis on test.
