# Analysis plan — When Does Corrective Retrieval Pay Off?

**Authors:** Davi Silva Eleuterio, Pedro Filipe Oliveira, Paulo Jorge Teixeira Matos
(CeDRI, SusTEC, Universidade Politécnica de Bragança)
**Status:** frozen after the development run and before any generation on the test sets.
**Date:** 2026-09-24

---

## 1. Purpose and history

Corrective retrieval-augmented generation (RAG) reformulates the query and retrieves
again when the first retrieval seems inadequate. Several recent systems choose the
correction by diagnosing the type of failure (ambiguous, complex, unanswerable); others
choose it without such labels. This study measures two things end-to-end comparisons
cannot separate: (i) how much a *perfect* failure-type label is worth for choosing the
correction, and (ii) whether the benefit of correction depends on how much of the
required evidence the first retrieval found (the retrieval state). The study proposes
no method.

**Disclosure of the pilot.** The hypotheses come from an exploratory pilot run by the
same authors in September 2026 on other questions (MuSiQue dev, HotpotQA validation,
2WikiMultiHopQA dev, PopQA, TriviaQA, SQuAD, ASQA dev). In the pilot, the oracle
failure-type label gave no detectable gain over no correction on 1,500 multi-hop
questions (+0.5 pp), and the gain of correction was concentrated in partial retrievals.
The authors therefore know the direction of the pilot results. This study tests the
hypotheses on questions never used in the pilot (all 18,856 pilot question ids are
excluded), with a new corpus version and a test run that has not started when this plan
is frozen.

## 2. Data

Construction code: `scripts/build_study.py`; audit: `scripts/audit_study.py`
(results in §9). Sources:

- Multi-hop: MuSiQue answerable **train** split, HotpotQA distractor validation,
  2WikiMultiHopQA dev.
- Single-hop: PopQA (long tail), TriviaQA test, SQuAD dev (Self-RAG / FlashRAG files).
- Ambiguous: ASQA **train** split; the question's passages are the evidence annotated in
  ASQA (the Wikipedia context of each disambiguated question and the annotators'
  knowledge passages).

Classes:
- **COMPLEX** — ≥ 2 annotated supporting paragraphs, all in the corpus.
- **ANSWERABLE** — single-hop, an answer-bearing passage in the corpus.
- **VAGUE** — ≥ 2 annotated interpretations, evidence for ≥ 2 distinct answers.
- **INSUFFICIENT_HOP / INSUFFICIENT_SINGLE** — every corpus passage containing a gold
  answer (≥ 3 characters) is excluded and any retrieved candidate containing one is
  dropped at query time; yes/no answers and answers under 3 characters are not used.

Sets (disjoint by base question id):

| set | composition |
|---|---|
| MH.test | 1,500 COMPLEX: 500 MuSiQue, 500 HotpotQA, 500 2WikiMultiHopQA |
| MIXED.test | 270 ANSWERABLE and 210 INSUFFICIENT_SINGLE (90 / 70 per single-hop source); 270 COMPLEX and 210 INSUFFICIENT_HOP (90 / 70 per multi-hop source); 150 VAGUE |
| MH.dev | 300 COMPLEX (100 per source) |
| MIXED.dev | 90 ANSWERABLE, 75 INSUFFICIENT_SINGLE, 90 COMPLEX, 75 INSUFFICIENT_HOP, 50 VAGUE |
| STATE_TRAIN | 2,890 rows for the secondary predictability analysis (no generation) |

Retrieval corpora: for each family (multi-hop, single-hop, ambiguous), the union of the
pilot family corpus and every passage of every study question of that family,
deduplicated by text hash (multi-hop 82,728, single-hop 71,239, ambiguous 34,593
passages).

## 3. Systems

Shared pipeline: BM25 and BGE-M3 dense retrieval (100 candidates each), reciprocal rank
fusion (k = 60), top 10 re-scored by `bge-reranker-v2-m3`; the generator sees the top 3.
A correction asks the LLM for one reformulation, retrieves for it, and answers from the
three initial passages plus the two best new passages. Temperature 0. Generators
(Ollama, 4-bit `q4_K_M`): **Llama-3.1-8B-Instruct (primary)**, Qwen-2.5-7B-Instruct,
Gemma-2-9B-Instruct.

Arms on MH sets: no correction with the top 3 and with the top 5; each strategy applied
to every question (rewrite, decompose, iterative decomposition, disambiguate — called
Always-Correct —, anchored rewrite); failure-type routing (VAGUE → disambiguate,
COMPLEX → decompose, INSUFFICIENT → anchored rewrite, NONE → rewrite) with the oracle
label, a random label and the generator as a prompted three-way judge.
Arms on MIXED sets: the same with abstention allowed (the generator may answer
UNANSWERABLE), plus no correction (top 5) without abstention.
Routers defined by a mapping (type → action, state → action) are evaluated by taking,
for each question, the result of the arm the mapping selects; no router is run
separately.

## 4. Variables

- **Accuracy**: a normalised gold answer is a substring of the normalised answer
  (lower case, no punctuation or articles). Abstentions count as incorrect.
- **Utility** (MIXED): 1 for a correct answer to an answerable question, 1 for an
  abstention on an unanswerable question, 0 otherwise.
- **Retrieval state** of the top 3 passages of the first retrieval (identical for every
  arm): coverage of the annotated evidence passages — 0 → NONE_FOUND, 1 → COMPLETE,
  otherwise PARTIAL; for questions without annotated passages, COMPLETE if a normalised
  gold answer (≥ 2 characters) occurs in a shown passage, otherwise NONE_FOUND.
- **Type** (MIXED): ANSWERABLE → NONE; COMPLEX → COMPLEX; VAGUE → VAGUE; both
  unanswerable classes → INSUFFICIENT.

## 5. Primary hypotheses

All three are tested with the primary generator. Holm correction over the three p
values, family-wise α = 0.05.

**H1 — the oracle failure-type label does not help choose the correction on multi-hop
questions.** On MH.test, the accuracy of failure-type routing with the oracle label
minus the accuracy of no correction (top 5), paired by question. Test: two one-sided
tests (normal approximation on the paired differences) with margin ±2 percentage
points. H1 is confirmed if the Holm-adjusted TOST p value is < 0.05.

**H2 — with types that vary, the best type-to-action mapping does not beat the best
label-free constant policy.** On MIXED.test, the utility of the oracle type label with
the type-to-action mapping selected on MIXED.dev, minus the utility of the constant
policy selected on MIXED.dev, paired by question. The actions are: answer from the top
5, or apply one of the five strategies (abstention allowed to the generator in every
action). Selection on MIXED.dev: for each type, the action with the highest mean
utility; the constant policy is the action with the highest mean utility over all
MIXED.dev questions; ties go to the action with fewer LLM calls, then to the order
answer, rewrite, decompose, iterative decomposition, disambiguate, anchored rewrite.
Test: TOST with margin ±2 pp; confirmed if the Holm-adjusted p value is < 0.05.
The selected mapping and constant for every generator are listed in §8. If the selected
mapping assigns the constant action to every type, the difference is zero by
construction; H2 is then reported as confirmed and the report states that the label
changed no decision.

**H3 — the benefit of correction depends on the retrieval state.** On MH.test, the
per-question gain g = correct(Always-Correct) − correct(no correction, top 5). Test:
difference between the mean gain in PARTIAL and the mean gain in COMPLETE, z test with
unequal variances, two-sided. H3 is confirmed if the Holm-adjusted p value is < 0.05 and
the difference is positive. (NONE_FOUND questions are excluded from this contrast.)

## 6. Secondary analyses (reported without multiplicity correction)

1. H1–H3 with Qwen-2.5 7B and Gemma-2 9B (same sets, same tests, unadjusted).
2. H3 adjusted: OLS of g on 1[PARTIAL], the best re-ranker score of the first
   retrieval and indicators of source × question type (HotpotQA bridge/comparison,
   2WikiMultiHopQA type, MuSiQue number of hops), HC3 standard errors, PARTIAL and
   COMPLETE questions only.
3. Gain of every strategy and of each failure-type router over no correction (top 5),
   overall, by retrieval state, and by source (McNemar, bootstrap 95% CIs).
4. Taxonomy value on MH.test: oracle route minus the best label-free arm (strategies and
   no correction). Judge loss: oracle route minus self-judge route; distribution of the
   judge's labels.
5. Oracle ceilings on MIXED.test: the dev-selected mapping plus "abstain on
   INSUFFICIENT"; routing by the true state (PARTIAL → Always-Correct, otherwise answer)
   with and without "abstain on NONE_FOUND"; state routing plus abstention exactly on
   the unanswerable questions (knowledge of unanswerability). Oracle-state routing on
   MH.test compared with Always-Correct.
6. Utility of every MIXED arm with its components (accuracy on answerable questions,
   false abstentions, correct abstentions) and by class; effect of allowing the generator
   to abstain (no correction, top 5, with vs without abstention).
7. Distribution of retrieval states overall, by source and within each class; uncorrected
   accuracy by state.
8. Coverage change: gain of Always-Correct and of decomposition on questions where the
   correction increased, kept or decreased the coverage of the evidence chain
   (descriptive; conditions on a post-treatment variable).
9. Metrics: H1–H3 with exact match and token F1 (Wilcoxon for F1), and with an LLM
   judge (Gemma-2 9B grading the Llama-3.1 8B answers, prompt in the repository);
   agreement and Cohen's κ between the judge and substring accuracy.
10. Predictability of the state (`scripts/predictability.py`): DistilBERT classifiers
    trained on STATE_TRAIN only (question + top-3 passages; question only), a logistic
    regression on the three re-ranker scores, and the majority class; accuracy and
    macro-F1 against the true state on MH.test and MIXED.test. Hyper-parameters fixed in
    the script (4 epochs, learning rate 3e-5, batch 16, maximum length 512, seed 42, 10%
    of parent questions held out for validation). No system routes on these predictions.
11. Cost: LLM calls and tokens per question for every arm.
12. Latency (`scripts/run_latency.sh`, after the test sweep): the first 100 MH.test
    questions with Llama-3.1 8B for no correction (top 5), Always-Correct, decomposition,
    iterative decomposition and the self-judge route; each arm with its own empty LLM and
    query-embedding caches, no retrieval cache and nothing else on the GPU; mean and 95th
    percentile seconds per question.

The two state classifiers of item 10 were trained before this plan was frozen (validation
on 289 held-out STATE_TRAIN rows: accuracy 0.775 and 0.696, macro-F1 0.642 and 0.529 for
question + passages and question only); their weights are among the frozen files.

Confidence intervals: percentile bootstrap over questions, 5,000 resamples, seed 42;
90% intervals are reported for equivalence. Paired binary outcomes: exact two-sided
McNemar test.

## 7. Execution rules

- Every arm is run on every question of a set; the runner is resumable and a row is
  written once. Rows with a technical error are re-run before the analysis; any error
  that remains is counted as an incorrect answer and reported.
- The test sweep stores the result of each retrieval on disk and reuses it across arms
  (`FTC_RETRIEVAL_CACHE=1`). Retrieval is deterministic; re-running 6 arms on 60 dev
  questions per set with the cache filled and then read gave rows identical to the dev
  rows produced without it (720/720; `results/analysis/retrieval_cache_check.json`).
- No interim analysis of test results: the analysis script is run on the test sets only
  after all arms of all generators have finished.
- The analysis script (`scripts/analyze_study.py`), the run script, the question sets
  and the prompts are those whose hashes are listed in §10. Any change after freezing is
  reported as a deviation in the article, with its reason.
- Results are reported whatever their direction, including hypotheses that are not
  confirmed.

## 8. Development run and selected mappings

The development run (MH.dev and MIXED.dev, all arms, the three generators) finished on
24 September 2026 with 20,400 rows and no technical errors. It was used to check the
pipeline and the analysis script, and to select the H2 mapping; its hypothesis tests
(n = 300 and 380) are not interpreted.

Selected on MIXED.dev (rule of §5, H2):

| generator | NONE | COMPLEX | VAGUE | INSUFFICIENT | constant policy |
|---|---|---|---|---|---|
| Llama-3.1 8B (primary) | decompose | anchored rewrite | disambiguate (Always-Correct) | answer from the top 5 | iterative decomposition |
| Qwen-2.5 7B | rewrite | disambiguate (Always-Correct) | answer from the top 5 | iterative decomposition | iterative decomposition |
| Gemma-2 9B | iterative decomposition | disambiguate (Always-Correct) | rewrite | rewrite | rewrite |

Mean utility on MIXED.dev by type and action (the data behind the selection):

*llama3-8b*

| type | answer from the top 5 | rewrite | decompose | iterative decomposition | disambiguate | anchored rewrite |
|---|---|---|---|---|---|---|
| COMPLEX | 23.3 | 25.6 | 24.4 | 27.8 | 26.7 | 27.8 |
| INSUFFICIENT | 62.0 | 56.0 | 56.7 | 59.3 | 54.7 | 59.3 |
| NONE | 74.4 | 75.6 | 77.8 | 77.8 | 75.6 | 76.7 |
| VAGUE | 60.0 | 62.0 | 62.0 | 64.0 | 64.0 | 62.0 |
| all | 55.5 | 54.2 | 54.7 | 56.8 | 54.2 | 56.3 |

*qwen2p5-7b*

| type | answer from the top 5 | rewrite | decompose | iterative decomposition | disambiguate | anchored rewrite |
|---|---|---|---|---|---|---|
| COMPLEX | 22.2 | 20.0 | 23.3 | 27.8 | 28.9 | 25.6 |
| INSUFFICIENT | 71.3 | 69.3 | 68.7 | 73.3 | 67.3 | 68.7 |
| NONE | 72.2 | 74.4 | 73.3 | 73.3 | 73.3 | 73.3 |
| VAGUE | 60.0 | 56.0 | 60.0 | 60.0 | 56.0 | 60.0 |
| all | 58.4 | 57.1 | 57.9 | 60.8 | 58.2 | 58.4 |

*gemma2-9b*

| type | answer from the top 5 | rewrite | decompose | iterative decomposition | disambiguate | anchored rewrite |
|---|---|---|---|---|---|---|
| COMPLEX | 32.2 | 36.7 | 36.7 | 40.0 | 43.3 | 42.2 |
| INSUFFICIENT | 62.7 | 64.7 | 62.0 | 62.0 | 62.7 | 60.7 |
| NONE | 74.4 | 76.7 | 76.7 | 78.9 | 73.3 | 74.4 |
| VAGUE | 64.0 | 68.0 | 66.0 | 62.0 | 62.0 | 64.0 |
| all | 58.4 | 61.3 | 60.0 | 60.8 | 60.5 | 60.0 |


## 9. Construction audit

Run before any generation (`data/suite/s1/audit.json`). No base question id is shared
with the pilot or between sets.

| check | MH.test | MIXED.test |
|---|---|---|
| full evidence chain in the corpus (COMPLEX) | 1500/1500 | 270/270 |
| answer-bearing passage in the corpus (ANSWERABLE / VAGUE) | — | 270/270 / 150/150 |
| evidence for ≥ 2 distinct answers (VAGUE) | — | 147/150 |
| answer retrievable after exclusion, querying with the question or the answer (INSUFFICIENT_HOP / _SINGLE) | — | 0/210 / 0/210 |
| answer string in the question's own passages outside the annotated chain (COMPLEX) | 249/1500 | 55/270 |
| answer string anywhere in the corpus outside the chain (COMPLEX; dominated by common strings) | 983/1500 | 180/270 |
| answers of ≤ 4 characters | 345/1500 | 230/1110 |

## 10. Frozen files (SHA-256)

The SHA-256 of every frozen file (109 files: code, question sets, corpora,
classifier weights, pilot exclusion list, development results) is listed in
`frozen_files.sha256`, registered with this plan. The most important ones:

| file | SHA-256 |
|---|---|
| `scripts/analyze_study.py` | `7c75d0e1df3b097335fb412a378cb7ee056613c1968bbf1df1678793681f94b7` |
| `scripts/run_study.py` | `1cd2af81627320ccb0ef3c345189f6ece413ba4ad556fa0813470ebb3195c986` |
| `src/loop.py` | `18bce14ed2db2928f425fbfb792e512109f1f11fd3add645c12e743b019c6d1d` |
| `src/prompts.py` | `b139dff2eae6d3da2fecec38090719a7eba9af1058453fab45c38cfd30a0faa3` |
| `src/state.py` | `d6edd61b43890567ddb2a960e62cab676a2a3fcacce90b0c11fcf2d1aafe3d18` |
| `config.py` | `003c50b63a1f9d397bb42b8cf43289491d2034dc19ed72f015f8a213143d4142` |
| `data/suite/s1/MH.test.jsonl` | `377b56159d86f9f8e3a9f69e45471a10aa19ccd15b1033514bbfb692668ba56a` |
| `data/suite/s1/MIXED.test.jsonl` | `3723fb6a13a10dd069931eebef2051b273181c3743f7989aa640500a348c54eb` |
| `data/suite/s1/MIXED.dev.jsonl` | `c5a0c812d3e94db6a342cf67b77bb82378c54e371472bd9204c9f20420c382e5` |
| `data/suite/s1/MH.dev.jsonl` | `e725438d7894598e1ea9efeb8cd1c5fc10cc3898239b4429369dd2f224fccc5c` |
