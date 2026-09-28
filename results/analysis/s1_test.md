# Study s1 — test analysis

- **H1** p=0.0592, Holm p=0.118 → not confirmed
- **H3** p=2.08e-07, Holm p=6.23e-07 → confirmed
- **H2** p=0.0967, Holm p=0.118 → not confirmed

## llama3-8b

- MH n=1500: H1 oracle type − no-corr(k5): -0.7 pp [-2.3, +1.0]; 90% [-2.0, +0.7]; p=0.482; TOST p=0.0592
- H3 gain PARTIAL +11.8 (n=845) vs COMPLETE +1.8 (n=612): diff +10.0 [+6.3, +13.8], p=2.08e-07
- H3 adjusted: +13.4 pp (SE 2.5)
- states: {'COMPLETE': 612, 'PARTIAL': 845, 'NONE_FOUND': 43}

| arm | acc | calls | errors |
|---|---|---|---|
| no_corr | 41.5 | 1.00 | 0 |
| no_corr_k5 | 41.9 | 1.00 | 0 |
| s_rewrite | 43.1 | 2.00 | 0 |
| s_decompose | 41.3 | 2.00 | 0 |
| s_decomp_iter | 44.1 | 3.00 | 0 |
| s_disambiguate | 49.6 | 2.00 | 0 |
| s_anchor | 47.0 | 2.00 | 0 |
| j_oracle | 41.3 | 2.00 | 0 |
| j_random | 45.9 | 2.00 | 0 |
| j_llm | 47.0 | 3.00 | 0 |

- MIXED n=1110 {'ANSWERABLE': 270, 'INSUFFICIENT_SINGLE': 210, 'COMPLEX': 270, 'INSUFFICIENT_HOP': 210, 'VAGUE': 150}
- dev mapping: {'COMPLEX': 'anchor', 'INSUFFICIENT': 'answer', 'NONE': 'decompose', 'VAGUE': 'disambiguate'}; constant: decomp_iter
- H2 mapping − constant: +0.8 pp [-1.0, +2.6]; 90% [-0.6, +2.3]; p=0.431; TOST p=0.0967

| arm | utility | acc ans. | false abst. | correct abst. | calls |
|---|---|---|---|---|---|
| no_corr_k5 | 36.8 | 59.1 | 0.0 | 0.0 | 1.00 |
| no_corr_k5_ab | 55.6 | 53.2 | 27.5 | 59.5 | 1.00 |
| s_rewrite_ab | 55.9 | 54.9 | 27.7 | 57.4 | 2.00 |
| s_decompose_ab | 55.7 | 53.3 | 26.7 | 59.5 | 2.00 |
| s_decomp_iter_ab | 56.1 | 54.5 | 26.2 | 58.8 | 3.00 |
| s_disambiguate_ab | 56.8 | 56.7 | 22.5 | 56.9 | 2.00 |
| s_anchor_ab | 55.9 | 55.2 | 24.8 | 57.1 | 2.00 |
| j_oracle_ab | 55.4 | 54.3 | 26.4 | 57.1 | 2.00 |
| j_random_ab | 56.1 | 55.1 | 24.5 | 57.9 | 2.00 |
| j_llm_ab | 55.9 | 55.2 | 24.8 | 57.1 | 3.00 |

## qwen2p5-7b

- MH n=1500: H1 oracle type − no-corr(k5): -0.5 pp [-2.0, +1.1]; 90% [-1.7, +0.8]; p=0.603; TOST p=0.0231
- H3 gain PARTIAL +12.7 (n=845) vs COMPLETE +0.5 (n=612): diff +12.2 [+8.4, +15.9], p=2.53e-10
- H3 adjusted: +14.1 pp (SE 2.6)
- states: {'COMPLETE': 612, 'PARTIAL': 845, 'NONE_FOUND': 43}

| arm | acc | calls | errors |
|---|---|---|---|
| no_corr | 36.9 | 1.00 | 0 |
| no_corr_k5 | 38.1 | 1.00 | 0 |
| s_rewrite | 39.5 | 2.00 | 0 |
| s_decompose | 37.7 | 2.00 | 0 |
| s_decomp_iter | 39.8 | 3.00 | 0 |
| s_disambiguate | 45.7 | 2.00 | 0 |
| s_anchor | 43.1 | 2.00 | 0 |
| j_oracle | 37.7 | 2.00 | 0 |
| j_random | 42.4 | 2.00 | 0 |
| j_llm | 43.3 | 3.00 | 0 |

- MIXED n=1110 {'ANSWERABLE': 270, 'INSUFFICIENT_SINGLE': 210, 'COMPLEX': 270, 'INSUFFICIENT_HOP': 210, 'VAGUE': 150}
- dev mapping: {'COMPLEX': 'disambiguate', 'INSUFFICIENT': 'decomp_iter', 'NONE': 'rewrite', 'VAGUE': 'answer'}; constant: decomp_iter
- H2 mapping − constant: +0.9 pp [-0.5, +2.3]; 90% [-0.3, +2.1]; p=0.253; TOST p=0.0606

| arm | utility | acc ans. | false abst. | correct abst. | calls |
|---|---|---|---|---|---|
| no_corr_k5 | 34.5 | 55.5 | 0.0 | 0.0 | 1.00 |
| no_corr_k5_ab | 57.8 | 47.8 | 33.2 | 74.3 | 1.00 |
| s_rewrite_ab | 57.1 | 47.8 | 34.5 | 72.4 | 2.00 |
| s_decompose_ab | 55.9 | 46.1 | 35.7 | 71.9 | 2.00 |
| s_decomp_iter_ab | 57.9 | 48.3 | 33.2 | 73.8 | 3.00 |
| s_disambiguate_ab | 58.6 | 50.6 | 30.0 | 71.9 | 2.00 |
| s_anchor_ab | 57.1 | 48.3 | 31.6 | 71.7 | 2.00 |
| j_oracle_ab | 56.6 | 47.4 | 34.5 | 71.7 | 2.00 |
| j_random_ab | 57.7 | 48.8 | 32.6 | 72.4 | 2.00 |
| j_llm_ab | 58.0 | 49.9 | 30.3 | 71.4 | 3.00 |

## gemma2-9b

- MH n=1500: H1 oracle type − no-corr(k5): +0.2 pp [-1.2, +1.7]; 90% [-1.0, +1.4]; p=0.852; TOST p=0.00592
- H3 gain PARTIAL +17.9 (n=845) vs COMPLETE +2.0 (n=612): diff +15.9 [+12.5, +19.3], p=0
- H3 adjusted: +18.8 pp (SE 2.5)
- states: {'COMPLETE': 612, 'PARTIAL': 845, 'NONE_FOUND': 43}

| arm | acc | calls | errors |
|---|---|---|---|
| no_corr | 44.4 | 1.00 | 0 |
| no_corr_k5 | 44.5 | 1.00 | 0 |
| s_rewrite | 45.9 | 2.00 | 0 |
| s_decompose | 44.7 | 2.00 | 0 |
| s_decomp_iter | 50.1 | 3.00 | 0 |
| s_disambiguate | 55.4 | 2.00 | 0 |
| s_anchor | 52.1 | 2.00 | 0 |
| j_oracle | 44.7 | 2.00 | 0 |
| j_random | 51.2 | 2.00 | 0 |
| j_llm | 51.7 | 3.00 | 0 |

- MIXED n=1110 {'ANSWERABLE': 270, 'INSUFFICIENT_SINGLE': 210, 'COMPLEX': 270, 'INSUFFICIENT_HOP': 210, 'VAGUE': 150}
- dev mapping: {'COMPLEX': 'disambiguate', 'INSUFFICIENT': 'rewrite', 'NONE': 'decomp_iter', 'VAGUE': 'rewrite'}; constant: rewrite
- H2 mapping − constant: +1.8 pp [+0.7, +3.0]; 90% [+0.9, +2.7]; p=0.00222; TOST p=0.363

| arm | utility | acc ans. | false abst. | correct abst. | calls |
|---|---|---|---|---|---|
| no_corr_k5 | 37.2 | 59.9 | 0.0 | 0.0 | 1.00 |
| no_corr_k5_ab | 57.3 | 56.1 | 25.9 | 59.3 | 1.00 |
| s_rewrite_ab | 58.6 | 57.1 | 25.4 | 61.2 | 2.00 |
| s_decompose_ab | 57.7 | 55.9 | 25.9 | 60.5 | 2.00 |
| s_decomp_iter_ab | 58.1 | 56.8 | 24.9 | 60.2 | 3.00 |
| s_disambiguate_ab | 59.5 | 59.7 | 21.3 | 59.0 | 2.00 |
| s_anchor_ab | 60.9 | 60.1 | 21.9 | 62.1 | 2.00 |
| j_oracle_ab | 58.6 | 56.4 | 25.5 | 62.1 | 2.00 |
| j_random_ab | 59.1 | 58.6 | 22.6 | 60.0 | 2.00 |
| j_llm_ab | 60.9 | 59.9 | 21.7 | 62.6 | 3.00 |
