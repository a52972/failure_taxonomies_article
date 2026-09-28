# Study s1 — dev analysis

- **H1** p=0.306, Holm p=0.421 → not confirmed
- **H3** p=0.0343, Holm p=0.103 → not confirmed
- **H2** p=0.21, Holm p=0.421 → not confirmed

## llama3-8b

- MH n=300: H1 oracle type − no-corr(k5): +1.0 pp [-3.0, +4.7]; 90% [-2.3, +4.3]; p=0.736; TOST p=0.306
- H3 gain PARTIAL +16.6 (n=157) vs COMPLETE +6.2 (n=129): diff +10.4 [+0.8, +19.9], p=0.0343
- H3 adjusted: +10.8 pp (SE 6.6)
- states: {'PARTIAL': 157, 'COMPLETE': 129, 'NONE_FOUND': 14}

| arm | acc | calls | errors |
|---|---|---|---|
| no_corr | 40.7 | 1.00 | 0 |
| no_corr_k5 | 40.3 | 1.00 | 0 |
| s_rewrite | 44.3 | 2.00 | 0 |
| s_decompose | 41.3 | 2.00 | 0 |
| s_decomp_iter | 45.3 | 3.00 | 0 |
| s_disambiguate | 51.7 | 2.00 | 0 |
| s_anchor | 45.7 | 2.00 | 0 |
| j_oracle | 41.3 | 2.00 | 0 |
| j_random | 50.3 | 2.00 | 0 |
| j_llm | 45.7 | 3.00 | 0 |

- MIXED n=380 {'ANSWERABLE': 90, 'INSUFFICIENT_SINGLE': 75, 'COMPLEX': 90, 'INSUFFICIENT_HOP': 75, 'VAGUE': 50}
- dev mapping: {'COMPLEX': 'anchor', 'INSUFFICIENT': 'answer', 'NONE': 'decompose', 'VAGUE': 'disambiguate'}; constant: decomp_iter
- H2 mapping − constant: +1.1 pp [-1.3, +3.4]; 90% [-0.8, +2.9]; p=0.503; TOST p=0.21

| arm | utility | acc ans. | false abst. | correct abst. | calls |
|---|---|---|---|---|---|
| no_corr_k5 | 33.9 | 56.1 | 0.0 | 0.0 | 1.00 |
| no_corr_k5_ab | 55.5 | 51.3 | 31.7 | 62.0 | 1.00 |
| s_rewrite_ab | 54.2 | 53.0 | 29.1 | 56.0 | 2.00 |
| s_decompose_ab | 54.7 | 53.5 | 29.6 | 56.7 | 2.00 |
| s_decomp_iter_ab | 56.8 | 55.2 | 27.8 | 59.3 | 3.00 |
| s_disambiguate_ab | 54.2 | 53.9 | 27.0 | 54.7 | 2.00 |
| s_anchor_ab | 56.3 | 54.3 | 28.3 | 59.3 | 2.00 |
| j_oracle_ab | 55.5 | 53.0 | 29.1 | 59.3 | 2.00 |
| j_random_ab | 54.5 | 52.6 | 29.1 | 57.3 | 2.00 |
| j_llm_ab | 56.3 | 54.3 | 28.3 | 59.3 | 3.00 |

## qwen2p5-7b

- MH n=300: H1 oracle type − no-corr(k5): -3.3 pp [-7.3, +0.3]; 90% [-6.7, +0.0]; p=0.132; TOST p=0.748
- H3 gain PARTIAL +13.4 (n=157) vs COMPLETE -3.9 (n=129): diff +17.3 [+8.7, +25.9], p=0.000101
- H3 adjusted: +21.1 pp (SE 6.2)
- states: {'PARTIAL': 157, 'COMPLETE': 129, 'NONE_FOUND': 14}

| arm | acc | calls | errors |
|---|---|---|---|
| no_corr | 35.7 | 1.00 | 0 |
| no_corr_k5 | 39.0 | 1.00 | 0 |
| s_rewrite | 38.7 | 2.00 | 0 |
| s_decompose | 35.7 | 2.00 | 0 |
| s_decomp_iter | 38.7 | 3.00 | 0 |
| s_disambiguate | 44.7 | 2.00 | 0 |
| s_anchor | 42.3 | 2.00 | 0 |
| j_oracle | 35.7 | 2.00 | 0 |
| j_random | 40.7 | 2.00 | 0 |
| j_llm | 42.7 | 3.00 | 0 |

- MIXED n=380 {'ANSWERABLE': 90, 'INSUFFICIENT_SINGLE': 75, 'COMPLEX': 90, 'INSUFFICIENT_HOP': 75, 'VAGUE': 50}
- dev mapping: {'COMPLEX': 'disambiguate', 'INSUFFICIENT': 'decomp_iter', 'NONE': 'rewrite', 'VAGUE': 'answer'}; constant: decomp_iter
- H2 mapping − constant: +0.5 pp [-1.6, +2.6]; 90% [-1.3, +2.4]; p=0.804; TOST p=0.081

| arm | utility | acc ans. | false abst. | correct abst. | calls |
|---|---|---|---|---|---|
| no_corr_k5 | 33.9 | 56.1 | 0.0 | 0.0 | 1.00 |
| no_corr_k5_ab | 58.4 | 50.0 | 35.2 | 71.3 | 1.00 |
| s_rewrite_ab | 57.1 | 49.1 | 36.1 | 69.3 | 2.00 |
| s_decompose_ab | 57.9 | 50.9 | 37.0 | 68.7 | 2.00 |
| s_decomp_iter_ab | 60.8 | 52.6 | 32.2 | 73.3 | 3.00 |
| s_disambiguate_ab | 58.2 | 52.2 | 30.4 | 67.3 | 2.00 |
| s_anchor_ab | 58.4 | 51.7 | 31.7 | 68.7 | 2.00 |
| j_oracle_ab | 57.6 | 50.4 | 35.2 | 68.7 | 2.00 |
| j_random_ab | 58.7 | 51.7 | 32.6 | 69.3 | 2.00 |
| j_llm_ab | 57.6 | 50.9 | 32.2 | 68.0 | 3.00 |

## gemma2-9b

- MH n=300: H1 oracle type − no-corr(k5): -0.7 pp [-4.3, +3.0]; 90% [-3.7, +2.3]; p=0.856; TOST p=0.233
- H3 gain PARTIAL +17.8 (n=157) vs COMPLETE +3.1 (n=129): diff +14.7 [+5.4, +24.3], p=0.00239
- H3 adjusted: +15.8 pp (SE 6.6)
- states: {'PARTIAL': 157, 'COMPLETE': 129, 'NONE_FOUND': 14}

| arm | acc | calls | errors |
|---|---|---|---|
| no_corr | 43.7 | 1.00 | 0 |
| no_corr_k5 | 44.7 | 1.00 | 0 |
| s_rewrite | 47.3 | 2.00 | 0 |
| s_decompose | 44.0 | 2.00 | 0 |
| s_decomp_iter | 49.7 | 3.00 | 0 |
| s_disambiguate | 56.0 | 2.00 | 0 |
| s_anchor | 52.0 | 2.00 | 0 |
| j_oracle | 44.0 | 2.00 | 0 |
| j_random | 53.7 | 2.00 | 0 |
| j_llm | 51.3 | 3.00 | 0 |

- MIXED n=380 {'ANSWERABLE': 90, 'INSUFFICIENT_SINGLE': 75, 'COMPLEX': 90, 'INSUFFICIENT_HOP': 75, 'VAGUE': 50}
- dev mapping: {'COMPLEX': 'disambiguate', 'INSUFFICIENT': 'rewrite', 'NONE': 'decomp_iter', 'VAGUE': 'rewrite'}; constant: rewrite
- H2 mapping − constant: +2.1 pp [-0.3, +4.7]; 90% [+0.0, +4.2]; p=0.152; TOST p=0.533

| arm | utility | acc ans. | false abst. | correct abst. | calls |
|---|---|---|---|---|---|
| no_corr_k5 | 35.8 | 59.1 | 0.0 | 0.0 | 1.00 |
| no_corr_k5_ab | 58.4 | 55.7 | 28.3 | 62.7 | 1.00 |
| s_rewrite_ab | 61.3 | 59.1 | 27.8 | 64.7 | 2.00 |
| s_decompose_ab | 60.0 | 58.7 | 27.8 | 62.0 | 2.00 |
| s_decomp_iter_ab | 60.8 | 60.0 | 25.7 | 62.0 | 3.00 |
| s_disambiguate_ab | 60.5 | 59.1 | 23.0 | 62.7 | 2.00 |
| s_anchor_ab | 60.0 | 59.6 | 24.8 | 60.7 | 2.00 |
| j_oracle_ab | 58.9 | 57.8 | 27.8 | 60.7 | 2.00 |
| j_random_ab | 58.4 | 57.8 | 26.1 | 59.3 | 2.00 |
| j_llm_ab | 60.5 | 59.6 | 25.2 | 62.0 | 3.00 |
