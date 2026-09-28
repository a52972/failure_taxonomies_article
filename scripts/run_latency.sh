#!/usr/bin/env bash
# Latency (docs/21, secondary): the first 100 MH.test questions, Llama-3.1 8B, each arm
# re-run with its own EMPTY LLM and query-embedding caches and no retrieval cache, with
# nothing else on the GPU. Run only after the test sweep has finished.
#
#   setsid nohup bash scripts/run_latency.sh > results/logs/s1_latency.nohup 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=results/logs/s1_latency.log
echo $$ > results/logs/s1_latency.pid
ARMS="no_corr_k5 s_disambiguate s_decompose s_decomp_iter j_llm"
STAMP=$(date +%Y%m%d_%H%M%S)
for arm in $ARMS; do
  echo "$(date '+%F %T') ==== latency $arm ====" | tee -a "$LOG"
  FTC_LLM_CACHE="data/cache/latency_${STAMP}/${arm}/llm" FTC_EMB_CACHE="data/cache/latency_${STAMP}/${arm}/emb" \
  FTC_RETRIEVAL_CACHE=0 python3 scripts/run_study.py --generator llama3-8b --split test --sets MH \
    --systems "$arm" --limit 100 --exp s1_latency >> "$LOG" 2>&1 || echo "FAILED $arm" | tee -a "$LOG"
done
echo "$(date '+%F %T') ==== LATENCY DONE ====" | tee -a "$LOG"
