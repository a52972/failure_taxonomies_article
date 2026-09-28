#!/usr/bin/env bash
# Study s1 sweeps (docs/20). One process, strictly sequential, resumable (the runner
# skips stored qids).
#
#   setsid nohup bash scripts/run_study.sh dev  > results/logs/s1_dev.nohup  2>&1 < /dev/null &
#   setsid nohup bash scripts/run_study.sh test > results/logs/s1_test.nohup 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.." || exit 1
SPLIT="${1:?usage: run_study.sh dev|test}"
LOG=results/logs/s1_${SPLIT}.log
GENS="llama3-8b qwen2p5-7b gemma2-9b"
# retrieval cache: verified identical on the dev set (results/analysis/retrieval_cache_check.json)
[ "$SPLIT" = test ] && export FTC_RETRIEVAL_CACHE=1
echo $$ > results/logs/s1_${SPLIT}.pid

log() { echo "$(date '+%F %T') ==== $* ====" | tee -a "$LOG"; }
unload() {
  for m in llama3.1:8b-instruct-q4_K_M qwen2.5:7b-instruct-q4_K_M gemma2:9b-instruct-q4_K_M; do
    curl -s http://ollama:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null
  done; sleep 5
}

for f in data/suite/s1/MH.${SPLIT}.jsonl data/suite/s1/MIXED.${SPLIT}.jsonl data/corpus_s1/multihop/emb.done \
         data/corpus_s1/single/emb.done data/corpus_s1/ambig/emb.done data/suite/s1/finalise.json; do
  [ -f "$f" ] || { log "MISSING: $f — aborted"; exit 1; }
done

for g in $GENS; do
  unload
  log "$g | $SPLIT | MH MIXED"
  if python3 scripts/run_study.py --generator "$g" --split "$SPLIT" --sets MH MIXED >> "$LOG" 2>&1; then log "OK"; else log "FAILED (continuing)"; fi
done

# second pass: re-runs only rows that ended with a technical error (docs/21 §7)
for g in $GENS; do
  unload
  log "$g | $SPLIT | error re-run pass"
  python3 scripts/run_study.py --generator "$g" --split "$SPLIT" --sets MH MIXED >> "$LOG" 2>&1 || log "FAILED (continuing)"
done

if [ "$SPLIT" = test ]; then
  unload
  log "judge_audit (gemma2-9b judges llama3-8b)"
  if python3 scripts/judge_audit.py --split test >> "$LOG" 2>&1; then log "OK"; else log "FAILED"; fi
  unload
  log "predictability: features and evaluation on MH.test and MIXED.test"
  python3 scripts/predictability.py features --sets MH.test MIXED.test >> "$LOG" 2>&1 && \
    python3 scripts/predictability.py evaluate --sets MH.test MIXED.test >> "$LOG" 2>&1 && log "OK" || log "FAILED"
  log "latency (uncached, first 100 MH.test questions)"
  unset FTC_RETRIEVAL_CACHE
  bash scripts/run_latency.sh >> "$LOG" 2>&1 && log "OK" || log "FAILED"
fi
unload
log "S1 $SPLIT DONE"
