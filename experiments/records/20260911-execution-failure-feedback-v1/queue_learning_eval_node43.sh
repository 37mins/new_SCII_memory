#!/usr/bin/env bash
# Wait for the already authorized r2 pair; start the separate evaluation clock later.
set -euo pipefail
test "${SC2_FEEDBACK_LEARNING_EVAL_AUTHORIZED:-0}" = 1
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
queue=${EVAL_QUEUE_DIR:?set unused queue status directory}
mkdir "${queue}"
date -u +%FT%TZ > "${queue}/queued_at.txt"
trap 'echo "$?" > "${queue}/done"; date -u +%FT%TZ > "${queue}/ended_at.txt"' EXIT
training=/root/new-scii-execution-feedback-20260911/rl4-serial-r2
wait_until=$(( $(date -d "$(cat "${training}/started_at.txt")" +%s) + 10860 ))
echo waiting_for_rl4 > "${queue}/status"
while [[ ! -f "${training}/.done" ]]; do
    test "$(date +%s)" -lt "${wait_until}"
    sleep 10
done
test "$(cat "${training}/.done")" = EXIT_CODE=0
for _ in {1..15}; do
    test -n "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)" || break
    sleep 2
done
echo checking_evaluation_preflight > "${queue}/status"
timeout --signal=INT --kill-after=15s 7125 \
    bash "${record_dir}/run_learning_eval_node43.sh"
echo evaluation_finished > "${queue}/status"
