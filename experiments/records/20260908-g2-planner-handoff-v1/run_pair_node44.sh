#!/usr/bin/env bash
# Start once under: timeout --signal=INT --kill-after=60s 3540 bash <this file>
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
test -z "$(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)"
pair_dir=${PAIR_DIR:?set an unused suite directory}
mkdir "${pair_dir}"
date -Is > "${pair_dir}/started_at.txt"
run_root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260908-g2-planner-handoff-v1/runs
mkdir -p "${run_root}"
run_arm() {
  local arm=$1
  local run_id=20260908-g2-planner-handoff-v1-spb2-l6-g2-${arm}-eval4-s0-r1
  set +e
  RUN_ID=${run_id} OUTPUT_DIR=${run_root}/${run_id} RAY_TMPDIR=/dev/shm/g2handoff_${arm}_r1 \
    bash "${record_dir}/run_eval.sh" "${arm}" > "${pair_dir}/${arm}.log" 2>&1
  local status=$?
  printf 'EXIT_CODE=%s\n' "${status}" > "${pair_dir}/${arm}.done"
  date -Is >> "${pair_dir}/${arm}.done"
  exit "${status}"
}
run_arm A &
a_pid=$!
run_arm B &
b_pid=$!
set +e
wait "${a_pid}"
a_status=$?
wait "${b_pid}"
b_status=$?
printf 'A_EXIT_CODE=%s\nB_EXIT_CODE=%s\n' "${a_status}" "${b_status}" > "${pair_dir}/pair.done"
date -Is >> "${pair_dir}/pair.done"
test "${a_status}" = 0 && test "${b_status}" = 0
