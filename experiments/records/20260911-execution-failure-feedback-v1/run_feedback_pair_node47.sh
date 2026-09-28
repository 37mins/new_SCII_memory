#!/usr/bin/env bash
# Wrap once in timeout --signal=INT --kill-after=15s 3585; no automatic retries.
set -euo pipefail
test "${SC2_FEEDBACK_AB_AUTHORIZED:-0}" = 1
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
control=${PAIR_DIR:?set unused suite directory}
mkdir "${control}"
exec > >(tee "${control}/console.log") 2>&1
trap 'status=$?; printf "EXIT_CODE=%s\n" "$status" > "${control}/.done"; date -u +%FT%TZ > "${control}/ended_at.txt"' EXIT
date -u +%FT%TZ | tee "${control}/started_at.txt"
deadline=$(( $(date +%s) + ${FEEDBACK_PAIR_SECONDS:-3580} ))
test -z "$(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)"
timeout 5s stat "${FEEDBACK_STORAGE_ROOT:-/mnt/data0601}/yangjinyue/new_SCII"
(cd "${source_dir}" && sha256sum -c configs/system-v1.sha256)
export PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${source_dir}:${FEEDBACK_PYTHON_OVERLAY:-/root/new-scii-spb2-level5-control/python}
"${PYTHON_BIN}" "${record_dir}/checkpoint_preflight.py" "${control}"
run_arm() {
    local arm=$1
    local remaining=$(( deadline - $(date +%s) - 30 ))
    test "${remaining}" -gt 0
    date -u +%FT%TZ > "${control}/${arm}-started_at.txt"
    set +e
    timeout --signal=INT --kill-after=10s "${remaining}" \
        bash "${record_dir}/run_feedback_eval_node47.sh" "${arm}" > "${control}/${arm}.log" 2>&1
    local status=$?
    printf 'EXIT_CODE=%s\n' "${status}" > "${control}/${arm}.done"
    date -u +%FT%TZ >> "${control}/${arm}.done"
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
printf 'A_EXIT_CODE=%s\nB_EXIT_CODE=%s\n' "${a_status}" "${b_status}" > "${control}/pair.done"
test "${a_status}" = 0 && test "${b_status}" = 0
