#!/usr/bin/env bash
# User-approved node44 retry: one shared 50-minute window for both arms.
set -euo pipefail
root=/root/new-scii-g2-planner-handoff-20260908
export SOURCE_SHA=ea0055ca6eff2a802ea5d07098e3f3d0e49b361f
code=${root}/code-ea0055c
record=${code}/experiments/records/20260908-g2-planner-handoff-v1
test "$(git -C "${code}" rev-parse HEAD)" = "${SOURCE_SHA}"
test -z "$(git -C "${code}" status --porcelain)"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
export SC2_PYTHON_OVERLAY=/root/new-scii-spb2-level5-control/python
export CPATH=/usr/include/python3.12
pair=${root}/pair-r3
mkdir "${pair}"
cp "${BASH_SOURCE[0]}" "${pair}/launcher.sh"
date -Is > "${pair}/started_at.txt"
export PAIR_DEADLINE=$(( $(date +%s) + 3000 ))
printf '%s\n' "${PAIR_DEADLINE}" > "${pair}/deadline_epoch.txt"
run_root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260908-g2-planner-handoff-v1/runs
run_arm() {
    local arm=$1 gpu_ids=$2
    local run_id=20260908-g2-planner-handoff-v1-spb2-l6-g2-${arm}-eval4-s0-r3
    CUDA_VISIBLE_DEVICES=${gpu_ids} RUN_ID=${run_id} OUTPUT_DIR=${run_root}/${run_id} \
        RAY_TMPDIR=/dev/shm/g2handoff_${arm}_r3 ARM_STATUS_PATH=${pair}/${arm}.done \
        bash "${record}/run_timed_arm.sh" "${arm}" > "${pair}/${arm}.log" 2>&1
}
run_arm A 0,1,2,3 &
a_pid=$!
run_arm B 4,5,6,7 &
b_pid=$!
set +e
wait "${a_pid}"; a_status=$?
wait "${b_pid}"; b_status=$?
printf 'A_EXIT_CODE=%s\nB_EXIT_CODE=%s\n' "${a_status}" "${b_status}" > "${pair}/pair.done"
date -Is >> "${pair}/pair.done"
test "${a_status}" = 0 && test "${b_status}" = 0
