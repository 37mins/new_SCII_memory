#!/usr/bin/env bash
# User-approved 50-minute pair; no retries or training. Runtime SHAs stay fixed.
set -euo pipefail
root=/root/new-scii-bo-segment-20260909
a_sha=0d7ba09ddbdf94d85d67b553bf7978f402cf2052
b_sha=a1f15af1553b36cc691c0a19b6be65ea9cd25bda
export SC2_PYTHON_OVERLAY=/root/new-scii-spb2-level5-control/python
export CPATH=/usr/include/python3.12:/usr/include
test -d "${SC2_PYTHON_OVERLAY}"
test -f /usr/include/python3.12/Python.h
for sha in "${a_sha}" "${b_sha}"; do
  code=${root}/code-${sha:0:7}
  test "$(git -C "${code}" rev-parse HEAD)" = "${sha}"
  test -z "$(git -C "${code}" status --porcelain)"
  (cd "${code}" && sha256sum -c configs/system-v1.sha256)
done
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
pair=${root}/pair-r1
mkdir "${pair}"
cp "${BASH_SOURCE[0]}" "${pair}/launcher.sh"
date -Is > "${pair}/started_at.txt"
export PAIR_DEADLINE=$(( $(date +%s) + 3000 ))
printf '%s\n' "${PAIR_DEADLINE}" > "${pair}/deadline_epoch.txt"
printf 'A_SHA=%s\nB_SHA=%s\nORCHESTRATION_SHA=%s\n' "${a_sha}" "${b_sha}" "${ORCHESTRATION_SHA:?set backed-up orchestration SHA}" > "${pair}/source.txt"
run_root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260909-bo-segment-protocol-v1/runs
wrapper=${root}/code-${b_sha:0:7}/experiments/records/20260909-bo-segment-protocol-v1/run_eval.sh
run_arm() {
  local arm=$1 gpu_ids=$2 sha=$3
  local run_id=20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-${arm}-eval4-s0-r1
  test ! -e "${run_root}/${run_id}"
  SOURCE_DIR=${root}/code-${sha:0:7} SOURCE_SHA=${sha} CUDA_VISIBLE_DEVICES=${gpu_ids} \
    RUN_ID=${run_id} OUTPUT_DIR=${run_root}/${run_id} RAY_TMPDIR=/dev/shm/boseg_${arm}_r1 \
    ARM_STATUS_PATH=${pair}/${arm}.done bash "${wrapper}" "${arm}" > "${pair}/${arm}.log" 2>&1
}
run_arm A 0,1,2,3 "${a_sha}" &
a_pid=$!
run_arm B 4,5,6,7 "${b_sha}" &
b_pid=$!
set +e
wait "${a_pid}"; a_status=$?
wait "${b_pid}"; b_status=$?
printf 'A_EXIT_CODE=%s\nB_EXIT_CODE=%s\n' "${a_status}" "${b_status}" > "${pair}/pair.done"
date -Is >> "${pair}/pair.done"
test "${a_status}" = 0 && test "${b_status}" = 0
