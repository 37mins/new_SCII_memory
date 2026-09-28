#!/usr/bin/env bash
# One B4 model, four GPUs, A/B four games each, <=45 minutes; never retry.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
export FEEDBACK_STORAGE_ROOT=${FEEDBACK_STORAGE_ROOT:-/root/new-scii-execution-feedback-local-20260911/shared}
export FEEDBACK_PYTHON_OVERLAY=/root/new-scii-spb-level5-96-20260908/python
export FEEDBACK_INCLUDE_ROOT=/root/new-scii-spb-level5-96-20260908/include
export FEEDBACK_EVAL_GPUS=0,1,2,3 FEEDBACK_LEARNING_MODEL=B4 FEEDBACK_SECOND_PLAN=1
export PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3} PYTHONDONTWRITEBYTECODE=1
export SC2_FEEDBACK_AB_AUTHORIZED=1
export PYTHONPATH=${record_dir}:${source_dir}/experiments/records/20260908-g2-planner-handoff-v1:${FEEDBACK_PYTHON_OVERLAY}
run_root=${FEEDBACK_STORAGE_ROOT}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs
export RUN_ID=${RUN_ID:-20260911-execution-failure-feedback-v1-spb2-B4-l6-second-plan-ab8-4gpu-s1098341402-r1}
export OUTPUT_DIR=${OUTPUT_DIR:-${run_root}/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/sp_0911_r1}
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exec bash "${record_dir}/run_feedback_eval_node47.sh" B
fi
test "${SC2_SECOND_PLAN_AUTHORIZED:-0}" = 1
test -z "$(git -C "${source_dir}" status --porcelain)"
test -z "$(nvidia-smi -i "${FEEDBACK_EVAL_GPUS}" --query-compute-apps=pid --format=csv,noheader)"
test ! -e "${OUTPUT_DIR}"
control=${CONTROL_DIR:?set unused control directory}
mkdir "${control}"
date -u +%FT%TZ > "${control}/started_at.txt"
deadline=$(( $(date +%s) + 2700 ))
finish() {
    status=$?
    echo "${status}" > "${control}/done"
    date -u +%FT%TZ > "${control}/ended_at.txt"
}
trap finish EXIT
CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" "${record_dir}/learning_eval.py" \
    --runs "${run_root}" --output "${control}/checkpoint-preflight.json"
remaining=$(( deadline - $(date +%s) - 15 ))
test "${remaining}" -gt 0
timeout --signal=INT --kill-after=15s "${remaining}" \
    bash "${record_dir}/run_feedback_eval_node47.sh" B > "${control}/run.log" 2>&1
