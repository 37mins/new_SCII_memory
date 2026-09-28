#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
result_root=${RESULT_ROOT:?set RESULT_ROOT to a node-local result directory}
model_path=${MODEL_PATH:?set MODEL_PATH to the verified Qwen3-4B-Instruct-2507 snapshot}
sc2_path=${SC2PATH:?set SC2PATH to the StarCraftII directory}
fivebo_root=${FIVEBO_ROOT:?set FIVEBO_ROOT to the verified g1-g5 directory}
python_bin=${PYTHON_BIN:-python3}
run_prefix=${RUN_PREFIX:-20260903-r6-natural-termination-ab-v1-qwen-instruct2507-l5-fivebo}

mkdir -p "${result_root}"

run_one() {
    local bo_key=$1 gpu=$2
    local run_id=${run_prefix}-${bo_key}-s0-r1
    CONFIG_PATH="${repo_dir}/configs/qwen3-4b-instruct-2507-level5.env" \
    MODEL_PATH="${model_path}" \
    SC2PATH="${sc2_path}" \
    FIVEBO_ROOT="${fivebo_root}" \
    PYTHON_BIN="${python_bin}" \
    RUN_ID="${run_id}" \
    OUTPUT_DIR="${result_root}/${run_id}" \
    RAY_TMPDIR="/dev/shm/r_${bo_key}_$$_l5" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    SC2_DIFFICULTY=Hard \
    SC2_ENV_SEED=0 \
    BO_PROFILE=fivebo \
    BO_KEY="${bo_key}" \
    EVAL_ONLY=1 \
    bash "${repo_dir}/scripts/train_sc2.sh"
}

pids=()
for spec in g1:0 g2:1 g3:2 g4:3 g5:4; do
    run_one "${spec%%:*}" "${spec##*:}" > "${result_root}/${run_prefix}-${spec%%:*}.launcher.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
printf 'STATUS=%s\n' "${status}" > "${result_root}/${run_prefix}.done"
exit "${status}"
