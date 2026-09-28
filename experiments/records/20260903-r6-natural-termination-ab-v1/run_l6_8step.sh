#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
run_id=20260903-r6-natural-termination-ab-v1-qwen-instruct2507-l6-rl8-s0-r1

export CONFIG_PATH="${repo_dir}/configs/qwen3-4b-instruct-2507-level5.env"
export MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the verified Qwen3-4B-Instruct-2507 snapshot}
export SC2PATH=${SC2PATH:?set SC2PATH to the StarCraftII directory}
export PYTHON_BIN=${PYTHON_BIN:-python3}
export RUN_ID=${RUN_ID:-${run_id}}
export OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a node-local unique run directory}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/r_l6_rl8_s0r1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export SC2_DIFFICULTY=Harder
export SC2_ENV_SEED=0
export SC2_MAX_STEPS=20
export SC2_ROLLOUT_N=8
export BO_PROFILE=canonical
export TOTAL_EPOCHS=8
export SAVE_FREQ=8

exec bash "${repo_dir}/scripts/train_sc2.sh"
