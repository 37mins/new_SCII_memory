#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
arm=${1:-corrected}
if [[ "${arm}" != corrected ]]; then
    echo "usage: run_arm.sh corrected" >&2
    exit 2
fi

export CONFIG_PATH="${repo_dir}/configs/qwen3-4b-instruct-2507-level5.env"
export MODEL_PATH=${MODEL_PATH:-/mnt/data0601/yangjinyue/new_SCII/assets/Qwen3-4B-Instruct-2507-cdbee75}
export SC2PATH=${SC2PATH:-/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII}
export RUN_ID="20260903-r6-natural-termination-ab-v1-qwen-instruct2507-l5-gymnasium-rl1-s0-r1"
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/data0601/yangjinyue/new_SCII/experiments/20260903-r6-natural-termination-ab-v1/runs/${RUN_ID}}
export SC2_MAX_STEPS=20
export SC2_ROLLOUT_N=8
export SC2_ENV_SEED=0
export TOTAL_EPOCHS=1
export SAVE_FREQ=1

exec bash "${repo_dir}/scripts/train_qwen_level5.sh"
