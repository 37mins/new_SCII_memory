#!/usr/bin/env bash
set -euo pipefail

record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES=4,5,6,7 N_GPUS_PER_NODE=4
if [[ "${DRY_RUN:-0}" != 1 && -n "$(nvidia-smi --id=4,5,6,7 --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "Approved GPUs are occupied; not starting." >&2
    exit 2
fi
export RUN_ID=${RUN_ID:-20260905-r6-terminal-policy-credit-v1-qwen-instruct2507-l5-legacy-4gpu-rl1-s0-r2}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/data0601/yangjinyue/new_SCII/experiments/20260905-r6-terminal-policy-credit-v1/runs/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/policy0905_r2}
exec bash "${record_dir}/../20260905-r6-natural-step-shaping-ab-v1/run_arm.sh" legacy
