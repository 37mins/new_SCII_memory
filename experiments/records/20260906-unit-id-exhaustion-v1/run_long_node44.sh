#!/usr/bin/env bash
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 N_GPUS_PER_NODE=8
export RUN_ID=${RUN_ID:-20260906-unit-id-exhaustion-v1-qwen-instruct2507-l6-8gpu-rl96-s0-r1}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-unit-id-exhaustion-v1/runs/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/unitid_l6_96_8_r1}
export SAVE_FREQ=8
exec bash "${record_dir}/../20260905-qwen-generation-position-v1/run_stage.sh" l6-96
