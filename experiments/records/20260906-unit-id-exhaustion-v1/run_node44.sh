#!/usr/bin/env bash
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export RUN_ID=${RUN_ID:-20260906-unit-id-exhaustion-v1-qwen-instruct2507-l6-4gpu-rl1-s0-r1}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-unit-id-exhaustion-v1/runs/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/unitid_l6_1_r1}
exec bash "${record_dir}/../20260905-qwen-generation-position-v1/run_stage.sh" l6-1
