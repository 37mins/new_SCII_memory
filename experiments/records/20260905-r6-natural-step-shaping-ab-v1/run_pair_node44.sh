#!/usr/bin/env bash
set -euo pipefail

record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES=4,5,6,7 N_GPUS_PER_NODE=4
for mode in legacy off; do
    if [[ -n "$(nvidia-smi --id=4,5,6,7 --query-compute-apps=pid --format=csv,noheader)" ]]; then
        echo "Selected GPUs are occupied; not starting ${mode}." >&2
        exit 2
    fi
    export RUN_ID="20260905-r6-natural-step-shaping-ab-v1-qwen-instruct2507-l5-${mode}-4gpu-rl1-s0-r1"
    export RAY_TMPDIR="/dev/shm/shp4_0905_${mode}_r1"
    export OUTPUT_DIR="/mnt/data0601/yangjinyue/new_SCII/experiments/20260905-r6-natural-step-shaping-ab-v1/runs/${RUN_ID}"
    bash "${record_dir}/run_arm.sh" "${mode}"
    /usr/bin/python3 - "${OUTPUT_DIR}/result.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
if result['status'] != 'completed':
    raise SystemExit('Run completion audit failed; stopping the pair')
PY
done
