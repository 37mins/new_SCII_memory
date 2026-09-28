#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_arm.sh legacy|off}
case "${mode}" in legacy|off) ;; *) echo "expected legacy or off" >&2; exit 2 ;; esac
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
export EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/root/new-scii-spb2-level5-control/python}
export PYTHONPATH="${repo_dir}:${EXTRA_PYTHONPATH}"
"${PYTHON_BIN}" - <<'PY'
import torch, transformers, ray, gymnasium
actual = (torch.__version__, transformers.__version__, ray.__version__, gymnasium.__version__)
expected = ('2.7.1+cu128', '4.55.2', '2.48.0', '1.2.3')
if actual != expected:
    raise RuntimeError(f'Validated dependency versions changed: {actual} != {expected}')
PY
export SC2_STEP_SHAPING_MODE="${mode}"
export MODEL_PATH=${MODEL_PATH:-/mnt/data0601/yangjinyue/new_SCII/assets/Qwen3-4B-Instruct-2507-cdbee75}
export SC2PATH=${SC2PATH:-/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII}
export RUN_ID=${RUN_ID:-20260905-r6-natural-step-shaping-ab-v1-qwen-instruct2507-l5-${mode}-rl1-s0-r1}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/data0601/yangjinyue/new_SCII/experiments/20260905-r6-natural-step-shaping-ab-v1/runs/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/shp0905_${mode}_r1}
export SC2_MAX_STEPS=0 SC2_ROLLOUT_N=8 SC2_ENV_SEED=0
export SC2_DIFFICULTY=Hard SC2_AI_BUILD=Air BO_PROFILE=canonical
export TOTAL_EPOCHS=1 SAVE_FREQ=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
unset SC2_GAME_TIME_LIMIT
exec timeout --signal=INT --kill-after=60s 7200 bash "${repo_dir}/scripts/train_qwen_instruct2507_level5.sh"
