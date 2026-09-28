#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export CUDA_VISIBLE_DEVICES=1,2 N_GPUS_PER_NODE=2
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    test -z "$(nvidia-smi --id=1,2 --query-compute-apps=pid --format=csv,noheader)"
fi
export PYTHON_BIN=/usr/bin/python3 EXTRA_PYTHONPATH=/root/new-scii-qwen-kl-metric-20260906/python
export PYTHONPATH=${repo_dir}:${EXTRA_PYTHONPATH} PYTHONDONTWRITEBYTECODE=1
export C_INCLUDE_PATH=/root/new-scii-qwen-kl-metric-20260906/include/python3.12:/root/new-scii-qwen-kl-metric-20260906/include
test -r "${C_INCLUDE_PATH%%:*}/Python.h"
"${PYTHON_BIN}" - <<'PY'
import torch, transformers, ray, gymnasium
assert (torch.__version__, transformers.__version__, ray.__version__, gymnasium.__version__) == ('2.7.1+cu128', '4.55.2', '2.48.0', '1.2.3')
PY
export MODEL_PATH=/mnt/data0601/yangjinyue/new_SCII/assets/Qwen3-4B-Instruct-2507-cdbee75
export SC2PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII
export RUN_ID=${RUN_ID:-20260906-qwen-kl-metric-v1-qwen-instruct2507-l6-2gpu-integrated-trace-rl1-s0-r8}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-qwen-kl-metric-v1/runs/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/qkl_int_l6_r8}
export SC2_MAX_STEPS=0 SC2_ROLLOUT_N=8 SC2_ENV_SEED=0
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air BO_PROFILE=canonical
export SC2_STEP_SHAPING_MODE=off EVAL_ONLY=0 TOTAL_EPOCHS=1 SAVE_FREQ=1
export PPO_DIAGNOSTICS=1 ROLLOUT_LOGPROB_DIAGNOSTICS=1
unset SC2_GAME_TIME_LIMIT
exec timeout --signal=INT --kill-after=60s 10800 bash "${repo_dir}/scripts/train_qwen_instruct2507_level5.sh"
