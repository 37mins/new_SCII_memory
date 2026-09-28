#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export CUDA_VISIBLE_DEVICES=4,5,6,7 N_GPUS_PER_NODE=4
if [[ "${DRY_RUN:-0}" != 1 && -n "$(nvidia-smi --id=4,5,6,7 --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "Approved GPUs are occupied; not starting." >&2
    exit 2
fi
export PYTHON_BIN=/usr/bin/python3
export EXTRA_PYTHONPATH=/root/new-scii-spb2-level5-control/python
export PYTHONPATH=${repo_dir}:${EXTRA_PYTHONPATH} PYTHONDONTWRITEBYTECODE=1
"${PYTHON_BIN}" - <<'PY'
import torch, transformers, ray, gymnasium
actual = (torch.__version__, transformers.__version__, ray.__version__, gymnasium.__version__)
expected = ('2.7.1+cu128', '4.55.2', '2.48.0', '1.2.3')
if actual != expected:
    raise RuntimeError(f'Validated dependency versions changed: {actual} != {expected}')
PY
export MODEL_PATH=/mnt/data0601/yangjinyue/new_SCII/assets/Qwen3-4B-Instruct-2507-cdbee75
export SC2PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII
export RUN_ID=${RUN_ID:-20260906-integrated-fixes-v1-qwen-instruct2507-l6-4gpu-rl4-s0-r1}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-integrated-fixes-v1/runs/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/sc2_int_l6_4_r1}
export SC2_MAX_STEPS=0 SC2_ROLLOUT_N=8 SC2_ENV_SEED=0
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air BO_PROFILE=canonical
export SC2_STEP_SHAPING_MODE=off EVAL_ONLY=0 TOTAL_EPOCHS=4 SAVE_FREQ=1
unset SC2_GAME_TIME_LIMIT
exec timeout --signal=INT --kill-after=60s 21600 bash "${repo_dir}/scripts/train_qwen_instruct2507_level5.sh"
