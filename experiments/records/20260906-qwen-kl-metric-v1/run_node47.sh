#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 N_GPUS_PER_NODE=8
if [[ "${DRY_RUN:-0}" != 1 && -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "Approved GPUs are occupied; not starting." >&2
    exit 2
fi
runtime=/dev/shm/spb-unitid-20260906
export PYTHON_BIN=/usr/bin/python3 EXTRA_PYTHONPATH=${runtime}/python
export PYTHONPATH=${repo_dir}:${EXTRA_PYTHONPATH} PYTHONDONTWRITEBYTECODE=1
export TMPDIR=${runtime}/tmp XDG_CACHE_HOME=${runtime}/cache
export HF_HOME=${XDG_CACHE_HOME}/huggingface TRITON_CACHE_DIR=${XDG_CACHE_HOME}/triton
export TORCHINDUCTOR_CACHE_DIR=${XDG_CACHE_HOME}/torchinductor
export TORCH_EXTENSIONS_DIR=${XDG_CACHE_HOME}/torch_extensions
export CPATH=${runtime}/include/python3.12:${runtime}/include
"${PYTHON_BIN}" - <<'PY'
import torch, transformers, ray, gymnasium
actual = (torch.__version__, transformers.__version__, ray.__version__, gymnasium.__version__)
expected = ('2.7.1+cu128', '4.55.2', '2.48.0', '1.2.3')
if actual != expected:
    raise RuntimeError(f'Validated dependency versions changed: {actual} != {expected}')
PY
export MODEL_PATH=/mnt/data0601/yangjinyue/new_SCII/assets/Qwen3-4B-Instruct-2507-cdbee75
export SC2PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII
export RUN_ID=${RUN_ID:-20260906-qwen-kl-metric-v1-qwen-instruct2507-l6-8gpu-trace-rl1-s0-r5}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-qwen-kl-metric-v1/runs/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/qkl_l6_trace_r5}
export SC2_MAX_STEPS=0 SC2_ROLLOUT_N=8 SC2_ENV_SEED=0
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air BO_PROFILE=canonical
export SC2_STEP_SHAPING_MODE=legacy EVAL_ONLY=0 PPO_DIAGNOSTICS=1
export TOTAL_EPOCHS=1 SAVE_FREQ=1
unset SC2_GAME_TIME_LIMIT
exec timeout --signal=INT --kill-after=60s 7200 bash "${repo_dir}/scripts/train_qwen_instruct2507_level5.sh"
