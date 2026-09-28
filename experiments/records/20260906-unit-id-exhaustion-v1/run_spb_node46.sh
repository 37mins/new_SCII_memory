#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 N_GPUS_PER_NODE=8
if [[ "${DRY_RUN:-0}" != 1 && -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "Approved GPUs are occupied; not starting." >&2
    exit 2
fi
export PYTHON_BIN=/usr/bin/python3
deps=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted
export EXTRA_PYTHONPATH=${SPB_PYTHON_DIR:-/root/new-scii-spb-unitid-20260906/python}:${deps}/flash-linear-attention_dev:${deps}/MoBA
export PYTHONPATH=${repo_dir}:${EXTRA_PYTHONPATH} PYTHONDONTWRITEBYTECODE=1
"${PYTHON_BIN}" - <<'PY'
import torch, transformers, ray, gymnasium, fla, moba, flash_attn
actual = (torch.__version__, transformers.__version__, ray.__version__, gymnasium.__version__)
expected = ('2.7.1+cu128', '4.55.2', '2.48.0', '1.2.3')
if actual != expected:
    raise RuntimeError(f'Validated dependency versions changed: {actual} != {expected}')
print('SpB dependencies:', actual, fla.__file__, moba.__file__, flash_attn.__file__)
PY
export MODEL_PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/models/Panyuqi/SpikingBrain-2.0-instruct
export SC2PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII
export RUN_ID=${RUN_ID:-20260906-unit-id-exhaustion-v1-spb2-instruct-l5-8gpu-rl96-drop0-s0-r1}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-unit-id-exhaustion-v1/runs/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/unitid_spb_l5_96_r1}
export SC2_MAX_STEPS=0 SC2_ROLLOUT_N=8 SC2_ENV_SEED=0
export SC2_DIFFICULTY=Hard SC2_AI_BUILD=Air BO_PROFILE=canonical
export SC2_STEP_SHAPING_MODE=legacy EVAL_ONLY=0 SPB_DISABLE_SWA_DROPOUT=1
export TOTAL_EPOCHS=96 SAVE_FREQ=8
unset SC2_GAME_TIME_LIMIT
exec timeout --signal=INT --kill-after=60s 259200 bash "${repo_dir}/scripts/train_spb_level5.sh"
