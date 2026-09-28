#!/usr/bin/env bash
set -euo pipefail

# Run the existing repair, not the current development checkout.
source_dir=${SOURCE_DIR:?set SOURCE_DIR to the clean 8ac1e19 deployment}
export RAY_TMPDIR=${RAY_TMPDIR:?set RAY_TMPDIR to a short unique path, e.g. /dev/shm/r6_nat_0905_r2}
test "$(git -C "${source_dir}" rev-parse HEAD)" = 8ac1e197aa11403a43f3afd5b7b204188f653c4f
export PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
export EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/root/new-scii-spb2-level5-control/python}
export PYTHONPATH="${source_dir}:${EXTRA_PYTHONPATH}"
"${PYTHON_BIN}" - <<'PY'
import torch, transformers, ray, gymnasium
actual = (torch.__version__, transformers.__version__, ray.__version__, gymnasium.__version__)
expected = ('2.7.1+cu128', '4.55.2', '2.48.0', '1.2.3')
if actual != expected:
    raise RuntimeError(f'Historical A/B dependencies changed: {actual} != {expected}')
print('Verified historical A/B dependencies:', actual, flush=True)
PY

export MODEL_PATH=${MODEL_PATH:-/mnt/data0601/yangjinyue/new_SCII/assets/Qwen3-4B-Instruct-2507-cdbee75}
export SC2PATH=${SC2PATH:-/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII}
export RUN_ID=${RUN_ID:-20260905-r6-natural-termination-ab-v1-qwen-instruct2507-l5-uncapped-rl1-s0-r2}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/data0601/yangjinyue/new_SCII/experiments/20260903-r6-natural-termination-ab-v1/runs/${RUN_ID}}
export SC2_MAX_STEPS=0 SC2_ROLLOUT_N=8 SC2_ENV_SEED=0
export SC2_DIFFICULTY=Hard SC2_AI_BUILD=Air BO_PROFILE=canonical
export TOTAL_EPOCHS=1 SAVE_FREQ=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset SC2_GAME_TIME_LIMIT
exec bash "${source_dir}/scripts/train_qwen_instruct2507_level5.sh"
