#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
project_root=/root/new-scii-system-v1-eval-20260831
source_sha=$(git -C "${repo_dir}" rev-parse --short=8 HEAD)
export PATH=/root/new-scii-spb2-rl-smoke/runtime/bin:${PATH}
export EXTRA_PYTHONPATH=/root/new-scii-spb2-rl-smoke/python:/root/new-scii-spb2-rl-smoke/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted/flash-linear-attention_dev:/root/new-scii-spb2-rl-smoke/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted/MoBA
export PYTHONPATH=${EXTRA_PYTHONPATH}
export CPATH=/root/new-scii-spb2-rl-smoke/sysroot/usr/include/python3.12:/root/new-scii-spb2-rl-smoke/sysroot/usr/include
export SC2PATH=/dev/shm/new_scii_level5_attn_ab_assets/StarCraftII
export FIVEBO_ROOT=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/repro/mse_micro_five_bo_20260711/fixed_bos
export SC2_DIFFICULTY=Hard SC2_ENV_SEED=0 EVAL_GPU_IDS='0 1 2 3 4'
cd "${repo_dir}"
status=0
if [[ "$#" == 0 ]]; then set -- qwen spb; fi
for model in "$@"; do
    [[ "${model}" == qwen || "${model}" == spb ]] || exit 2
    if [[ "${model}" == qwen ]]; then
        export MODEL_PATH=/dev/shm/new_scii_full15_sft_v1/models/qwen
        export CONFIG_PATH=${repo_dir}/configs/qwen3-4b-level5.env
    else
        export MODEL_PATH=/root/new-scii-spb2-rl-smoke/model
        export CONFIG_PATH=${repo_dir}/configs/spikingbrain2-level5.env
    fi
    export RUN_ID=20260831-system-v1-${model}-fivebo-l5-s0-${source_sha}
    export RESULT_ROOT=${project_root}/runs/${RUN_ID}
    bash scripts/eval_five_bo.sh || status=1
done
exit "${status}"
