#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
mode=${1:?choose initial-eval, train8, or step8-eval}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 N_GPUS_PER_NODE=8
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
fi
export PYTHON_BIN=/usr/bin/python3 EXTRA_PYTHONPATH=/root/new-scii-spb2-level5-control/python
export PYTHONPATH=${repo_dir}:${EXTRA_PYTHONPATH} PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CONFIG_PATH=${repo_dir}/configs/qwen3-4b-instruct-2507-level5.env
export MODEL_PATH=/mnt/data0601/yangjinyue/new_SCII/assets/Qwen3-4B-Instruct-2507-cdbee75
export SC2PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII
export SC2_MAX_STEPS=0 SC2_ROLLOUT_N=8 SC2_ENV_SEED=0
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air BO_PROFILE=canonical SC2_STEP_SHAPING_MODE=off
export TOTAL_EPOCHS=8 SAVE_FREQ=1 EVAL_ONLY=0
export PPO_DIAGNOSTICS=1 ROLLOUT_LOGPROB_DIAGNOSTICS=0
unset SC2_GAME_TIME_LIMIT
suite=20260906-integrated-fixes-v1-qwen-instruct2507-l6-8gpu-fp32-rl96-gate8-s0-r4
root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-integrated-fixes-v1/runs
overrides=(
    trainer.total_training_steps=96
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp32
    +actor_rollout_ref.actor.fp32_normalization=True
    +actor_rollout_ref.ref.fsdp_config.model_dtype=fp32
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=fp32
    +actor_rollout_ref.ref.fp32_normalization=True
)
case "${mode}" in
    train8) tag=train8; wall_seconds=43200 ;;
    initial-eval|step8-eval)
        tag=${mode}; wall_seconds=10800
        export EVAL_ONLY=1 PPO_DIAGNOSTICS=0
        overrides+=(trainer.n_gpus_per_node=8 actor_rollout_ref.actor.ppo_mini_batch_size=8
            env.rollout.n=8
            +env.val_rollout_n=8 actor_rollout_ref.rollout.val_kwargs.n=8 ray_init.num_cpus=32
            env.prompt_response_log_max_prompt_chars=0)
        if [[ "${mode}" == step8-eval ]]; then
            checkpoint=${root}/${suite}-train8/checkpoints/global_step_8
            overrides+=(trainer.resume_mode=resume_path trainer.resume_from_path="${checkpoint}")
        fi
        ;;
    *) echo "Unknown mode: ${mode}" >&2; exit 2 ;;
esac
export RUN_ID=${RUN_ID:-${suite}-${tag}}
export OUTPUT_DIR=${OUTPUT_DIR:-${root}/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/sc2_real_r4_${tag}}
"${PYTHON_BIN}" - <<'PY'
import torch, transformers, ray, gymnasium
assert (torch.__version__, transformers.__version__, ray.__version__, gymnasium.__version__) == ('2.7.1+cu128', '4.55.2', '2.48.0', '1.2.3')
PY
exec timeout --signal=INT --kill-after=60s "${wall_seconds}" \
    bash "${repo_dir}/scripts/train_sc2.sh" "${overrides[@]}"
