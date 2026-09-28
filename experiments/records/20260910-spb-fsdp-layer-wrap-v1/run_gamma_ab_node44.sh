#!/usr/bin/env bash
# One arm, eight new global steps from step4; parent gameplay and loss are frozen.
set -euo pipefail
arm=${1:?A or B}
case "${arm}" in A) gamma=0.95 ;; B) gamma=1.0 ;; *) exit 2 ;; esac
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 N_GPUS_PER_NODE=8
export PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
run_root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs
parent=${run_root}/20260910-spb-fsdp-layer-wrap-v1-spb2-l6-g2-opening-quota-rl96-8gpu-s0-r1
export CONFIG_PATH=${source_dir}/configs/spikingbrain2-level5.env
deps=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted
export EXTRA_PYTHONPATH=${record_dir}:${source_dir}/experiments/records/20260906-integrated-fixes-v1:/root/new-scii-spb2-level5-control/python:${deps}/flash-linear-attention_dev:${deps}/MoBA
export PYTHONPATH=${source_dir}:${EXTRA_PYTHONPATH}
export CPATH=/usr/include/python3.12:/usr/include
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export MODEL_PATH=${MODEL_PATH:-/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/models/Panyuqi/SpikingBrain-2.0-instruct}
export SC2PATH=${SC2PATH:-/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII}
export SPB_DISABLE_SWA_DROPOUT=1
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air SC2_ENV_SEED=0 SC2_MAX_STEPS=0 SC2_ROLLOUT_N=8
export SC2_FIXED_BO_PREFIX_LIMIT=-1 SC2_STEP_SHAPING_MODE=off BO_PROFILE=canonical EVAL_ONLY=0
export SC2_ORIGINAL_BO_SEGMENT=1 SC2_CLARIFY_BO_QUOTA=1
export TOTAL_EPOCHS=8 SAVE_FREQ=4 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=192
export SC2_PLAN_SEGMENT_MAX=6 SC2_PLAN_BULLET_MAX=6
export PPO_DIAGNOSTICS=0 ROLLOUT_LOGPROB_DIAGNOSTICS=0
export SC2_SEED_PYTHON_RANDOM=1 SC2_RESET_RETRIES=1
unset SC2_GAME_TIME_LIMIT SC2_RNG_PREFIX_TARGET
export RUN_ID=${RUN_ID:-20260910-spb-fsdp-layer-wrap-v1-spb2-l6-g2-gamma${gamma}-s4plus8-8gpu-s0-r2}
export OUTPUT_DIR=${OUTPUT_DIR:-${run_root}/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/sc2_gamma_${arm}_0911_r2}
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    test "${SC2_GAMMA_AB_AUTHORIZED:-0}" = 1
    test -z "$(git -C "${source_dir}" status --porcelain)"
    test -z "$(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)"
    (cd "${source_dir}" && sha256sum -c configs/system-v1.sha256)
fi
exec bash "${source_dir}/scripts/train_sc2.sh" \
    +ray_init.runtime_env.worker_process_setup_hook=gamma_ab.install \
    "+ray_init.runtime_env.env_vars.SC2_SEED_PYTHON_RANDOM='1'" \
    "+ray_init.runtime_env.env_vars.SC2_RESET_RETRIES='1'" \
    "+ray_init.runtime_env.env_vars.SC2_PRESERVE_VALID_BO='1'" \
    "+ray_init.runtime_env.env_vars.SC2_ENABLE_AIR_OPTIONS='1'" \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp32 \
    +actor_rollout_ref.actor.fp32_normalization=True \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=fp32 \
    +actor_rollout_ref.ref.fp32_normalization=True \
    ++actor_rollout_ref.rollout.fsdp_layer_wrap=True \
    ++env.prompt_response_log_max_prompt_chars=0 \
    trainer.total_training_steps=96 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${parent}/checkpoints/global_step_4" \
    trainer.del_local_ckpt_after_load=False \
    algorithm.gamma="${gamma}" \
    ++ray_init.num_cpus=32
