#!/usr/bin/env bash
# One Level6 training arm; the pair controller owns the three-hour limit.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
case "${1:?A or B}" in
    A) export CUDA_VISIBLE_DEVICES=${FEEDBACK_RL_GPUS:-0,1,2,3} SC2_EXECUTION_FAILURE_FEEDBACK=0 ;;
    B) export CUDA_VISIBLE_DEVICES=${FEEDBACK_RL_GPUS:-4,5,6,7} SC2_EXECUTION_FAILURE_FEEDBACK=1 ;;
    *) exit 2 ;;
esac
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    test "${SC2_FEEDBACK_RL_AUTHORIZED:-0}" = 1
    test -z "$(git -C "${source_dir}" status --porcelain)"
    test -z "$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader)"
fi
storage=${FEEDBACK_STORAGE_ROOT:-/root/new-scii-execution-feedback-local-20260911/shared}
assets=${storage}/yangjinyue/sc2-verl-agent-experiments/assets
deps=${assets}/repro/stageguard_v7_legacy_contract_20260806/source/extracted
export PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3} N_GPUS_PER_NODE=4
export CONFIG_PATH=${source_dir}/configs/spikingbrain2-level5.env
python_overlay=${FEEDBACK_PYTHON_OVERLAY:-/root/new-scii-spb-level5-96-20260908/python}
include_root=${FEEDBACK_INCLUDE_ROOT:-/root/new-scii-spb-level5-96-20260908/include}
export EXTRA_PYTHONPATH=${source_dir}/experiments/records/20260906-integrated-fixes-v1:${python_overlay}:${deps}/flash-linear-attention_dev:${deps}/MoBA
export PYTHONPATH=${source_dir}:${EXTRA_PYTHONPATH}
export CPATH=${include_root}/python3.12:${include_root}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MODEL_PATH=${MODEL_PATH:-${assets}/models/Panyuqi/SpikingBrain-2.0-instruct}
export SC2PATH=${SC2PATH:-${assets}/sc2/StarCraftII}
export SPB_DISABLE_SWA_DROPOUT=1
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air SC2_ENV_SEED=0 SC2_MAX_STEPS=0 SC2_ROLLOUT_N=8
export SC2_FIXED_BO_PREFIX_LIMIT=-1 SC2_STEP_SHAPING_MODE=off BO_PROFILE=canonical EVAL_ONLY=0
export SC2_ORIGINAL_BO_SEGMENT=1 SC2_CLARIFY_BO_QUOTA=1
export TOTAL_EPOCHS=4 SAVE_FREQ=4 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=192
resume_args=(trainer.resume_mode=disable)
if [[ "${FEEDBACK_CONTINUE_A4:-0}" == 1 ]]; then
    test "${1}" = A
    export EXTRA_PYTHONPATH=${record_dir}:${source_dir}/experiments/records/20260910-spb-fsdp-layer-wrap-v1:${EXTRA_PYTHONPATH}
    parent=${storage}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/20260911-execution-failure-feedback-v1-spb2-original-l6-g2-A-rl4-4gpu-s0-r2
    resume_args=(+ray_init.runtime_env.worker_process_setup_hook=continue_a4.install
        trainer.resume_mode=resume_path trainer.resume_from_path="${parent}/checkpoints/global_step_4"
        trainer.del_local_ckpt_after_load=False)
    export RUN_ID=${RUN_ID:-20260911-execution-failure-feedback-v1-spb2-A4-l6-s4plus4-4gpu-s0-r1}
    export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/a4t_0911_r1}
fi
if [[ "${FEEDBACK_A4_SCENE1:-0}" == 1 ]]; then
    test "${FEEDBACK_CONTINUE_A4:-0}" = 1
    test -z "${FEEDBACK_RL_TEMPERATURE:-}"
    export TOTAL_EPOCHS=1 SAVE_FREQ=1
    resume_args[0]=+ray_init.runtime_env.worker_process_setup_hook=continue_a4.install_scene
fi
if [[ "${FEEDBACK_A4_COVERAGE:-0}" == 1 ]]; then
    test "${FEEDBACK_CONTINUE_A4:-0}" = 1
    test "${FEEDBACK_A4_SCENE1:-0}" = 0
    test -z "${FEEDBACK_RL_TEMPERATURE:-}"
    resume_args[0]=+ray_init.runtime_env.worker_process_setup_hook=continue_a4.install_coverage
fi
if [[ -n "${FEEDBACK_A4_SIGNAL_ARM:-}" ]]; then
    test "${FEEDBACK_A4_COVERAGE:-0}" = 1
    case "${FEEDBACK_A4_SIGNAL_ARM}" in A|B) ;; *) exit 2 ;; esac
    resume_args[0]=+ray_init.runtime_env.worker_process_setup_hook=continue_a4.install_signal_${FEEDBACK_A4_SIGNAL_ARM}
fi
temperature_args=()
if [[ -n "${FEEDBACK_RL_TEMPERATURE:-}" ]]; then
    test "${FEEDBACK_CONTINUE_A4:-0}" = 1
    case "${FEEDBACK_RL_TEMPERATURE}" in 0.4|0.8) ;; *) exit 2 ;; esac
    temperature_args=(actor_rollout_ref.rollout.temperature="${FEEDBACK_RL_TEMPERATURE}")
fi
export SC2_PLAN_SEGMENT_MAX=6 SC2_PLAN_BULLET_MAX=6
export PPO_DIAGNOSTICS=0 ROLLOUT_LOGPROB_DIAGNOSTICS=0
export SC2_SEED_PYTHON_RANDOM=1 SC2_RESET_RETRIES=1
unset SC2_GAME_TIME_LIMIT SC2_RNG_PREFIX_TARGET
export RUN_ID=${RUN_ID:-20260911-execution-failure-feedback-v1-spb2-original-l6-g2-${1}-rl4-4gpu-s0-r1}
export OUTPUT_DIR=${OUTPUT_DIR:-${storage}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/sc2_feedback_rl4_${1}_0911_r1}
export TMPDIR=${RAY_TMPDIR} TRITON_CACHE_DIR=${RAY_TMPDIR}/triton HF_HOME=${RAY_TMPDIR}/hf
exec bash "${source_dir}/scripts/train_sc2.sh" \
    "+ray_init.runtime_env.env_vars.SC2_SEED_PYTHON_RANDOM='1'" \
    "+ray_init.runtime_env.env_vars.SC2_RESET_RETRIES='1'" \
    "+ray_init.runtime_env.env_vars.SC2_PRESERVE_VALID_BO='1'" \
    "+ray_init.runtime_env.env_vars.SC2_ENABLE_AIR_OPTIONS='1'" \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp32 \
    +actor_rollout_ref.actor.fp32_normalization=True \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=fp32 \
    +actor_rollout_ref.ref.fp32_normalization=True \
    ++env.prompt_response_log_max_prompt_chars=0 \
    trainer.total_training_steps=96 \
    ++ray_init.num_cpus=32 \
    ++actor_rollout_ref.rollout.fsdp_layer_wrap=True \
    "${resume_args[@]}" \
    ${temperature_args[@]+"${temperature_args[@]}"} \
    "${@:2}"
