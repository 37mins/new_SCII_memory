#!/usr/bin/env bash
# One arm, under the pair's shared one-hour ceiling. Never automatically retry.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
case "${1:?A or B}" in
    A) export CUDA_VISIBLE_DEVICES=0,1,2,3 SC2_EXECUTION_FAILURE_FEEDBACK=0 ;;
    B) export CUDA_VISIBLE_DEVICES=4,5,6,7 SC2_EXECUTION_FAILURE_FEEDBACK=1 ;;
    *) exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=${FEEDBACK_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES}}
export N_GPUS_PER_NODE=4 PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    test "${SC2_FEEDBACK_AB_AUTHORIZED:-0}" = 1
    test -z "$(git -C "${source_dir}" status --porcelain)"
    test -z "$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader)"
fi
storage_root=${FEEDBACK_STORAGE_ROOT:-/mnt/data0601}
parent=${storage_root}/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs/20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1
checkpoint=${parent}/checkpoints/global_step_24
eval_hook=feedback_eval.install
eval_games=4
resume_args=(trainer.resume_mode=resume_path trainer.resume_from_path="${checkpoint}")
if [[ -n "${FEEDBACK_LEARNING_MODEL:-}" ]]; then
    export SC2_EXECUTION_FAILURE_FEEDBACK=1
    eval_hook=learning_eval.install
    eval_games=8
    case "${FEEDBACK_LEARNING_MODEL}" in
        original) resume_args=(trainer.resume_mode=disable) ;;
        A4|B4)
            parent=${storage_root}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/20260911-execution-failure-feedback-v1-spb2-original-l6-g2-${FEEDBACK_LEARNING_MODEL:0:1}-rl4-4gpu-s0-r2
            checkpoint=${parent}/checkpoints/global_step_4
            resume_args=(trainer.resume_mode=resume_path trainer.resume_from_path="${checkpoint}") ;;
        A8)
            parent=${storage_root}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/20260911-execution-failure-feedback-v1-spb2-A4-l6-s4plus4-4gpu-s0-r1
            checkpoint=${parent}/checkpoints/global_step_8
            resume_args=(trainer.resume_mode=resume_path trainer.resume_from_path="${checkpoint}") ;;
        T08A8)
            parent=${storage_root}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/20260911-execution-failure-feedback-v1-spb2-A4-l6-temp08-rl4-4gpu-s0-r1
            checkpoint=${parent}/checkpoints/global_step_8
            resume_args=(trainer.resume_mode=resume_path trainer.resume_from_path="${checkpoint}") ;;
        A5scene)
            test "${FEEDBACK_SCENE_EVAL:-0}" = 1
            parent=${storage_root}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/20260911-execution-failure-feedback-v1-spb2-A4-l6-scene1098341402-rl1-4gpu-r1
            checkpoint=${parent}/checkpoints/global_step_5
            resume_args=(trainer.resume_mode=resume_path trainer.resume_from_path="${checkpoint}") ;;
        SceneA8|SceneB8)
            test "${FEEDBACK_SCENE_POSTEVAL:-0}" = 1
            checkpoint=${FEEDBACK_POST_CHECKPOINT:?set verified scene checkpoint}
            resume_args=(trainer.resume_mode=resume_path trainer.resume_from_path="${checkpoint}") ;;
        *) exit 2 ;;
    esac
fi
if [[ "${FEEDBACK_SECOND_PLAN:-0}" == 1 ]]; then
    test "${FEEDBACK_LEARNING_MODEL:-}" = B4
    case "${FEEDBACK_SECOND_PLAN_VARIANT:-recorded}" in
        recorded) eval_hook=second_plan_eval.install ;;
        marine_only) eval_hook=second_plan_eval.install_marine_only ;;
        *) exit 2 ;;
    esac
fi
eval_temperature=0.4
temperature_args=()
if [[ -n "${FEEDBACK_TEMPERATURE_AB:-}" ]]; then
    test "${FEEDBACK_LEARNING_MODEL:-}" = A4
    test "${FEEDBACK_SECOND_PLAN:-0}" = 0
    case "${FEEDBACK_TEMPERATURE_AB}" in 0.4|0.8) ;; *) exit 2 ;; esac
    eval_temperature=${FEEDBACK_TEMPERATURE_AB}
    eval_hook=temperature_eval.install
    temperature_args=("+ray_init.runtime_env.env_vars.SC2_FEEDBACK_TEMPERATURE_AB='${eval_temperature}'")
fi
if [[ -n "${FEEDBACK_FIRST_ATTACK_THRESHOLD:-}" ]]; then
    test "${FEEDBACK_LEARNING_MODEL:-}" = A4
    test "${FEEDBACK_SECOND_PLAN:-0}" = 0
    test -z "${FEEDBACK_TEMPERATURE_AB:-}"
    case "${FEEDBACK_FIRST_ATTACK_THRESHOLD}" in 20|40) ;; *) exit 2 ;; esac
    eval_hook=first_attack_eval.install
    eval_games=4
    temperature_args=("+ray_init.runtime_env.env_vars.SC2_FIRST_ATTACK_THRESHOLD_AB='${FEEDBACK_FIRST_ATTACK_THRESHOLD}'")
fi
scene_args=()
if [[ "${FEEDBACK_SCENE_EVAL:-0}" == 1 ]]; then
    case "${FEEDBACK_LEARNING_MODEL:-}" in A4|A5scene) ;; *) exit 2 ;; esac
    test "${FEEDBACK_SECOND_PLAN:-0}" = 0
    test -z "${FEEDBACK_TEMPERATURE_AB:-}${FEEDBACK_FIRST_ATTACK_THRESHOLD:-}"
    export SC2_EXECUTION_FAILURE_FEEDBACK=0
    case "${FEEDBACK_SCENE_SET:-single}" in
        single) eval_hook=learning_eval.install_scene ;;
        historical8) eval_hook=learning_eval.install_multiseed ;;
        screening)
            test "${FEEDBACK_LEARNING_MODEL}" = A4
            case "${SC2_SCREEN_SCENE_INDEX:?set scene index 0-7}" in [0-7]) ;; *) exit 2 ;; esac
            eval_hook=learning_eval.install_screen_scene
            scene_args=("+ray_init.runtime_env.env_vars.SC2_SCREEN_SCENE_INDEX='${SC2_SCREEN_SCENE_INDEX}'") ;;
        *) exit 2 ;;
    esac
fi
export CONFIG_PATH=${source_dir}/configs/spikingbrain2-level5.env
if [[ "${FEEDBACK_SCENE_POSTEVAL:-0}" == 1 ]]; then
    case "${FEEDBACK_LEARNING_MODEL:-}" in A4|SceneA8|SceneB8) ;; *) exit 2 ;; esac
    test "${FEEDBACK_SCENE_EVAL:-0}" = 0
    test "${FEEDBACK_SECOND_PLAN:-0}" = 0
    test -z "${FEEDBACK_TEMPERATURE_AB:-}${FEEDBACK_FIRST_ATTACK_THRESHOLD:-}${FEEDBACK_PHASE2_AIR:-}"
    export SC2_EXECUTION_FAILURE_FEEDBACK=0
    checkpoint=${FEEDBACK_POST_CHECKPOINT:?set verified scene checkpoint}
    resume_args=(trainer.resume_mode=resume_path trainer.resume_from_path="${checkpoint}")
    eval_hook=learning_eval.install_scene_posteval
    eval_games=8
    if [[ -n "${FEEDBACK_SCENE_TEMPERATURE:-}" ]]; then
        test "${FEEDBACK_LEARNING_MODEL}" = SceneB8
        case "${FEEDBACK_SCENE_TEMPERATURE}" in 0.4|0.8) ;; *) exit 2 ;; esac
        eval_temperature=${FEEDBACK_SCENE_TEMPERATURE}
        eval_hook=temperature_eval.install_scene_b8
        temperature_args=("+ray_init.runtime_env.env_vars.SC2_SCENE_TEMPERATURE='${eval_temperature}'")
    fi
fi
if [[ -n "${FEEDBACK_SCENE_TEMPERATURE:-}" ]]; then test "${FEEDBACK_SCENE_POSTEVAL:-0}" = 1; fi
experiment_extra_path=
if [[ -n "${FEEDBACK_PHASE2_AIR:-}" ]]; then
    test "${FEEDBACK_LEARNING_MODEL:-}" = A4
    test "${FEEDBACK_SCENE_EVAL:-0}" = 0
    test "${FEEDBACK_SECOND_PLAN:-0}" = 0
    test -z "${FEEDBACK_TEMPERATURE_AB:-}${FEEDBACK_FIRST_ATTACK_THRESHOLD:-}"
    case "${FEEDBACK_PHASE2_AIR}" in A|B) ;; *) exit 2 ;; esac
    export SC2_EXECUTION_FAILURE_FEEDBACK=0
    eval_hook=phase2_eval.install
    eval_games=4
    experiment_extra_path=${source_dir}/experiments/records/20260913-phase2-air-options-v1:
    temperature_args=("+ray_init.runtime_env.env_vars.SC2_PHASE2_AIR_AB='${FEEDBACK_PHASE2_AIR}'")
fi
deps=${storage_root}/yangjinyue/sc2-verl-agent-experiments/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted
python_overlay=${FEEDBACK_PYTHON_OVERLAY:-/root/new-scii-spb2-level5-control/python}
include_root=${FEEDBACK_INCLUDE_ROOT:-/dev/shm/spb-unitid-20260906/include}
export EXTRA_PYTHONPATH=${experiment_extra_path}${record_dir}:${source_dir}/experiments/records/20260910-spb-fsdp-layer-wrap-v1:${source_dir}/experiments/records/20260908-g2-planner-handoff-v1:${source_dir}/experiments/records/20260906-integrated-fixes-v1:${python_overlay}:${deps}/flash-linear-attention_dev:${deps}/MoBA
export PYTHONPATH=${source_dir}:${EXTRA_PYTHONPATH}
export CPATH=${include_root}/python3.12:${include_root}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MODEL_PATH=${MODEL_PATH:-${storage_root}/yangjinyue/sc2-verl-agent-experiments/assets/models/Panyuqi/SpikingBrain-2.0-instruct}
export SC2PATH=${SC2PATH:-${storage_root}/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII}
export SPB_DISABLE_SWA_DROPOUT=1
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air SC2_ENV_SEED=0 SC2_MAX_STEPS=0 SC2_ROLLOUT_N=4
export SC2_FIXED_BO_PREFIX_LIMIT=-1 SC2_STEP_SHAPING_MODE=off BO_PROFILE=canonical EVAL_ONLY=1
export SC2_ORIGINAL_BO_SEGMENT=1 SC2_CLARIFY_BO_QUOTA=1
export TOTAL_EPOCHS=1 SAVE_FREQ=-1 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=192
export SC2_PLAN_SEGMENT_MAX=6 SC2_PLAN_BULLET_MAX=6
export PPO_DIAGNOSTICS=0 ROLLOUT_LOGPROB_DIAGNOSTICS=0
export SC2_SEED_PYTHON_RANDOM=1 SC2_RESET_RETRIES=1
unset SC2_GAME_TIME_LIMIT SC2_RNG_PREFIX_TARGET
replicate=${FEEDBACK_REPLICATE:-r1}
export RUN_ID=${RUN_ID:-20260911-execution-failure-feedback-v1-spb2-l7s24-l6-g2-${1}-eval4-s0-${replicate}}
export OUTPUT_DIR=${OUTPUT_DIR:-${storage_root}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/sc2_feedback_${1}_0911_${replicate}}
# Keep new temporary/cache writes off node47's nearly full system disk.
export TMPDIR=${RAY_TMPDIR} TRITON_CACHE_DIR=${RAY_TMPDIR}/triton HF_HOME=${RAY_TMPDIR}/hf
exec bash "${source_dir}/scripts/train_sc2.sh" \
    +ray_init.runtime_env.worker_process_setup_hook="${eval_hook}" \
    "+ray_init.runtime_env.env_vars.SC2_SEED_PYTHON_RANDOM='1'" \
    "+ray_init.runtime_env.env_vars.SC2_RESET_RETRIES='1'" \
    "+ray_init.runtime_env.env_vars.SC2_PRESERVE_VALID_BO='1'" \
    "+ray_init.runtime_env.env_vars.SC2_ENABLE_AIR_OPTIONS='1'" \
    "++ray_init.runtime_env.env_vars.SC2_ROLLOUT_N='${eval_games}'" \
    "+ray_init.runtime_env.env_vars.SC2_FEEDBACK_LEARNING_MODEL='${FEEDBACK_LEARNING_MODEL:-}'" \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp32 \
    +actor_rollout_ref.actor.fp32_normalization=True \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=fp32 \
    +actor_rollout_ref.ref.fp32_normalization=True \
    ++actor_rollout_ref.rollout.fsdp_layer_wrap=True \
    ++env.prompt_response_log_max_prompt_chars=0 \
    trainer.total_training_steps=96 \
    trainer.n_gpus_per_node=4 actor_rollout_ref.actor.ppo_mini_batch_size=8 env.rollout.n="${eval_games}" \
    +env.val_rollout_n="${eval_games}" actor_rollout_ref.rollout.val_kwargs.n="${eval_games}" \
    "${resume_args[@]}" \
    trainer.del_local_ckpt_after_load=False \
    actor_rollout_ref.rollout.temperature="${eval_temperature}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${eval_temperature}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.top_k=0 \
    ++ray_init.num_cpus=32 \
    ${scene_args[@]+"${scene_args[@]}"} \
    ${temperature_args[@]+"${temperature_args[@]}"}
