#!/usr/bin/env bash
# One eight-GPU evaluation, A/B interleaved, at most45 minutes including exit.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 N_GPUS_PER_NODE=8 PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
storage=/mnt/data0601
python_root=/root/new-scii-spb2-level5-control/python
include_root=/usr/include
if [[ "${SC2_COMMAND_TRACE_VALIDATION:-0}" == 1 ]]; then
    storage=/root/new-scii-execution-feedback-local-20260911/shared
    python_root=/root/new-scii-spb-level5-96-20260908/python
    include_root=/root/new-scii-spb-level5-96-20260908/include
fi
run_root=${storage}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    if [[ "${SC2_COMMAND_TRACE_VALIDATION:-0}" == 1 ]]; then
        test "${SC2_COMMAND_TRACE_AUTHORIZED:-0}" = 1
    else
        test "${SC2_SUPPLY_ORDER_AUTHORIZED:-0}" = 1
    fi
    test -z "$(git -C "${source_dir}" status --porcelain)"
    test -z "$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader)"
    control=${CONTROL_DIR:?set unused control directory}
    mkdir "${control}"
    exec > >(tee "${control}/console.log") 2>&1
    trap 'status=$?; printf "EXIT_CODE=%s\n" "$status" > "${control}/.done"; date -u +%FT%TZ > "${control}/ended_at.txt"' EXIT
    date -u +%FT%TZ | tee "${control}/started_at.txt"
    timeout 5s stat "${run_root}"
    (cd "${source_dir}" && sha256sum -c configs/system-v1.sha256)
fi
assets=${storage}/yangjinyue/sc2-verl-agent-experiments/assets
deps=${assets}/repro/stageguard_v7_legacy_contract_20260806/source/extracted
export CONFIG_PATH=${source_dir}/configs/spikingbrain2-level5.env
export EXTRA_PYTHONPATH=${record_dir}:${source_dir}/experiments/records/20260910-spb-fsdp-layer-wrap-v1:${source_dir}/experiments/records/20260906-integrated-fixes-v1:${python_root}:${deps}/flash-linear-attention_dev:${deps}/MoBA
export PYTHONPATH=${source_dir}:${EXTRA_PYTHONPATH}
export CPATH=${include_root}/python3.12:${include_root} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MODEL_PATH=${MODEL_PATH:-${assets}/models/Panyuqi/SpikingBrain-2.0-instruct}
export SC2PATH=${SC2PATH:-${assets}/sc2/StarCraftII}
export SPB_DISABLE_SWA_DROPOUT=1 SC2_EXECUTION_FAILURE_FEEDBACK=1
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air SC2_ENV_SEED=0 SC2_MAX_STEPS=0 SC2_ROLLOUT_N=8
export SC2_FIXED_BO_PREFIX_LIMIT=-1 SC2_STEP_SHAPING_MODE=off BO_PROFILE=canonical EVAL_ONLY=1
export SC2_ORIGINAL_BO_SEGMENT=1 SC2_CLARIFY_BO_QUOTA=1
export TOTAL_EPOCHS=1 SAVE_FREQ=-1 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=192
export SC2_PLAN_SEGMENT_MAX=6 SC2_PLAN_BULLET_MAX=6
export PPO_DIAGNOSTICS=0 ROLLOUT_LOGPROB_DIAGNOSTICS=0 SC2_SEED_PYTHON_RANDOM=1 SC2_RESET_RETRIES=1
unset SC2_GAME_TIME_LIMIT SC2_RNG_PREFIX_TARGET
export RUN_ID=${RUN_ID:-20260911-execution-failure-feedback-v1-spb2-original-l6-g2-supply-order-ab8-8gpu-s52620-r1}
export OUTPUT_DIR=${OUTPUT_DIR:-${run_root}/${RUN_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/sc2_supply_order_0911_r1}
export TMPDIR=${RAY_TMPDIR} TRITON_CACHE_DIR=${RAY_TMPDIR}/triton HF_HOME=${RAY_TMPDIR}/hf
trace_args=()
if [[ "${SC2_COMMAND_TRACE_VALIDATION:-0}" == 1 ]]; then
    export SC2_COMMAND_TRACE_DIR=${OUTPUT_DIR}
    trace_args=(+ray_init.runtime_env.env_vars.SC2_COMMAND_TRACE_DIR="${SC2_COMMAND_TRACE_DIR}")
fi
runner=(timeout --signal=INT --kill-after=30s 2640 bash)
if [[ "${DRY_RUN:-0}" == 1 ]]; then runner=(bash); fi
"${runner[@]}" "${source_dir}/scripts/train_sc2.sh" \
    +ray_init.runtime_env.worker_process_setup_hook=${SC2_SUPPLY_EVAL_HOOK:-supply_order_eval.install} \
    "+ray_init.runtime_env.env_vars.SC2_SEED_PYTHON_RANDOM='1'" \
    "+ray_init.runtime_env.env_vars.SC2_RESET_RETRIES='1'" \
    "+ray_init.runtime_env.env_vars.SC2_PRESERVE_VALID_BO='1'" \
    "+ray_init.runtime_env.env_vars.SC2_ENABLE_AIR_OPTIONS='1'" \
    "++ray_init.runtime_env.env_vars.SC2_ROLLOUT_N='8'" \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp32 \
    +actor_rollout_ref.actor.fp32_normalization=True \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=fp32 \
    +actor_rollout_ref.ref.fp32_normalization=True \
    ++actor_rollout_ref.rollout.fsdp_layer_wrap=True ++env.prompt_response_log_max_prompt_chars=0 \
    trainer.total_training_steps=96 trainer.resume_mode=disable \
    trainer.n_gpus_per_node=8 actor_rollout_ref.actor.ppo_mini_batch_size=8 env.rollout.n=8 \
    +env.val_rollout_n=8 actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.temperature=0.4 actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 actor_rollout_ref.rollout.val_kwargs.top_k=0 \
    ++ray_init.num_cpus=64 \
    ${trace_args[@]+"${trace_args[@]}"}
