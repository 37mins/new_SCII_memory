#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_dir}"
config_path=${CONFIG_PATH:-${repo_dir}/configs/qwen3-4b-instruct-2507-level5.env}
source "${config_path}"
export SC2_STEP_SHAPING_MODE
case "${SC2_STEP_SHAPING_MODE}" in
    legacy|off) ;;
    *) echo "SC2_STEP_SHAPING_MODE must be legacy or off" >&2; exit 2 ;;
esac
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the registered original model snapshot}
SC2PATH=${SC2PATH:?set SC2PATH to the StarCraftII directory}
MODEL_PATH=$(cd "${MODEL_PATH}" && pwd)
SC2PATH=$(cd "${SC2PATH}" && pwd)
model_source=${MODEL_PATH}
RUN_ID=${RUN_ID:-${MODEL_KIND}-level5-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_DIR=${OUTPUT_DIR:-${repo_dir}/runs/${RUN_ID}}
DRY_RUN=${DRY_RUN:-0}
VERIFY_ASSETS=${VERIFY_ASSETS:-1}
EVAL_ONLY=${EVAL_ONLY:-0}
if [[ "${EVAL_ONLY}" == 1 ]]; then SC2_ROLLOUT_N=1; fi

if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "RUN_ID must contain only letters, digits, dots, underscores or hyphens" >&2
    exit 2
fi
if [[ "${DRY_RUN}" != 1 && -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    echo "Refusing to train from a dirty source tree: ${repo_dir}" >&2
    exit 2
fi
if [[ "${VERIFY_ASSETS}" != 1 && "${DRY_RUN}" != 1 ]]; then
    echo "Asset verification may be skipped only for DRY_RUN=1" >&2
    exit 2
fi
case "${BO_PROFILE}" in
    canonical) bo_path="${repo_dir}/agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/knowledge/BO/replay_observations_Flat128_d4_TvT_Macro_g2_Result.Victory_twobase.json" ;;
    rush) bo_path="${repo_dir}/data/rush_bo.json" ;;
    fivebo)
        bo_path=$("${PYTHON_BIN}" -c 'import json,sys; from pathlib import Path; print(Path(sys.argv[2]) / json.load(open(sys.argv[1]))[sys.argv[3]][0])' \
            "${repo_dir}/configs/five-bo.json" "${FIVEBO_ROOT:?set FIVEBO_ROOT}" "${BO_KEY:?set BO_KEY to g1..g5}")
        ;;
    *) echo "Unknown BO_PROFILE: ${BO_PROFILE}" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "${OUTPUT_DIR}")"
# Atomic creation refuses existing runs, including empty directories.
mkdir "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)
export SC2_EPISODE_OUTCOME_LOG="${OUTPUT_DIR}/episode_outcomes.jsonl"
mkdir "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/logs"

finish_run() {
    status=$?
    trap - EXIT
    "${PYTHON_BIN}" "${repo_dir}/scripts/run_record.py" finish --output "${OUTPUT_DIR}" --exit-code "${status}" || {
        if [[ "${status}" == 0 ]]; then status=1; fi
    }
    exit "${status}"
}
trap finish_run EXIT

if [[ "${VERIFY_ASSETS}" == 1 ]]; then
    bo_args=()
    if [[ "${BO_PROFILE}" == fivebo ]]; then bo_args=(--bo-key "${BO_KEY}" --bo "${bo_path}"); fi
    "${PYTHON_BIN}" "${repo_dir}/scripts/verify_assets.py" \
        --model "${MODEL_PATH}" --model-manifest "${repo_dir}/${MODEL_MANIFEST}" \
        --sc2 "${SC2PATH}" --bo-profile "${BO_PROFILE}" "${bo_args[@]}" 2>&1 | tee "${OUTPUT_DIR}/preflight.log"
fi
if [[ "${MODEL_KIND}" == spb ]]; then
    MODEL_PATH="${OUTPUT_DIR}/model"
    if [[ "${DRY_RUN}" != 1 ]]; then
        "${PYTHON_BIN}" "${repo_dir}/scripts/prepare_spb_model.py" \
            --source "${model_source}" --destination "${MODEL_PATH}" 2>&1 | tee -a "${OUTPUT_DIR}/preflight.log"
    fi
fi
ray_tmpdir=${RAY_TMPDIR:-/dev/shm/sc2_${RUN_ID}}
if [[ "${DRY_RUN}" != 1 ]]; then mkdir "${ray_tmpdir}"; fi

export PYTHONPATH="${repo_dir}${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export VERL_ATTN_IMPLEMENTATION=eager
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export RAY_USAGE_STATS_ENABLED=0
export WANDB_MODE=${WANDB_MODE:-disabled}
export SC2PATH
export SC2_SOCKET_TIMEOUT=900
export SC2_ACCEPT_TIMEOUT=60
export SC2_RESET_RETRIES=5
export SC2_RESET_ATTEMPT_TIMEOUT=900
export SC2_RESET_CONCURRENCY=8
export SC2_LOCAL_EXECUTOR_FALLBACK=1
export SC2_FAIL_FAST_ON_ENV_ERROR=1
export SC2_PRE_ACTION_TERMINAL_DRAIN_TIMEOUT=0.5
export SC2_TERMINAL_ACK_TIMEOUT=900.0
export SC2_DIFFICULTY
export SC2_EXPECTED_DIFFICULTY=${SC2_DIFFICULTY}
export SC2_AI_BUILD
export SC2_ENV_SEED
export SC2_MAX_STEPS
export SC2_ROLLOUT_N
export SC2_RUNTIME_LOG_ROOT="${OUTPUT_DIR}/sc2-logs"
export SC2_BO_ADDRESS=${bo_path}
export SC2_FIXED_BO_PREFIX_LIMIT=${SC2_FIXED_BO_PREFIX_LIMIT:-0}
export SC2_STRATEGY_PROFILE=marine_pressure
export SC2_AUTO_TRAIN_MARINES=0
export SC2_MAX_FORCED_ATTACKS=0
export SC2_LLM_ATTACK_MIN_UNITS=5
export SC2_FIRST_LLM_ATTACK_MIN_UNITS=5
export SC2_REINFORCEMENT_MIN_UNITS=5
export SC2_REINFORCEMENT_TIMER=88
export SC2_ATTACK_ORDER_POSITION=after_build
export SC2_ATTACK_OPTION_ACCOUNT_ISSUED=1
export SC2_PLAN_SEGMENT_MAX
export SC2_PLAN_BULLET_MAX
export SC2_ORIGINAL_BO_SEGMENT=${SC2_ORIGINAL_BO_SEGMENT:-0}
export SC2_CLARIFY_BO_QUOTA=${SC2_CLARIFY_BO_QUOTA:-0}
export SC2_EXECUTION_FAILURE_FEEDBACK=${SC2_EXECUTION_FAILURE_FEEDBACK:-0}
export SC2_PLANNER_PROTOCOL=legacy
export SC2_ENABLE_NEWEXEC_MICRO=0
export SC2_NEWEXEC_BASIC_HOOLIGAN=1
export SC2_NEWEXEC_HOOLIGAN_BIO=1
export SC2_NEWEXEC_HOOLIGAN_ALLY_POWER_RADIUS=8
export SC2_NEWEXEC_HOOLIGAN_RETREAT=0
export SC2_NEWEXEC_REQUIRE_TECH_UNIT=0
export SC2_NEWEXEC_DEFENSE_MICRO=0
export SC2_NEWEXEC_SPECIAL_ABILITIES=1
export SC2_NEWEXEC_FIGHT_MICRO=1
export SC2_ATTACK_TARGET_UNITS_FIRST=0
export SC2_ATTACK_ENGAGED_RETARGET_IDLE_ONLY=1
export SC2_HOOLIGAN_STRUCTURE_FOCUS=guarded

runtime_env=(
    SC2PATH SC2_DIFFICULTY SC2_EXPECTED_DIFFICULTY SC2_AI_BUILD SC2_ENV_SEED
    SC2_MAX_STEPS SC2_ROLLOUT_N SC2_RUNTIME_LOG_ROOT SC2_BO_ADDRESS SC2_STRATEGY_PROFILE
    SC2_FIXED_BO_PREFIX_LIMIT SC2_EXECUTION_FAILURE_FEEDBACK
    SC2_EPISODE_OUTCOME_LOG SC2_STEP_SHAPING_MODE
    SC2_AUTO_TRAIN_MARINES SC2_MAX_FORCED_ATTACKS SC2_LLM_ATTACK_MIN_UNITS
    SC2_FIRST_LLM_ATTACK_MIN_UNITS SC2_REINFORCEMENT_MIN_UNITS
    SC2_REINFORCEMENT_TIMER SC2_ATTACK_ORDER_POSITION
    SC2_ATTACK_OPTION_ACCOUNT_ISSUED SC2_PLAN_SEGMENT_MAX SC2_PLAN_BULLET_MAX
    SC2_PLANNER_PROTOCOL SC2_ORIGINAL_BO_SEGMENT SC2_CLARIFY_BO_QUOTA SC2_ENABLE_NEWEXEC_MICRO SC2_NEWEXEC_BASIC_HOOLIGAN
    SC2_NEWEXEC_HOOLIGAN_BIO SC2_NEWEXEC_HOOLIGAN_ALLY_POWER_RADIUS
    SC2_NEWEXEC_HOOLIGAN_RETREAT SC2_NEWEXEC_REQUIRE_TECH_UNIT
    SC2_NEWEXEC_DEFENSE_MICRO SC2_NEWEXEC_SPECIAL_ABILITIES
    SC2_NEWEXEC_FIGHT_MICRO SC2_ATTACK_TARGET_UNITS_FIRST
    SC2_ATTACK_ENGAGED_RETARGET_IDLE_ONLY SC2_HOOLIGAN_STRUCTURE_FOCUS
    SC2_FAIL_FAST_ON_ENV_ERROR SC2_ACCEPT_TIMEOUT
    SC2_PRE_ACTION_TERMINAL_DRAIN_TIMEOUT SC2_TERMINAL_ACK_TIMEOUT
)
ray_args=()
for name in "${runtime_env[@]}"; do
    ray_args+=("+ray_init.runtime_env.env_vars.${name}='${!name}'")
done

command=("${PYTHON_BIN}" -m verl.trainer.main_ppo \
    +ray_init.address=local \
    +ray_init.include_dashboard=False \
    +ray_init._temp_dir="${ray_tmpdir}" \
    +ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}" \
    +ray_init.runtime_env.env_vars.TMPDIR="${ray_tmpdir}" \
    +ray_init.runtime_env.env_vars.RAY_TMPDIR="${ray_tmpdir}" \
    +ray_init.runtime_env.env_vars.SC2_TRAIN_SEEDS_LOG_PATH="${OUTPUT_DIR}/training_seeds.jsonl" \
    "${ray_args[@]}" \
    algorithm.adv_estimator=gigpo \
    data.train_files="${repo_dir}/data/train.parquet" \
    data.val_files="${repo_dir}/data/test.parquet" \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.attn_implementation=eager \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.temperature=0.4 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.top_k=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    algorithm.filter_groups.enable=False \
    algorithm.filter_groups.max_num_gen_batches=1 \
    env.env_name=SC2 \
    env.seed="${SC2_ENV_SEED}" \
    env.max_steps="${SC2_MAX_STEPS}" \
    env.rollout.n="${SC2_ROLLOUT_N}" \
    env.history_length=0 \
    +env.sc2_memory_mode=off \
    +env.forbid_thinking_output=True \
    +env.strict_legacy_plan_contract=True \
    +env.enable_prompt_response_log=True \
    +env.prompt_response_log_path="${OUTPUT_DIR}/prompt_response.jsonl" \
    +env.prompt_response_log_max_prompt_chars=0 \
    +env.prompt_response_log_train_only=True \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=sc2-linear-attention \
    trainer.experiment_name="${RUN_ID}" \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq=-1 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.resume_mode=disable \
    hydra.run.dir="${OUTPUT_DIR}/hydra" \
    hydra.output_subdir=null
)

if [[ "${MODEL_KIND}" == spb && "${SPB_DISABLE_SWA_DROPOUT:-0}" == 1 ]]; then
    command+=(+actor_rollout_ref.model.override_config.swa_dropout=0.0)
fi
if [[ "${PPO_DIAGNOSTICS:-0}" == 1 ]]; then
    command+=(+actor_rollout_ref.actor.ppo_diagnostics_dir="${OUTPUT_DIR}/ppo_diagnostics")
fi
if [[ "${ROLLOUT_LOGPROB_DIAGNOSTICS:-0}" == 1 ]]; then
    command+=(+actor_rollout_ref.rollout.capture_log_probs=True)
fi

if [[ "${EVAL_ONLY}" == 1 ]]; then
    # Reuse the frozen training path; only change evaluation/budget controls.
    for i in "${!command[@]}"; do
        case "${command[$i]}" in
            actor_rollout_ref.actor.ppo_mini_batch_size=*) command[$i]=actor_rollout_ref.actor.ppo_mini_batch_size=1 ;;
            trainer.n_gpus_per_node=*) command[$i]=trainer.n_gpus_per_node=1 ;;
            trainer.save_freq=*) command[$i]=trainer.save_freq=-1 ;;
            trainer.total_epochs=*) command[$i]=trainer.total_epochs=1 ;;
            trainer.val_before_train=*) command[$i]=trainer.val_before_train=True ;;
            trainer.val_only=*) command[$i]=trainer.val_only=True ;;
            +env.prompt_response_log_train_only=*) command[$i]=+env.prompt_response_log_train_only=False ;;
            +env.prompt_response_log_max_prompt_chars=*) command[$i]=+env.prompt_response_log_max_prompt_chars=20000 ;;
            actor_rollout_ref.rollout.val_kwargs.temperature=*) command[$i]=actor_rollout_ref.rollout.val_kwargs.temperature=0.0 ;;
        esac
    done
    command+=(actor_rollout_ref.rollout.val_kwargs.do_sample=False
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0
        +ray_init.object_store_memory=1073741824
        +ray_init.num_cpus=8
        +ray_init.runtime_env.env_vars.SC2_EVAL_SEEDS_LOG_PATH="${OUTPUT_DIR}/evaluation_seeds.jsonl")
fi

# Explicit experiment overrides use Hydra's existing last-value precedence.
# The exact command (including these overrides) is saved in effective_config.json.
command+=("$@")

record_args=()
if [[ "${DRY_RUN}" == 1 ]]; then record_args+=(--dry-run); fi
"${PYTHON_BIN}" "${repo_dir}/scripts/run_record.py" start \
    --output "${OUTPUT_DIR}" --config "${config_path}" \
    --bo "${bo_path}" --bo-profile "${BO_PROFILE}" \
    --model-source "${model_source}" --model-manifest "${repo_dir}/${MODEL_MANIFEST}" \
    --model-id "${MODEL_ID}" --model-revision "${MODEL_REVISION}" \
    "${record_args[@]}" -- "${command[@]}"
if [[ "${DRY_RUN}" == 1 ]]; then
    trap - EXIT
    echo "Dry run recorded in ${OUTPUT_DIR}; no model, GPU or SC2 process started."
    exit 0
fi
"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/console.log"
