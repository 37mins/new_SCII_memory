#!/usr/bin/env bash
set -euo pipefail

BO_KEY=${BO_KEY:?BO_KEY must be g1, g2, g3, g4, or g5}
RUN_ID=${RUN_ID:?RUN_ID is required}
CODE_ROOT=${CODE_ROOT:?CODE_ROOT is required}
RESULT_ROOT=${RESULT_ROOT:?RESULT_ROOT is required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
SC2PATH=${SC2PATH:?SC2PATH is required}
ENV_SEED=${ENV_SEED:?ENV_SEED is required}
DIFFICULTY_NAME=${DIFFICULTY_NAME:?DIFFICULTY_NAME is required}
COMPAT_PATH=${COMPAT_PATH:?COMPAT_PATH is required}
SOURCE_COMMIT=${SOURCE_COMMIT:?SOURCE_COMMIT is required}
DEPLOYMENT_TAR_SHA=${DEPLOYMENT_TAR_SHA:?DEPLOYMENT_TAR_SHA is required}
ENABLE_NEWEXEC_MICRO=${ENABLE_NEWEXEC_MICRO:-1}

case "${ENABLE_NEWEXEC_MICRO}" in
  0|1) ;;
  *) echo "ENABLE_NEWEXEC_MICRO must be 0 or 1" >&2; exit 2 ;;
esac

case "${BO_KEY}" in
  g1) BO_SHA=5e83321f7f0859bae9848c3787c0a1b29dbf21237ab970c784627d51167163a0 ;;
  g2) BO_SHA=304b1f4ff12fe30c5e6d5f7b8eaef8ed6823ca65afed3a3706cae10e3639c494 ;;
  g3) BO_SHA=b3ba854f5083a9dc9aa43bc9a025bfb3c66a9518c634e31bc74500a21009569c ;;
  g4) BO_SHA=22058ecc1026284ab76630a611d4dac01fe787cb625578189437455fc39cb0a8 ;;
  g5) BO_SHA=e964de6e3ba92b809cab978b78e04067ab63eb1da7d8a6e8518fe7a6d2040e5a ;;
  *) echo "unsupported BO_KEY=${BO_KEY}" >&2; exit 2 ;;
esac

BO_PATH=${BO_PATH:?BO_PATH is required}
RESULT_DIR=${RESULT_ROOT}/${RUN_ID}-${BO_KEY}-s${ENV_SEED}
RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/ray_${RUN_ID}_${BO_KEY}}
SCRIPT_PLAYER=${CODE_ROOT}/agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/players/script_player.py
EXPECTED_SCRIPT_PLAYER_SHA=76354ff908b47f8c400e2cf279a3af5f49e98a3da6afb53018143b579f28c6da

test ! -e "${RESULT_DIR}"
mkdir -p "${RESULT_DIR}" "${RAY_TMPDIR}"
cp "${COMPAT_PATH}" "${RESULT_DIR}/compat.txt"
printf 'SOURCE_COMMIT=%s\nDEPLOYMENT_TAR_SHA=%s\n' \
  "${SOURCE_COMMIT}" "${DEPLOYMENT_TAR_SHA}" \
  > "${RESULT_DIR}/deployment.env"
test "$(sha256sum "${BO_PATH}" | awk '{print $1}')" = "${BO_SHA}"
test "$(sha256sum "${SCRIPT_PLAYER}" | awk '{print $1}')" = "${EXPECTED_SCRIPT_PLAYER_SHA}"
test -s "${MODEL_PATH}/config.json"
test -s "${SC2PATH}/maps/Melee/Flat128.SC2Map"

cat > "${RESULT_DIR}/effective.env" <<EOF
RUN_ID=${RUN_ID}
BO_KEY=${BO_KEY}
BO_SHA=${BO_SHA}
ENV_SEED=${ENV_SEED}
MODEL_PATH=${MODEL_PATH}
MODEL_PROJECT_SFT=0
MODEL_PRIOR_PROJECT_RL=0
DIFFICULTY=${DIFFICULTY_NAME}
AI_BUILD=Air
MAP=Flat128
TRAINER_VAL_ONLY=true
TRAINING_UPDATES=0
ROLLOUTS=1
SC2_ENABLE_NEWEXEC_MICRO=${ENABLE_NEWEXEC_MICRO}
SC2_ATTACK_TARGET_UNITS_FIRST=0
SC2_STRATEGY_PROFILE=marine_pressure
SC2_AUTO_TRAIN_MARINES=0
SC2_MAX_FORCED_ATTACKS=0
SC2_PLANNER_PROTOCOL=legacy
SC2_PLAN_SEGMENT_MAX=6
SC2_PLAN_BULLET_MAX=6
SCRIPT_PLAYER_SHA=${EXPECTED_SCRIPT_PLAYER_SHA}
EOF

export PYTHONPATH="${CODE_ROOT}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export VERL_ATTN_IMPLEMENTATION=eager
export RAY_USAGE_STATS_ENABLED=0
export WANDB_MODE=disabled
export SC2PATH
export SC2_SOCKET_TIMEOUT=900
export SC2_ACCEPT_TIMEOUT=60
export SC2_RESET_RETRIES=5
export SC2_RESET_ATTEMPT_TIMEOUT=900
export SC2_RESET_CONCURRENCY=3
export SC2_RESET_LOCK_DIR=${RESULT_ROOT}/reset_locks
export SC2_LOCAL_EXECUTOR_FALLBACK=1
export SC2_FAIL_FAST_ON_ENV_ERROR=1
export SC2_PRE_ACTION_TERMINAL_DRAIN_TIMEOUT=0.5
export SC2_TERMINAL_ACK_TIMEOUT=900.0
export SC2_DIFFICULTY=${DIFFICULTY_NAME}
export SC2_EXPECTED_DIFFICULTY=${DIFFICULTY_NAME}
export SC2_AI_BUILD=Air
export SC2_ENV_SEED=${ENV_SEED}
export SC2_MAX_STEPS=20
export SC2_ROLLOUT_N=1
export SC2_BO_ADDRESS=${BO_PATH}
export SC2_GAME_TIME_LIMIT=900
export SC2_STRATEGY_PROFILE=marine_pressure
export SC2_AUTO_TRAIN_MARINES=0
export SC2_MAX_FORCED_ATTACKS=0
export SC2_LLM_ATTACK_MIN_UNITS=5
export SC2_FIRST_LLM_ATTACK_MIN_UNITS=5
export SC2_REINFORCEMENT_MIN_UNITS=5
export SC2_REINFORCEMENT_TIMER=88
export SC2_ATTACK_ORDER_POSITION=after_build
export SC2_ATTACK_OPTION_ACCOUNT_ISSUED=1
export SC2_PLAN_SEGMENT_MAX=6
export SC2_PLAN_BULLET_MAX=6
export SC2_PLANNER_PROTOCOL=legacy
export SC2_ENABLE_NEWEXEC_MICRO=${ENABLE_NEWEXEC_MICRO}
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
  SC2_MAX_STEPS SC2_ROLLOUT_N SC2_BO_ADDRESS SC2_GAME_TIME_LIMIT
  SC2_STRATEGY_PROFILE SC2_AUTO_TRAIN_MARINES SC2_MAX_FORCED_ATTACKS
  SC2_LLM_ATTACK_MIN_UNITS SC2_FIRST_LLM_ATTACK_MIN_UNITS
  SC2_REINFORCEMENT_MIN_UNITS SC2_REINFORCEMENT_TIMER
  SC2_ATTACK_ORDER_POSITION SC2_ATTACK_OPTION_ACCOUNT_ISSUED
  SC2_PLAN_SEGMENT_MAX SC2_PLAN_BULLET_MAX SC2_PLANNER_PROTOCOL
  SC2_ENABLE_NEWEXEC_MICRO SC2_NEWEXEC_BASIC_HOOLIGAN
  SC2_NEWEXEC_HOOLIGAN_BIO SC2_NEWEXEC_HOOLIGAN_ALLY_POWER_RADIUS
  SC2_NEWEXEC_HOOLIGAN_RETREAT SC2_NEWEXEC_REQUIRE_TECH_UNIT
  SC2_NEWEXEC_DEFENSE_MICRO SC2_NEWEXEC_SPECIAL_ABILITIES
  SC2_NEWEXEC_FIGHT_MICRO SC2_ATTACK_TARGET_UNITS_FIRST
  SC2_ATTACK_ENGAGED_RETARGET_IDLE_ONLY SC2_HOOLIGAN_STRUCTURE_FOCUS
  SC2_LOCAL_EXECUTOR_FALLBACK SC2_FAIL_FAST_ON_ENV_ERROR SC2_ACCEPT_TIMEOUT
  SC2_PRE_ACTION_TERMINAL_DRAIN_TIMEOUT SC2_TERMINAL_ACK_TIMEOUT
)
ray_args=()
for name in "${runtime_env[@]}"; do
  ray_args+=("+ray_init.runtime_env.env_vars.${name}='${!name}'")
done

cd "${CODE_ROOT}"
set +e
timeout 7200s python -m verl.trainer.main_ppo \
  +ray_init.address=local \
  +ray_init.include_dashboard=False \
  +ray_init._temp_dir="${RAY_TMPDIR}" \
  +ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}" \
  +ray_init.runtime_env.env_vars.TMPDIR="${RAY_TMPDIR}" \
  +ray_init.runtime_env.env_vars.RAY_TMPDIR="${RAY_TMPDIR}" \
  "${ray_args[@]}" \
  algorithm.adv_estimator=gigpo \
  data.train_files="${CODE_ROOT}/data/train.parquet" \
  data.val_files="${CODE_ROOT}/data/test.parquet" \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=4096 \
  data.max_response_length=96 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.trust_remote_code=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  +actor_rollout_ref.model.attn_implementation=eager \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.multi_turn.enable=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.temperature=0.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  env.env_name=SC2 \
  env.seed="${ENV_SEED}" \
  env.max_steps=20 \
  env.rollout.n=1 \
  env.history_length=0 \
  +env.sc2_memory_mode=off \
  +env.forbid_thinking_output=True \
  +env.strict_legacy_plan_contract=True \
  +env.enable_prompt_response_log=True \
  +env.prompt_response_log_path="${RESULT_DIR}/prompt_response.jsonl" \
  +env.prompt_response_log_max_prompt_chars=3000 \
  +env.prompt_response_log_train_only=False \
  env.resources_per_worker.num_cpus=0.1 \
  trainer.logger="['console']" \
  trainer.project_name=sc2-qwen-eval \
  trainer.experiment_name="${RUN_ID}-${BO_KEY}" \
  trainer.default_local_dir="${RESULT_DIR}/checkpoints" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.resume_mode=disable \
  > "${RESULT_DIR}/console.log" 2>&1
process_status=$?
set -e

replay=$(find "${CODE_ROOT}/logs" -type f -name replay.SC2Replay -print 2>/dev/null | head -1 || true)
if [[ -n "${replay}" ]]; then
  cp "${replay}" "${RESULT_DIR}/${RUN_ID}-${BO_KEY}-s${ENV_SEED}.SC2Replay"
  status=${process_status}
else
  status=20
fi

printf 'STATUS=%s\nPROCESS_STATUS=%s\nREPLAY=%s\n' \
  "${status}" "${process_status}" "${replay}" > "${RESULT_DIR}/done"
exit "${status}"
