#!/usr/bin/env bash
# Single-game acceptance of the standalone no-policy evaluation finish.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=/root/new-scii-garrison-wave-20260908/code-e1e25ce
test "$(git -C "${source_dir}" rev-parse HEAD)" = e1e25ce1d14362b209e34ec73871ebaeb63538a6
test -z "$(git -C "${source_dir}" status --porcelain)"
gpu=1
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  test -z "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader)"
fi
export CUDA_VISIBLE_DEVICES=${gpu} PYTHON_BIN=/usr/bin/python3
export CONFIG_PATH=${source_dir}/configs/spikingbrain2-level5.env
deps=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted
export EXTRA_PYTHONPATH=${source_dir}/experiments/records/20260906-integrated-fixes-v1:/root/new-scii-spb2-level5-control/python:${deps}/flash-linear-attention_dev:${deps}/MoBA
export CPATH=/usr/include/python3.12
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MODEL_PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/models/Panyuqi/SpikingBrain-2.0-instruct
export SPB_DISABLE_SWA_DROPOUT=1
export SC2PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air SC2_ENV_SEED=0 SC2_MAX_STEPS=0
export SC2_STEP_SHAPING_MODE=off BO_PROFILE=canonical EVAL_ONLY=1
export SC2_PLAN_SEGMENT_MAX=6 SC2_PLAN_BULLET_MAX=6 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=96
export PPO_DIAGNOSTICS=0 ROLLOUT_LOGPROB_DIAGNOSTICS=0
export SC2_SEED_PYTHON_RANDOM=1 SC2_RESET_RETRIES=1
unset SC2_GAME_TIME_LIMIT SC2_RNG_PREFIX_TARGET
export RUN_ID=${CANDIDATE_RUN_ID:-20260908-garrison-wave-ownership-v1-spb2-l6-g2-s2621640384-evalfinish-r2}
run_root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260908-garrison-wave-ownership-v1/runs
export OUTPUT_DIR=${CANDIDATE_OUTPUT_DIR:-${run_root}/${RUN_ID}}
export RAY_TMPDIR=/dev/shm/sc2_garrison_evalfinish_20260908_r2
set +e
timeout --signal=TERM --kill-after=30s 1800 bash "${source_dir}/scripts/train_sc2.sh" \
  +ray_init.runtime_env.worker_process_setup_hook=r4_pair_seed.install \
  "+ray_init.runtime_env.env_vars.SC2_SEED_PYTHON_RANDOM='1'" \
  "+ray_init.runtime_env.env_vars.SC2_RESET_RETRIES='1'" \
  "+ray_init.runtime_env.env_vars.SC2_PRESERVE_VALID_BO='1'" \
  "+ray_init.runtime_env.env_vars.SC2_ENABLE_AIR_OPTIONS='1'" \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp32 \
  +actor_rollout_ref.actor.fp32_normalization=True \
  +actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 \
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=fp32 \
  +actor_rollout_ref.ref.fp32_normalization=True \
  ++env.prompt_response_log_max_prompt_chars=0
status=$?
set -e
if [[ -d "${OUTPUT_DIR}" ]]; then
  mkdir "${OUTPUT_DIR}/candidate_orchestration"
  cp "${BASH_SOURCE[0]}" "${record_dir}/evaluation-only-finish.md" \
    "${source_dir}/experiments/records/20260906-integrated-fixes-v1/r4_pair_seed.py" \
    "${OUTPUT_DIR}/candidate_orchestration/"
  git -C "${source_dir}" rev-parse HEAD > "${OUTPUT_DIR}/candidate_orchestration/source_sha.txt"
  printf 'EXIT_CODE=%s\n' "${status}" > "${OUTPUT_DIR}/candidate_orchestration/.done"
fi
exit "${status}"
