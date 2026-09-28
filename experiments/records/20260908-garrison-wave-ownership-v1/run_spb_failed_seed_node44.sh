#!/usr/bin/env bash
# Original SpB at historical failed seed43567: eight games, no PPO.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=/root/new-scii-garrison-wave-20260908/code-e1e25ce
test "$(git -C "${source_dir}" rev-parse HEAD)" = e1e25ce1d14362b209e34ec73871ebaeb63538a6
test -z "$(git -C "${source_dir}" status --porcelain)"
export CUDA_VISIBLE_DEVICES=1,2,3,4 N_GPUS_PER_NODE=4 PYTHON_BIN=/usr/bin/python3
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  test -z "$(nvidia-smi -i 1,2,3,4 --query-compute-apps=pid --format=csv,noheader)"
fi
export CONFIG_PATH=${source_dir}/configs/spikingbrain2-level5.env
deps=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted
export EXTRA_PYTHONPATH=${record_dir}:${source_dir}/experiments/records/20260906-integrated-fixes-v1:/root/new-scii-spb2-level5-control/python:${deps}/flash-linear-attention_dev:${deps}/MoBA
export PYTHONPATH=${source_dir}:${EXTRA_PYTHONPATH}
export CPATH=/usr/include/python3.12
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MODEL_PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/models/Panyuqi/SpikingBrain-2.0-instruct
export SPB_DISABLE_SWA_DROPOUT=1
export SC2PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air SC2_ENV_SEED=0 SC2_MAX_STEPS=0
export SC2_STEP_SHAPING_MODE=off BO_PROFILE=canonical EVAL_ONLY=1
export TOTAL_EPOCHS=1 SAVE_FREQ=-1
export SC2_PLAN_SEGMENT_MAX=6 SC2_PLAN_BULLET_MAX=6 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=96
export PPO_DIAGNOSTICS=0 ROLLOUT_LOGPROB_DIAGNOSTICS=0
export SC2_SEED_PYTHON_RANDOM=1 SC2_RESET_RETRIES=1
unset SC2_GAME_TIME_LIMIT SC2_RNG_PREFIX_TARGET
export RUN_ID=${CANDIDATE_RUN_ID:-20260908-garrison-wave-ownership-v1-spb2-original-l6-g2-s43567-eval8-r1}
run_root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260908-garrison-wave-ownership-v1/runs
export OUTPUT_DIR=${CANDIDATE_OUTPUT_DIR:-${run_root}/${RUN_ID}}
export RAY_TMPDIR=${CANDIDATE_RAY_TMPDIR:-/dev/shm/sc2_spb_failed_seed_0908_r1}
set +e
timeout --signal=INT --kill-after=60s 3600 bash "${source_dir}/scripts/train_sc2.sh" \
  +ray_init.runtime_env.worker_process_setup_hook=spb_failed_seed_eval.install \
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
  ++env.prompt_response_log_max_prompt_chars=0 \
  trainer.n_gpus_per_node=4 actor_rollout_ref.actor.ppo_mini_batch_size=8 env.rollout.n=8 \
  +env.val_rollout_n=8 actor_rollout_ref.rollout.val_kwargs.n=8 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
  actor_rollout_ref.rollout.val_kwargs.top_k=0 \
  ++ray_init.num_cpus=32
status=$?
set -e
if [[ -d "${OUTPUT_DIR}" ]]; then
  mkdir "${OUTPUT_DIR}/candidate_orchestration"
  cp "${BASH_SOURCE[0]}" "${record_dir}/spb_failed_seed_eval.py" \
    "${record_dir}/failed-seed-control.md" "${OUTPUT_DIR}/candidate_orchestration/"
  git -C "${record_dir}" rev-parse HEAD > "${OUTPUT_DIR}/candidate_orchestration/orchestration_sha.txt"
  printf 'EXIT_CODE=%s\n' "${status}" > "${OUTPUT_DIR}/candidate_orchestration/.done"
fi
exit "${status}"
