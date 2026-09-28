#!/usr/bin/env bash
# One arm of the approved paired evaluation, wrapped by the shared 60-minute budget.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set the frozen source SHA}"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  test -z "$(git -C "${source_dir}" status --porcelain)"
fi
case "${1:?arm A or B}" in
  A) export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} SC2_FIXED_BO_PREFIX_LIMIT=0 ;;
  B) export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7} SC2_FIXED_BO_PREFIX_LIMIT=${SC2_HANDOFF_B_PREFIX:-138} ;;
  *) exit 2 ;;
esac
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  test -z "$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader)"
fi
export N_GPUS_PER_NODE=${SC2_EVAL_N_GPUS:-4} PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
export CONFIG_PATH=${source_dir}/configs/spikingbrain2-level5.env
deps=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted
export EXTRA_PYTHONPATH=${record_dir}:${source_dir}/experiments/records/20260906-integrated-fixes-v1:${SC2_PYTHON_OVERLAY:-/root/new-scii-spb2-level5-control/python}:${deps}/flash-linear-attention_dev:${deps}/MoBA
export PYTHONPATH=${source_dir}:${EXTRA_PYTHONPATH}
export CPATH=${CPATH:-/usr/include/python3.12}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MODEL_PATH=${MODEL_PATH:-/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/models/Panyuqi/SpikingBrain-2.0-instruct}
export SPB_DISABLE_SWA_DROPOUT=1
export SC2PATH=${SC2PATH:-/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII}
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air SC2_ENV_SEED=0 SC2_MAX_STEPS=0
export SC2_STEP_SHAPING_MODE=off BO_PROFILE=canonical EVAL_ONLY=1
export TOTAL_EPOCHS=1 SAVE_FREQ=-1
export SC2_PLAN_SEGMENT_MAX=6 SC2_PLAN_BULLET_MAX=6 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=${SC2_EVAL_RESPONSE_LENGTH:-96}
export PPO_DIAGNOSTICS=0 ROLLOUT_LOGPROB_DIAGNOSTICS=0
export SC2_SEED_PYTHON_RANDOM=1 SC2_RESET_RETRIES=1
unset SC2_GAME_TIME_LIMIT SC2_RNG_PREFIX_TARGET
export RUN_ID=${RUN_ID:?set a unique arm RUN_ID}
export OUTPUT_DIR=${OUTPUT_DIR:?set an unused arm output directory}
export RAY_TMPDIR=${RAY_TMPDIR:?set a short unique Ray directory}
exec bash "${source_dir}/scripts/train_sc2.sh" \
  +ray_init.runtime_env.worker_process_setup_hook=handoff_eval.install \
  "+ray_init.runtime_env.env_vars.SC2_SEED_PYTHON_RANDOM='1'" \
  "+ray_init.runtime_env.env_vars.SC2_RESET_RETRIES='1'" \
  "+ray_init.runtime_env.env_vars.SC2_PRESERVE_VALID_BO='1'" \
  "+ray_init.runtime_env.env_vars.SC2_ENABLE_AIR_OPTIONS='1'" \
  "++ray_init.runtime_env.env_vars.SC2_ROLLOUT_N='4'" \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp32 \
  +actor_rollout_ref.actor.fp32_normalization=True \
  +actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 \
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=fp32 \
  +actor_rollout_ref.ref.fp32_normalization=True \
  ++env.prompt_response_log_max_prompt_chars=0 \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" actor_rollout_ref.actor.ppo_mini_batch_size=8 env.rollout.n="${N_GPUS_PER_NODE}" \
  +env.val_rollout_n=4 actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
  actor_rollout_ref.rollout.val_kwargs.top_k=0 \
  ++ray_init.num_cpus=32
