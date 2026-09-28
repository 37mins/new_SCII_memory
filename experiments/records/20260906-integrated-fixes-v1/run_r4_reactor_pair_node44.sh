#!/usr/bin/env bash
# One natural full game per arm, using the existing frozen evaluation entry.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
orchestration_dir=$(cd "${record_dir}/../../.." && pwd)
test -z "$(git -C "${orchestration_dir}" status --porcelain)"
case "${1:?choose control, slots or spb}" in
  control) source_sha=836460a8533b6f0429f9e9d401ac9715adc55f5b; gpu=0 ;;
  slots) source_sha=c15d02b08085ddfcf20cff5517d10ceae9fa2af0; gpu=1 ;;
  spb) source_sha=c15d02b08085ddfcf20cff5517d10ceae9fa2af0; gpu=0 ;;
  *) exit 2 ;;
esac
source_dir=/root/new-scii-r4-python-rng-20260907/code-${source_sha:0:7}
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${source_sha}"
test -z "$(git -C "${source_dir}" status --porcelain)"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  test -z "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader)"
fi
export CUDA_VISIBLE_DEVICES=${gpu} PYTHON_BIN=/usr/bin/python3
export CONFIG_PATH=${source_dir}/configs/qwen3-4b-instruct-2507-level5.env
export EXTRA_PYTHONPATH=${record_dir}:/root/new-scii-spb2-level5-control/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MODEL_PATH=/mnt/data0601/yangjinyue/new_SCII/assets/Qwen3-4B-Instruct-2507-cdbee75
if [[ "${1}" == spb ]]; then
  export CONFIG_PATH=${source_dir}/configs/spikingbrain2-level5.env
  export MODEL_PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/models/Panyuqi/SpikingBrain-2.0-instruct
  deps=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted
  export EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH}:${deps}/flash-linear-attention_dev:${deps}/MoBA
  export SPB_DISABLE_SWA_DROPOUT=1 CPATH=/usr/include/python3.12
fi
export SC2PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air SC2_ENV_SEED=0 SC2_MAX_STEPS=0
export SC2_STEP_SHAPING_MODE=off BO_PROFILE=canonical EVAL_ONLY=1
export SC2_PLAN_SEGMENT_MAX=6 SC2_PLAN_BULLET_MAX=6 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=96
export PPO_DIAGNOSTICS=0 ROLLOUT_LOGPROB_DIAGNOSTICS=0
export SC2_SEED_PYTHON_RANDOM=1 SC2_RESET_RETRIES=1
unset SC2_GAME_TIME_LIMIT SC2_RNG_PREFIX_TARGET
export RUN_ID=20260906-integrated-fixes-v1-r4-reactor-pair-${1}-qwen2507-l6-g2-s2621640384-r2
if [[ "${1}" == spb ]]; then
  export RUN_ID=20260906-integrated-fixes-v1-r4-reactor-pair-slots-spb2-l6-g2-s2621640384-r1
fi
run_root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-integrated-fixes-v1/runs
export OUTPUT_DIR=${PAIR_OUTPUT_DIR:-${run_root}/${RUN_ID}}
export RAY_TMPDIR=/dev/shm/sc2_pair_${1}_20260907_r2
if [[ "${1}" == spb ]]; then export RAY_TMPDIR=/dev/shm/sc2_pair_spb_20260907_r1; fi
set +e
timeout --signal=TERM --kill-after=30s 1800 bash "${source_dir}/scripts/train_sc2.sh" \
  +ray_init.runtime_env.worker_process_setup_hook=r4_pair_seed.install \
  "+ray_init.runtime_env.env_vars.SC2_SEED_PYTHON_RANDOM='1'" \
  "+ray_init.runtime_env.env_vars.SC2_RESET_RETRIES='1'" \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp32 \
  +actor_rollout_ref.actor.fp32_normalization=True \
  +actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 \
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=fp32 \
  +actor_rollout_ref.ref.fp32_normalization=True \
  ++env.prompt_response_log_max_prompt_chars=0
status=$?
set -e
if [[ -d "${OUTPUT_DIR}" ]]; then
  mkdir "${OUTPUT_DIR}/pair_orchestration"
  cp "${BASH_SOURCE[0]}" "${record_dir}/r4_pair_seed.py" "${record_dir}/r4-reactor-paired-eval.md" "${OUTPUT_DIR}/pair_orchestration/"
  git -C "${orchestration_dir}" rev-parse HEAD > "${OUTPUT_DIR}/pair_orchestration/source_sha.txt"
  printf 'EXIT_CODE=%s\n' "${status}" > "${OUTPUT_DIR}/pair_orchestration/.done"
fi
exit "${status}"
