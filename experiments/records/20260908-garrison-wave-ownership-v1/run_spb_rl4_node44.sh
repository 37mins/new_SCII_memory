#!/usr/bin/env bash
# Frozen r3 gameplay: initial evaluation -> four global RL steps -> paired evaluation.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode=${1:?choose suite, initial-eval, train4, or step4-eval}
if [[ "${mode}" == suite ]]; then
  for stage in initial-eval train4 step4-eval; do
    bash "${BASH_SOURCE[0]}" "${stage}"
  done
  exit 0
fi
case "${mode}" in initial-eval|train4|step4-eval) ;; *) exit 2 ;; esac
source_dir=/root/new-scii-garrison-wave-20260908/code-e1e25ce
test "$(git -C "${source_dir}" rev-parse HEAD)" = e1e25ce1d14362b209e34ec73871ebaeb63538a6
test -z "$(git -C "${source_dir}" status --porcelain)"
export CUDA_VISIBLE_DEVICES=6,7 N_GPUS_PER_NODE=2 PYTHON_BIN=/usr/bin/python3
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  test -z "$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader)"
fi
export CONFIG_PATH=${source_dir}/configs/spikingbrain2-level5.env
deps=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted
export EXTRA_PYTHONPATH=${source_dir}/experiments/records/20260906-integrated-fixes-v1:/root/new-scii-spb2-level5-control/python:${deps}/flash-linear-attention_dev:${deps}/MoBA
export PYTHONPATH=${source_dir}:${EXTRA_PYTHONPATH}
export CPATH=/usr/include/python3.12
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MODEL_PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/models/Panyuqi/SpikingBrain-2.0-instruct
export SPB_DISABLE_SWA_DROPOUT=1
export SC2PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII
export SC2_DIFFICULTY=Harder SC2_AI_BUILD=Air SC2_ENV_SEED=0 SC2_MAX_STEPS=0 SC2_ROLLOUT_N=8
export SC2_STEP_SHAPING_MODE=off BO_PROFILE=canonical
export TOTAL_EPOCHS=4 SAVE_FREQ=1 EVAL_ONLY=0
export SC2_PLAN_SEGMENT_MAX=6 SC2_PLAN_BULLET_MAX=6 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=96
export PPO_DIAGNOSTICS=0 ROLLOUT_LOGPROB_DIAGNOSTICS=0
export SC2_SEED_PYTHON_RANDOM=1 SC2_RESET_RETRIES=1
unset SC2_GAME_TIME_LIMIT SC2_RNG_PREFIX_TARGET
suite=20260908-garrison-wave-ownership-v1-spb2-l6-g2-rl4-2gpu-s0-r4
run_root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260908-garrison-wave-ownership-v1/runs
export RUN_ID=${CANDIDATE_RUN_ID:-${suite}-${mode}}
export OUTPUT_DIR=${CANDIDATE_OUTPUT_DIR:-${run_root}/${RUN_ID}}
export RAY_TMPDIR=${CANDIDATE_RAY_TMPDIR:-/dev/shm/sc2_spb_rl4_0908_r4_${mode}}

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  "${PYTHON_BIN}" - "${mode}" "${run_root}/${suite}" <<'PY'
import json, sys
from pathlib import Path
import torch, transformers, ray, gymnasium
assert (torch.__version__, transformers.__version__, ray.__version__, gymnasium.__version__) == ('2.7.1+cu128', '4.55.2', '2.48.0', '1.2.3')
mode, prefix = sys.argv[1:]
if mode != 'initial-eval':
    initial = Path(prefix + '-initial-eval')
    outcome = json.loads((initial / '.done').read_text())
    assert outcome['status'] == 'completed' and outcome['training_updates'] == 0
    seeds = [json.loads(line) for line in (initial / 'evaluation_seeds.jsonl').read_text().splitlines()]
    assert seeds == [{'base_seed': 1000, 'reset_index': 0, 'seeds': [2807211443, 882774615, 494016583, 2621640384, 4081499387, 44124510, 2071061852, 1549697793]}]
if mode == 'step4-eval':
    train = Path(prefix + '-train4')
    outcome = json.loads((train / '.done').read_text())
    assert outcome['status'] == 'completed' and outcome['completed_steps'] == 4
    for step in range(1, 5):
        checkpoint = train / 'checkpoints' / f'global_step_{step}'
        files = [checkpoint / 'data.pt'] + [checkpoint / 'actor' / f'{kind}_world_size_2_rank_{rank}.pt'
                for kind in ('model', 'optim', 'extra_state') for rank in range(2)]
        assert all(path.is_file() and path.stat().st_size > 0 for path in files)
PY
fi

overrides=(
  "+ray_init.runtime_env.env_vars.SC2_SEED_PYTHON_RANDOM='1'"
  "+ray_init.runtime_env.env_vars.SC2_RESET_RETRIES='1'"
  "+ray_init.runtime_env.env_vars.SC2_PRESERVE_VALID_BO='1'"
  "+ray_init.runtime_env.env_vars.SC2_ENABLE_AIR_OPTIONS='1'"
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp32
  +actor_rollout_ref.actor.fp32_normalization=True
  +actor_rollout_ref.ref.fsdp_config.model_dtype=fp32
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=fp32
  +actor_rollout_ref.ref.fp32_normalization=True
  ++env.prompt_response_log_max_prompt_chars=0
  trainer.total_training_steps=4
  trainer.n_gpus_per_node=2 actor_rollout_ref.actor.ppo_mini_batch_size=8 env.rollout.n=8
  ++ray_init.num_cpus=32
)
wall_seconds=14400
if [[ "${mode}" != train4 ]]; then
  export EVAL_ONLY=1
  wall_seconds=3600
  overrides+=(+env.val_rollout_n=8 actor_rollout_ref.rollout.val_kwargs.n=8)
fi
if [[ "${mode}" == step4-eval ]]; then
  overrides+=(trainer.resume_mode=resume_path
    trainer.resume_from_path="${run_root}/${suite}-train4/checkpoints/global_step_4")
fi
set +e
timeout --signal=INT --kill-after=60s "${wall_seconds}" bash "${source_dir}/scripts/train_sc2.sh" "${overrides[@]}"
status=$?
set -e
if [[ -d "${OUTPUT_DIR}" ]]; then
  mkdir "${OUTPUT_DIR}/candidate_orchestration"
  cp "${BASH_SOURCE[0]}" "${record_dir}/spb-rl4.md" "${OUTPUT_DIR}/candidate_orchestration/"
  git -C "${record_dir}" rev-parse HEAD > "${OUTPUT_DIR}/candidate_orchestration/orchestration_sha.txt"
  printf 'EXIT_CODE=%s\n' "${status}" > "${OUTPUT_DIR}/candidate_orchestration/.done"
fi
if [[ "${status}" == 0 && "${mode}" == step4-eval && "${DRY_RUN:-0}" != 1 ]]; then
  cmp "${run_root}/${suite}-initial-eval/evaluation_seeds.jsonl" "${OUTPUT_DIR}/evaluation_seeds.jsonl"
fi
exit "${status}"
