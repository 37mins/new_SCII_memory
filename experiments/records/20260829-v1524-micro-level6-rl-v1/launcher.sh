#!/usr/bin/env bash
set -uo pipefail

NODE_TAG=${NODE_TAG:?NODE_TAG is required}
GPU_ID=${GPU_ID:?GPU_ID is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
NEW_REPO_COMMIT=${NEW_REPO_COMMIT:?NEW_REPO_COMMIT is required}
EFFECTIVE_CONFIG_SHA=${EFFECTIVE_CONFIG_SHA:?EFFECTIVE_CONFIG_SHA is required}
EVAL_LAUNCHER_PATCH_SHA=${EVAL_LAUNCHER_PATCH_SHA:?EVAL_LAUNCHER_PATCH_SHA is required}
FULL_NODE_PATCH_SHA=${FULL_NODE_PATCH_SHA:?FULL_NODE_PATCH_SHA is required}
VALIDATION_SEED_PATCH_SHA=${VALIDATION_SEED_PATCH_SHA:?VALIDATION_SEED_PATCH_SHA is required}
QWEN_RL_PATCH_SHA=${QWEN_RL_PATCH_SHA:?QWEN_RL_PATCH_SHA is required}
TRAINING_SEED_PATCH_SHA=${TRAINING_SEED_PATCH_SHA:?TRAINING_SEED_PATCH_SHA is required}
HF_ROLLOUT_OFFLOAD_PATCH_SHA=${HF_ROLLOUT_OFFLOAD_PATCH_SHA:?HF_ROLLOUT_OFFLOAD_PATCH_SHA is required}
CHAT_TEMPLATE_CONTROL_PATCH_SHA=${CHAT_TEMPLATE_CONTROL_PATCH_SHA:?CHAT_TEMPLATE_CONTROL_PATCH_SHA is required}
CHAT_TEMPLATE_CANARY_SHA=${CHAT_TEMPLATE_CANARY_SHA:?CHAT_TEMPLATE_CANARY_SHA is required}
RL_REWARD_BALANCE_PATCH_SHA=${RL_REWARD_BALANCE_PATCH_SHA:?RL_REWARD_BALANCE_PATCH_SHA is required}
QWEN_BASE_CHECKPOINT_PATCH_SHA=${QWEN_BASE_CHECKPOINT_PATCH_SHA:?QWEN_BASE_CHECKPOINT_PATCH_SHA is required}
CANONICAL_LAUNCHER_PATCH_SHA=${CANONICAL_LAUNCHER_PATCH_SHA:?CANONICAL_LAUNCHER_PATCH_SHA is required}
CANONICAL_PLANNER_SHA=${CANONICAL_PLANNER_SHA:?CANONICAL_PLANNER_SHA is required}
STRICT_MANAGER_PATCH_SHA=${STRICT_MANAGER_PATCH_SHA:?STRICT_MANAGER_PATCH_SHA is required}
RUNTIME_PROVENANCE_PATCH_SHA=${RUNTIME_PROVENANCE_PATCH_SHA:?RUNTIME_PROVENANCE_PATCH_SHA is required}
CONTAINER_IMAGE_ID=${CONTAINER_IMAGE_ID:?CONTAINER_IMAGE_ID is required}
CONTAINER_MOUNT_ROOT=${CONTAINER_MOUNT_ROOT:?CONTAINER_MOUNT_ROOT is required}
DEPLOYMENT_MANIFEST_SHA=${DEPLOYMENT_MANIFEST_SHA:?DEPLOYMENT_MANIFEST_SHA is required}
MICRO=${MICRO:-old}
ENV_SEED=${ENV_SEED:?ENV_SEED is required}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
TRAIN_ROLLOUT_N=${TRAIN_ROLLOUT_N:-8}
SAVE_FREQ=${SAVE_FREQ:-20}
TARGET_STEP=${TARGET_STEP:-96}
REQUIRE_CHECKPOINT=${REQUIRE_CHECKPOINT:-1}
DIFFICULTY_NAME=${DIFFICULTY_NAME:?DIFFICULTY_NAME is required}
DIFFICULTY_LEVEL=${DIFFICULTY_LEVEL:?DIFFICULTY_LEVEL is required}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-192}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
EXPERIMENT_TAG=${EXPERIMENT_TAG:?EXPERIMENT_TAG is required}
WANDB_GROUP=${WANDB_GROUP:-99-audit/qwen-compact-contract-canary}
MODE=control

case "${MICRO}" in
  old)
    ENABLE_NEWEXEC_MICRO=0
    ;;
  v1524)
    ENABLE_NEWEXEC_MICRO=1
    ;;
  *)
    echo "unsupported MICRO=${MICRO}; expected old or v1524" >&2
    exit 2
    ;;
esac

REV=level${DIFFICULTY_LEVEL}_qwen3_4b_base_nosft_noskill_compact6_v7_newa100
ROOT=${RUNTIME_ROOT_OVERRIDE:?RUNTIME_ROOT_OVERRIDE is required}
CODE=${ROOT}/code
PROJECT_ROOT=${PROJECT_ROOT_OVERRIDE:?PROJECT_ROOT_OVERRIDE is required}
SHARED=${PROJECT_ROOT}/runs
RESULT_ROOT=${RESULT_ROOT_OVERRIDE:-${SHARED}/20260807-qwen-tenlevel-compact-contract}
RUN_NAME=qwen3_4b_g2_${EXPERIMENT_TAG}_level${DIFFICULTY_LEVEL}_from_base_${MICRO}micro_${NODE_TAG}_seed${ENV_SEED}_step${TARGET_STEP}_resp${MAX_RESPONSE_LENGTH}_${REV}_20260816
LOG_DIR=${RESULT_ROOT}/logs/${RUN_NAME}
CKPT_DIR=${RESULT_ROOT}/checkpoints/${RUN_NAME}
AUDIT_DIR=${RESULT_ROOT}/audit/${RUN_NAME}
DONE=${RESULT_ROOT}/${RUN_NAME}.done
# Ray appends long session/socket suffixes, so keep this prefix well below
# Linux's 107-byte AF_UNIX path limit even when NODE_TAG is descriptive.
RAY_TAG=${RAY_TAG:-$(printf '%s' "${NODE_TAG}" | sha256sum | cut -c1-10)}
RAY_TMPDIR=/dev/shm/r_${RAY_TAG}
CACHE_DIR=${ROOT}/cache
PYTHON_SITE_ROOT=${PYTHON_SITE_ROOT_OVERRIDE:-}

SOURCE_REPO=${PROJECT_ROOT}/git/worktrees/stageguard-v7-legacy-contract
RUNNER_REL=examples/gigpo_trainer/run_sc2_spb2_gigpo_0095_newmicro_skill_phase2_8gpu_100ep_20260724.sh
RUNNER=${CODE}/${RUNNER_REL}
V7_ASSETS=${PROJECT_ROOT}/assets/repro/stageguard_v7_legacy_contract_20260806
FROZEN_RUNTIME_TAR=${V7_ASSETS}/runtime/frozen_runtime_source.tar.gz
EXPECTED_FROZEN_RUNTIME_TAR_SHA=016772ab0cd8f8cbc3d017150cb1b7c17cb3f40984accdfd14a2e8b6a9b76cb0
PATCH_AGENTS=${PATCH_AGENTS_OVERRIDE:-${SOURCE_REPO}/agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/agents}
PATCH_MANAGER=${PATCH_MANAGER_OVERRIDE:-${SOURCE_REPO}/agent_system/environments/strategic_sc2_env_manager_v2.py}
PATCH_STRATEGIST_SHA=3abb1fe0ec718919f1b897f1b4747685a5548830508082394b3126bbd70b7bbe
PATCH_COMMANDER_SHA=0ffd91b4f913bb2f69bd02c88b9274fc718da9103b2a1ffd61ae496e1ae6c592
PATCH_PLANNER_SHA=221aac4f71b6219b01ff4c0425b2a204ae0f700a92c11088e9acfdd8e94b8346
PATCH_MANAGER_SHA=1feeb6e7ab45b21d4daab9f46ee7c9466d1bc0faccfb943388acdc2c99d219c8
PYTHON_BIN=${PYTHON_BIN_OVERRIDE:-${PROJECT_ROOT}/envs/verl-agent-v7-exact/bin/python}
BASE_MODEL=${MODEL_ASSET_ROOT:-${PROJECT_ROOT}/assets/models/Qwen/checkpoint-1248_verlfix}
BASE_MODEL_MANIFEST=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen3_4b_base_checkpoint.sha256
EXPECTED_BASE_CONFIG_SHA=8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a
EXPECTED_BASE_TOKENIZER_CONFIG_SHA=d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101
EXPECTED_BASE_MODEL_INDEX_SHA=6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2
EXPECTED_BASE_MODEL_MANIFEST_SHA=82fdeae910670892e5c70b177ca5ec30d3d69090416e34480be6d63880f7a56c
DATA_ROOT=${V7_ASSETS}/frozen_inputs
FLA_ROOT=${V7_ASSETS}/source/extracted/flash-linear-attention_dev
MOBA_ROOT=${V7_ASSETS}/source/extracted/MoBA
SC2PATH=${SC2PATH_OVERRIDE:-${PROJECT_ROOT}/assets/sc2/StarCraftII}
EXPECTED_GIT_COMMIT=e61a23b8c642cb1b88d67f11f1f6d9a96a65ef31
RUNNER_PATCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runner_new_a100.patch
VALIDATION_SEED_PATCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/independent_validation_seeds.patch
TRAINING_SEED_PATCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/training_seed_ledger.patch
HF_ROLLOUT_OFFLOAD_PATCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/verl_hf_rollout_offload.patch
RL_REWARD_BALANCE_PATCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rl_reward_balance.patch
CANONICAL_PLANNER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/canonical_planner.py
STRICT_MANAGER_PATCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/strict_legacy_plan_validator.patch

EXPECTED_VERL_TREE_SHA=aa81b037c7972bd62886403c01a92c398a8b4abc28d863aeaa0c1a1b874dcc4b
EXPECTED_AGENT_TREE_SHA=7ea801c8d6d758479bd26269b6289a7df71f4217b66796e3cbd6dd0511c8377b
EXPECTED_GIGPO_TREE_SHA=7098f59713be946888a4e303b346d4ff43d220f15323d6ddea0b9c7f612cd089
EXPECTED_RUNNER_SHA=330e21b745862f80672d034182aa550d428f3124bba8f1a33f628256ba55a4ae
EXPECTED_SCRIPT_PLAYER_SHA=76354ff908b47f8c400e2cf279a3af5f49e98a3da6afb53018143b579f28c6da
EXPECTED_TRAIN_DATA_SHA=7c8f6b07eacef5e06963b475c3c36e90196da870fae095b12c2ea0f7faf10912
EXPECTED_TEST_DATA_SHA=4dbe5e3ff68df3fbf76e91284b28b06c52bf7495193013a4bf4ed7ca417a0cd7
WANDB_ENV=${PROJECT_ROOT}/secrets/wandb.env

tree_sha() {
  local root=$1
  find "${root}" -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum \
    | sed "s#${root}/##" \
    | sha256sum \
    | awk '{print $1}'
}

write_failure() {
  local status=$1
  local reason=$2
  mkdir -p "$(dirname "${DONE}")"
  {
    echo "STATUS=${status}"
    echo "REASON=${reason}"
    echo "NODE_TAG=${NODE_TAG}"
    echo "RAY_TAG=${RAY_TAG}"
    echo "RAY_TMPDIR=${RAY_TMPDIR}"
    echo "MICRO=${MICRO}"
    echo "ENV_SEED=${ENV_SEED}"
    echo "EXPERIMENT_TAG=${EXPERIMENT_TAG}"
    echo "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
    echo "BASE_MODEL=${BASE_MODEL}"
  } > "${DONE}"
}

mkdir -p "${ROOT}" "${LOG_DIR}" "${CKPT_DIR}" "${AUDIT_DIR}" "${RAY_TMPDIR}" "${CACHE_DIR}"

frozen_runtime_tar_sha=$(sha256sum "${FROZEN_RUNTIME_TAR}" 2>/dev/null | awk '{print $1}')
source_verl_sha=${EXPECTED_VERL_TREE_SHA}
source_agent_sha=${EXPECTED_AGENT_TREE_SHA}
source_gigpo_sha=${EXPECTED_GIGPO_TREE_SHA}
source_runner_sha=${EXPECTED_RUNNER_SHA}
base_model_config_sha=$(sha256sum "${BASE_MODEL}/config.json" 2>/dev/null | awk '{print $1}')
base_model_tokenizer_config_sha=$(sha256sum "${BASE_MODEL}/tokenizer_config.json" 2>/dev/null | awk '{print $1}')
base_model_index_sha=$(sha256sum "${BASE_MODEL}/model.safetensors.index.json" 2>/dev/null | awk '{print $1}')
base_model_manifest_sha=$(sha256sum "${BASE_MODEL_MANIFEST}" 2>/dev/null | awk '{print $1}')
base_model_layout_sha=$(find "${BASE_MODEL}" -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null | LC_ALL=C sort | sha256sum | awk '{print $1}')
train_data_sha=$(sha256sum "${DATA_ROOT}/train.parquet" 2>/dev/null | awk '{print $1}')
test_data_sha=$(sha256sum "${DATA_ROOT}/test.parquet" 2>/dev/null | awk '{print $1}')

if [[ "${frozen_runtime_tar_sha}" != "${EXPECTED_FROZEN_RUNTIME_TAR_SHA}" || \
      "${train_data_sha}" != "${EXPECTED_TRAIN_DATA_SHA}" || \
      "${test_data_sha}" != "${EXPECTED_TEST_DATA_SHA}" || \
      "${base_model_config_sha}" != "${EXPECTED_BASE_CONFIG_SHA}" || \
      "${base_model_tokenizer_config_sha}" != "${EXPECTED_BASE_TOKENIZER_CONFIG_SHA}" || \
      "${base_model_index_sha}" != "${EXPECTED_BASE_MODEL_INDEX_SHA}" || \
      "${base_model_manifest_sha}" != "${EXPECTED_BASE_MODEL_MANIFEST_SHA}" || \
      -z "${base_model_layout_sha}" || \
      -z "${train_data_sha}" || -z "${test_data_sha}" ]] || \
   ! (cd "${BASE_MODEL}" && sha256sum --status -c "${BASE_MODEL_MANIFEST}"); then
  write_failure 2 source_hash_mismatch
  exit 2
fi

if [[ ! -s "${BASE_MODEL}/config.json" || ! -s "${BASE_MODEL}/tokenizer_config.json" || \
      ! -s "${BASE_MODEL}/model.safetensors.index.json" || \
      "$(find "${BASE_MODEL}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l)" -ne 3 || \
      ! -s "${DATA_ROOT}/train.parquet" || ! -s "${DATA_ROOT}/test.parquet" || \
      ! -s "${SC2PATH}/maps/Melee/Flat128.SC2Map" ]]; then
  write_failure 3 runtime_assets_incomplete
  exit 3
fi

export WANDB_ENTITY=13981456278-casia
export WANDB_PROJECT=sc2-rl
export WANDB_MODE=disabled
export WANDB_RUN_GROUP="${WANDB_GROUP}"
export WANDB_RUN_ID=$(printf '%s' "${RUN_NAME}" | sha256sum | cut -c1-32)
export WANDB_RESUME=never

if [[ "$(sha256sum "${PATCH_AGENTS}/strategist.py" | awk '{print $1}')" != "${PATCH_STRATEGIST_SHA}" || \
      "$(sha256sum "${PATCH_AGENTS}/commander.py" | awk '{print $1}')" != "${PATCH_COMMANDER_SHA}" || \
      "$(sha256sum "${PATCH_AGENTS}/planner.py" | awk '{print $1}')" != "${PATCH_PLANNER_SHA}" || \
      "$(sha256sum "${PATCH_MANAGER}" | awk '{print $1}')" != "${PATCH_MANAGER_SHA}" ]]; then
  write_failure 10 stageguard_patch_hash_mismatch
  exit 10
fi

if [[ "${RESULT_ROOT}" == /mnt/data0601/* ]]; then
  avail_kb=$(df -Pk /mnt/data0601 | awk 'NR==2 {print $4}')
  if [[ -z "${avail_kb}" || "${avail_kb}" -lt 314572800 ]]; then
    write_failure 4 shared_storage_below_300gb
    exit 4
  fi
fi

compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')
if [[ -n "${compute_pids}" ]]; then
  write_failure 5 full_node_gpu_compute_process_present
  exit 5
fi

sc2_pids=$(pgrep -f '[/]SC2_x64' || true)
if [[ -n "${sc2_pids}" ]]; then
  write_failure 16 full_node_sc2_process_present
  exit 16
fi

if [[ -e "${CODE}" ]]; then
  write_failure 6 stale_frozen_code_root
  exit 6
fi
tar -C "${ROOT}" -xzf "${FROZEN_RUNTIME_TAR}"
if [[ ! -d "${CODE}/agent_system" || ! -d "${CODE}/verl" || ! -d "${CODE}/gigpo" ]]; then
  write_failure 11 frozen_runtime_extract_failed
  exit 11
fi
# The base archive was created after an import preflight, so it contains only
# generated bytecode in addition to the hashed source tree. Remove that bytecode
# from this run-owned tmpfs copy before verifying and importing the source.
find "${CODE}" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "${CODE}" -type f -name '*.pyc' -delete
AGENTS_DIR=${CODE}/agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/agents
cp "${PATCH_AGENTS}/strategist.py" "${AGENTS_DIR}/strategist.py"
cp "${PATCH_AGENTS}/commander.py" "${AGENTS_DIR}/commander.py"
cp "${PATCH_AGENTS}/planner.py" "${AGENTS_DIR}/planner.py"
cp "${PATCH_MANAGER}" "${CODE}/agent_system/environments/strategic_sc2_env_manager_v2.py"

frozen_verl_sha=$(tree_sha "${CODE}/verl")
frozen_gigpo_sha=$(tree_sha "${CODE}/gigpo")
frozen_runner_sha=$(sha256sum "${RUNNER}" | awk '{print $1}')
script_player_sha=$(sha256sum "${CODE}/agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/players/script_player.py" | awk '{print $1}')
BO=${CODE}/agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/knowledge/BO/replay_observations_Flat128_d4_TvT_Macro_g2_Result.Victory_twobase.json
bo_sha=$(sha256sum "${BO}" 2>/dev/null | awk '{print $1}')
frozen_strategist_sha=$(sha256sum "${AGENTS_DIR}/strategist.py" | awk '{print $1}')
frozen_commander_sha=$(sha256sum "${AGENTS_DIR}/commander.py" | awk '{print $1}')
frozen_planner_sha=$(sha256sum "${AGENTS_DIR}/planner.py" | awk '{print $1}')
frozen_manager_sha=$(sha256sum "${CODE}/agent_system/environments/strategic_sc2_env_manager_v2.py" | awk '{print $1}')

if [[ "${frozen_verl_sha}" != "${EXPECTED_VERL_TREE_SHA}" || \
      "${frozen_gigpo_sha}" != "${EXPECTED_GIGPO_TREE_SHA}" || \
      "${frozen_runner_sha}" != "${EXPECTED_RUNNER_SHA}" || \
      "${script_player_sha}" != "${EXPECTED_SCRIPT_PLAYER_SHA}" || \
      "${frozen_strategist_sha}" != "${PATCH_STRATEGIST_SHA}" || \
      "${frozen_commander_sha}" != "${PATCH_COMMANDER_SHA}" || \
      "${frozen_planner_sha}" != "${PATCH_PLANNER_SHA}" || \
      "${frozen_manager_sha}" != "${PATCH_MANAGER_SHA}" || \
      ! -x "${PYTHON_BIN}" || ! -s "${RUNNER}" || ! -s "${BO}" || \
      -z "${bo_sha}" ]]; then
  write_failure 7 frozen_runtime_preflight_failed
  exit 7
fi

runner_patch_sha=$(sha256sum "${RUNNER_PATCH}" 2>/dev/null | awk '{print $1}')
if [[ -z "${runner_patch_sha}" ]] || ! patch --batch --forward -d "${CODE}" -p1 < "${RUNNER_PATCH}"; then
  write_failure 12 new_a100_runner_patch_failed
  exit 12
fi
sed -i -E '/^(export WANDB_API_KEY=|: "\$\{WANDB_API_KEY:)/c\export WANDB_MODE=${WANDB_MODE:-disabled}' "${RUNNER}"
patched_runner_sha=$(sha256sum "${RUNNER}" | awk '{print $1}')
if [[ "${patched_runner_sha}" == "${EXPECTED_RUNNER_SHA}" ]]; then
  write_failure 13 new_a100_runner_patch_not_applied
  exit 13
fi

sed -i -E 's/^export SC2_RESET_CONCURRENCY=1$/export SC2_RESET_CONCURRENCY=8/' "${RUNNER}"
if ! grep -qx 'export SC2_RESET_CONCURRENCY=8' "${RUNNER}"; then
  write_failure 15 full_node_reset_concurrency_patch_failed
  exit 15
fi

validation_seed_patch_sha=$(sha256sum "${VALIDATION_SEED_PATCH}" 2>/dev/null | awk '{print $1}')
if [[ "${validation_seed_patch_sha}" != "${VALIDATION_SEED_PATCH_SHA}" ]] || \
   ! patch --batch --forward -d "${CODE}" -p1 < "${VALIDATION_SEED_PATCH}"; then
  write_failure 14 independent_validation_seed_patch_failed
  exit 14
fi

hf_rollout_offload_patch_sha=$(sha256sum "${HF_ROLLOUT_OFFLOAD_PATCH}" 2>/dev/null | awk '{print $1}')
if [[ "${hf_rollout_offload_patch_sha}" != "${HF_ROLLOUT_OFFLOAD_PATCH_SHA}" ]] || \
   ! patch --batch --forward -d "${CODE}" -p1 < "${HF_ROLLOUT_OFFLOAD_PATCH}"; then
  write_failure 20 hf_rollout_offload_patch_failed
  exit 20
fi

training_seed_patch_sha=$(sha256sum "${TRAINING_SEED_PATCH}" 2>/dev/null | awk '{print $1}')
if [[ "${training_seed_patch_sha}" != "${TRAINING_SEED_PATCH_SHA}" ]] || \
   ! patch --batch --forward -d "${CODE}" -p1 < "${TRAINING_SEED_PATCH}"; then
  write_failure 17 training_seed_ledger_patch_failed
  exit 17
fi

rl_reward_balance_patch_sha=$(sha256sum "${RL_REWARD_BALANCE_PATCH}" 2>/dev/null | awk '{print $1}')
if [[ "${rl_reward_balance_patch_sha}" != "${RL_REWARD_BALANCE_PATCH_SHA}" ]] || \
   ! patch --ignore-whitespace --batch --forward -d "${CODE}" -p1 < "${RL_REWARD_BALANCE_PATCH}"; then
  write_failure 21 rl_reward_balance_patch_failed
  exit 21
fi

strict_manager_patch_sha=$(sha256sum "${STRICT_MANAGER_PATCH}" 2>/dev/null | awk '{print $1}')
if [[ "${strict_manager_patch_sha}" != "${STRICT_MANAGER_PATCH_SHA}" ]] || \
   ! patch --batch --forward -d "${CODE}" -p1 < "${STRICT_MANAGER_PATCH}"; then
  write_failure 22 strict_legacy_plan_validator_patch_failed
  exit 22
fi

canonical_planner_sha=$(sha256sum "${CANONICAL_PLANNER}" 2>/dev/null | awk '{print $1}')
if [[ "${canonical_planner_sha}" != "${CANONICAL_PLANNER_SHA}" ]] || \
   ! cp "${CANONICAL_PLANNER}" "${AGENTS_DIR}/planner.py"; then
  write_failure 23 canonical_planner_install_failed
  exit 23
fi
final_planner_sha=$(sha256sum "${AGENTS_DIR}/planner.py" 2>/dev/null | awk '{print $1}')
if [[ "${final_planner_sha}" != "${CANONICAL_PLANNER_SHA}" ]]; then
  write_failure 23 canonical_planner_install_failed
  exit 23
fi

if ! PYTHONPATH="${CODE}:${PYTHON_SITE_ROOT}" "${PYTHON_BIN}" -c \
  'from gigpo import core_gigpo; from verl.trainer.ppo.ray_trainer import RayPPOTrainer'; then
  write_failure 8 frozen_python_import_preflight_failed
  exit 8
fi

{
  echo "EVALUATION_ONLY=0"
  echo "RUN_ID=${RUN_ID}"
  echo "NODE_TAG=${NODE_TAG}"
  echo "GPU_ID=${GPU_ID}"
  echo "NEW_REPO_COMMIT=${NEW_REPO_COMMIT}"
  echo "NEW_REPO_DIRTY=0"
  echo "EFFECTIVE_CONFIG_SHA=${EFFECTIVE_CONFIG_SHA}"
  echo "EVAL_LAUNCHER_PATCH_SHA=${EVAL_LAUNCHER_PATCH_SHA}"
  echo "FULL_NODE_PATCH_SHA=${FULL_NODE_PATCH_SHA}"
  echo "VALIDATION_SEED_PATCH_SHA=${validation_seed_patch_sha}"
  echo "QWEN_RL_PATCH_SHA=${QWEN_RL_PATCH_SHA}"
  echo "TRAINING_SEED_PATCH_SHA=${training_seed_patch_sha}"
  echo "HF_ROLLOUT_OFFLOAD_PATCH_SHA=${hf_rollout_offload_patch_sha}"
  echo "CHAT_TEMPLATE_CONTROL_PATCH_SHA=${CHAT_TEMPLATE_CONTROL_PATCH_SHA}"
  echo "CHAT_TEMPLATE_CANARY_SHA=${CHAT_TEMPLATE_CANARY_SHA}"
  echo "ENABLE_THINKING=0"
  echo "FORBID_THINKING_OUTPUT=1"
  echo "RL_REWARD_BALANCE_PATCH_SHA=${rl_reward_balance_patch_sha}"
  echo "QWEN_BASE_CHECKPOINT_PATCH_SHA=${QWEN_BASE_CHECKPOINT_PATCH_SHA}"
  echo "CANONICAL_LAUNCHER_PATCH_SHA=${CANONICAL_LAUNCHER_PATCH_SHA}"
  echo "CANONICAL_PLANNER_SHA=${canonical_planner_sha}"
  echo "STRICT_MANAGER_PATCH_SHA=${strict_manager_patch_sha}"
  echo "FINAL_PLANNER_SHA=${final_planner_sha}"
  echo "RUNTIME_PROVENANCE_PATCH_SHA=${RUNTIME_PROVENANCE_PATCH_SHA}"
  echo "CONTAINER_IMAGE_ID=${CONTAINER_IMAGE_ID}"
  echo "CONTAINER_MOUNT_ROOT=${CONTAINER_MOUNT_ROOT}"
  echo "DEPLOYMENT_MANIFEST_SHA=${DEPLOYMENT_MANIFEST_SHA}"
  echo "STRICT_LEGACY_PLAN_CONTRACT=1"
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "PYTHON_SITE_ROOT=${PYTHON_SITE_ROOT}"
  echo "RUNTIME_ROOT=${ROOT}"
  echo "INDEPENDENT_VALIDATION_SEEDS=1"
  echo "GIT_COMMIT=${EXPECTED_GIT_COMMIT}"
  echo "MICRO=${MICRO}"
  echo "ENABLE_NEWEXEC_MICRO=${ENABLE_NEWEXEC_MICRO}"
  echo "ENV_SEED=${ENV_SEED}"
  echo "EXPERIMENT_TAG=${EXPERIMENT_TAG}"
  echo "BASE_MODEL=${BASE_MODEL}"
  echo "BASE_MODEL_ID=Qwen/Qwen3-4B"
  echo "BASE_MODEL_ARCHITECTURE=Qwen3ForCausalLM"
  echo "BASE_MODEL_INITIALIZATION=public_base_snapshot_no_sft_no_rl"
  echo "BASE_MODEL_CONFIG_SHA=${base_model_config_sha}"
  echo "BASE_MODEL_TOKENIZER_CONFIG_SHA=${base_model_tokenizer_config_sha}"
  echo "BASE_MODEL_INDEX_SHA=${base_model_index_sha}"
  echo "BASE_MODEL_MANIFEST_SHA=${base_model_manifest_sha}"
  echo "BASE_MODEL_LAYOUT_SHA=${base_model_layout_sha}"
  echo "TRAIN_DATA_SHA=${train_data_sha}"
  echo "TEST_DATA_SHA=${test_data_sha}"
  echo "FROZEN_DATA_ROOT=${DATA_ROOT}"
  echo "BASE_FROZEN_RUNTIME_TAR=${FROZEN_RUNTIME_TAR}"
  echo "BASE_FROZEN_RUNTIME_TAR_SHA=${frozen_runtime_tar_sha}"
  echo "SOURCE_VERL_TREE_SHA=${source_verl_sha}"
  echo "FROZEN_VERL_TREE_SHA=${frozen_verl_sha}"
  echo "SOURCE_AGENT_TREE_SHA=${source_agent_sha}"
  echo "SOURCE_GIGPO_TREE_SHA=${source_gigpo_sha}"
  echo "FROZEN_GIGPO_TREE_SHA=${frozen_gigpo_sha}"
  echo "RUNNER_SHA=${frozen_runner_sha}"
  echo "RUNNER_PATCH_SHA=${runner_patch_sha}"
  echo "PATCHED_RUNNER_SHA=${patched_runner_sha}"
  echo "SCRIPT_PLAYER_SHA=${script_player_sha}"
  echo "STRATEGIST_SHA=${frozen_strategist_sha}"
  echo "COMMANDER_SHA=${frozen_commander_sha}"
  echo "PLANNER_SHA=${frozen_planner_sha}"
  echo "MANAGER_SHA=${frozen_manager_sha}"
  echo "BO=${BO}"
  echo "BO_SHA=${bo_sha}"
  echo "MEMORY_MODE=off"
  echo "QUERY_POLICY=never"
  echo "QUERY_TRAIN=0"
  echo "ENABLE_INTERVENTION=0"
  echo "DIFFICULTY=${DIFFICULTY_NAME}"
  echo "DIFFICULTY_LEVEL=${DIFFICULTY_LEVEL}"
  echo "AI_BUILD=Air"
  echo "SC2_STRATEGY_PROFILE=marine_pressure"
  echo "SC2_AUTO_TRAIN_MARINES=0"
  echo "SC2_MAX_FORCED_ATTACKS=0"
  echo "SC2_PLAN_SEGMENT_MAX=6"
  echo "SC2_PLAN_BULLET_MAX=6"
  echo "SC2_LLM_ATTACK_MIN_UNITS=5"
  echo "SC2_FIRST_LLM_ATTACK_MIN_UNITS=5"
  echo "SC2_REINFORCEMENT_MIN_UNITS=5"
  echo "SC2_REINFORCEMENT_TIMER=88"
  echo "HISTORICAL_BO_BYTE_IDENTICAL=0"
  echo "DIAGNOSTIC_SCOPE=current_g2_bo_with_historical_compact_planning_contract"
  echo "SC2_PLANNER_PROTOCOL=legacy"
  echo "SC2_NEWEXEC_BASIC_HOOLIGAN=1"
  echo "SC2_NEWEXEC_HOOLIGAN_RETREAT=0"
  echo "SC2_NEWEXEC_REQUIRE_TECH_UNIT=0"
  echo "SC2_ATTACK_TARGET_UNITS_FIRST=0"
  echo "START_STEP=0"
  echo "TARGET_STEP=${TARGET_STEP}"
  echo "SAVE_FREQ=${SAVE_FREQ}"
  echo "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
  echo "VAL_ROLLOUT_N=${VAL_ROLLOUT_N}"
  echo "TRAIN_ROLLOUT_N=${TRAIN_ROLLOUT_N}"
  echo "MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}"
  echo "REQUIRE_CHECKPOINT=${REQUIRE_CHECKPOINT}"
  echo "WANDB_ENTITY=${WANDB_ENTITY}"
  echo "WANDB_PROJECT=${WANDB_PROJECT}"
  echo "WANDB_GROUP=${WANDB_GROUP}"
  echo "WANDB_RUN_ID=${WANDB_RUN_ID}"
} > "${AUDIT_DIR}/source_manifest.txt"
tar -C "${ROOT}" -czf "${AUDIT_DIR}/frozen_runtime_source.tar.gz" code
sha256sum "${AUDIT_DIR}/frozen_runtime_source.tar.gz" > "${AUDIT_DIR}/frozen_runtime_source.tar.gz.sha256"

{
  echo "STATUS=RUNNING"
  cat "${AUDIT_DIR}/source_manifest.txt"
} > "${DONE}"

set +e
timeout 604800s env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  REPO_DIR="${CODE}" \
  PYTHONPATH="${CODE}:${FLA_ROOT}:${MOBA_ROOT}:${PYTHON_SITE_ROOT}" \
  TMPDIR="${RAY_TMPDIR}" \
  RAY_TMPDIR="${RAY_TMPDIR}" \
  XDG_CACHE_HOME="${CACHE_DIR}" \
  HF_HOME="${CACHE_DIR}/huggingface" \
  TORCH_HOME="${CACHE_DIR}/torch" \
  SPB2_MODEL="${BASE_MODEL}" \
  SPB2_ROOT="${V7_ASSETS}/source/extracted" \
  PYTHON_BIN="${PYTHON_BIN}" \
  SC2PATH="${SC2PATH}" \
  RUN_NAME="${RUN_NAME}" \
  OUTPUT_ROOT="${RESULT_ROOT}" \
  LOG_DIR="${LOG_DIR}" \
  CKPT_DIR="${CKPT_DIR}" \
  DATA_DIR="${DATA_ROOT}" \
  SC2_DIFFICULTY="${DIFFICULTY_NAME}" \
  SC2_EXPECTED_DIFFICULTY="${DIFFICULTY_NAME}" \
  SC2_ENV_SEED="${ENV_SEED}" \
  SC2_EVAL_SEEDS_LOG_PATH="${AUDIT_DIR}/evaluation_seeds.jsonl" \
  SC2_TRAIN_SEEDS_LOG_PATH="${AUDIT_DIR}/training_seeds.jsonl" \
  SC2_AI_BUILD=Air \
  SC2_MAX_STEPS=20 \
  SC2_ROLLOUT_N="${TRAIN_ROLLOUT_N}" \
  SC2_BO_ADDRESS="${BO}" \
  SC2_STRATEGY_PROFILE=marine_pressure \
  SC2_AUTO_TRAIN_MARINES=0 \
  SC2_MAX_FORCED_ATTACKS=0 \
  SC2_LLM_ATTACK_MIN_UNITS=5 \
  SC2_FIRST_LLM_ATTACK_MIN_UNITS=5 \
  SC2_REINFORCEMENT_MIN_UNITS=5 \
  SC2_REINFORCEMENT_TIMER=88 \
  SC2_ATTACK_ORDER_POSITION=after_build \
  SC2_ATTACK_OPTION_ACCOUNT_ISSUED=1 \
  SC2_PLAN_SEGMENT_MAX=6 \
  SC2_PLAN_BULLET_MAX=6 \
  SC2_PLANNER_PROTOCOL=legacy \
  SC2_ENABLE_NEWEXEC_MICRO="${ENABLE_NEWEXEC_MICRO}" \
  SC2_NEWEXEC_BASIC_HOOLIGAN=1 \
  SC2_NEWEXEC_HOOLIGAN_BIO=1 \
  SC2_NEWEXEC_HOOLIGAN_ALLY_POWER_RADIUS=8 \
  SC2_NEWEXEC_HOOLIGAN_RETREAT=0 \
  SC2_NEWEXEC_REQUIRE_TECH_UNIT=0 \
  SC2_NEWEXEC_DEFENSE_MICRO=0 \
  SC2_NEWEXEC_SPECIAL_ABILITIES=1 \
  SC2_NEWEXEC_FIGHT_MICRO=1 \
  SC2_ATTACK_TARGET_UNITS_FIRST=0 \
  SC2_ATTACK_ENGAGED_RETARGET_IDLE_ONLY=1 \
  SC2_HOOLIGAN_STRUCTURE_FOCUS=guarded \
  HISTORY_LENGTH=0 \
  SC2_MEMORY_MODE=off \
  SC2_TOOL_MEMORY_PHASE=off \
  SC2_TOOL_QUERY_TRAIN=0 \
  SC2_TOOL_QUERY_POLICY=never \
  SC2_TOOL_STARTUP_QUERY_STEPS=1 \
  SC2_TOOL_QUERY_COOLDOWN=4 \
  SC2_TOOL_QUERY_STALL_STEPS=5 \
  SC2_TOOL_ATTACK_QUERY_MIN_ARMY=8 \
  SC2_TOOL_ATTACK_READY_MIN_ARMY=12 \
  SC2_TOOL_MAX_QUERIES_PER_EPISODE=1 \
  SC2_TOOL_ASSET_ONLINE_UPDATE=0 \
  SC2_TOOL_FREEZE_RETRIEVAL_AFTER_INIT=1 \
  SC2_TOOL_TRAINING_MODEL_REFLECTION=0 \
  SC2_TOOL_REFLECTION_ENABLED=0 \
  SC2_ASSET_MEMORY_LOG=0 \
  DATA_MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
  ENABLE_PROMPT_RESPONSE_LOG=1 \
  PROMPT_RESPONSE_LOG_MAX_PROMPT_CHARS=3000 \
  KILL_OLD_SC2_MAIN_THREAD=0 \
  bash "${RUNNER}" \
    +ray_init._temp_dir="${RAY_TMPDIR}" \
    +ray_init.runtime_env.env_vars.TMPDIR="${RAY_TMPDIR}" \
    +ray_init.runtime_env.env_vars.RAY_TMPDIR="${RAY_TMPDIR}" \
    +ray_init.runtime_env.env_vars.SC2_EVAL_SEEDS_LOG_PATH="${AUDIT_DIR}/evaluation_seeds.jsonl" \
    +ray_init.runtime_env.env_vars.SC2_TRAIN_SEEDS_LOG_PATH="${AUDIT_DIR}/training_seeds.jsonl" \
    +ray_init.runtime_env.env_vars.SC2_PLAN_BULLET_MAX="'6'" \
    +ray_init.runtime_env.env_vars.SC2_PLANNER_PROTOCOL="'legacy'" \
    trainer.total_epochs="${TARGET_STEP}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq=-1 \
    trainer.logger="['console']" \
    trainer.project_name=sc2-rl \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    +env.forbid_thinking_output=True \
    +env.strict_legacy_plan_contract=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}" \
    +env.val_rollout_n="${VAL_ROLLOUT_N}" \
    env.prompt_response_log_train_only=True
status=$?

OUT_FILE=${LOG_DIR}/test-${RUN_NAME}.txt
CHECKPOINT=${CKPT_DIR}/global_step_${TARGET_STEP}
final_model_shards=$(find "${CHECKPOINT}" -type f -path '*/actor/model_world_size_8_rank_*.pt' 2>/dev/null | wc -l | tr -d ' ' || true)
final_optim_shards=$(find "${CHECKPOINT}" -type f -path '*/actor/optim_world_size_8_rank_*.pt' 2>/dev/null | wc -l | tr -d ' ' || true)
final_extra_shards=$(find "${CHECKPOINT}" -type f -path '*/actor/extra_state_world_size_8_rank_*.pt' 2>/dev/null | wc -l | tr -d ' ' || true)
checkpoint_complete=0
if [[ "${status}" == 0 && -s "${CHECKPOINT}/data.pt" && "${final_model_shards}" == 8 && "${final_optim_shards}" == 8 && "${final_extra_shards}" == 8 ]]; then
  checkpoint_complete=1
fi

{
  echo "STATUS=${status}"
  echo "CHECKPOINT_COMPLETE=${checkpoint_complete}"
  cat "${AUDIT_DIR}/source_manifest.txt"
  echo "OUTPUT_LOG=${OUT_FILE}"
  echo "CHECKPOINT=${CHECKPOINT}"
  echo "MODEL_SHARDS=${final_model_shards}"
  echo "OPTIM_SHARDS=${final_optim_shards}"
  echo "EXTRA_STATE_SHARDS=${final_extra_shards}"
  echo "DATA_PT_EXISTS=$([[ -s "${CHECKPOINT}/data.pt" ]] && echo 1 || echo 0)"
} > "${DONE}"

if [[ "${REQUIRE_CHECKPOINT}" == 1 && "${checkpoint_complete}" != 1 ]]; then
  exit 9
fi
if [[ "${status}" != 0 ]]; then
  exit "${status}"
fi
exit 0
