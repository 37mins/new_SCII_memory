#!/usr/bin/env bash
set -euo pipefail

SOURCE_COMMIT=${SOURCE_COMMIT:?SOURCE_COMMIT is required}
DEPLOYMENT_TAR_SHA=${DEPLOYMENT_TAR_SHA:?DEPLOYMENT_TAR_SHA is required}
RECORD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments
RUN_ID=20260830-v1524-micro-level5-rl-v1-qwen-base-l5-full96-s0-r1
RUN_ROOT=${PROJECT_ROOT}/runs/20260830-v1524-micro-level5-rl-v1/${RUN_ID}
RUNTIME_ROOT=/dev/shm/v1524_l5_rl_s0_r1

test ! -e "${RUN_ROOT}"
test ! -e "${RUNTIME_ROOT}"
cd "${RECORD_DIR}"
sha256sum --status -c record.sha256
mkdir -p "${RUN_ROOT}"
cp -a "${RECORD_DIR}" "${RUN_ROOT}/record"

export NODE_TAG=sc2-a100-42
export GPU_ID=0,1,2,3,4,5,6,7
export RUN_ID
export RAY_TAG=v1524l5s0
export NEW_REPO_COMMIT=${SOURCE_COMMIT}
export EFFECTIVE_CONFIG_SHA=$(sha256sum "${RECORD_DIR}/effective_config.json" | awk '{print $1}')
export EVAL_LAUNCHER_PATCH_SHA=$(sha256sum "${RECORD_DIR}/launcher_eval_only.patch" | awk '{print $1}')
export FULL_NODE_PATCH_SHA=$(sha256sum "${RECORD_DIR}/launcher_full_node.patch" | awk '{print $1}')
export VALIDATION_SEED_PATCH_SHA=$(sha256sum "${RECORD_DIR}/independent_validation_seeds.patch" | awk '{print $1}')
export QWEN_RL_PATCH_SHA=$(sha256sum "${RECORD_DIR}/launcher_qwen_rl.patch" | awk '{print $1}')
export TRAINING_SEED_PATCH_SHA=$(sha256sum "${RECORD_DIR}/training_seed_ledger.patch" | awk '{print $1}')
export HF_ROLLOUT_OFFLOAD_PATCH_SHA=$(sha256sum "${RECORD_DIR}/verl_hf_rollout_offload.patch" | awk '{print $1}')
export CHAT_TEMPLATE_CONTROL_PATCH_SHA=$(sha256sum "${RECORD_DIR}/launcher_disable_thinking.patch" | awk '{print $1}')
export CHAT_TEMPLATE_CANARY_SHA=$(sha256sum "${RECORD_DIR}/chat_template_canary.json" | awk '{print $1}')
export RL_REWARD_BALANCE_PATCH_SHA=$(sha256sum "${RECORD_DIR}/rl_reward_balance.patch" | awk '{print $1}')
export QWEN_BASE_CHECKPOINT_PATCH_SHA=$(sha256sum "${RECORD_DIR}/qwen_base_checkpoint.patch" | awk '{print $1}')
export CANONICAL_LAUNCHER_PATCH_SHA=$(sha256sum "${RECORD_DIR}/launcher_canonical_planner.patch" | awk '{print $1}')
export CANONICAL_PLANNER_SHA=$(sha256sum "${RECORD_DIR}/canonical_planner.py" | awk '{print $1}')
export STRICT_MANAGER_PATCH_SHA=$(sha256sum "${RECORD_DIR}/strict_legacy_plan_validator.patch" | awk '{print $1}')
export RUNTIME_PROVENANCE_PATCH_SHA=$(sha256sum "${RECORD_DIR}/launcher_runtime_provenance.patch" | awk '{print $1}')
export CONTAINER_IMAGE_ID=not-used-on-shared-node
export CONTAINER_MOUNT_ROOT=${PROJECT_ROOT}
export DEPLOYMENT_MANIFEST_SHA=${DEPLOYMENT_TAR_SHA}
export PROJECT_ROOT_OVERRIDE=${PROJECT_ROOT}
export PYTHON_SITE_ROOT_OVERRIDE=${PROJECT_ROOT}/envs/verl-agent-v7-exact/lib/python3.12/site-packages
export MODEL_ASSET_ROOT=${PROJECT_ROOT}/assets/models/Qwen/Qwen3-4B
export RUNTIME_ROOT_OVERRIDE=${RUNTIME_ROOT}
export RESULT_ROOT_OVERRIDE=${RUN_ROOT}/results
export MICRO=v1524
export ENV_SEED=0
export VAL_ROLLOUT_N=1
export TRAIN_ROLLOUT_N=8
export SAVE_FREQ=96
export TARGET_STEP=96
export REQUIRE_CHECKPOINT=1
export MAX_PROMPT_LENGTH=4096
export DIFFICULTY_NAME=Hard
export DIFFICULTY_LEVEL=5
export MAX_RESPONSE_LENGTH=96
export EXPERIMENT_TAG=qwen_base_l5_canonical_v1524_full96
export WANDB_GROUP=disabled

exec bash "${RUN_ROOT}/record/launcher.sh"
