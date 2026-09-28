#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${RUN_ID:?RUN_ID is required}
DEPLOYMENT_TAR=${DEPLOYMENT_TAR:?DEPLOYMENT_TAR is required}
SOURCE_COMMIT=${SOURCE_COMMIT:?SOURCE_COMMIT is required}
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/data0601/yangjinyue/sc2-verl-agent-experiments}
RESULT_ROOT=${RESULT_ROOT:-${PROJECT_ROOT}/runs/20260829-v1524-micro-level6-fivebo}
FROZEN_RUNTIME_TAR=${FROZEN_RUNTIME_TAR:-${PROJECT_ROOT}/assets/repro/stageguard_v7_legacy_contract_20260806/runtime/frozen_runtime_source.tar.gz}
V1524_SOURCE=${V1524_SOURCE:-${PROJECT_ROOT}/assets/repro/v1524_micro_20260725/script_player.v1524.no_merge_into_attacking.py}
FIVEBO_ROOT=${FIVEBO_ROOT:-${PROJECT_ROOT}/assets/repro/mse_micro_five_bo_20260711/fixed_bos}
MODEL_PATH=${MODEL_PATH:-${PROJECT_ROOT}/assets/models/Qwen/Qwen3-4B}
SC2PATH=${SC2PATH:-${PROJECT_ROOT}/assets/sc2/StarCraftII}
PYTHON_BIN=${PYTHON_BIN:-${PROJECT_ROOT}/envs/verl-agent-v7-exact/bin/python}
DIFFICULTY_NAME=${DIFFICULTY_NAME:-Harder}
COMMON_SEED=${COMMON_SEED:-}
ENABLE_NEWEXEC_MICRO=${ENABLE_NEWEXEC_MICRO:-1}
RECORD_REL=experiments/records/20260829-v1524-micro-level6-fivebo
SCRIPT_REL=agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/players/script_player.py

test -x "${PYTHON_BIN}"
test "$(sha256sum "${FROZEN_RUNTIME_TAR}" | awk '{print $1}')" = 016772ab0cd8f8cbc3d017150cb1b7c17cb3f40984accdfd14a2e8b6a9b76cb0
test "$(sha256sum "${V1524_SOURCE}" | awk '{print $1}')" = 8c1c48126e444ecf604a7ac79989c5c9e8dd06eaca31a5e8786b0e5f16ae4e8c

mkdir -p "${RESULT_ROOT}"

run_one() {
  local bo=$1 seed=$2 gpu=$3 bo_file=$4
  local runtime=/dev/shm/${RUN_ID}_${bo}
  local frozen=/dev/shm/${RUN_ID}_${bo}_frozen
  local result_dir=${RESULT_ROOT}/${RUN_ID}-${bo}-s${seed}
  local ray_tag
  ray_tag=$(printf '%s' "${RUN_ID}-${bo}" | sha256sum | cut -c1-8)

  test ! -e "${runtime}"
  test ! -e "${frozen}"
  test ! -e "${result_dir}"
  mkdir -p "${runtime}/code" "${frozen}"
  tar -C "${runtime}/code" -xzf "${DEPLOYMENT_TAR}"
  tar -C "${frozen}" -xzf "${FROZEN_RUNTIME_TAR}" "code/${SCRIPT_REL}"
  cp "${frozen}/code/${SCRIPT_REL}" "${runtime}/code/${SCRIPT_REL}"

  "${PYTHON_BIN}" "${runtime}/code/${RECORD_REL}/verify_v1524_compat.py" \
    "${runtime}/code/${SCRIPT_REL}" "${V1524_SOURCE}" \
    > "${runtime}/compat.txt"

  CUDA_VISIBLE_DEVICES=${gpu} \
  PATH="$(dirname "${PYTHON_BIN}"):${PATH}" \
  BO_KEY=${bo} \
  BO_PATH="${FIVEBO_ROOT}/${bo_file}" \
  ENV_SEED=${seed} \
  DIFFICULTY_NAME=${DIFFICULTY_NAME} \
  ENABLE_NEWEXEC_MICRO=${ENABLE_NEWEXEC_MICRO} \
  RUN_ID=${RUN_ID} \
  CODE_ROOT="${runtime}/code" \
  RESULT_ROOT=${RESULT_ROOT} \
  MODEL_PATH=${MODEL_PATH} \
  SC2PATH=${SC2PATH} \
  COMPAT_PATH="${runtime}/compat.txt" \
  SOURCE_COMMIT=${SOURCE_COMMIT} \
  DEPLOYMENT_TAR_SHA="$(sha256sum "${DEPLOYMENT_TAR}" | awk '{print $1}')" \
  RAY_TMPDIR="/dev/shm/r_${ray_tag}" \
  bash "${runtime}/code/${RECORD_REL}/launcher.sh"
}

run_one g1 "${COMMON_SEED:-752}" 0 replay_observations_Flat128_d4_TvT_Macro_g1_Result.Victory_t.json & p1=$!
run_one g2 "${COMMON_SEED:-757}" 1 replay_observations_Flat128_d4_TvT_Macro_g2_Result.Victory_t.json & p2=$!
run_one g3 "${COMMON_SEED:-756}" 2 replay_observations_Flat128_d4_TvT_Macro_g3_Result.Victory_n.json & p3=$!
run_one g4 "${COMMON_SEED:-781}" 3 replay_observations_Flat128_d4_TvT_Macro_g4_Result.Victory_s.json & p4=$!
run_one g5 "${COMMON_SEED:-755}" 4 replay_observations_Flat128_d4_TvT_Macro_g5_Result.Victory_n.json & p5=$!

status=0
for pid in "${p1}" "${p2}" "${p3}" "${p4}" "${p5}"; do
  wait "${pid}" || status=1
done
printf 'STATUS=%s\n' "${status}" > "${RESULT_ROOT}/${RUN_ID}.done"
exit "${status}"
