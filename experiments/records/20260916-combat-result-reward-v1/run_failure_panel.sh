#!/usr/bin/env bash
# 48 answers on four GPUs; no games, optimizer steps, checkpoints or retries.
set -euo pipefail
test "${SC2_FAILURE_PANEL_AUTHORIZED:-0}" = 1
record=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
: "${PANEL_OUTPUT:?unused output directory}"
: "${PANEL_TMPDIR:?unused temporary directory}"
: "${MODEL_PATH:?registered original SpB}"
: "${SC2PATH:?existing SC2 assets, checked but not launched}"
: "${FEEDBACK_STORAGE_ROOT:?prepared read-only A4 mirror}"
: "${FEEDBACK_PYTHON_OVERLAY:?verified dependency path}"
: "${FEEDBACK_INCLUDE_ROOT:?verified headers}"
: "${PYTHON_BIN:?verified Python}"
export CUDA_VISIBLE_DEVICES=4,5,6,7
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
test ! -e "${PANEL_TMPDIR}"
mkdir "${PANEL_OUTPUT}" "${PANEL_TMPDIR}"
date -u +%FT%TZ > "${PANEL_OUTPUT}/started_at.txt"
deadline=$(( $(date +%s) + 1800 ))
echo "${deadline}" > "${PANEL_OUTPUT}/deadline_unix.txt"
finish() { rc=$?; echo "EXIT_CODE=${rc}" > "${PANEL_OUTPUT}/.done"; date -u +%FT%TZ >> "${PANEL_OUTPUT}/.done"; }
trap finish EXIT
export TMPDIR=${PANEL_TMPDIR} TRITON_CACHE_DIR=${PANEL_TMPDIR}/triton HF_MODULES_CACHE=${PANEL_TMPDIR}/hf_modules
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CPATH=${FEEDBACK_INCLUDE_ROOT}/python3.12:${FEEDBACK_INCLUDE_ROOT}
deps=${FEEDBACK_STORAGE_ROOT}/yangjinyue/sc2-verl-agent-experiments/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted
export PYTHONPATH=${source_dir}:${source_dir}/tests:${FEEDBACK_PYTHON_OVERLAY}:${deps}/flash-linear-attention_dev:${deps}/MoBA
export SC2_ORIGINAL_BO_SEGMENT=1 SC2_CLARIFY_BO_QUOTA=1 SPB_DISABLE_SWA_DROPOUT=1
cd "${source_dir}"
sha256sum -c configs/system-v1.sha256 > "${PANEL_OUTPUT}/source-check.log"
# Metadata-only launch preparation, reusing the exact training configuration.
timeout 45s env DRY_RUN=1 VERIFY_ASSETS=0 RUN_ID=failure-panel-config \
    OUTPUT_DIR="${PANEL_OUTPUT}/config" RAY_TMPDIR="${PANEL_TMPDIR}/unused-ray" \
    bash "${record}/run_arm.sh" A > "${PANEL_OUTPUT}/config-prepare.log" 2>&1
timeout 120s "${PYTHON_BIN}" "${source_dir}/scripts/prepare_spb_model.py" \
    --source "${MODEL_PATH}" --destination "${PANEL_OUTPUT}/config/model" > "${PANEL_OUTPUT}/model-prepare.log" 2>&1
remaining=$(( deadline - $(date +%s) - 20 )); test "${remaining}" -gt 0
timeout --signal=INT --kill-after=20s "${remaining}" "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=4 --max_restarts=0 \
    "${source_dir}/experiments/records/20260911-execution-failure-feedback-v1/fixed_feedback_compare.py" \
    --source-run "${PANEL_OUTPUT}/config" --failure-panel "${record}/failure-panel.json" \
    --output "${PANEL_OUTPUT}" > "${PANEL_OUTPUT}/console.log" 2>&1
