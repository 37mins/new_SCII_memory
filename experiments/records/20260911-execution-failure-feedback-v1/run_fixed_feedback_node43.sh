#!/usr/bin/env bash
# Approved four-GPU, 20-minute attempt; two saved contexts, no SC2/PPO/retry.
set -euo pipefail
record=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
test "${FIXED_FEEDBACK_AUTHORIZED:-0}" = 1
export CUDA_VISIBLE_DEVICES=0,1,2,3
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
storage=/root/new-scii-execution-feedback-local-20260911/shared
runs=${storage}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs
source_run=${runs}/20260911-execution-failure-feedback-v1-spb2-l7s24-l6-g2-B-eval4-s0-r3
export RUN_ID=20260911-execution-failure-feedback-v1-spb2-original-vs-l7s24-fixed2x4-4gpu-r1
output=${runs}/${RUN_ID}
export TMPDIR=/dev/shm/sc2_fixed_feedback_0911_r1
test ! -e "${output}"
test ! -e "${TMPDIR}"
mkdir "${output}" "${TMPDIR}"
date -Is > "${output}/started_at.txt"
deadline=$(( $(date +%s) + 1200 ))
printf '%s\n' "${deadline}" > "${output}/deadline_epoch.txt"
cp "${BASH_SOURCE[0]}" "${output}/launcher.sh"
export HF_MODULES_CACHE=${TMPDIR}/hf_modules TRITON_CACHE_DIR=${TMPDIR}/triton
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
include=/root/new-scii-spb-level5-96-20260908/include
export CPATH=${include}/python3.12:${include}
deps=${storage}/yangjinyue/sc2-verl-agent-experiments/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted
export PYTHONPATH=${source_dir}:${source_dir}/tests:/root/new-scii-spb-level5-96-20260908/python:${deps}/flash-linear-attention_dev:${deps}/MoBA
export SC2_ORIGINAL_BO_SEGMENT=1 SC2_CLARIFY_BO_QUOTA=1
cd "${source_dir}"
sha256sum -c configs/system-v1.sha256 > "${output}/source-check.log"
set +e
remaining=$(( deadline - $(date +%s) - 20 ))
timeout --signal=INT --kill-after=20s "${remaining}" /usr/bin/python3 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 --max_restarts=0 \
  "${record}/fixed_feedback_compare.py" --source-run "${source_run}" --output "${output}" \
  > "${output}/console.log" 2>&1
status=$?
printf 'EXIT_CODE=%s\n' "${status}" > "${output}/.done"
date -Is >> "${output}/.done"
exit "${status}"
