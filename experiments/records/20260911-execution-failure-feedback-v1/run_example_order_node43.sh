#!/usr/bin/env bash
# Fixed-input probe; explicit per-mode authorization, no SC2/PPO/retry.
set -euo pipefail
if test "${ORIGINAL_A4_PROBE:-0}" = 1; then
    test "${ORIGINAL_A4_AUTHORIZED:-0}" = 1
    export RUN_ID=20260911-execution-failure-feedback-v1-spb2-original-for-A4-fixed1x8-4gpu-r1
    export TMPDIR=/dev/shm/sc2_original_a4_0912_r1
    seconds=900
else
    test "${EXAMPLE_ORDER_AUTHORIZED:-0}" = 1
    export RUN_ID=20260911-execution-failure-feedback-v1-spb2-A4-exampleorder-fixed1x8-4gpu-r1
    export TMPDIR=/dev/shm/sc2_example_order_0912_r1
    seconds=1200
fi
record=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
export CUDA_VISIBLE_DEVICES=0,1,2,3
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
storage=/root/new-scii-execution-feedback-local-20260911/shared
runs=${storage}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs
source_run=${runs}/20260911-execution-failure-feedback-v1-spb2-A4-l6-firstattack40-eval4-s42613-r1
output=${runs}/${RUN_ID}
test ! -e "${output}"
test ! -e "${TMPDIR}"
mkdir "${output}" "${TMPDIR}"
date -Is > "${output}/started_at.txt"
deadline=$(( $(date +%s) + seconds ))
printf '%s\n' "${deadline}" > "${output}/deadline_epoch.txt"
finish() {
    status=$?
    printf 'EXIT_CODE=%s\n' "${status}" > "${output}/.done"
    date -Is >> "${output}/.done"
}
trap finish EXIT
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
if test "${ORIGINAL_A4_PROBE:-0}" = 1; then
    paired_run=${runs}/20260911-execution-failure-feedback-v1-spb2-A4-exampleorder-fixed1x8-4gpu-r1
    probe_args=(--paired-a4-run "${paired_run}")
    CUDA_VISIBLE_DEVICES= /usr/bin/python3 "${record}/original_a4_probe.py" \
      --source-run "${source_run}" "${probe_args[@]}" > "${output}/input-cpu-check.log"
else
    probe_args=(--example-order)
    CUDA_VISIBLE_DEVICES= /usr/bin/python3 "${record}/example_order_probe.py" \
      --source-run "${source_run}" > "${output}/input-cpu-check.log"
fi
remaining=$(( deadline - $(date +%s) - 20 ))
test "${remaining}" -gt 0
timeout --signal=INT --kill-after=20s "${remaining}" /usr/bin/python3 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 --max_restarts=0 \
  "${record}/fixed_feedback_compare.py" "${probe_args[@]}" --source-run "${source_run}" --output "${output}" \
  > "${output}/console.log" 2>&1
