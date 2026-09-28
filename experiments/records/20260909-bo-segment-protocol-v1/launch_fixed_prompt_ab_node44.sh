#!/usr/bin/env bash
# One approved 30-minute attempt, four GPUs; saved inputs only, no SC2/PPO.
set -euo pipefail
record=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
(cd "${source_dir}" && sha256sum -c configs/system-v1.sha256)
export CUDA_VISIBLE_DEVICES=4,5,6,7
test -z "$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader)"
export RUN_ID=20260909-bo-segment-protocol-v1-spb2-fixed41-quota-prompt-ab-4gpu-s0-r1
run_root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260909-bo-segment-protocol-v1/runs
output=${run_root}/${RUN_ID}
source_run=${run_root}/20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-B192-8gpu-eval4-s0-r1
test ! -e "${output}"
test ! -e /dev/shm/sc2_prompt_ab_r1
mkdir "${output}" /dev/shm/sc2_prompt_ab_r1
date -Is > "${output}/started_at.txt"
deadline=$(( $(date +%s) + 1800 ))
printf '%s\n' "${deadline}" > "${output}/deadline_epoch.txt"
cp "${BASH_SOURCE[0]}" "${output}/launcher.sh"
export TMPDIR=/dev/shm/sc2_prompt_ab_r1
export HF_MODULES_CACHE=${TMPDIR}/hf_modules TRITON_CACHE_DIR=${TMPDIR}/triton
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CPATH=/usr/include/python3.12:/usr/include
deps=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/repro/stageguard_v7_legacy_contract_20260806/source/extracted
export PYTHONPATH=${source_dir}:${source_dir}/tests:/root/new-scii-spb2-level5-control/python:${deps}/flash-linear-attention_dev:${deps}/MoBA
export SC2_ORIGINAL_BO_SEGMENT=1
cd "${source_dir}"
set +e
remaining=$(( deadline - $(date +%s) - 60 ))
timeout --signal=INT --kill-after=60s "${remaining}" /usr/bin/python3 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 --max_restarts=0 \
  "${record}/fixed_prompt_ab.py" --source-run "${source_run}" --output "${output}" \
  > "${output}/console.log" 2>&1
status=$?
printf 'EXIT_CODE=%s\n' "${status}" > "${output}/.done"
date -Is >> "${output}/.done"
exit "${status}"
