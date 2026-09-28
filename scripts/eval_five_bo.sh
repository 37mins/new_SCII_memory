#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${RUN_ID:?set a unique suite RUN_ID}"
: "${FIVEBO_ROOT:?set the directory holding the five registered BO files}"
result_root=${RESULT_ROOT:-${repo_dir}/runs/${RUN_ID}}
mkdir -p "$(dirname "${result_root}")"
mkdir "${result_root}"
read -r -a gpu_ids <<< "${EVAL_GPU_IDS:-0 1 2 3 4}"
[[ "${#gpu_ids[@]}" == 5 ]] || { echo 'EVAL_GPU_IDS must contain five GPU IDs' >&2; exit 2; }
pids=()
for i in 0 1 2 3 4; do
    bo_key=g$((i + 1))
    id=${RUN_ID}-${bo_key}
    ray_tag=$(printf '%s' "${id}" | shasum -a 256 | cut -c1-12)
    CUDA_VISIBLE_DEVICES=${gpu_ids[$i]} BO_PROFILE=fivebo BO_KEY=${bo_key} EVAL_ONLY=1 \
        RUN_ID=${id} OUTPUT_DIR=${result_root}/${bo_key} RAY_TMPDIR=/dev/shm/r_${ray_tag} \
        timeout 7200s bash "${repo_dir}/scripts/train_sc2.sh" > "${result_root}/${bo_key}.driver.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
printf 'STATUS=%s\n' "${status}" > "${result_root}/suite.done"
exit "${status}"
