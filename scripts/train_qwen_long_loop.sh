#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CONFIG_PATH="${repo_dir}/configs/qwen3-4b-instruct-2507-level5.env"
export SC2_DIFFICULTY=Harder
export SC2_MEMORY_MODE=long_loop
export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS_PER_NODE=4
export MODEL_PATH="${MODEL_PATH:-${repo_dir}/assets/Qwen3-4B-Instruct-2507}"
export SC2PATH="${SC2PATH:-/raid/ghf/StarCraftII}"

wait_for_selected_gpus() {
	if [[ "${DRY_RUN:-0}" == 1 ]]; then
		printf 'DRY_RUN=1; skipping GPU memory wait.\n'
		return
	fi
	if ! command -v nvidia-smi >/dev/null 2>&1; then
		printf 'nvidia-smi is required for the GPU memory check.\n' >&2
		return 2
	fi

	local visible_gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
	local expected_gpus="${N_GPUS_PER_NODE:-8}"
	local -a gpu_ids
	IFS=',' read -r -a gpu_ids <<< "${visible_gpus}"
	if [[ "${#gpu_ids[@]}" != "${expected_gpus}" ]]; then
		printf 'Expected %s CUDA_VISIBLE_DEVICES entries, got %s: %s\n' "${expected_gpus}" "${#gpu_ids[@]}" "${visible_gpus}" >&2
		return 2
	fi

	local threshold_mib=35840
	local all_ready free_mib gpu_id
	while true; do
		all_ready=1
		for gpu_id in "${gpu_ids[@]}"; do
			gpu_id="${gpu_id//[[:space:]]/}"
			if ! free_mib=$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null); then
				printf 'Could not query free memory for GPU %s. Check CUDA_VISIBLE_DEVICES.\n' "${gpu_id}" >&2
				return 2
			fi
			free_mib="${free_mib//[[:space:]]/}"
			if [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
				printf 'Unexpected free-memory result for GPU %s: %s\n' "${gpu_id}" "${free_mib}" >&2
				return 2
			fi
			if (( free_mib <= threshold_mib )); then
				printf 'GPU %s has %s MiB free; need more than %s MiB.\n' "${gpu_id}" "${free_mib}" "${threshold_mib}"
				all_ready=0
			fi
		done

		if (( all_ready )); then
			printf 'All %s selected GPUs have more than 35 GiB free. Starting training.\n' "${expected_gpus}"
			return 0
		fi
		printf 'Waiting 30 minutes before checking GPU memory again.\n'
		sleep 1800
	done
}

wait_for_selected_gpus
exec bash "${repo_dir}/scripts/train_sc2.sh" "$@"
