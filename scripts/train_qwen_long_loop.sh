#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CONFIG_PATH="${repo_dir}/configs/qwen3-4b-instruct-2507-level5.env"
export SC2_DIFFICULTY=Harder
export SC2_MEMORY_MODE=long_loop
export MODEL_PATH="${MODEL_PATH:-${repo_dir}/assets/Qwen3-4B-Instruct-2507}"
export SC2PATH="${SC2PATH:-/raid/ghf/StarCraftII}"
exec bash "${repo_dir}/scripts/train_sc2.sh" "$@"
