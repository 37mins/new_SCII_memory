#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CONFIG_PATH="${repo_dir}/configs/spikingbrain2-level5.env"
exec bash "${repo_dir}/scripts/train_sc2.sh"
