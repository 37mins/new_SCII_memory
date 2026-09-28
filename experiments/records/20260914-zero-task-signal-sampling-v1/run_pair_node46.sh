#!/usr/bin/env bash
# Approved four-card relocation, native A4 shards; no checkpoint conversion.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=/dev/shm/sc2-v2-pair-node46-20260915-r1
export SIGNAL_DEADLINE_UNIX=$(cat "${root}/deadline_unix.txt")
export SIGNAL_CONTROL_DIR=${root}/pair-r2
export SIGNAL_REPLICATE=r2 SC2_SCENE_SAMPLING=0
export FEEDBACK_STORAGE_ROOT=${root}/shared
export FEEDBACK_PYTHON_OVERLAY=/root/new-scii-spb-unitid-20260906/python
export FEEDBACK_INCLUDE_ROOT=/usr/include PYTHON_BIN=/usr/bin/python3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=${root}/hf TRITON_CACHE_DIR=${root}/triton
remaining=$(( SIGNAL_DEADLINE_UNIX - $(date +%s) - 15 ))
test "${remaining}" -gt 0
exec timeout --signal=INT --kill-after=15s "${remaining}" \
    bash "${record_dir}/run_pair_node43.sh"
