#!/usr/bin/env bash
set -euo pipefail
export R4_REPLAY_NODE=47
export R4_REPLAY_PYTHON_DIR=/dev/shm/spb-unitid-20260906/python
export CPATH=/dev/shm/spb-unitid-20260906/include/python3.12:/dev/shm/spb-unitid-20260906/include
export R4_REPLAY_ATTEMPT=2 R4_REPLAY_MAX_SECONDS=1680
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_r4_first_update_node46.sh" "$@"
