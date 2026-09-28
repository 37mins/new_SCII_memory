#!/usr/bin/env bash
# Approved node change: same experiment, new IDs, at most 57 minutes total.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export FEEDBACK_PYTHON_OVERLAY=/root/new-scii-spb-level5-96-20260908/python
export FEEDBACK_INCLUDE_ROOT=/root/new-scii-spb-level5-96-20260908/include
export FEEDBACK_REPLICATE=${FEEDBACK_REPLICATE:-r2} FEEDBACK_PAIR_SECONDS=3400
exec timeout --signal=INT --kill-after=15s 3405 \
    bash "${record_dir}/run_feedback_pair_node47.sh"
