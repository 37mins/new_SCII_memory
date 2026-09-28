#!/usr/bin/env bash
# Same B4/prefix; second answer differs only in Marine count. No PPO or retry.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export FEEDBACK_SECOND_PLAN_VARIANT=marine_only
export RUN_ID=${RUN_ID:-20260911-execution-failure-feedback-v1-spb2-B4-l6-marine-count-ab8-4gpu-s1098341402-r1}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/mc_0911_r1}
exec bash "${record_dir}/run_second_plan_node43.sh"
