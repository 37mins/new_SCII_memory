#!/usr/bin/env bash
# Approved eight-GPU/15-minute issuer-order recording; no PPO/checkpoint/retry.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SC2_COMMAND_TRACE_VALIDATION=1
export RUN_ID=20260911-execution-failure-feedback-v1-spb2-original-l6-g2-commandtrace-ab8-8gpu-s52620-r2
export RAY_TMPDIR=/dev/shm/ct_0912_r2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exec bash "${record_dir}/run_supply_order_node44.sh"
fi
test "${SC2_COMMAND_TRACE_AUTHORIZED:-0}" = 1
test ! -e "${RAY_TMPDIR}"
exec timeout --signal=INT --kill-after=15s 885 bash "${record_dir}/run_supply_order_node44.sh"
