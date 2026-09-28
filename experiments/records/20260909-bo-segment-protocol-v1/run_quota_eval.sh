#!/usr/bin/env bash
# Both arms use the original-count protocol; only the quota prompt flag differs.
set -euo pipefail
record=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "${1:?arm A or B}" in
  A) export SC2_CLARIFY_BO_QUOTA=0 ;;
  B) export SC2_CLARIFY_BO_QUOTA=1 ;;
  *) exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=4,5,6,7 SC2_EVAL_N_GPUS=4 SC2_EVAL_RESPONSE_LENGTH=192
exec bash "${record}/run_eval.sh" B
