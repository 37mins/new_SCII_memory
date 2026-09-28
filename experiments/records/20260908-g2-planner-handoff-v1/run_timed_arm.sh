#!/usr/bin/env bash
# Independent four-GPU arm, sharing an absolute deadline across the two nodes.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
remaining=$(( ${PAIR_DEADLINE:?set shared UTC epoch deadline} - $(date +%s) - 60 ))
test "${remaining}" -gt 0 && test "${remaining}" -le 3540
status_path=${ARM_STATUS_PATH:?set a new status file}
test ! -e "${status_path}"
date -Is
set +e
timeout --signal=INT --kill-after=60s "${remaining}" bash "${record_dir}/run_eval.sh" "${1:?arm A or B}"
arm_status=$?
printf 'EXIT_CODE=%s\n' "${arm_status}" > "${status_path}"
date -Is >> "${status_path}"
exit "${arm_status}"
