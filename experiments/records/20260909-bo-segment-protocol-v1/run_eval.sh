#!/usr/bin/env bash
# Proposed pure evaluation only; a fresh GPU authorization is required.
set -euo pipefail
source_dir=${SOURCE_DIR:?set an independent frozen source directory}
case "${1:?arm A or B}" in
  A) export SC2_ORIGINAL_BO_SEGMENT=0 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} ;;
  B) export SC2_ORIGINAL_BO_SEGMENT=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7} ;;
  *) exit 2 ;;
esac
# Reuse the existing four-game seed/model/environment entry for BOTH arms.
# Its B selects the handoff value below; it does not select our protocol arm.
export SC2_HANDOFF_B_PREFIX=-1
entry=${source_dir}/experiments/records/20260908-g2-planner-handoff-v1/run_eval.sh
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exec bash "${entry}" B
fi
remaining=$(( ${PAIR_DEADLINE:?set the authorized shared UTC deadline} - $(date +%s) - 60 ))
test "${remaining}" -gt 0 && test "${remaining}" -le 2940
status_path=${ARM_STATUS_PATH:?set a new status file}
test ! -e "${status_path}"
set +e
timeout --signal=INT --kill-after=60s "${remaining}" bash "${entry}" B
arm_status=$?
printf 'EXIT_CODE=%s\n' "${arm_status}" > "${status_path}"
date -Is >> "${status_path}"
exit "${arm_status}"
