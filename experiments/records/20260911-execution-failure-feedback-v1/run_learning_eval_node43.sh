#!/usr/bin/env bash
# Retry proposal: same evaluations, <=119 minutes after the failed 32-second launch.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "${SC2_FEEDBACK_LEARNING_EVAL_AUTHORIZED:-0}" = 1
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
export FEEDBACK_STORAGE_ROOT=/root/new-scii-execution-feedback-local-20260911/shared
export FEEDBACK_PYTHON_OVERLAY=/root/new-scii-spb-level5-96-20260908/python
export FEEDBACK_INCLUDE_ROOT=/root/new-scii-spb-level5-96-20260908/include
export FEEDBACK_EVAL_GPUS=0,1,2,3
export SC2_FEEDBACK_AB_AUTHORIZED=1 PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${record_dir}:${source_dir}/experiments/records/20260908-g2-planner-handoff-v1:${FEEDBACK_PYTHON_OVERLAY}
runs=${FEEDBACK_STORAGE_ROOT}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs
test -z "$(nvidia-smi -i "${FEEDBACK_EVAL_GPUS}" --query-compute-apps=pid --format=csv,noheader)"
control=${EVAL_CONTROL_DIR:?set unused absolute control directory}
mkdir "${control}"
date -u +%FT%TZ > "${control}/started_at.txt"
deadline=$(( $(date +%s) + 7140 ))
finish() {
    status=$?
    date -u +%FT%TZ > "${control}/ended_at.txt"
    echo "${status}" > "${control}/done"
}
trap finish EXIT
# Does not read large tensors; distributed loading is checked later by the hook.
CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" "${record_dir}/learning_eval.py" \
    --runs "${runs}" --output "${control}/checkpoint-preflight.json"
for label in original A4 B4; do
    export FEEDBACK_LEARNING_MODEL=${label}
    export RUN_ID=20260911-execution-failure-feedback-v1-spb2-${label}-l6-feedback-on-eval8-r2
    export OUTPUT_DIR=${runs}/${RUN_ID}
    export RAY_TMPDIR=/dev/shm/fl_${label}_0911_r2
    remaining=$(( deadline - $(date +%s) - 15 ))
    test "${remaining}" -gt 0
    echo "${label}" > "${control}/current_model.txt"
    date -u +%FT%TZ > "${control}/${label}-started_at.txt"
    # No concurrent checkpoint load/save on this host; no automatic retry.
    timeout --signal=INT --kill-after=15s "${remaining}" \
        bash "${record_dir}/run_feedback_eval_node47.sh" B > "${control}/${label}.log" 2>&1
    "${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
result = json.loads((run / 'result.json').read_text())
proof = json.loads((run / 'checkpoint_evaluation.json').read_text())
assert result['status'] == 'completed' and result['process_exit_code'] == 0
assert result['training_updates'] == 0 and result['episode_outcomes']['natural_games'] == 8
assert proof['val_only'] and proof['optimizer_updates'] == 0 and proof['games'] == 8
PY
    date -u +%FT%TZ > "${control}/${label}-completed_at.txt"
    # A completed Ray process can take a few seconds to release its context.
    for _ in {1..15}; do
        test -n "$(nvidia-smi -i "${FEEDBACK_EVAL_GPUS}" --query-compute-apps=pid --format=csv,noheader)" || break
        sleep 2
    done
done
