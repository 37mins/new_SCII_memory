#!/usr/bin/env bash
# Approved A4 temperature comparison: four GPUs, <=45 minutes across both arms.
set -euo pipefail
test "${SC2_TEMPERATURE_AB_AUTHORIZED:-0}" = 1
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
control=${TEMPERATURE_CONTROL_DIR:?set unused control directory}
mkdir "${control}"
date -u +%FT%TZ > "${control}/started_at.txt"
deadline=$(( $(date +%s) + 2700 ))
finish() {
    status=$?
    date -u +%FT%TZ > "${control}/ended_at.txt"
    echo "${status}" > "${control}/done"
}
trap finish EXIT
export FEEDBACK_STORAGE_ROOT=/root/new-scii-execution-feedback-local-20260911/shared
export FEEDBACK_PYTHON_OVERLAY=/root/new-scii-spb-level5-96-20260908/python
export FEEDBACK_INCLUDE_ROOT=/root/new-scii-spb-level5-96-20260908/include
export FEEDBACK_EVAL_GPUS=0,1,2,3 FEEDBACK_LEARNING_MODEL=A4
export SC2_FEEDBACK_AB_AUTHORIZED=1 PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${record_dir}:${source_dir}/experiments/records/20260908-g2-planner-handoff-v1:${FEEDBACK_PYTHON_OVERLAY}
unset FEEDBACK_SECOND_PLAN FEEDBACK_SECOND_PLAN_VARIANT FEEDBACK_CONTINUE_A4
runs=${FEEDBACK_STORAGE_ROOT}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs
test -z "$(nvidia-smi -i "${FEEDBACK_EVAL_GPUS}" --query-compute-apps=pid --format=csv,noheader)"
(cd "${source_dir}" && sha256sum -c configs/system-v1.sha256)
CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" "${record_dir}/learning_eval.py" \
    --runs "${runs}" --output "${control}/checkpoint-preflight.json"
for temperature in 0.4 0.8; do
    suffix=${temperature/./}
    export FEEDBACK_TEMPERATURE_AB=${temperature}
    export RUN_ID=20260911-execution-failure-feedback-v1-spb2-A4-l6-temp${suffix}-eval8-s42613-r1
    export OUTPUT_DIR=${runs}/${RUN_ID} RAY_TMPDIR=/dev/shm/t${suffix}_0912_r1
    test ! -e "${OUTPUT_DIR}"
    remaining=$(( deadline - $(date +%s) - 15 ))
    test "${remaining}" -gt 0
    echo "T${temperature}" > "${control}/stage.txt"
    date -u +%FT%TZ > "${control}/T${suffix}-started_at.txt"
    timeout --signal=INT --kill-after=15s "${remaining}" \
        bash "${record_dir}/run_feedback_eval_node47.sh" B > "${control}/T${suffix}.log" 2>&1
    "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${SOURCE_SHA}" "${temperature}" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); r=json.loads((p/'result.json').read_text())
m=json.loads((p/'manifest.json').read_text()); proof=json.loads((p/'checkpoint_evaluation.json').read_text())
assert r == json.loads((p/'.done').read_text())
assert m['source_commit']==sys.argv[2] and not m['dirty']
assert r['status']=='completed' and r['process_exit_code']==0 and r['training_updates']==0
assert r['episode_outcomes']['natural_games']==8 and r['metrics_finite']
assert proof['loaded_global_step']==4 and proof['checkpoint_load_returned']
assert proof['val_only'] and proof['optimizer_updates']==0 and proof['feedback']=='1'
assert proof['game_seeds']==[42613]*8 and proof['temperature']==float(sys.argv[3])
actual=[json.loads(line)['seeds'] for line in (p/'evaluation_seeds.jsonl').read_text().splitlines()]
assert actual==[[42613]*8]
PY
    date -u +%FT%TZ > "${control}/T${suffix}-completed_at.txt"
    for _ in {1..15}; do
        test -n "$(nvidia-smi -i "${FEEDBACK_EVAL_GPUS}" --query-compute-apps=pid --format=csv,noheader)" || break
        sleep 2
    done
done
echo completed > "${control}/stage.txt"
