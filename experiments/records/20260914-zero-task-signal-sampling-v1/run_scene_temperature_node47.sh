#!/usr/bin/env bash
# Approved: 47 GPU4-7, SceneB8 T0.4 then T0.8, 8 games each, 45 minutes total.
set -euo pipefail
test "${SC2_SCENE_TEMPERATURE_AUTHORIZED:-0}" = 1
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
control=${SCENE_TEMPERATURE_CONTROL:?set unused control directory}
mkdir "${control}"
deadline=$(( $(date +%s) + 2700 ))
echo "${deadline}" > "${control}/deadline_unix.txt"
date -u +%FT%TZ > "${control}/started_at.txt"
finish_suite() { rc=$?; date -u +%FT%TZ > "${control}/ended_at.txt"; echo "${rc}" > "${control}/done"; }
trap finish_suite EXIT
old_record=${source_dir}/experiments/records/20260911-execution-failure-feedback-v1
export FEEDBACK_STORAGE_ROOT=/mnt/data0601
export FEEDBACK_PYTHON_OVERLAY=/dev/shm/sc2-scene-B-retry-20260915-r2/deps/python
export FEEDBACK_INCLUDE_ROOT=/dev/shm/sc2-scene-B-retry-20260915-r2/deps/include
export PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=${control}
export PYTHONPATH=${record_dir}:${old_record}:${source_dir}/experiments/records/20260908-g2-planner-handoff-v1:${FEEDBACK_PYTHON_OVERLAY}
export FEEDBACK_SCENE_POSTEVAL=1 SC2_FEEDBACK_AB_AUTHORIZED=1 FEEDBACK_LEARNING_MODEL=SceneB8 FEEDBACK_EVAL_GPUS=4,5,6,7
unset RAY_ADDRESS FEEDBACK_SCENE_EVAL FEEDBACK_SECOND_PLAN FEEDBACK_SECOND_PLAN_VARIANT
unset FEEDBACK_TEMPERATURE_AB FEEDBACK_FIRST_ATTACK_THRESHOLD FEEDBACK_PHASE2_AIR SC2_COMMAND_TRACE_DIR
unset FEEDBACK_SCENE_TEMPERATURE
runs=/mnt/data0601/yangjinyue/new_SCII/experiments/20260914-zero-task-signal-sampling-v1/runs
checker=${record_dir}/scene_temperature_check.py
entry=${old_record}/run_feedback_eval_node47.sh
test -z "$(nvidia-smi -i "${FEEDBACK_EVAL_GPUS}" --query-compute-apps=pid --format=csv,noheader)"
test -z "$(pgrep -x raylet || true)"
"${PYTHON_BIN}" - <<'PY' > "${control}/resource-preflight.json"
import json,pathlib,subprocess
mem={s.split(':')[0]:int(s.split()[1]) for s in pathlib.Path('/proc/meminfo').read_text().splitlines()}
assert mem['MemAvailable'] > 600*1024**2, 'Need host-memory headroom for one model instance'
print(json.dumps(dict(mem_available_kib=mem['MemAvailable'],gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))))
PY
timeout 10s stat "${runs}"
timeout 30s env CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" "${checker}" preflight > "${control}/checkpoint-preflight.json"
export FEEDBACK_POST_CHECKPOINT=$("${PYTHON_BIN}" "${record_dir}/scene_posteval_check.py" checkpoint SceneB8)
cd "${source_dir}"
sha256sum -c configs/system-v1.sha256 > "${control}/system-hashes.log"
timeout 90s env -u FEEDBACK_SCENE_POSTEVAL CUDA_VISIBLE_DEVICES= PYTHONPATH="${source_dir}/tests:${PYTHONPATH}" "${PYTHON_BIN}" -B -m unittest test_scene_temperature test_scene_posteval -v > "${control}/cpu-tests.log" 2>&1
for temperature in 0.4 0.8; do
    label=${temperature/./}
    timeout 30s env DRY_RUN=1 VERIFY_ASSETS=0 FEEDBACK_SCENE_TEMPERATURE="${temperature}" OUTPUT_DIR="${control}/dry-${label}" RUN_ID="dry-${label}" RAY_TMPDIR="/dev/shm/st${label}_0915_r1" bash "${entry}" A > "${control}/dry-${label}.log" 2>&1
done
"${PYTHON_BIN}" "${checker}" compare "${control}/dry-04" "${control}/dry-08" > "${control}/dry-config-comparison.json"
arm() (
    temperature=${1}; label=${temperature/./}
    finish_arm() { rc=$?; date -u +%FT%TZ > "${control}/${label}-ended_at.txt"; echo "${rc}" > "${control}/${label}.done"; }
    trap finish_arm EXIT
    export FEEDBACK_SCENE_TEMPERATURE=${temperature}
    export RUN_ID=20260914-zero-task-signal-sampling-v1-spb2-SceneB8-l6-temp${label}-eval8-s1436145665-r1
    export OUTPUT_DIR=${runs}/${RUN_ID} RAY_TMPDIR=/dev/shm/st${label}_0915_r1
    test ! -e "${OUTPUT_DIR}"; test ! -e "${RAY_TMPDIR}"
    for _ in {1..15}; do
        test -n "$(nvidia-smi -i "${FEEDBACK_EVAL_GPUS}" --query-compute-apps=pid --format=csv,noheader)" || break
        sleep 2
    done
    test -z "$(nvidia-smi -i "${FEEDBACK_EVAL_GPUS}" --query-compute-apps=pid --format=csv,noheader)"
    remaining=$((deadline - $(date +%s) - 45)); test "${remaining}" -gt 0
    date -u +%FT%TZ > "${control}/${label}-started_at.txt"
    timeout --signal=INT --kill-after=15s "${remaining}" bash "${entry}" A > "${control}/${label}.log" 2>&1
    timeout 20s "${PYTHON_BIN}" "${checker}" finish "${temperature}" "${OUTPUT_DIR}" > "${control}/${label}-acceptance.json"
)
echo evaluating > "${control}/stage.txt"
arm 0.4
arm 0.8
"${PYTHON_BIN}" "${checker}" compare "${runs}/20260914-zero-task-signal-sampling-v1-spb2-SceneB8-l6-temp04-eval8-s1436145665-r1" "${runs}/20260914-zero-task-signal-sampling-v1-spb2-SceneB8-l6-temp08-eval8-s1436145665-r1" > "${control}/actual-config-comparison.json"
"${PYTHON_BIN}" - "${control}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); results=[json.loads((p/f'{label}-acceptance.json').read_text()) for label in ['04','08']]
assert results[0]['sampling_states']==results[1]['sampling_states']
with (p/'suite-acceptance.json').open('x') as f: json.dump(dict(games=16,training_updates=0,common_sampling_states=True,arms=results),f,indent=2)
PY
echo completed > "${control}/stage.txt"
