#!/usr/bin/env bash
# Approved eight GPUs, three x eight games, one shared 60-minute deadline.
set -euo pipefail
test "${SC2_SCENE_POSTEVAL_AUTHORIZED:-0}" = 1
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
control=${SCENE_POST_CONTROL:?set unused control directory}
old_record=${source_dir}/experiments/records/20260911-execution-failure-feedback-v1
export FEEDBACK_STORAGE_ROOT=/mnt/data0601
export FEEDBACK_PYTHON_OVERLAY=/dev/shm/sc2-scene-B-retry-20260915-r2/deps/python
export FEEDBACK_INCLUDE_ROOT=/dev/shm/sc2-scene-B-retry-20260915-r2/deps/include
export PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=${old_record}:${source_dir}/experiments/records/20260908-g2-planner-handoff-v1:${FEEDBACK_PYTHON_OVERLAY}
export FEEDBACK_SCENE_POSTEVAL=1 SC2_FEEDBACK_AB_AUTHORIZED=1
unset RAY_ADDRESS FEEDBACK_SCENE_EVAL FEEDBACK_SECOND_PLAN FEEDBACK_SECOND_PLAN_VARIANT
unset FEEDBACK_TEMPERATURE_AB FEEDBACK_FIRST_ATTACK_THRESHOLD FEEDBACK_PHASE2_AIR SC2_COMMAND_TRACE_DIR
runs=/mnt/data0601/yangjinyue/new_SCII/experiments/20260914-zero-task-signal-sampling-v1/runs
checker=${record_dir}/scene_posteval_check.py
entry=${old_record}/run_feedback_eval_node47.sh
if [[ "${1:-suite}" == arm ]]; then
    export FEEDBACK_LEARNING_MODEL=${2} FEEDBACK_EVAL_GPUS=${3}
    label=${2}
    deadline=$(<"${control}/deadline_unix.txt")
    finish_arm() { rc=$?; date -u +%FT%TZ > "${control}/${label}-ended_at.txt"; echo "${rc}" > "${control}/${label}.done"; }
    trap finish_arm EXIT
    export FEEDBACK_POST_CHECKPOINT=$("${PYTHON_BIN}" "${checker}" checkpoint "${label}")
    export RUN_ID=20260914-zero-task-signal-sampling-v1-spb2-${label}-l6-panel-eval8-4gpu-r1
    export OUTPUT_DIR=${runs}/${RUN_ID} RAY_TMPDIR=/dev/shm/pe_${label}_0915_r1
    test ! -e "${OUTPUT_DIR}"
    test ! -e "${RAY_TMPDIR}"
    for _ in {1..15}; do
        test -n "$(nvidia-smi -i "${FEEDBACK_EVAL_GPUS}" --query-compute-apps=pid --format=csv,noheader)" || break
        sleep 2
    done
    test -z "$(nvidia-smi -i "${FEEDBACK_EVAL_GPUS}" --query-compute-apps=pid --format=csv,noheader)"
    remaining=$((deadline - $(date +%s) - 45)); test "${remaining}" -gt 0
    date -u +%FT%TZ > "${control}/${label}-started_at.txt"
    timeout --signal=INT --kill-after=15s "${remaining}" bash "${entry}" A > "${control}/${label}.log" 2>&1
    timeout 20s "${PYTHON_BIN}" "${checker}" finish "${label}" "${OUTPUT_DIR}" > "${control}/${label}-acceptance.json"
    exit 0
fi
mkdir "${control}"
export TMPDIR=${control}
deadline=$(( $(date +%s) + 3600 ))
echo "${deadline}" > "${control}/deadline_unix.txt"
date -u +%FT%TZ > "${control}/started_at.txt"
finish() { rc=$?; date -u +%FT%TZ > "${control}/ended_at.txt"; echo "${rc}" > "${control}/done"; }
trap finish EXIT
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$(pgrep -x raylet || true)"
"${PYTHON_BIN}" - <<'PY' > "${control}/resource-preflight.json"
import json,pathlib,subprocess
mem={s.split(':')[0]:int(s.split()[1]) for s in pathlib.Path('/proc/meminfo').read_text().splitlines()}
assert mem['MemAvailable'] > 800*1024**2, 'Need headroom for two model instances'
print(json.dumps(dict(mem_available_kib=mem['MemAvailable'],gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))))
PY
timeout 20s stat "${runs}"
timeout 30s env CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" "${checker}" preflight > "${control}/checkpoint-preflight.json"
(cd "${source_dir}" && sha256sum -c configs/system-v1.sha256) > "${control}/system-hashes.log"
cd "${source_dir}"
timeout 90s env -u FEEDBACK_SCENE_POSTEVAL CUDA_VISIBLE_DEVICES= PYTHONPATH="${source_dir}/tests:${PYTHONPATH}" "${PYTHON_BIN}" -B -m unittest test_scene_posteval test_scene_screening test_feedback_learning_eval -v > "${control}/cpu-tests.log" 2>&1
for label in A4 SceneA8 SceneB8; do
    checkpoint=$("${PYTHON_BIN}" "${checker}" checkpoint "${label}")
    timeout 30s env DRY_RUN=1 VERIFY_ASSETS=0 FEEDBACK_LEARNING_MODEL="${label}" FEEDBACK_POST_CHECKPOINT="${checkpoint}" FEEDBACK_EVAL_GPUS=0,1,2,3 \
        OUTPUT_DIR="${control}/dry-${label}" RUN_ID="dry-${label}" RAY_TMPDIR="/dev/shm/pe_${label}_0915_r1" \
        bash "${entry}" A > "${control}/dry-${label}.log" 2>&1
done
"${PYTHON_BIN}" "${checker}" compare "${control}/dry-A4" "${control}/dry-SceneA8" "${control}/dry-SceneB8" > "${control}/dry-config-comparison.json"
echo evaluating > "${control}/stage.txt"
(trap - EXIT; bash "$0" arm A4 0,1,2,3; bash "$0" arm SceneB8 0,1,2,3) & left=$!
(trap - EXIT; bash "$0" arm SceneA8 4,5,6,7) & right=$!
left_rc=0; right_rc=0
wait "${left}" || left_rc=$?
wait "${right}" || right_rc=$?
test "${left_rc}" = 0 && test "${right_rc}" = 0
"${PYTHON_BIN}" "${checker}" compare "${runs}/20260914-zero-task-signal-sampling-v1-spb2-A4-l6-panel-eval8-4gpu-r1" "${runs}/20260914-zero-task-signal-sampling-v1-spb2-SceneA8-l6-panel-eval8-4gpu-r1" "${runs}/20260914-zero-task-signal-sampling-v1-spb2-SceneB8-l6-panel-eval8-4gpu-r1" > "${control}/actual-config-comparison.json"
"${PYTHON_BIN}" - "${control}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); results=[json.loads((p/f'{label}-acceptance.json').read_text()) for label in ['A4','SceneA8','SceneB8']]
assert all(r['sampling_states']==results[0]['sampling_states'] for r in results)
with (p/'suite-acceptance.json').open('x') as f: json.dump(dict(games=24, training_updates=0, common_sampling_states=True, arms=results),f,indent=2)
PY
echo completed > "${control}/stage.txt"
