#!/usr/bin/env bash
# Serial A/B, four games each, one shared 45-minute deadline, no PPO.
set -euo pipefail
test "${SC2_PHASE2_AIR_AUTHORIZED:-0}" = 1
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
feedback_dir=${source_dir}/experiments/records/20260911-execution-failure-feedback-v1
export FEEDBACK_STORAGE_ROOT=/root/new-scii-execution-feedback-local-20260911/shared
export FEEDBACK_PYTHON_OVERLAY=/root/new-scii-spb-level5-96-20260908/python
export FEEDBACK_INCLUDE_ROOT=/root/new-scii-spb-level5-96-20260908/include
export FEEDBACK_EVAL_GPUS=0,1,2,3 FEEDBACK_LEARNING_MODEL=A4
export SC2_FEEDBACK_AB_AUTHORIZED=1 PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${record_dir}:${feedback_dir}:${source_dir}/experiments/records/20260908-g2-planner-handoff-v1:${FEEDBACK_PYTHON_OVERLAY}
unset FEEDBACK_SCENE_EVAL FEEDBACK_SECOND_PLAN FEEDBACK_SECOND_PLAN_VARIANT FEEDBACK_TEMPERATURE_AB FEEDBACK_FIRST_ATTACK_THRESHOLD SC2_COMMAND_TRACE_DIR
test -z "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)"
control=${EVAL_CONTROL_DIR:?set unused control directory}
mkdir "${control}"
deadline=$(( $(date +%s) + 2700 ))
finish() {
    status=$?
    date -u +%FT%TZ > "${control}/ended_at.txt"
    echo "${status}" > "${control}/done"
}
trap finish EXIT
date -u +%FT%TZ > "${control}/started_at.txt"
echo "${deadline}" > "${control}/deadline_unix.txt"
old_runs=${FEEDBACK_STORAGE_ROOT}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs
runs=${FEEDBACK_STORAGE_ROOT}/yangjinyue/new_SCII/experiments/20260913-phase2-air-options-v1/runs
# Metadata and file sizes only; actual distributed restoration is checked below.
CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" - "${old_runs}" "${control}" <<'PY'
import json, sys
from pathlib import Path
from continue_a4 import PARENT_NAME, PARENT_SHA
runs,control=map(Path,sys.argv[1:]); run=runs/PARENT_NAME
result=json.loads((run/'result.json').read_text())
manifest=json.loads((run/'manifest.json').read_text())
assert result==json.loads((run/'.done').read_text())
assert result['status']=='completed' and result['process_exit_code']==0 and result['checkpoint_complete']
assert result['last_global_step']==result['target_steps']==4
assert manifest['source_commit']==PARENT_SHA and not manifest['dirty'] and manifest['world_size']==4
assert manifest['model_manifest_sha256']=='40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c'
for name,meta in result['checkpoint_files'].items():
    assert (run/name).stat().st_size==meta['bytes']
with (control/'checkpoint-preflight.json').open('x') as stream:
    json.dump(dict(run=str(run),source_sha=PARENT_SHA,step=4,shards_stat_passed=True),stream,indent=2)
PY
for arm in A B; do
    export FEEDBACK_PHASE2_AIR=${arm}
    export RUN_ID=20260913-phase2-air-options-v1-spb2-A4-l6-${arm}-eval4-4gpu-r1
    export OUTPUT_DIR=${runs}/${RUN_ID} RAY_TMPDIR=/dev/shm/p2_${arm}_0913_r1
    test ! -e "${OUTPUT_DIR}"
    test ! -e "${RAY_TMPDIR}"
    test -z "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)"
    remaining=$(( deadline - $(date +%s) - 15 ))
    test "${remaining}" -gt 0
    echo "${arm}" > "${control}/current_arm.txt"
    timeout --signal=INT --kill-after=15s "${remaining}" \
        bash "${feedback_dir}/run_feedback_eval_node47.sh" A > "${control}/${arm}.log" 2>&1
    "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${SOURCE_SHA}" <<'PY'
import json,sys
from pathlib import Path
from phase2_eval import GAME_SEEDS
run=Path(sys.argv[1]); result=json.loads((run/'result.json').read_text())
proof=json.loads((run/'checkpoint_evaluation.json').read_text())
manifest=json.loads((run/'manifest.json').read_text())
assert result==json.loads((run/'.done').read_text())
assert result['status']=='completed' and result['process_exit_code']==0
assert result['training_updates']==0 and result['episode_outcomes']['natural_games']==4
assert manifest['source_commit']==sys.argv[2] and not manifest['dirty']
assert proof['val_only'] and proof['optimizer_updates']==0 and proof['feedback']=='0'
assert proof['loaded_global_step']==4 and proof['game_seeds']==GAME_SEEDS
seeds=[json.loads(x)['seeds'] for x in (run/'evaluation_seeds.jsonl').read_text().splitlines()]
assert seeds==[GAME_SEEDS]
for rank in range(4):
    seed=json.loads((run/f'sampling-rank{rank}.json').read_text())
    assert seed['rank']==seed['seed']==rank
PY
    date -u +%FT%TZ > "${control}/${arm}-completed_at.txt"
    for _ in {1..15}; do
        test -n "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)" || break
        sleep 2
    done
done
echo completed > "${control}/stage.txt"
