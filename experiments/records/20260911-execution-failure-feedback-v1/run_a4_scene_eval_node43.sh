#!/usr/bin/env bash
# A4 versus scene step5, 16 evaluation games, one shared 45-minute limit.
set -euo pipefail
test "${SC2_A4_SCENE_EVAL_AUTHORIZED:-0}" = 1
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
export FEEDBACK_STORAGE_ROOT=/root/new-scii-execution-feedback-local-20260911/shared
export FEEDBACK_PYTHON_OVERLAY=/root/new-scii-spb-level5-96-20260908/python
export FEEDBACK_INCLUDE_ROOT=/root/new-scii-spb-level5-96-20260908/include
export FEEDBACK_EVAL_GPUS=0,1,2,3 FEEDBACK_SCENE_EVAL=1
export FEEDBACK_SCENE_SET=${FEEDBACK_SCENE_SET:-single}
case "${FEEDBACK_SCENE_SET}" in
    single) scenario_name=scene1098341402; ray_prefix=se ;;
    historical8)
        test "${SC2_A4_MULTI_EVAL_AUTHORIZED:-0}" = 1
        scenario_name=multiseed; ray_prefix=me ;;
    *) exit 2 ;;
esac
export SC2_FEEDBACK_AB_AUTHORIZED=1 PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${record_dir}:${source_dir}/experiments/records/20260908-g2-planner-handoff-v1:${FEEDBACK_PYTHON_OVERLAY}
unset FEEDBACK_SECOND_PLAN FEEDBACK_SECOND_PLAN_VARIANT FEEDBACK_TEMPERATURE_AB FEEDBACK_FIRST_ATTACK_THRESHOLD SC2_COMMAND_TRACE_DIR
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
runs=${FEEDBACK_STORAGE_ROOT}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs
# Only metadata/stat, not tensor loading. Existing distributed loader is still required.
CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" - "${runs}" "${control}" <<'PY'
import json, sys
from pathlib import Path
from continue_a4 import PARENT_NAME, PARENT_SHA
runs,control=map(Path,sys.argv[1:]); proof={}
for label,name,sha,step in (
    ('A4',PARENT_NAME,PARENT_SHA,4),
    ('A5scene','20260911-execution-failure-feedback-v1-spb2-A4-l6-scene1098341402-rl1-4gpu-r1',
     '1689ba46f6307b8a4d0a7825d78b3fda40e587ea',5)):
    run=runs/name; result=json.loads((run/'result.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    assert result==json.loads((run/'.done').read_text())
    assert result['status']=='completed' and result['process_exit_code']==0 and result['checkpoint_complete']
    assert result['last_global_step']==step and result['target_steps']==step
    assert manifest['source_commit']==sha and not manifest['dirty'] and manifest['world_size']==4
    assert manifest['model_manifest_sha256']=='40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c'
    for name,meta in result['checkpoint_files'].items():
        assert (run/name).stat().st_size==meta['bytes']
    proof[label]=dict(run=run.name,source_sha=sha,step=step,shards_stat_passed=True)
with (control/'checkpoint-preflight.json').open('x') as stream:
    json.dump(proof,stream,indent=2)
PY
for label in A4 A5scene; do
    export FEEDBACK_LEARNING_MODEL=${label}
    export RUN_ID=20260911-execution-failure-feedback-v1-spb2-${label}-l6-${scenario_name}-eval8-4gpu-r1
    export OUTPUT_DIR=${runs}/${RUN_ID} RAY_TMPDIR=/dev/shm/${ray_prefix}_${label}_0913_r1
    test ! -e "${OUTPUT_DIR}"
    remaining=$(( deadline - $(date +%s) - 15 ))
    test "${remaining}" -gt 0
    echo "${label}" > "${control}/current_model.txt"
    timeout --signal=INT --kill-after=15s "${remaining}" \
        bash "${record_dir}/run_feedback_eval_node47.sh" A > "${control}/${label}.log" 2>&1
    "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${label}" "${SOURCE_SHA}" <<'PY'
import json,os,sys
from pathlib import Path
from learning_eval import GAME_SEEDS
expected_seeds=GAME_SEEDS if os.environ['FEEDBACK_SCENE_SET']=='historical8' else [1098341402]*8
run=Path(sys.argv[1]); result=json.loads((run/'result.json').read_text())
proof=json.loads((run/'checkpoint_evaluation.json').read_text())
manifest=json.loads((run/'manifest.json').read_text())
assert result==json.loads((run/'.done').read_text())
assert result['status']=='completed' and result['process_exit_code']==0
assert result['training_updates']==0 and result['episode_outcomes']['natural_games']==8
assert manifest['source_commit']==sys.argv[3] and not manifest['dirty']
assert proof['val_only'] and proof['optimizer_updates']==0 and proof['feedback']=='0'
assert proof['loaded_global_step']==(4 if sys.argv[2]=='A4' else 5)
assert proof['game_seeds']==expected_seeds
seeds=[json.loads(x)['seeds'] for x in (run/'evaluation_seeds.jsonl').read_text().splitlines()]
assert seeds==[expected_seeds]
for rank in range(4):
    seed=json.loads((run/f'sampling-rank{rank}.json').read_text())
    assert seed['rank']==seed['seed']==rank
PY
    date -u +%FT%TZ > "${control}/${label}-completed_at.txt"
    for _ in {1..15}; do
        test -n "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)" || break
        sleep 2
    done
done
echo completed > "${control}/stage.txt"
