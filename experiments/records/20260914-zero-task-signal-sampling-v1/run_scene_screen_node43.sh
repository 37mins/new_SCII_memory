#!/usr/bin/env bash
# Approved: after pair-r1 ends, four GPUs, 8 scenes x 8 games, <=90min total.
set -euo pipefail
test "${SC2_SCENE_SCREEN_AUTHORIZED:-0}" = 1
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
test "$(cat /root/new-scii-zero-signal-20260914/pair-r1/done)" = 0
test "$(cat /root/new-scii-zero-signal-20260914/pair-r1/stage.txt)" = completed
test -z "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)"
control=${SCREEN_CONTROL_DIR:?set unused control directory}
mkdir "${control}"
deadline=$(( $(date +%s) + 5400 ))
finish() {
    status=$?
    date -u +%FT%TZ > "${control}/ended_at.txt"
    echo "${status}" > "${control}/done"
}
trap finish EXIT
date -u +%FT%TZ > "${control}/started_at.txt"
echo "${deadline}" > "${control}/deadline_unix.txt"
old_record=${source_dir}/experiments/records/20260911-execution-failure-feedback-v1
export FEEDBACK_STORAGE_ROOT=/root/new-scii-execution-feedback-local-20260911/shared
export FEEDBACK_PYTHON_OVERLAY=/root/new-scii-spb-level5-96-20260908/python
export FEEDBACK_INCLUDE_ROOT=/root/new-scii-spb-level5-96-20260908/include
export FEEDBACK_EVAL_GPUS=0,1,2,3 FEEDBACK_LEARNING_MODEL=A4
export FEEDBACK_SCENE_EVAL=1 FEEDBACK_SCENE_SET=screening
export SC2_FEEDBACK_AB_AUTHORIZED=1 PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${old_record}:${source_dir}/experiments/records/20260908-g2-planner-handoff-v1:${FEEDBACK_PYTHON_OVERLAY}
unset FEEDBACK_SECOND_PLAN FEEDBACK_SECOND_PLAN_VARIANT FEEDBACK_TEMPERATURE_AB FEEDBACK_FIRST_ATTACK_THRESHOLD FEEDBACK_PHASE2_AIR SC2_COMMAND_TRACE_DIR
runs=/mnt/data0601/yangjinyue/new_SCII/experiments/20260914-zero-task-signal-sampling-v1/runs
timeout --kill-after=1s 5s stat "${runs}"
timeout --signal=INT --kill-after=5s 25s env CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" - "${FEEDBACK_STORAGE_ROOT}" "${control}" <<'PY'
import json, sys
from pathlib import Path
from continue_a4 import PARENT_NAME, PARENT_SHA
storage, control = map(Path, sys.argv[1:])
parent = storage/'yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs'/PARENT_NAME
result = json.loads((parent/'result.json').read_text())
manifest = json.loads((parent/'manifest.json').read_text())
assert result == json.loads((parent/'.done').read_text())
assert result['status']=='completed' and result['process_exit_code']==0 and result['checkpoint_complete']
assert result['last_global_step']==4 and manifest['source_commit']==PARENT_SHA
assert not manifest['dirty'] and manifest['world_size']==4
assert manifest['model_manifest_sha256']=='40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c'
for name, meta in result['checkpoint_files'].items():
    assert (parent/name).stat().st_size==meta['bytes']
with (control/'checkpoint-preflight.json').open('x') as stream:
    json.dump(dict(parent=str(parent), source=PARENT_SHA, step=4,
                   shard_sizes_verified=True, distributed_load_still_required=True), stream, indent=2)
PY
for index in 0 1 2 3 4 5 6 7; do
    export SC2_SCREEN_SCENE_INDEX=${index}
    seed=$("${PYTHON_BIN}" -c "from learning_eval import SCREEN_GAME_SEEDS; print(SCREEN_GAME_SEEDS[${index}])")
    export RUN_ID=20260914-zero-task-signal-sampling-v1-spb2-A4-l6-screen-s${seed}-eval8-4gpu-r1
    export OUTPUT_DIR=${runs}/${RUN_ID} RAY_TMPDIR=/dev/shm/ss_${index}_0914_r1
    timeout --kill-after=1s 5s test ! -e "${OUTPUT_DIR}"
    test ! -e "${RAY_TMPDIR}"
    for _ in {1..15}; do
        test -n "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)" || break
        sleep 2
    done
    test -z "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)"
    remaining=$(( deadline - $(date +%s) - 15 ))
    test "${remaining}" -gt 0
    echo "scene-${index}-seed-${seed}" > "${control}/stage.txt"
    date -u +%FT%TZ > "${control}/scene-${index}-started_at.txt"
    timeout --signal=INT --kill-after=15s "${remaining}" \
        bash "${old_record}/run_feedback_eval_node47.sh" A > "${control}/scene-${index}.log" 2>&1
    timeout --signal=INT --kill-after=5s 25s "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${SOURCE_SHA}" "${seed}" <<'PY'
import json, sys
from pathlib import Path
run=Path(sys.argv[1]); seed=int(sys.argv[3])
result=json.loads((run/'result.json').read_text())
proof=json.loads((run/'checkpoint_evaluation.json').read_text())
manifest=json.loads((run/'manifest.json').read_text())
assert result==json.loads((run/'.done').read_text())
assert result['status']=='completed' and result['process_exit_code']==0
assert result['training_updates']==0 and result['episode_outcomes']['natural_games']==8
assert not result['episode_outcomes']['truncated']
assert manifest['source_commit']==sys.argv[2] and not manifest['dirty']
assert proof['val_only'] and proof['optimizer_updates']==0 and proof['loaded_global_step']==4
assert proof['feedback']=='0' and proof['game_seeds']==[seed]*8
assert proof['checkpoint'].endswith('A-rl4-4gpu-s0-r2/checkpoints/global_step_4')
actual=[json.loads(row)['seeds'] for row in (run/'evaluation_seeds.jsonl').read_text().splitlines()]
assert actual==[[seed]*8]
for rank in range(4):
    sampling=json.loads((run/f'sampling-rank{rank}.json').read_text())
    assert sampling['seed']==sampling['rank']==rank
PY
    date -u +%FT%TZ > "${control}/scene-${index}-completed_at.txt"
done
echo completed > "${control}/stage.txt"
