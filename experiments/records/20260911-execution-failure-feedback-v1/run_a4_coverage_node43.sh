#!/usr/bin/env bash
# Authorized four-GPU serial A/B: four new global steps per arm, shared 3h limit.
set -euo pipefail
test "${SC2_A4_COVERAGE_AUTHORIZED:-0}" = 1
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
control=${A4_CONTROL_DIR:?set unused control directory}
mkdir "${control}"
deadline=$(( $(date +%s) + 10800 ))
finish() {
    status=$?
    date -u +%FT%TZ > "${control}/ended_at.txt"
    echo "${status}" > "${control}/done"
}
trap finish EXIT
date -u +%FT%TZ > "${control}/started_at.txt"
echo "${deadline}" > "${control}/deadline_unix.txt"
export FEEDBACK_STORAGE_ROOT=/root/new-scii-execution-feedback-local-20260911/shared
export FEEDBACK_PYTHON_OVERLAY=/root/new-scii-spb-level5-96-20260908/python
export FEEDBACK_RL_GPUS=0,1,2,3 SC2_FEEDBACK_RL_AUTHORIZED=1 FEEDBACK_CONTINUE_A4=1
export PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${record_dir}:${source_dir}/experiments/records/20260908-g2-planner-handoff-v1:${FEEDBACK_PYTHON_OVERLAY}
unset FEEDBACK_A4_SCENE1 FEEDBACK_RL_TEMPERATURE FEEDBACK_TEMPERATURE_AB
unset FEEDBACK_SECOND_PLAN FEEDBACK_SECOND_PLAN_VARIANT FEEDBACK_LEARNING_MODEL SC2_COMMAND_TRACE_DIR
(cd "${source_dir}" && sha256sum -c configs/system-v1.sha256)
local_runs=${FEEDBACK_STORAGE_ROOT}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs
runs=/mnt/data0601/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs
timeout --signal=INT --kill-after=5s 30s env CUDA_VISIBLE_DEVICES= \
    "${PYTHON_BIN}" - "${local_runs}" "${runs}" "${control}" <<'PY'
import json, os, sys
from pathlib import Path
from learning_eval import checkpoint_preflight
local_runs, runs, control = map(Path, sys.argv[1:])
proof = checkpoint_preflight(local_runs)
fs = os.statvfs(runs)
if fs.f_bavail*fs.f_frsize < 160*1024**3:
    raise RuntimeError('Need 160GiB free on shared storage for two checkpoints and reserve')
with (runs/(control.name+'-write-probe.bin')).open('xb') as stream:
    stream.write(bytes(1024**2)); stream.flush(); os.fsync(stream.fileno())
with (control/'checkpoint-preflight.json').open('x') as stream:
    json.dump(proof['A'], stream, indent=2)
PY
for arm in A B; do
    # Both arms use the feedback-OFF training entry and the same parent A4.
    export FEEDBACK_A4_COVERAGE=0
    if [[ "${arm}" == B ]]; then export FEEDBACK_A4_COVERAGE=1; fi
    export RUN_ID=20260911-execution-failure-feedback-v1-spb2-A4-l6-coverage-${arm}-rl4-4gpu-r1
    export OUTPUT_DIR=${runs}/${RUN_ID} RAY_TMPDIR=/dev/shm/cv_${arm}_0913_r1
    test ! -e "${OUTPUT_DIR}"
    for _ in {1..15}; do
        test -n "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)" || break
        sleep 2
    done
    test -z "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)"
    echo "training-${arm}" > "${control}/stage.txt"
    remaining=$(( deadline - $(date +%s) - 15 ))
    test "${remaining}" -gt 0
    timeout --signal=INT --kill-after=15s "${remaining}" \
        bash "${record_dir}/run_feedback_train4_node43.sh" A > "${control}/train-${arm}.log" 2>&1
    timeout --signal=INT --kill-after=5s 30s "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${SOURCE_SHA}" "${arm}" <<'PY'
import json, sys
from pathlib import Path
from continue_a4 import COVERAGE_SEEDS, NEXT_SEEDS
p = Path(sys.argv[1]); r = json.loads((p/'.done').read_text())
m = json.loads((p/'manifest.json').read_text())
assert r == json.loads((p/'result.json').read_text())
assert m['source_commit'] == sys.argv[2] and not m['dirty']
assert r['status'] == 'completed' and r['process_exit_code'] == 0
assert r['start_step'] == 4 and r['last_global_step'] == r['target_steps'] == 8
assert r['completed_steps'] == 4 and r['checkpoint_complete'] and r['metrics_finite']
assert r['episode_outcomes']['natural_games'] == 32 and not r['empty_policy_batches']
expected = NEXT_SEEDS if sys.argv[3] == 'A' else COVERAGE_SEEDS
assert [json.loads(s)['seeds'] for s in (p/'training_seeds.jsonl').read_text().splitlines()] == [[s]*8 for s in expected]
proof = json.loads((p/'checkpoints/a4_continuation.json').read_text())
assert proof['model_optimizer_load_returned'] and proof['new_run_global_steps'] == [5,6,7,8]
assert proof['game_seed_per_new_step'] == expected
PY
    date -u +%FT%TZ > "${control}/train-${arm}-completed_at.txt"
done
echo completed > "${control}/stage.txt"
