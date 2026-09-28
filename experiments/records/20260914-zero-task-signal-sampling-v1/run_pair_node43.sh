#!/usr/bin/env bash
# Confirmed September14: four GPUs, serial A/B, one 3h limit, no retry.
set -euo pipefail
replicate=${SIGNAL_REPLICATE:-r1}
[[ "${replicate}" =~ ^r[1-9][0-9]*$ ]] || exit 2
export SC2_SCENE_SAMPLING=${SC2_SCENE_SAMPLING:-0}
case "${SC2_SCENE_SAMPLING}" in
    0) test "${SC2_ZERO_SIGNAL_AUTHORIZED:-0}" = 1; variant=pair-${replicate}; ray_prefix=zs; ray_date=0914 ;;
    1) test "${SC2_SCENE_SAMPLING_AUTHORIZED:-0}" = 1; variant=scene-train-${replicate}; ray_prefix=ss; ray_date=0915 ;;
    *) exit 2 ;;
esac
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
control=${SIGNAL_CONTROL_DIR:?set unused control directory}
mkdir "${control}"
deadline=$(( $(date +%s) + 10800 ))
if [[ -n "${SIGNAL_DEADLINE_UNIX:-}" ]]; then
    [[ "${SIGNAL_DEADLINE_UNIX}" =~ ^[0-9]+$ ]]
    test "${SIGNAL_DEADLINE_UNIX}" -le "${deadline}"
    deadline=${SIGNAL_DEADLINE_UNIX}
fi
finish() {
    status=$?
    date -u +%FT%TZ > "${control}/ended_at.txt"
    echo "${status}" > "${control}/done"
}
trap finish EXIT
date -u +%FT%TZ > "${control}/started_at.txt"
echo "${deadline}" > "${control}/deadline_unix.txt"
export FEEDBACK_STORAGE_ROOT=${FEEDBACK_STORAGE_ROOT:-/root/new-scii-execution-feedback-local-20260911/shared}
export PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3} PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${source_dir}/experiments/records/20260911-execution-failure-feedback-v1:${FEEDBACK_PYTHON_OVERLAY:-/root/new-scii-spb-level5-96-20260908/python}
runs=/mnt/data0601/yangjinyue/new_SCII/experiments/20260914-zero-task-signal-sampling-v1/runs
timeout --kill-after=1s 5s stat /mnt/data0601/yangjinyue/new_SCII
timeout --kill-after=1s 10s mkdir -p "${runs}"
timeout --signal=INT --kill-after=5s 30s env CUDA_VISIBLE_DEVICES= \
    "${PYTHON_BIN}" - "${FEEDBACK_STORAGE_ROOT}" "${runs}" "${control}" "${variant}" <<'PY'
import json, os, sys
from pathlib import Path
from continue_a4 import PARENT_NAME, PARENT_SHA
storage, runs, control = map(Path, sys.argv[1:4])
parent = storage/'yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs'/PARENT_NAME
manifest = json.loads((parent/'manifest.json').read_text())
result = json.loads((parent/'.done').read_text())
assert result == json.loads((parent/'result.json').read_text())
assert manifest['source_commit'] == PARENT_SHA and not manifest['dirty'] and manifest['world_size'] == 4
assert result['status'] == 'completed' and result['process_exit_code'] == 0
assert result['completed_steps'] == 4 and result['checkpoint_complete']
checkpoint = parent/'checkpoints/global_step_4'
sizes = {}
for rank in range(4):
    for kind, size in [('model', 5555629511), ('optim', 10332463827), ('extra_state', None)]:
        name = f'actor/{kind}_world_size_4_rank_{rank}.pt'
        actual = (checkpoint/name).stat().st_size
        assert actual == size if size else actual > 10000
        sizes[name] = actual
assert (checkpoint/'data.pt').stat().st_size > 0
fs = os.statvfs(runs)
assert fs.f_bavail*fs.f_frsize >= 160*1024**3, 'Need 160GiB shared free space including reserve'
with (runs/(sys.argv[4]+'-write-probe.bin')).open('xb') as stream:
    stream.write(bytes(1024**2)); stream.flush(); os.fsync(stream.fileno())
with (control/'checkpoint-preflight.json').open('x') as stream:
    json.dump(dict(parent=str(parent), files=sizes, actual_distributed_load_still_required=True), stream, indent=2)
PY
for arm in A B; do
    export RUN_ID=20260914-zero-task-signal-sampling-v1-spb2-A4-l6-${arm}-32games-4gpu-${replicate}
    if [[ "${SC2_SCENE_SAMPLING}" == 1 ]]; then
        export RUN_ID=20260914-zero-task-signal-sampling-v1-spb2-A4-l6-scene-${arm}-rl4-4gpu-${replicate}
    fi
    export OUTPUT_DIR=${runs}/${RUN_ID} RAY_TMPDIR=/dev/shm/${ray_prefix}_${arm}_${ray_date}_${replicate}
    timeout --kill-after=1s 5s test ! -e "${OUTPUT_DIR}"
    test ! -e "${RAY_TMPDIR}"
    for _ in {1..15}; do
        test -n "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)" || break
        sleep 2
    done
    test -z "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)"
    echo "training-${arm}" > "${control}/stage.txt"
    remaining=$(( deadline - $(date +%s) - 15 ))
    test "${remaining}" -gt 0
    timeout --signal=INT --kill-after=15s "${remaining}" \
        bash "${record_dir}/run_arm_node43.sh" "${arm}" > "${control}/train-${arm}.log" 2>&1
    timeout --signal=INT --kill-after=5s 25s "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${SOURCE_SHA}" "${arm}" "${SC2_SCENE_SAMPLING}" <<'PY'
import json, sys
from pathlib import Path
from continue_a4 import COVERAGE_SEEDS, SIGNAL_SCENE_SEEDS
scene_sampling = sys.argv[4] == '1'
expected_seeds = SIGNAL_SCENE_SEEDS[sys.argv[3]] if scene_sampling else COVERAGE_SEEDS
p = Path(sys.argv[1]); result = json.loads((p/'.done').read_text())
manifest = json.loads((p/'manifest.json').read_text())
assert manifest['source_commit'] == sys.argv[2] and not manifest['dirty']
assert result == json.loads((p/'result.json').read_text())
assert result['status'] == 'completed' and result['process_exit_code'] == 0
assert result['start_step'] == 4 and 1 <= result['completed_steps'] <= 4
assert result['checkpoint_complete'] and result['metrics_finite']
assert result['episode_outcomes']['natural_games'] == 32 and not result['empty_policy_batches']
assert result['signal_sampling']['sampled_batches'] == 4
assert result['signal_sampling']['skip_zero_signal'] == (sys.argv[3] == 'B' and not scene_sampling)
if scene_sampling:
    assert result['completed_steps'] == 4 and result['last_global_step'] == 8
    assert result['signal_sampling']['skipped_batches'] == 0
assert [json.loads(s)['seeds'] for s in (p/'training_seeds.jsonl').read_text().splitlines()] == [[s]*8 for s in expected_seeds]
proof = json.loads((p/'checkpoints/a4_continuation.json').read_text())
assert proof['model_optimizer_load_returned'] and proof['game_seed_per_collection'] == expected_seeds
PY
    date -u +%FT%TZ > "${control}/train-${arm}-completed_at.txt"
done
echo completed > "${control}/stage.txt"
