#!/usr/bin/env bash
# A4 -> global5, one declared scene, one 45-minute ceiling; no evaluation/retry.
set -euo pipefail
test "${SC2_A4_SCENE1_AUTHORIZED:-0}" = 1
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
control=${A4_CONTROL_DIR:?set unused control directory}
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
export FEEDBACK_STORAGE_ROOT=/root/new-scii-execution-feedback-local-20260911/shared
export FEEDBACK_PYTHON_OVERLAY=/root/new-scii-spb-level5-96-20260908/python
export FEEDBACK_INCLUDE_ROOT=/root/new-scii-spb-level5-96-20260908/include
export FEEDBACK_RL_GPUS=0,1,2,3 SC2_FEEDBACK_RL_AUTHORIZED=1
export PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${record_dir}:${source_dir}/experiments/records/20260908-g2-planner-handoff-v1:${FEEDBACK_PYTHON_OVERLAY}
unset FEEDBACK_SECOND_PLAN FEEDBACK_SECOND_PLAN_VARIANT FEEDBACK_LEARNING_MODEL FEEDBACK_TEMPERATURE_AB FEEDBACK_RL_TEMPERATURE SC2_COMMAND_TRACE_DIR
test -z "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)"
(cd "${source_dir}" && sha256sum -c configs/system-v1.sha256)
runs=${FEEDBACK_STORAGE_ROOT}/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs
CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" - "${runs}" "${control}" <<'PY'
import json, os, sys
from pathlib import Path
from learning_eval import checkpoint_preflight
runs, control = map(Path, sys.argv[1:])
proof = checkpoint_preflight(runs)
fs = os.statvfs(runs)
if fs.f_bavail*fs.f_frsize < 90*1024**3:
    raise RuntimeError('Need 90GiB free for one checkpoint and safety reserve')
with (control/'write-probe.bin').open('xb') as stream:
    stream.write(bytes(1024**2)); stream.flush(); os.fsync(stream.fileno())
with (control/'checkpoint-preflight.json').open('x') as stream:
    json.dump(proof['A'], stream, indent=2)
PY
export FEEDBACK_CONTINUE_A4=1 FEEDBACK_A4_SCENE1=1
export RUN_ID=20260911-execution-failure-feedback-v1-spb2-A4-l6-scene1098341402-rl1-4gpu-r1
export OUTPUT_DIR=${runs}/${RUN_ID} RAY_TMPDIR=/dev/shm/a4s_0913_r1
test ! -e "${OUTPUT_DIR}"
echo training > "${control}/stage.txt"
remaining=$(( deadline - $(date +%s) - 15 ))
test "${remaining}" -gt 0
timeout --signal=INT --kill-after=15s "${remaining}" \
    bash "${record_dir}/run_feedback_train4_node43.sh" A > "${control}/train.log" 2>&1
"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${SOURCE_SHA}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); result = json.loads((p/'.done').read_text())
manifest = json.loads((p/'manifest.json').read_text())
assert result == json.loads((p/'result.json').read_text())
assert manifest['source_commit'] == sys.argv[2] and not manifest['dirty']
assert result['status'] == 'completed' and result['process_exit_code'] == 0
assert result['start_step'] == 4 and result['last_global_step'] == result['target_steps'] == 5
assert result['completed_steps'] == 1 and result['checkpoint_complete'] and result['metrics_finite']
assert result['episode_outcomes']['natural_games'] == 8 and not result['empty_policy_batches']
seeds = [json.loads(line)['seeds'] for line in (p/'training_seeds.jsonl').read_text().splitlines()]
assert seeds == [[1098341402]*8]
proof = json.loads((p/'checkpoints/a4_continuation.json').read_text())
assert proof['model_optimizer_load_returned'] and proof['new_run_global_steps'] == [5]
assert proof['game_seed_per_new_step'] == [1098341402]
PY
echo completed > "${control}/stage.txt"
