#!/usr/bin/env bash
# Node47's approved GPU4-7, from an independent frozen checkout.
set -euo pipefail
test "${SC2_COMBAT_REWARD_AUTHORIZED:-0}" = 1
record=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
: "${COMBAT_CONTROL_DIR:?unused control directory}"
: "${COMBAT_RUNS_DIR:?new run root}"
: "${COMBAT_REPLICATE:?unique replicate}"
[[ "${COMBAT_REPLICATE}" =~ ^r[1-9][0-9]*$ ]]
mkdir "${COMBAT_CONTROL_DIR}"
deadline=$(( $(date +%s) + 10800 ))
date -u +%FT%TZ > "${COMBAT_CONTROL_DIR}/started_at.txt"
echo "${deadline}" > "${COMBAT_CONTROL_DIR}/deadline_unix.txt"
finish() {
    status=$?
    echo "${status}" > "${COMBAT_CONTROL_DIR}/exit_code"
    date -u +%FT%TZ > "${COMBAT_CONTROL_DIR}/ended_at.txt"
}
trap finish EXIT
for arm in A B; do
    test -z "$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader)"
    export RUN_ID=20260916-combat-result-reward-v1-spb2-A4-l6-${arm}-rl4-4gpu-${COMBAT_REPLICATE}
    export OUTPUT_DIR=${COMBAT_RUNS_DIR}/${RUN_ID}
    export RAY_TMPDIR=/dev/shm/combat_${arm}_0916_${COMBAT_REPLICATE}
    test ! -e "${OUTPUT_DIR}"
    test ! -e "${RAY_TMPDIR}"
    remaining=$(( deadline - $(date +%s) - 15 ))
    test "${remaining}" -gt 0
    echo "training-${arm}" > "${COMBAT_CONTROL_DIR}/stage.txt"
    timeout --signal=INT --kill-after=15s "${remaining}" \
        bash "${record}/run_arm.sh" "${arm}" > "${COMBAT_CONTROL_DIR}/${arm}.log" 2>&1
    "${PYTHON_BIN:?}" - "${OUTPUT_DIR}" "${arm}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
r = json.loads((p/'.done').read_text())
assert r == json.loads((p/'result.json').read_text())
assert r['status'] == 'completed' and r['process_exit_code'] == 0
assert r['start_step'] == 4 and r['last_global_step'] == 8 and r['completed_steps'] == 4
assert r['checkpoint_complete'] and r['metrics_finite'] and not r['empty_policy_batches']
assert r['episode_outcomes']['natural_games'] == 32
assert r['signal_sampling']['sampled_batches'] == 4 and not r['signal_sampling']['skip_zero_signal']
seeds = [1098341402,42613,17089,3236598124]
assert [json.loads(s)['seeds'] for s in (p/'training_seeds.jsonl').read_text().splitlines()] == [[s]*8 for s in seeds]
proof = json.loads((p/'checkpoints/a4_continuation.json').read_text())
assert proof['model_optimizer_load_returned'] and proof['game_seed_per_collection'] == seeds
overrides = json.loads((p/'effective_config.json').read_text())['overrides']
scale = float(overrides['+ray_init.runtime_env.env_vars.SC2_COMBAT_REWARD_SCALE'].strip("'\""))
assert scale == (0 if sys.argv[2]=='A' else .001)
PY
done
echo completed > "${COMBAT_CONTROL_DIR}/stage.txt"
