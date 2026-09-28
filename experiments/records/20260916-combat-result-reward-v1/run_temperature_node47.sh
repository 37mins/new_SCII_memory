#!/usr/bin/env bash
# One separately approved C or D: four GPUs, four new steps, 90 minutes, no retry.
set -euo pipefail
arm=${1:-C}
case "${arm}" in
    C) test "${SC2_COMBAT_TEMPERATURE_AUTHORIZED:-0}" = 1
       suffix=C-temp08; gamma=0.95; ray_name=combat_C_t08_0916_r1 ;;
    D) test "${SC2_COMBAT_GAMMA_AUTHORIZED:-0}" = 1
       suffix=D-temp08-gamma1; gamma=1.0; ray_name=combat_D_g1_0916_r1 ;;
    *) exit 2 ;;
esac
record=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
: "${COMBAT_CONTROL_DIR:?unused control directory}"
export RUN_ID=20260916-combat-result-reward-v1-spb2-A4-l6-${suffix}-rl4-4gpu-r1
export OUTPUT_DIR=/mnt/data0601/yangjinyue/new_SCII/experiments/20260916-combat-result-reward-v1/runs/${RUN_ID}
export RAY_TMPDIR=/dev/shm/${ray_name}
test ! -e "${OUTPUT_DIR}"
test ! -e "${RAY_TMPDIR}"
test -z "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)"
awk '/MemAvailable/ {exit !($2 > 600*1024*1024)}' /proc/meminfo
mkdir "${COMBAT_CONTROL_DIR}"
deadline=$(( $(date +%s) + 5400 ))
date -u +%FT%TZ > "${COMBAT_CONTROL_DIR}/started_at.txt"
echo "${deadline}" > "${COMBAT_CONTROL_DIR}/deadline_unix.txt"
finish() {
    rc=$?
    echo "${rc}" > "${COMBAT_CONTROL_DIR}/exit_code"
    date -u +%FT%TZ > "${COMBAT_CONTROL_DIR}/ended_at.txt"
}
trap finish EXIT
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "${COMBAT_CONTROL_DIR}/gpus.csv"
export FEEDBACK_RL_GPUS=0,1,2,3 SC2_COMBAT_REWARD_AUTHORIZED=1
export FEEDBACK_STORAGE_ROOT=/mnt/data0601/yangjinyue/new_SCII/staging/20260915-scene-B-r2/shared
export FEEDBACK_PYTHON_OVERLAY=${FEEDBACK_PYTHON_OVERLAY:-/dev/shm/sc2-scene-B-retry-20260915-r2/deps/python}
export FEEDBACK_INCLUDE_ROOT=${FEEDBACK_INCLUDE_ROOT:-/dev/shm/sc2-scene-B-retry-20260915-r2/deps/include}
export PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1
export MODEL_PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/models/Panyuqi/SpikingBrain-2.0-instruct
export SC2PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII
unset RAY_ADDRESS SC2_COMBAT_REWARD_SCALE
echo "training-${arm}" > "${COMBAT_CONTROL_DIR}/stage.txt"
remaining=$(( deadline - $(date +%s) - 15 ))
test "${remaining}" -gt 0
timeout --signal=INT --kill-after=15s "${remaining}" \
    bash "${record}/run_arm.sh" "${arm}" > "${COMBAT_CONTROL_DIR}/${arm}.log" 2>&1
"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${gamma}" <<'PY'
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
seeds = [1098341402, 42613, 17089, 3236598124]
assert [json.loads(s)['seeds'] for s in (p/'training_seeds.jsonl').read_text().splitlines()] == [[s]*8 for s in seeds]
proof = json.loads((p/'checkpoints/a4_continuation.json').read_text())
assert proof['model_optimizer_load_returned'] and proof['game_seed_per_collection'] == seeds
cfg = json.loads((p/'effective_config.json').read_text())['overrides']
assert float(cfg['+ray_init.runtime_env.env_vars.SC2_COMBAT_REWARD_SCALE'].strip("'\"")) == .001
assert float(cfg['actor_rollout_ref.rollout.temperature']) == .8
assert float(cfg['algorithm.gamma']) == float(sys.argv[2])
PY
echo completed > "${COMBAT_CONTROL_DIR}/stage.txt"
