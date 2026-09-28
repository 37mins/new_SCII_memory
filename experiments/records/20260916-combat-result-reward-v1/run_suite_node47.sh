#!/usr/bin/env bash
# Approved relocation replaces node197: fixed inputs first, then serial RL A/B.
set -euo pipefail
test "${SC2_NODE47_SUITE_AUTHORIZED:-0}" = 1
record=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
: "${SUITE_CONTROL:?unused controller path}"
mkdir "${SUITE_CONTROL}"
finish() { rc=$?; date -u +%FT%TZ > "${SUITE_CONTROL}/ended_at.txt"; echo "$rc" > "${SUITE_CONTROL}/exit_code"; }
trap finish EXIT
export FEEDBACK_STORAGE_ROOT=/mnt/data0601/yangjinyue/new_SCII/staging/20260915-scene-B-r2/shared
export FEEDBACK_PYTHON_OVERLAY=/dev/shm/sc2-scene-B-retry-20260915-r2/deps/python
export FEEDBACK_INCLUDE_ROOT=/dev/shm/sc2-scene-B-retry-20260915-r2/deps/include
export PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1
export MODEL_PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/models/Panyuqi/SpikingBrain-2.0-instruct
export SC2PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII
export PYTHONPATH=${source_dir}:${source_dir}/tests:${FEEDBACK_PYTHON_OVERLAY}
unset RAY_ADDRESS SC2_COMBAT_REWARD_SCALE
test -z "$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader)"
test -z "$(pgrep -x raylet || true)"
timeout 20s env CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" - <<'PY' > "${SUITE_CONTROL}/resource-preflight.json"
import json, pathlib, subprocess
m={s.split(':')[0]:int(s.split()[1]) for s in pathlib.Path('/proc/meminfo').read_text().splitlines()}
assert m['MemAvailable'] > 600*1024**2, 'Need one-model host-memory headroom'
print(json.dumps(dict(mem_available_kib=m['MemAvailable'],gpus=[4,5,6,7],
    snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))))
PY
cd "${source_dir}"
timeout 60s env CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" -B -m unittest \
    test_sc2_combat_reward test_combat_experiment test_r6_terminal_policy_credit \
    test_a4_continuation -v > "${SUITE_CONTROL}/cpu-tests.log" 2>&1
sha256sum -c configs/system-v1.sha256 > "${SUITE_CONTROL}/system-hashes.log"
runs=/mnt/data0601/yangjinyue/new_SCII/experiments/20260916-combat-result-reward-v1/runs
mkdir -p "${runs}"
export PANEL_OUTPUT=${runs}/20260916-combat-result-reward-v1-spb2-original-A4-fixed3x8-4gpu-r1
export PANEL_TMPDIR=/dev/shm/sc2_combat_panel_0916_r1
export SC2_FAILURE_PANEL_AUTHORIZED=1
echo fixed-input-panel > "${SUITE_CONTROL}/stage.txt"
timeout --signal=INT --kill-after=15s 1785 bash "${record}/run_failure_panel.sh" > "${SUITE_CONTROL}/panel.log" 2>&1
"${PYTHON_BIN}" - "${PANEL_OUTPUT}" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); r=json.loads((p/'result.json').read_text())
assert (p/'.done').read_text().startswith('EXIT_CODE=0\n')
assert r['status']=='completed' and r['games']==r['optimizer_updates']==0
assert r['failure_panel']['paired_input_and_sampling_rng'] and len(r['failure_panel']['contexts'])==3
assert all(a['answers']==24 for a in r['arms'].values())
PY
# No global stop/cleanup command: refuse overlap or foreign GPU use.
test -z "$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader)"
export SC2_COMBAT_REWARD_AUTHORIZED=1 COMBAT_CONTROL_DIR=${SUITE_CONTROL}/reward-pair
export COMBAT_RUNS_DIR=${runs} COMBAT_REPLICATE=r1
echo reward-rl-pair > "${SUITE_CONTROL}/stage.txt"
timeout --signal=INT --kill-after=15s 10785 bash "${record}/run_pair.sh" > "${SUITE_CONTROL}/pair.log" 2>&1
echo completed > "${SUITE_CONTROL}/stage.txt"
