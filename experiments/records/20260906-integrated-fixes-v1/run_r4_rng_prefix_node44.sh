#!/usr/bin/env bash
# Two sequential prefixes by default; an explicit repeat list allows one acceptance game.
# No retries, models, Ray or CUDA.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${repo_dir}"
test -z "$(git status --porcelain)"
sha=$(git rev-parse HEAD)
sha256sum --check configs/system-v1.sha256 >/dev/null
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=${repo_dir}:/root/new-scii-spb2-level5-control/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export SC2PATH=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets/sc2/StarCraftII
set -a
source configs/system-v1.env
set +a
export SC2_DIFFICULTY=Harder SC2_EXPECTED_DIFFICULTY=Harder SC2_AI_BUILD=Air
export SC2_ENV_SEED=2621640384 SC2_STEP_SHAPING_MODE=off SC2_GAME_TIME_LIMIT=120
export SC2_SEED_PYTHON_RANDOM=1 SC2_LOCAL_EXECUTOR_FALLBACK=1
export SC2_SOCKET_TIMEOUT=240 SC2_ACCEPT_TIMEOUT=60 SC2_RESET_RETRIES=1
export SC2_RESET_ATTEMPT_TIMEOUT=270 SC2_RESET_CONCURRENCY=1 SC2_FAIL_FAST_ON_ENV_ERROR=1
export SC2_RNG_PREFIX_TARGET=${1:-120}
case "${SC2_RNG_PREFIX_TARGET}" in
    120) tag=prefix120; wall_seconds=300 ;;
    first-planner)
        tag=preplanner; wall_seconds=600
        export SC2_GAME_TIME_LIMIT=1200 SC2_SOCKET_TIMEOUT=540 SC2_RESET_ATTEMPT_TIMEOUT=570
        ;;
    *) echo 'Choose 120 or first-planner' >&2; exit 2 ;;
esac
tag=${SC2_RNG_PREFIX_RUN_TAG:-${tag}}
export SC2_PRE_ACTION_TERMINAL_DRAIN_TIMEOUT=0.5 SC2_TERMINAL_ACK_TIMEOUT=900.0
export SC2_BO_ADDRESS=${repo_dir}/agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/knowledge/BO/replay_observations_Flat128_d4_TvT_Macro_g2_Result.Victory_twobase.json
# Same gameplay settings as r4's frozen scripts/train_sc2.sh (no strategic changes).
export SC2_STRATEGY_PROFILE=marine_pressure SC2_AUTO_TRAIN_MARINES=0 SC2_MAX_FORCED_ATTACKS=0
export SC2_LLM_ATTACK_MIN_UNITS=5 SC2_FIRST_LLM_ATTACK_MIN_UNITS=5 SC2_REINFORCEMENT_MIN_UNITS=5
export SC2_REINFORCEMENT_TIMER=88 SC2_ATTACK_ORDER_POSITION=after_build SC2_ATTACK_OPTION_ACCOUNT_ISSUED=1
export SC2_PLANNER_PROTOCOL=legacy SC2_ENABLE_NEWEXEC_MICRO=0 SC2_NEWEXEC_BASIC_HOOLIGAN=1
export SC2_NEWEXEC_HOOLIGAN_BIO=1 SC2_NEWEXEC_HOOLIGAN_ALLY_POWER_RADIUS=8 SC2_NEWEXEC_HOOLIGAN_RETREAT=0
export SC2_NEWEXEC_REQUIRE_TECH_UNIT=0 SC2_NEWEXEC_DEFENSE_MICRO=0 SC2_NEWEXEC_SPECIAL_ABILITIES=1
export SC2_NEWEXEC_FIGHT_MICRO=1 SC2_ATTACK_TARGET_UNITS_FIRST=0
export SC2_ATTACK_ENGAGED_RETARGET_IDLE_ONLY=1 SC2_HOOLIGAN_STRUCTURE_FOCUS=guarded
root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-integrated-fixes-v1/runs
for repeat in ${SC2_RNG_PREFIX_REPEATS:-1 2}; do
    export RUN_ID=20260906-integrated-fixes-v1-r4-python-rng-${tag}-s2621640384-repeat${repeat}-r1
    export OUTPUT_DIR=${root}/${RUN_ID}
    mkdir "${OUTPUT_DIR}"
    export SC2_RUNTIME_LOG_ROOT=${OUTPUT_DIR}/sc2-logs
    export SC2_EPISODE_OUTCOME_LOG=${OUTPUT_DIR}/episode_outcomes.jsonl
    date -Is > "${OUTPUT_DIR}/started_at.txt"
    printf '%s\n' "${sha}" > "${OUTPUT_DIR}/source_sha.txt"
    cp "${BASH_SOURCE[0]}" "${OUTPUT_DIR}/launcher.sh"
    /usr/bin/python3 -B - <<'PY' > "${OUTPUT_DIR}/effective_config.json"
import json, os, platform, socket
from importlib.metadata import version
print(json.dumps(dict(node=socket.gethostname(), python=platform.python_version(),
    burnysc2=version('burnysc2'), env={k:v for k,v in os.environ.items()
    if (k.startswith('SC2_') or k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS'))
    and not any(s in k for s in ('KEY','TOKEN','SECRET'))}), indent=2))
PY
    set +e
    timeout --signal=INT --kill-after=30s "${wall_seconds}" /usr/bin/python3 -B \
        experiments/records/20260906-integrated-fixes-v1/r4_rng_prefix.py \
        > "${OUTPUT_DIR}/driver.log" 2>&1
    status=$?
    set -e
    printf 'EXIT_CODE=%s\n' "${status}" > "${OUTPUT_DIR}/.done"
    date -Is > "${OUTPUT_DIR}/ended_at.txt"
    sha256sum "${OUTPUT_DIR}/driver.log" > "${OUTPUT_DIR}/artifacts.sha256"
    if [[ ${status} != 0 ]]; then exit "${status}"; fi
done
