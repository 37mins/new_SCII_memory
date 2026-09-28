#!/usr/bin/env bash
# Invoke under timeout --signal=INT --kill-after=15s 10785. No retries or promotion.
set -euo pipefail
deadline=$(( $(date +%s) + 10800 ))
test "${SC2_FEEDBACK_RL_AUTHORIZED:-0}" = 1
case "${FEEDBACK_RL_SERIAL:-0}" in 0|1) ;; *) exit 2 ;; esac
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
control=${PAIR_DIR:?set unused control directory}
mkdir "${control}"
exec > >(tee "${control}/console.log") 2>&1
finish_pair() {
    local status=$?
    if [[ -n "${monitor_pid:-}" ]]; then kill "${monitor_pid}" 2>/dev/null || true; fi
    if [[ -f "${control}/pair.status" ]]; then cp "${control}/pair.status" "${control}/pair.done"; fi
    printf 'EXIT_CODE=%s\n' "$status" > "${control}/.done"
    date -u +%FT%TZ > "${control}/ended_at.txt"
}
trap finish_pair EXIT
date -u +%FT%TZ | tee "${control}/started_at.txt"
cards=0,1,2,3,4,5,6,7
if [[ "${FEEDBACK_RL_SERIAL:-0}" == 1 ]]; then cards=0,1,2,3; fi
test -z "$(nvidia-smi -i "${cards}" --query-compute-apps=pid --format=csv,noheader)"
(cd "${source_dir}" && sha256sum -c configs/system-v1.sha256)
PYTHONPATH=/root/new-scii-spb-level5-96-20260908/python CUDA_VISIBLE_DEVICES= \
    PYTHONDONTWRITEBYTECODE=1 timeout 30s /usr/bin/python3 - "${control}" <<'PY'
import os, sys
from pathlib import Path
import torch, transformers, ray, gymnasium
assert (torch.__version__, transformers.__version__, ray.__version__, gymnasium.__version__) == (
    '2.7.1+cu128', '4.55.2', '2.48.0', '1.2.3')
assert not torch.cuda.is_initialized()
root = Path(sys.argv[1])
fs = os.statvfs(root)
free = fs.f_bavail * fs.f_frsize
if free < 180 * 1024**3:
    raise RuntimeError(f'Need 180GiB free for two checkpoints and reserve, got {free}')
with (root / 'write-probe.bin').open('xb') as stream:
    stream.write(bytes(1024**2)); stream.flush(); os.fsync(stream.fileno())
print('Pinned dependencies and local write/fsync verified; free bytes:', free, flush=True)
PY
if [[ "${FEEDBACK_RL_SERIAL:-0}" == 1 ]]; then
    /usr/bin/python3 "${record_dir}/watch_host_memory.py" "${control}" &
    monitor_pid=$!
fi
run_arm() (
    local arm=$1 remaining=$(( deadline - $(date +%s) - 60 ))
    test "${remaining}" -gt 0
    if [[ "${FEEDBACK_RL_SERIAL:-0}" == 1 ]]; then
        export FEEDBACK_RL_GPUS=0,1,2,3
        export RUN_ID=20260911-execution-failure-feedback-v1-spb2-original-l6-g2-${arm}-rl4-4gpu-s0-r2
        export RAY_TMPDIR=/dev/shm/sc2_feedback_rl4_${arm}_0911_r2
        printf '%s\n' "${arm}" > "${control}/current_arm.txt"
    fi
    date -u +%FT%TZ > "${control}/${arm}-started_at.txt"
    set +e
    timeout --signal=INT --kill-after=30s "${remaining}" \
        bash "${record_dir}/run_feedback_train4_node43.sh" "${arm}" > "${control}/${arm}.log" 2>&1
    local status=$?
    printf 'EXIT_CODE=%s\n' "${status}" > "${control}/${arm}.done"
    date -u +%FT%TZ >> "${control}/${arm}.done"
    exit "${status}"
)
if [[ "${FEEDBACK_RL_SERIAL:-0}" == 1 ]]; then
    printf 'A=pending\nB=pending\n' > "${control}/pair.status"
    for arm in A B; do
        printf '%s=running\n' "${arm}" >> "${control}/pair.status"
        arm_status=0
        run_arm "${arm}" || arm_status=$?
        printf '%s_EXIT_CODE=%s\n' "${arm}" "${arm_status}" >> "${control}/pair.status"
        test "${arm_status}" = 0
        /usr/bin/python3 - "${FEEDBACK_STORAGE_ROOT:-/root/new-scii-execution-feedback-local-20260911/shared}" "${arm}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]) / ('yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/20260911-execution-failure-feedback-v1-spb2-original-l6-g2-' + sys.argv[2] + '-rl4-4gpu-s0-r2/.done')
done = json.loads(path.read_text())
assert done['status'] == 'completed' and done['completed_steps'] == 4
assert done['process_exit_code'] == 0 and done['checkpoint_complete'] and done['metrics_finite']
PY
        printf '%s_CHECKPOINT=complete\n' "${arm}" >> "${control}/pair.status"
        if [[ "${arm}" == A ]]; then
            for attempt in 1 2 3 4 5 6; do
                test -n "$(nvidia-smi -i "${cards}" --query-compute-apps=pid --format=csv,noheader)" || break
                sleep 5
            done
            test -z "$(nvidia-smi -i "${cards}" --query-compute-apps=pid --format=csv,noheader)"
        fi
    done
    exit 0
fi
run_arm A &
a_pid=$!
run_arm B &
b_pid=$!
set +e
wait "${a_pid}"
a_status=$?
wait "${b_pid}"
b_status=$?
printf 'A_EXIT_CODE=%s\nB_EXIT_CODE=%s\n' "${a_status}" "${b_status}" > "${control}/pair.done"
test "${a_status}" = 0 && test "${b_status}" = 0
