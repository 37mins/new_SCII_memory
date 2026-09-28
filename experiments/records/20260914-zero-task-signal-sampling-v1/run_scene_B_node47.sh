#!/usr/bin/env bash
# One B-only retry; the shared staging start sets the entire two-hour deadline.
set -euo pipefail
test "${SC2_SCENE_B_RETRY_AUTHORIZED:-0}" = 1
stage=/mnt/data0601/yangjinyue/new_SCII/staging/20260915-scene-B-r2
root=/dev/shm/sc2-scene-B-retry-20260915-r2
control=${root}/control
mkdir "${control}"
deadline=$(cat "${stage}/deadline_unix.txt")
finish() {
    rc=$?
    date -u +%FT%TZ > "${control}/ended_at.txt"
    echo "${rc}" > "${control}/done"
}
trap finish EXIT
bound() {
    remaining=$((deadline - $(date +%s) - 20))
    test "${remaining}" -gt 0
    timeout --signal=INT --kill-after=15s "${remaining}" "$@"
}
date -u +%FT%TZ > "${control}/started_at.txt"
echo waiting-for-staging > "${control}/stage.txt"
while ! test -f "${stage}/stage.done"; do
    test "$(date +%s)" -lt "${deadline}"
    sleep 5
done
test "$(cat "${stage}/stage.done")" = 0
echo extracting-verified-dependencies > "${control}/stage.txt"
bound sha256sum -c "${stage}/python.tar.sha256"
mkdir "${root}/deps" "${root}/tmp"
bound tar -C "${root}/deps" -xf "${stage}/python.tar"
export SOURCE_SHA=8c2f0f8b103dab054a612ce95a041e4dde9c7c8c
source_dir=${root}/code-8c2f0f8
bound git clone --no-checkout "${stage}/runtime-8c2f0f8.bundle" "${source_dir}"
git -C "${source_dir}" checkout --detach "${SOURCE_SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
cd "${source_dir}"
bound git bundle verify "${stage}/runtime-8c2f0f8.bundle"
export FEEDBACK_PYTHON_OVERLAY=${root}/deps/python FEEDBACK_INCLUDE_ROOT=${root}/deps/include
export FEEDBACK_STORAGE_ROOT=${stage}/shared
export PYTHON_BIN=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=${root}/tmp PYTHONPATH=${source_dir}:${FEEDBACK_PYTHON_OVERLAY}
export CPATH=${FEEDBACK_INCLUDE_ROOT}/python3.12:${FEEDBACK_INCLUDE_ROOT}
echo cpu-validation > "${control}/stage.txt"
bound env CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" -B -m unittest discover -s tests -p test_signal_sampling_record.py -v > "${control}/cpu-signal.log" 2>&1
bound env CUDA_VISIBLE_DEVICES= "${PYTHON_BIN}" -B -m unittest discover -s tests -p test_a4_continuation.py -v > "${control}/cpu-continuation.log" 2>&1
bound sha256sum -c configs/system-v1.sha256 > "${control}/system-hashes.log"
export SC2_SCENE_SAMPLING=1 SC2_SCENE_SAMPLING_AUTHORIZED=1
export RUN_ID=20260914-zero-task-signal-sampling-v1-spb2-A4-l6-scene-B-rl4-4gpu-r2
export OUTPUT_DIR=/mnt/data0601/yangjinyue/new_SCII/experiments/20260914-zero-task-signal-sampling-v1/runs/${RUN_ID}
export RAY_TMPDIR=/dev/shm/sc2_scene_B_0915_r2
entry=${source_dir}/experiments/records/20260914-zero-task-signal-sampling-v1/run_arm_node43.sh
bound env DRY_RUN=1 VERIFY_ASSETS=0 RUN_ID=scene-B-r2-dry OUTPUT_DIR="${control}/scene-B-r2-dry" bash "${entry}" B > "${control}/dry-run.log" 2>&1
bound "${PYTHON_BIN}" "${root}/check_scene_B_retry.py" compare "${control}/scene-B-r2-dry" > "${control}/config-comparison.json"
test -z "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)"
test -z "$(pgrep -x raylet || true)"
bound "${PYTHON_BIN}" - "${control}" <<'PY'
import json,os,subprocess,sys
from pathlib import Path
mem={s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
assert mem['MemAvailable'] > 600*1024**2,'Insufficient host-memory headroom'
record=dict(mem_available_kib=mem['MemAvailable'],gpus=[0,1,2,3],
            gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
            environment={k:os.environ[k] for k in ['SOURCE_SHA','FEEDBACK_STORAGE_ROOT','FEEDBACK_PYTHON_OVERLAY','FEEDBACK_INCLUDE_ROOT','CPATH','RAY_TMPDIR']})
with (Path(sys.argv[1])/'resource-preflight.json').open('x') as f: json.dump(record,f,indent=2)
PY
test ! -e "${OUTPUT_DIR}"
test ! -e "${RAY_TMPDIR}"
echo training-B > "${control}/stage.txt"
date -u +%FT%TZ > "${control}/train-started_at.txt"
bound bash "${entry}" B > "${control}/train-B.log" 2>&1
bound "${PYTHON_BIN}" "${root}/check_scene_B_retry.py" compare "${OUTPUT_DIR}" > "${control}/actual-config-comparison.json"
bound "${PYTHON_BIN}" "${root}/check_scene_B_retry.py" finish "${OUTPUT_DIR}" > "${control}/acceptance.json"
echo completed > "${control}/stage.txt"
