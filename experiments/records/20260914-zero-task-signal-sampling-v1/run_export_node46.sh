#!/usr/bin/env bash
# CPU export is included in the authorized three-hour two-GPU experiment budget.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
test "${SC2_TWO_GPU_AUTHORIZED:-0}" = 1
export_gpu=${SC2_EXPORT_GPU:-1}
[[ "${export_gpu}" =~ ^[0-7]$ ]] || exit 2
root=/dev/shm/sc2-a4-two-gpu-20260915-r1
control=${root}/export-r3
mkdir "${control}"
finish() {
    rc=$?
    date -u +%FT%TZ > "${control}/ended_at.txt"
    printf '%s\n' "${rc}" > "${control}/done"
}
trap finish EXIT
test -z "$(nvidia-smi -i "${export_gpu}" --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=${export_gpu} PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_HOME=${root}/hf
assets=/mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets
deps=${assets}/repro/stageguard_v7_legacy_contract_20260806/source/extracted
export PYTHONPATH=/root/new-scii-spb-unitid-20260906/python:${deps}/flash-linear-attention_dev:${deps}/MoBA
parent=/mnt/data0601/yangjinyue/new_SCII/staging/20260915-scene-B-r2/shared/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/20260911-execution-failure-feedback-v1-spb2-original-l6-g2-A-rl4-4gpu-s0-r2
timeout 5s stat "${parent}"
ln -s "${assets}/models/Panyuqi/SpikingBrain-2.0-instruct" "${control}/model_source"
/usr/bin/python3 - "${root}" "${parent}" "${SOURCE_SHA}" "${control}" <<'PY'
import datetime, json, os, sys, time
from pathlib import Path
root, parent = map(Path, sys.argv[1:3])
assert json.loads((parent/'manifest.json').read_text())['source_commit'] == '361cb7ae7fd86fe8811d6fcb6946d0241bf93e11'
result = json.loads((parent/'.done').read_text())
assert result['process_exit_code'] == 0 and result['checkpoint_complete'] and result['completed_steps'] == 4
free = os.statvfs(root)
assert free.f_bavail*free.f_frsize > 80*1024**3
mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
assert int(mem['MemAvailable'].split()[0])*1024 > 350*1024**3
budget = json.loads((root/'budget.json').read_text())
assert budget['deadline_unix'] > time.time(), 'Original three-hour budget exhausted'
with (Path(sys.argv[4])/'launch.json').open('x') as out:
    json.dump(dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        deadline_unix=budget['deadline_unix'], source_sha=sys.argv[3],
        parent=str(parent), export_context_gpus=[int(os.environ['CUDA_VISIBLE_DEVICES'])], model_path_alias_only=True,
        model_forward_calls=0, optimizer_updates=0, automatic_retry=False), out, indent=2)
PY
remaining=$(/usr/bin/python3 -c 'import json,time,sys; print(min(1800,int(json.load(open(sys.argv[1]))["deadline_unix"]-time.time())))' "${root}/budget.json")
test "${remaining}" -gt 0
timeout --signal=INT --kill-after=10s "${remaining}" /usr/bin/python3 -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=4 \
    "${record_dir}/export_a4_full_cpu.py" --checkpoint "${parent}/checkpoints/global_step_4" \
    --model "${control}/model_source" --output "${control}/full"
