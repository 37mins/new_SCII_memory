#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
source_dir=${1:?frozen ca7885d runtime directory}
source_sha=ca7885d886aaaafb39fc105b1d5d46cddf4bcc5a
replay_node=${R4_REPLAY_NODE:-46}
python_dir=${R4_REPLAY_PYTHON_DIR:-/root/new-scii-spb-unitid-20260906/python}
attempt=${R4_REPLAY_ATTEMPT:-1}
max_seconds=${R4_REPLAY_MAX_SECONDS:-1800}
test -z "$(git -C "${repo_dir}" status --porcelain --untracked-files=normal)"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-integrated-fixes-v1/runs
historical=${root}/20260906-integrated-fixes-v1-qwen-instruct2507-l6-8gpu-fp32-rl96-gate8-s0-r4-train8
run_id=20260906-integrated-fixes-v1-r4-step2-first-update-8gpu-node${replay_node}-r${attempt}
output=${root}/${run_id}
timeout 5s stat "${historical}" >/dev/null
mkdir "${output}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=${source_dir}:${python_dir}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python VERL_ATTN_IMPLEMENTATION=eager
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/dev/shm/sc2_r4_first_update_node${replay_node}_r${attempt}
mkdir "${TMPDIR}"
export TRITON_CACHE_DIR=${TMPDIR}/triton TORCHINDUCTOR_CACHE_DIR=${TMPDIR}/inductor
finish() {
    status=$?
    trap - EXIT
    /usr/bin/python3 -B - "${output}" "${status}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p, code = Path(sys.argv[1]), int(sys.argv[2])
tails = {}
for f in p.glob('rank-*.jsonl'):
    lines = [json.loads(s) for s in f.read_text().splitlines(keepends=True) if s.endswith('\n')]
    tails[f.name] = lines[-1]['event'] if lines else None
complete = code == 0 and len(tails) == 8 and all(v == 'completed' for v in tails.values())
result = dict(status='completed' if complete else 'failed', process_exit_code=code,
    ended_at=datetime.now(timezone.utc).isoformat(), rank_tails=tails, sc2_started=False,
    diagnostic_only=True, max_optimizer_updates=1)
for name in ('result.json', '.done'):
    with (p / name).open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
PY
    exit "${status}"
}
trap finish EXIT
cp "${BASH_SOURCE[0]}" "${output}/launcher.sh"
/usr/bin/python3 -B - "${historical}" "${output}" "${repo_dir}" "${source_dir}" "${source_sha}" "${replay_node}" "${max_seconds}" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys
from datetime import datetime, timezone
run, output, repo, source = map(pathlib.Path, sys.argv[1:5])
sha = sys.argv[5]
manifest = json.loads((run/'manifest.json').read_text())
assert manifest['source_commit'] == sha and manifest['world_size'] == 8
assert manifest['model_id'] == 'Qwen/Qwen3-4B-Instruct-2507'
assert (run/'checkpoints/latest_checkpointed_iteration.txt').exists()
last = int((run/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip())
pending = [p.name for p in (run/'checkpoints').glob('global_step_*') if int(p.name.split('_')[-1]) > last]
assert not pending, f'Current RL is saving checkpoint {pending}; do not contend with it'
files = {}
for rel in subprocess.check_output(['git','-C',str(repo),'ls-tree','-r','--name-only',sha], text=True).splitlines():
    # Verify frozen executable/importable source, never the newest worktree.
    if rel.endswith(('.py', '.yaml', '.yml')):
        expected = subprocess.check_output(['git','-C',str(repo),'show',f'{sha}:{rel}'])
        digest = hashlib.sha256((source/rel).read_bytes()).hexdigest()
        assert digest == hashlib.sha256(expected).hexdigest(), rel
        files[rel] = digest
checkpoint = run/'checkpoints/global_step_1/actor'
parts = [checkpoint/f'{kind}_world_size_8_rank_{r}.pt' for kind in ('model','optim','extra_state') for r in range(8)]
assert all(p.is_file() and p.stat().st_size > 0 for p in parts)
record = dict(run_id=output.name, status='starting', started_at=datetime.now(timezone.utc).isoformat(),
    diagnostic_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
    runtime_commit=sha, historical_run=str(run), checkpoint=str(checkpoint),
    model=manifest, source_sha256=files, gpu_selection='0,1,2,3,4,5,6,7', node=f'sc2-a100-{sys.argv[6]}',
    max_gpu_seconds=int(sys.argv[7]), compiler_include_path=os.environ.get('CPATH'),
    max_optimizer_updates=1, samples_updated=8, samples_checked=184,
    checkpoint_files={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in parts},
    authorization=f'2026-09-07: eight GPUs / {sys.argv[7]} seconds / one update; node47 r2 user approved only CPATH and a new RUN',
    parameter_evidence='16 deterministic blocks per flat shard; gradients/moments/parameter delta plus full-shard gradient norms',
    sc2_started=False, numerical_or_loss_changes=False)
(output/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
PY
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -B - <<'PY'
import torch, transformers, ray
assert (torch.__version__, transformers.__version__, ray.__version__) == ('2.7.1+cu128','4.55.2','2.48.0')
PY
# Keep child shutdown inside the authorized per-attempt time budget.
timeout --signal=INT --kill-after=15s "$((max_seconds - 15))" /usr/bin/python3 -B -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 -- \
    "${repo_dir}/experiments/records/20260906-integrated-fixes-v1/replay_r4_first_update.py" \
    --source "${source_dir}" --run "${historical}" --output "${output}" \
    --checkpoint "${historical}/checkpoints/global_step_1/actor" \
    2>&1 | tee "${output}/console.log"
