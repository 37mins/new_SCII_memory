#!/usr/bin/env bash
# Two sequential four-GPU evaluation arms, 16 games total, no automatic retry.
set -euo pipefail
test "${SC2_TEMPERATURE_AB_AUTHORIZED:-0}" = 1
record=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record}" rev-parse --show-toplevel)
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
test -z "$(git -C "${source_dir}" status --porcelain)"
control=/root/new-scii-spb-fsdp-20260910/temperature-ab-r1
mkdir "${control}"
exec > >(tee "${control}/console.log") 2>&1
trap 'status=$?; printf "EXIT_CODE=%s\n" "$status" > "${control}/.done"; date -u +%FT%TZ > "${control}/ended_at.txt"' EXIT
date -u +%FT%TZ | tee "${control}/started_at.txt"
deadline=$(( $(date +%s) + 3600 ))
export PYTHON_BIN=/usr/bin/python3
export PYTHONPATH=${source_dir}:/root/new-scii-spb2-level5-control/python
export PYTHONDONTWRITEBYTECODE=1
test -z "$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader)"
timeout 5s stat /mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs
(cd "${source_dir}" && sha256sum -c configs/system-v1.sha256)
"${PYTHON_BIN}" - <<'PY'
import torch, transformers, ray, gymnasium
import json, hashlib
from pathlib import Path
actual = (torch.__version__, transformers.__version__, ray.__version__, gymnasium.__version__)
assert actual == ('2.7.1+cu128', '4.55.2', '2.48.0', '1.2.3'), actual
print('Dependency verification:', actual, flush=True)
run = Path('/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs/20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1')
manifest = json.loads((run/'manifest.json').read_text())
assert manifest['source_commit'] == '90b3fdc2fb7fadc1230861306b20c27540d9b736'
assert manifest['world_size'] == 4
checkpoint = run/'checkpoints/global_step_24'
state = torch.load(checkpoint/'sc2_state.pt', map_location='cpu', weights_only=False)
assert state['global_step'] == 24
files = {}
for rank in range(4):
    for kind, expected in [('model',5555629511),('optim',10332463827),('extra_state',None)]:
        path = checkpoint/f'actor/{kind}_world_size_4_rank_{rank}.pt'
        size = path.stat().st_size
        assert size == expected if expected else size > 10000
        files[str(path.relative_to(checkpoint))] = size
assert (checkpoint/'data.pt').is_file()
assert json.loads((checkpoint/'actor/config.json').read_text())['swa_dropout'] == 0
proof = dict(checkpoint=str(checkpoint), source_sha=manifest['source_commit'], files=files,
             manifest_sha256=hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),
             size_and_metadata_preflight_only=True, actual_distributed_load_still_required=True)
with Path('/root/new-scii-spb-fsdp-20260910/temperature-ab-r1/checkpoint_preflight.json').open('x') as stream:
    json.dump(proof, stream, indent=2)
print('Checkpoint metadata verified: step24; actual load still required', flush=True)
PY
for arm in A B; do
    remaining=$(( deadline - $(date +%s) - 60 ))
    test "${remaining}" -gt 0
    # Do not override an occupied-card check if another task takes these GPUs.
    test -z "$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader)"
    date -u +%FT%TZ > "${control}/${arm}-started_at.txt"
    set +e
    timeout --signal=INT --kill-after=60s "${remaining}" \
        bash "${record}/run_temperature_eval_node44.sh" "${arm}"
    status=$?
    set -e
    printf 'EXIT_CODE=%s\n' "${status}" > "${control}/${arm}.done"
    if [[ "${status}" != 0 ]]; then exit "${status}"; fi
done
