#!/usr/bin/env bash
# Eight GPU A then B, at most eight hours total. No retry or long-run promotion.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
test "${SC2_GAMMA_AB_AUTHORIZED:-0}" = 1
test -z "$(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)"
control=/root/new-scii-spb-fsdp-20260910/gamma-ab-r2
mkdir "${control}"
exec > >(tee "${control}/console.log") 2>&1
date -u +%FT%TZ | tee "${control}/started_at.txt"
trap 'status=$?; printf "EXIT_CODE=%s\n" "$status" > "${control}/.done"; date -u +%FT%TZ > "${control}/ended_at.txt"' EXIT
deadline=$(( $(date +%s) + 28800 ))
for arm in A B; do
    remaining=$(( deadline - $(date +%s) - 60 ))
    test "${remaining}" -gt 0
    date -u +%FT%TZ > "${control}/${arm}-started_at.txt"
    timeout --signal=INT --kill-after=60s "${remaining}" \
        bash "${record_dir}/run_gamma_ab_node44.sh" "${arm}"
    if [[ "${arm}" == A ]]; then gamma=0.95; else gamma=1.0; fi
    run=/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs/20260910-spb-fsdp-layer-wrap-v1-spb2-l6-g2-gamma${gamma}-s4plus8-8gpu-s0-r2
    "${PYTHON_BIN:-/usr/bin/python3}" - "${run}" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
result = json.loads((run/'result.json').read_text())
assert result == json.loads((run/'.done').read_text())
assert result['status'] == 'completed' and result['process_exit_code'] == 0
assert result['start_step'] == 4 and result['last_global_step'] == 12
assert result['completed_steps'] == 8 and result['checkpoint_complete']
assert result['episode_outcomes']['natural_games'] == 64
seeds = [json.loads(x) for x in (run/'training_seeds.jsonl').read_text().splitlines()]
expected = [2732, 43567, 42613, 52416, 45891, 21243, 30403, 32103]
assert len(seeds) == 8
assert all(row['reset_index'] == i and row['seeds'] == [s]*8
           for i, (row,s) in enumerate(zip(seeds,expected)))
print('Gamma arm accepted as completed, not learning-improvement acceptance:', run.name)
PY
done
