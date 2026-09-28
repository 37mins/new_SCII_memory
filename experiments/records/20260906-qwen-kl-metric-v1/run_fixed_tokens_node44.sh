#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${repo_dir}"
test -z "$(git status --porcelain --untracked-files=normal)"
mode=${1:?choose a diagnostic mode documented in the experiment record}
case "${mode}" in
    paths) export CUDA_VISIBLE_DEVICES=0,1,2,3; suffix=paths-opt0-r11; probe_args=(--max-updates 0) ;;
    updates) export CUDA_VISIBLE_DEVICES=4,5,6,7; suffix=updates-opt4-r13; probe_args=(--max-updates 4 --skip-paths) ;;
    precision-bf16) export CUDA_VISIBLE_DEVICES=0,1,2,3; suffix=param-bf16-opt0-r14; probe_args=(--max-updates 0 --param-dtype bf16) ;;
    precision-fp32) export CUDA_VISIBLE_DEVICES=4,5,6,7; suffix=param-fp32-opt0-r15; probe_args=(--max-updates 0 --param-dtype fp32) ;;
    norm-legacy) export CUDA_VISIBLE_DEVICES=0,1,2,3; suffix=norm-legacy-opt0-r16; probe_args=(--max-updates 0 --param-dtype fp32) ;;
    norm-fp32) export CUDA_VISIBLE_DEVICES=4,5,6,7; suffix=norm-fp32-opt0-r17; probe_args=(--max-updates 0 --param-dtype fp32 --fp32-normalization) ;;
    drift-legacy) export CUDA_VISIBLE_DEVICES=0,1,2,3; suffix=norm-legacy-opt4-r18; probe_args=(--max-updates 4 --param-dtype fp32 --skip-paths --recompute-start) ;;
    drift-fp32) export CUDA_VISIBLE_DEVICES=4,5,6,7; suffix=norm-fp32-opt4-r19; probe_args=(--max-updates 4 --param-dtype fp32 --fp32-normalization --skip-paths --recompute-start) ;;
    residual-bf16) export CUDA_VISIBLE_DEVICES=0,1,2,3; suffix=residual-hfbf16-opt0-r20; probe_args=(--max-updates 0 --param-dtype fp32 --fp32-normalization) ;;
    residual-fp32) export CUDA_VISIBLE_DEVICES=4,5,6,7; suffix=residual-hffp32-opt0-r21; probe_args=(--max-updates 0 --param-dtype fp32 --fp32-normalization --hf-full-precision) ;;
    head-control) export CUDA_VISIBLE_DEVICES=0,1,2,3; suffix=head-control-opt0-r24; probe_args=(--max-updates 0 --param-dtype fp32 --fp32-normalization) ;;
    head-observe) export CUDA_VISIBLE_DEVICES=4,5,6,7; suffix=head-observe-opt0-r25; probe_args=(--max-updates 0 --param-dtype fp32 --fp32-normalization --observe-fp32-head) ;;
    layers-control) export CUDA_VISIBLE_DEVICES=0,1,2,3; suffix=layers-reduction-default-opt0-r26; probe_args=(--max-updates 0 --param-dtype fp32 --fp32-normalization --trace-layers) ;;
    layers-reduction) export CUDA_VISIBLE_DEVICES=4,5,6,7; suffix=layers-reduction-fp32-opt0-r27; probe_args=(--max-updates 0 --param-dtype fp32 --fp32-normalization --trace-layers --disable-bf16-reduced-reduction) ;;
    rms-control) export CUDA_VISIBLE_DEVICES=0,1,2,3; suffix=rms-native-opt0-r28; probe_args=(--max-updates 0 --param-dtype fp32 --fp32-normalization --trace-layers) ;;
    rms-rowwise) export CUDA_VISIBLE_DEVICES=4,5,6,7; suffix=rms-rowwise-opt0-r29; probe_args=(--max-updates 0 --param-dtype fp32 --fp32-normalization --trace-layers --rowwise-rms-norm) ;;
    model-control) export CUDA_VISIBLE_DEVICES=0,1,2,3; suffix=model-bf16-opt4-r30; probe_args=(--max-updates 4 --param-dtype fp32 --fp32-normalization --recompute-start) ;;
    model-fp32) export CUDA_VISIBLE_DEVICES=4,5,6,7; suffix=model-fp32-opt4-r31; probe_args=(--max-updates 4 --param-dtype fp32 --fp32-normalization --recompute-start --model-fp32) ;;
    full-updates) export CUDA_VISIBLE_DEVICES=0,1,2,3; suffix=norm-fp32-opt35-r32; probe_args=(--max-updates 35 --param-dtype fp32 --fp32-normalization --skip-paths --recompute-start) ;;
    full-updates-45) export CUDA_VISIBLE_DEVICES=0,1,2,3; suffix=norm-fp32-opt35-r33; probe_args=(--max-updates 35 --param-dtype fp32 --fp32-normalization --skip-paths --recompute-start) ;;
    *) echo "Unknown fixed-token diagnostic mode: ${mode}" >&2; exit 2 ;;
esac
test -z "$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader)"
export PYTHONPATH=${repo_dir}:/root/new-scii-spb2-level5-control/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python VERL_ATTN_IMPLEMENTATION=eager
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
record_root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-qwen-kl-metric-v1/runs
historical_run=${record_root}/20260906-qwen-kl-metric-v1-qwen-instruct2507-l6-4gpu-integrated-trace-rl1-s0-r7
run_id=20260906-qwen-kl-metric-v1-qwen-instruct2507-fixedtokens-4gpu-${suffix}
output=${record_root}/${run_id}
mkdir "${output}"
export TMPDIR=/dev/shm/qkl_fixedtokens_${suffix}
mkdir "${TMPDIR}"
export TRITON_CACHE_DIR=${TMPDIR}/triton TORCHINDUCTOR_CACHE_DIR=${TMPDIR}/inductor

finish() {
    status=$?
    trap - EXIT
    /usr/bin/python3 -B - "${output}" "${status}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root, status = Path(sys.argv[1]), int(sys.argv[2])
tails = {}
for path in root.glob('rank-*.jsonl'):
    rows = [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]
    tails[path.name] = rows[-1]['event'] if rows else None
complete = len(tails) == 4 and all(value == 'completed' for value in tails.values())
result = dict(status='completed' if status == 0 and complete else 'failed',
    process_exit_code=status, completed_ranks=tails, ended_at=datetime.now(timezone.utc).isoformat(),
    sc2_started=False, resumable_checkpoint=False)
(root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
(root/'.done').write_text(result['status']+'\n')
PY
    exit "${status}"
}
trap finish EXIT
git rev-parse HEAD > "${output}/source_commit.txt"
cp "${BASH_SOURCE[0]}" "${output}/launcher.sh"
/usr/bin/python3 -B - "${output}" "${mode}" "${historical_run}" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
root, mode, historical_run = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
manifest = dict(run_id=root.name, mode=mode, status='preflight',
    started_at=datetime.now(timezone.utc).isoformat(), source_commit=(root/'source_commit.txt').read_text().strip(),
    historical_run=historical_run, gpu_selection=os.environ['CUDA_VISIBLE_DEVICES'], world_size=4,
    max_optimizer_updates=35 if mode.startswith('full-updates') else 4 if mode=='updates' or mode.startswith(('drift-', 'model-')) else 0, max_gpu_seconds=2700 if mode=='full-updates-45' else 1800,
    param_dtype='fp32' if mode.startswith(('norm-', 'drift-', 'residual-', 'head-', 'layers-', 'rms-', 'model-', 'full-')) else mode.removeprefix('precision-') if mode.startswith('precision-') else None,
    fp32_normalization=mode in ('norm-fp32', 'drift-fp32') or mode.startswith(('residual-', 'head-', 'layers-', 'rms-', 'model-', 'full-')), recompute_start=mode.startswith(('drift-', 'model-', 'full-')),
    hf_full_precision=mode=='residual-fp32', observe_fp32_head=mode=='head-observe',
    trace_layers=mode.startswith(('layers-', 'rms-')), disable_bf16_reduced_reduction=mode=='layers-reduction', rowwise_rms_norm=mode=='rms-rowwise',
    model_fp32=mode=='model-fp32',
    full_historical_batch=mode.startswith('full-updates'),
    user_authorization='2026-09-06 evening: complete historical batch with 35 optimizer updates before any more candidates; 45-minute rerun requires the separately requested approval' if mode.startswith('full-updates') else '2026-09-06: continue resolving cached-path residual' if mode.startswith(('layers-', 'rms-')) else '2026-09-06: approved temperature precision A/B and bounded drift diagnosis',
    model_id='Qwen/Qwen3-4B-Instruct-2507', model_revision='cdbee75f17c01a7cc42f958dc650907174af0554',
    model_path='/mnt/data0601/yangjinyue/new_SCII/assets/Qwen3-4B-Instruct-2507-cdbee75',
    project_sft=False, resumed_checkpoint=False, sc2_started=False)
(root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
PY
cp configs/qwen3-4b-instruct-2507.sha256 "${output}/model.sha256"
sha256sum -c configs/system-v1.sha256 > "${output}/preflight.log"
(cd /mnt/data0601/yangjinyue/new_SCII/assets/Qwen3-4B-Instruct-2507-cdbee75 && \
    sha256sum -c "${output}/model.sha256") >> "${output}/preflight.log"
/usr/bin/python3 -B - <<'PY'
import torch, transformers, ray
assert (torch.__version__, transformers.__version__, ray.__version__) == ('2.7.1+cu128','4.55.2','2.48.0')
PY
probe_timeout=1800
if [[ "${mode}" == full-updates-45 ]]; then probe_timeout=2700; fi
if [[ "${mode}" == rms-rowwise ]]; then
    timeout --signal=INT --kill-after=15s 180 /usr/bin/python3 -B tests/test_qwen_rowwise_norm.py --gpu > "${output}/rms-kernel-check.json"
    probe_timeout=1600
fi
timeout --signal=INT --kill-after=15s "${probe_timeout}" /usr/bin/python3 -B -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=4 -- \
    "${repo_dir}/experiments/records/20260906-qwen-kl-metric-v1/replay_fixed_tokens.py" \
    --source "${repo_dir}" --run "${historical_run}" --output "${output}" "${probe_args[@]}" \
    2>&1 | tee "${output}/console.log"
