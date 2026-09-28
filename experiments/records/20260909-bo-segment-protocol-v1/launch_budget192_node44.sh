#!/usr/bin/env bash
# Authorized eight-card, four-game B192 evaluation. One attempt, 30 minutes total.
set -euo pipefail
record=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SOURCE_DIR=$(git -C "${record}" rev-parse --show-toplevel)
export SOURCE_SHA=${SOURCE_SHA:?set frozen source SHA}
test "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" = "${SOURCE_SHA}"
test -z "$(git -C "${SOURCE_DIR}" status --porcelain)"
(cd "${SOURCE_DIR}" && sha256sum -c configs/system-v1.sha256)
export SC2_PYTHON_OVERLAY=/root/new-scii-spb2-level5-control/python
export CPATH=/usr/include/python3.12:/usr/include
test -d "${SC2_PYTHON_OVERLAY}"
test -f /usr/include/python3.12/Python.h
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 SC2_EVAL_N_GPUS=8 SC2_EVAL_RESPONSE_LENGTH=192
export RUN_ID=20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-B192-8gpu-eval4-s0-r1
export OUTPUT_DIR=/mnt/data0601/yangjinyue/new_SCII/experiments/20260909-bo-segment-protocol-v1/runs/${RUN_ID}
export RAY_TMPDIR=/dev/shm/boseg192_8g_r1
test ! -e "${OUTPUT_DIR}"
test ! -e "${RAY_TMPDIR}"
control=/root/new-scii-bo-segment-20260909/budget192-8gpu-r1
mkdir "${control}"
cp "${BASH_SOURCE[0]}" "${control}/launcher.sh"
printf '%s\n' "${SOURCE_SHA}" > "${control}/source.txt"
date -Is > "${control}/started_at.txt"
export PAIR_DEADLINE=$(( $(date +%s) + 1800 ))
printf '%s\n' "${PAIR_DEADLINE}" > "${control}/deadline_epoch.txt"
export ARM_STATUS_PATH=${control}/run.done
exec bash "${record}/run_eval.sh" B
