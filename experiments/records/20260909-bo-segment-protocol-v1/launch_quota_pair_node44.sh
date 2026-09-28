#!/usr/bin/env bash
# Approved four-card serial pair: 50 minutes total, no PPO and no retries.
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
test -z "$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader)"
control=/root/new-scii-bo-segment-20260909/quota-pair-r1
run_root=/mnt/data0601/yangjinyue/new_SCII/experiments/20260909-bo-segment-protocol-v1/runs
for arm in A B; do
  test ! -e "${run_root}/20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-quota-${arm}-eval4-s0-r1"
  test ! -e "/dev/shm/quota_${arm}_r1"
done
mkdir "${control}"
cp "${BASH_SOURCE[0]}" "${control}/launcher.sh"
printf '%s\n' "${SOURCE_SHA}" > "${control}/source.txt"
date -Is > "${control}/started_at.txt"
export PAIR_DEADLINE=$(( $(date +%s) + 3000 ))
printf '%s\n' "${PAIR_DEADLINE}" > "${control}/deadline_epoch.txt"
finish_pair() {
  status=$?
  printf 'EXIT_CODE=%s\n' "${status}" > "${control}/pair.done"
  date -Is >> "${control}/pair.done"
}
trap finish_pair EXIT
for arm in A B; do
  if [[ "${arm}" == B ]]; then
    # A failed arm exits through set -e; do not start B or retry it.
    test $(( PAIR_DEADLINE - $(date +%s) )) -ge 1200
    for check in {1..30}; do
      test -z "$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader)" && break
      sleep 1
    done
  fi
  test -z "$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader)"
  export RUN_ID=20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-quota-${arm}-eval4-s0-r1
  export OUTPUT_DIR=${run_root}/${RUN_ID} RAY_TMPDIR=/dev/shm/quota_${arm}_r1
  export ARM_STATUS_PATH=${control}/${arm}.done
  bash "${record}/run_quota_eval.sh" "${arm}" > "${control}/${arm}.log" 2>&1
done
