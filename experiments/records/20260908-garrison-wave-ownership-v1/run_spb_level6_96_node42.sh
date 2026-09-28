#!/usr/bin/env bash
# Same frozen eight-GPU training entry; Level6 and node42 paths only.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SPB_DEPLOY_ROOT=/root/new-scii-spb-level6-96-20260908 SC2_DIFFICULTY=Harder
export SPB_RUN_RECORD=${record_dir}/spb-rl4.md
export CANDIDATE_RUN_ID=${CANDIDATE_RUN_ID:-20260908-garrison-wave-ownership-v1-spb2-l6-g2-rl96-8gpu-s0-r6}
export CANDIDATE_RAY_TMPDIR=${CANDIDATE_RAY_TMPDIR:-/dev/shm/sc2_spb_l6_96_0908_r6}
exec bash "${record_dir}/run_spb_level5_96_node43.sh"
