#!/usr/bin/env bash
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Node 47's root filesystem is full; only relocate writable caches, not KV-cache behavior.
runtime=/dev/shm/spb-unitid-20260906
export SPB_PYTHON_DIR=${runtime}/python
export TMPDIR=${runtime}/tmp XDG_CACHE_HOME=${runtime}/cache
export HF_HOME=${XDG_CACHE_HOME}/huggingface
export TRITON_CACHE_DIR=${XDG_CACHE_HOME}/triton
export TORCHINDUCTOR_CACHE_DIR=${XDG_CACHE_HOME}/torchinductor
export TORCH_EXTENSIONS_DIR=${XDG_CACHE_HOME}/torch_extensions
export CPATH=${runtime}/include/python3.12:${runtime}/include
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"
exec bash "${record_dir}/run_spb_node46.sh"
