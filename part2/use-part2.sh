#!/usr/bin/env bash
# Source from Bash after activating the Python 3.12 reconstruction environment.
: "${P2_WORK:?Set P2_WORK to a writable working directory}"
: "${P2_DATA_ROOT:?Set P2_DATA_ROOT to the directory containing the three course scenes}"
: "${P2_SRC:=$P2_WORK/i3dgs}"

mkdir -p -- "$P2_WORK/cache/cupy" "$P2_WORK/tmp" || return
P2_WORK=$(cd -- "$P2_WORK" && pwd -P) || return
P2_SRC=$(cd -- "$P2_SRC" && pwd -P) || return
P2_DATA_ROOT=$(cd -- "$P2_DATA_ROOT" && pwd -P) || return
export P2_WORK P2_SRC P2_DATA_ROOT
export TORCH_HOME="$P2_WORK/weights/torch"
export XDG_CACHE_HOME="$P2_WORK/cache"
export CUPY_CACHE_DIR="$P2_WORK/cache/cupy"
export TMPDIR="$P2_WORK/tmp"
