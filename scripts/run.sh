#!/usr/bin/env bash
# Run a Rust binary with the vendored libtorch (CUDA).
# libtorch 2.13.0 has a linking quirk: libtorch_cuda.so is dropped by the
# linker's --as-needed, so it must be preloaded at runtime.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$ROOT/libtorch/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ROOT/libtorch/lib/libtorch_cuda.so:${LD_PRELOAD:-}"
exec "$@"
