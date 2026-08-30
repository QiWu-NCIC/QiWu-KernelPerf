#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?dataset root}
MANIFEST=${2:?manifest path}
PROXY=${3:?proxy URL}
python3 "$HOME/yjk/KernelPerf/scripts/download_matrix_manifest.py" \
  --manifest "$MANIFEST" \
  --root "$ROOT" \
  --proxy "$PROXY"
echo "downloaded $(find "$ROOT" -type f -name '*.mtx' | wc -l) Matrix Market files"
