#!/usr/bin/env bash
# TabPFN v2 classifier checkpoint (Prior Labs, Apache 2.0 with attribution),
# pinned to a fixed commit of Prior-Labs/TabPFN-v2-clf.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=f851f2a3c941544733b712d8c0f96dfae9b28862
SHA256=f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f
OUT=submission/weights/tabpfn-v2-classifier.ckpt
mkdir -p "$(dirname "$OUT")"
if [ ! -f "$OUT" ]; then
  curl -fL "https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/$COMMIT/tabpfn-v2-classifier.ckpt" -o "$OUT"
fi
echo "$SHA256  $OUT" | shasum -a 256 -c -
