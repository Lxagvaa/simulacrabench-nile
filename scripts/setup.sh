#!/usr/bin/env bash
# One-time setup: environment, starter kit, sandbox data, TabPFN checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync
./scripts/fetch_weights.sh
[ -d starter_kit ] || git clone --depth 1 https://github.com/SituatedEvals/public.git starter_kit
cd starter_kit
for s in unicef world_bank unhcr sample; do
  uv run python make_sandbox.py --schema data/$s.json --out _sandbox/$s
done
echo "Done. Next: make eval"
