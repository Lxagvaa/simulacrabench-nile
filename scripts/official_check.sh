#!/usr/bin/env bash
# Run the organizers' grader exactly as the platform would (fresh venv, network cut),
# on all three sandbox instruments. Slower than `make eval`, but authoritative for
# "does it run / is the output legal". Add --docker to mimic the container limits.
set -euo pipefail
cd "$(dirname "$0")/../starter_kit"
PHASE="${PHASE:-1}"
for s in unicef world_bank unhcr; do
  uv run python score.py --submission ../submission --data _sandbox/$s \
      --schema data/$s.json --phase "$PHASE" "$@"
done
