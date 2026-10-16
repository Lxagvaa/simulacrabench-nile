#!/usr/bin/env bash
# Package submission/ into dist/submission.zip (main.py at the archive root) and validate it.
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/fetch_weights.sh
mkdir -p dist
rm -f dist/submission.zip
( cd submission && zip -rq ../dist/submission.zip . -x '*__pycache__*' -x '*.pyc' -x '.DS_Store' )
uv run python starter_kit/tools/check_submission_zip.py dist/submission.zip
ls -lh dist/submission.zip
