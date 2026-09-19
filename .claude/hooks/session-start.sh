#!/usr/bin/env bash
# Prepare the environment so tests and the mock pipeline run immediately.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 0

if [ ! -d .venv ]; then
  uv venv --python 3.11 >/dev/null 2>&1 || exit 0
fi
uv pip install -q -e '.[dev]' >/dev/null 2>&1

echo "parlay-engine ready: .venv/bin/python -m pytest | python run_pipeline.py --sport nfl --mock"
