#!/bin/bash
# Run one WP-P3 config, resumable (re-running continues from its checkpoint; result JSON written atomically).
# Usage (from anywhere): outputs/session3/wpP3/run_one.sh <name>
# Env: PYTHON (default <repo>/.venv/bin/python), NUMBA_NUM_THREADS (default 2; must match the config's threads)
# Log: outputs/session3/wpP3/logs/<name>.log   Checkpoints: checkpoints/session3/wpP3/<name>/ (never committed)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
NAME="$1"
CONF="outputs/session3/wpP3/configs"
RES="outputs/session3/wpP3/results"
PY="${PYTHON:-$REPO/.venv/bin/python}"
cd "$REPO"
mkdir -p "$RES" outputs/session3/wpP3/logs "checkpoints/session3/wpP3/$NAME"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-2}"
CMD=("$PY" -m swirl.engine.run "$CONF/$NAME.json" --out "$RES/$NAME.json"
     --checkpoint-dir "checkpoints/session3/wpP3/$NAME" --verbose)
if command -v caffeinate >/dev/null 2>&1; then
    CMD=(caffeinate -i "${CMD[@]}")
fi
exec "${CMD[@]}" >> "outputs/session3/wpP3/logs/$NAME.log" 2>&1
