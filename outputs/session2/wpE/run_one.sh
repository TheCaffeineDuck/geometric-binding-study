#!/bin/bash
# Run one WP-E config, resumable (re-running continues from its checkpoint; the result JSON is
# written atomically with full parameters). Same line on macOS and Linux: the engine command has
# no macOS-only parts; caffeinate wraps it only where it exists.
# Usage (from anywhere): outputs/session2/wpE/run_one.sh <name> [configs_dir] [results_dir]
# Env: PYTHON (default <repo>/.venv/bin/python), NUMBA_NUM_THREADS (default 2; the configs use 2 threads)
# Log: outputs/session2/wpE/logs/<name>.log   Checkpoints: checkpoints/session2/wpE/<name>/ (never committed)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
NAME="$1"
CONF="${2:-outputs/session2/wpE/configs}"
RES="${3:-outputs/session2/wpE/results}"
PY="${PYTHON:-$REPO/.venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "python not found at $PY; set PYTHON (from a worktree: the main repo's .venv/bin/python)" >&2
    exit 2
fi
cd "$REPO"
mkdir -p "$RES" outputs/session2/wpE/logs "checkpoints/session2/wpE/$NAME"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-2}"
CMD=("$PY" -m swirl.engine.run "$CONF/$NAME.json" --out "$RES/$NAME.json"
     --checkpoint-dir "checkpoints/session2/wpE/$NAME" --verbose)
if command -v caffeinate >/dev/null 2>&1; then
    CMD=(caffeinate -i "${CMD[@]}")
fi
exec "${CMD[@]}" >> "outputs/session2/wpE/logs/$NAME.log" 2>&1
