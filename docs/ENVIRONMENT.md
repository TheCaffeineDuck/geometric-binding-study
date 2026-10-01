# Environment (WP0, 2026-10-01)

| Item | Value |
|---|---|
| Chip | Apple M4 |
| Performance cores (hw.perflevel0.physicalcpu) | 4 |
| Efficiency cores (hw.perflevel1.physicalcpu) | 6 |
| RAM | 16 GiB (17179869184 bytes) |
| macOS | 26.5.2 |
| Python | 3.12.14 (/opt/homebrew/bin/python3.12), venv at `<repo root>/.venv` |

## Packages (installed with uv, no others beyond their dependencies)

| Package | Version |
|---|---|
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| numba | 0.68.0 |
| llvmlite | 0.50.0 |
| matplotlib | 3.11.2 |
| pytest | 9.1.1 |

Numba smoke test (parallel prange) passed; default numba threads = 10.

## Rules

- Set `NUMBA_NUM_THREADS=1` before importing numba in multiprocessing runs.
- Parallel workers = performance cores (4 on this Mac).
- From a worktree, call the venv Python by absolute path: `/Users/aaronramos/Desktop/SWIRL/.venv/bin/python`.
