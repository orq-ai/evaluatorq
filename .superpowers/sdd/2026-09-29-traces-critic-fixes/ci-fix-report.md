# CI fix report

Updated the explorer table signature to satisfy Ruff formatting, excluded ignored `.superpowers` snapshot worktrees from basedpyright, and fixed three real checkout type errors in the trace finder views, run store, and dashboard test.

Checks passed: `uv run ruff check src`, `uv run ruff format --check src`, and bare `uv run basedpyright` (0 errors; 2 pre-existing warnings in `tests/trace_finder/test_classifier.py`). Focused tests passed: `uv run pytest tests/dashboard/test_finder.py tests/trace_finder/test_run_store.py` (99 passed).
