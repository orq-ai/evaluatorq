# Stale hydration test fix

The hydration-controlled explorer tests now pass `warm_trajectories=False` on loads that would otherwise await their deliberately blocked hydrator before the test can release it. The tests still trigger and assert the same explicit hydration, cache, and generation behavior.

Validation: `uv run pytest tests/trace_finder/test_explorer.py -v` passed all 17 tests; `uv run ruff check src` passed. Bare `uv run basedpyright` reported 0 errors and 2 existing warnings in `tests/trace_finder/test_classifier.py` and exited 1.
