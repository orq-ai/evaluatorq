# Stale hydration test fix

The hydration-controlled explorer tests now pass `warm_trajectories=False` on loads that would otherwise await their deliberately blocked hydrator before the test can release it. The tests still trigger and assert the same explicit hydration, cache, and generation behavior.

Validation: `uv run pytest tests/trace_finder/test_explorer.py -v` passed all 17 tests; `uv run ruff check src` passed. The two classifier test helpers now construct default dimensions inside the helper call instead of in parameter defaults. `uv run pytest tests/trace_finder/test_classifier.py -v` passed all 40 tests, and bare `uv run basedpyright` passed with 0 errors and 0 warnings.
