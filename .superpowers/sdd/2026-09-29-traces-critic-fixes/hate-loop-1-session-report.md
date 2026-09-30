# Browser session isolation follow-up

Added a regression test that blocks the fake provider search for two browser sessions and verifies both searches start before either is released. `ExplorerStore.load()` starts background work and returns without waiting, so no production change was needed for this concurrency finding.

Validation passed: the two-session warmup test and first-request warmup test (2 passed), `uv run ruff check src`, `uv run ruff format --check src`, and bare `uv run basedpyright`.
