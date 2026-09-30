# RES-1680 Task 8 implementation report

Implemented opt-in asynchronous Jev classification for tool names absent from `SignalsConfig.tool_roles`. The classifier uses the configured model, submits each tool once with the first observed call arguments, preserves configured roles, and records failed, abstained, or invalid classifications as `other` with a tool-specific warning. The synchronous `compute_signals` path remains model-free. The function is exported from `evaluatorq.signals`.

Added fake-judge tests for explicit-role skipping, sample selection, config immutability, and failure, abstention, and invalid-role fallback. No network calls are used by the tests.

Checks: `uv run ruff check src` passed; `uv run ruff format --check src` passed; `uv run pytest -m 'not integration' tests/signals tests/formats` passed with 294 passed and 4 skipped. `uv run basedpyright` reported 30 errors, all under the gitignored `.context/signals-src` research reference copy; it reported no errors in the implementation or tests.
