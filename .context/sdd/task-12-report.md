Task 12 implemented catalogue pricing for Insights embedding calls. `_embed_batch` extracts response usage, records token usage on the span, and prices the single-call usage through the Orq model catalogue. `embed_texts` adds each batch's priced or unpriced usage to the optional `UsageLedger` and emits one model-specific warning per `embed_texts` call when returned usage is unpriced. `_build_dimension` now forwards the run ledger.

Catalogue check: with the existing project `.env`, `openai/text-embedding-3-small` returned `total_cost=0.00002` for 1,000 input tokens.

Tests and checks: `uv run pytest tests/insights -v` passed (169 passed); `uv run ruff check src` passed; `uv run ruff format --check src` passed; `uv run basedpyright` passed (0 errors, 0 warnings, 0 notes); `git diff --check` passed.

Concerns: none.

Commit SHA: bfb0d1f7.
