# Task 3: Dense trace table

## Result

Implemented the approved trace table ordering and cell presentation in commit `83b17719` (`feat: densify trace explorer table`). The default table now shows Time, Trace / agent, Status, Model, input tokens, output tokens, cache, cost, duration, and AI match in that order. Status is visible as Success or Error, trace and agent details share one escaped cell, match values remain escaped and visible when classification results exist, and numeric columns use the existing right-aligned tabular-number styling.

The existing toolbar and trajectory renderer were preserved. The only `explorer_views.py` behavior change is the AI match cell label and styling; pre-existing trajectory edits were excluded from the commit.

## Test-driven implementation

Red command: `uv run pytest tests/dashboard/test_explorer.py tests/trace_finder/test_columns.py -q`.

Red result: 2 failed and 65 passed. The default-column order assertion failed because the registry still used the old order, and the combined trace-cell assertion failed because the `trace` column did not exist.

Green command: `uv run pytest tests/dashboard/test_explorer.py tests/trace_finder/test_columns.py -q`.

Green result: 68 passed in 1.75s. The suite covers default order, combined and escaped trace / agent text, visible status, visible escaped AI match text, and numeric cell alignment.

Lint command: `uv run ruff check src/evaluatorq/dashboard/trace_finder src/evaluatorq/trace_finder/columns.py`.

Lint result: all checks passed.

## Files and commit

Commit `83b17719` changes `src/evaluatorq/trace_finder/columns.py`, `src/evaluatorq/dashboard/trace_finder/explorer_views.py`, `src/evaluatorq/dashboard/styles.py`, `tests/trace_finder/test_columns.py`, and `tests/dashboard/test_explorer.py`.

Patch staging was used. The cached diff was checked before committing and contained only the Task 3 hunks. Other sessions' dirty trajectory, cache-label, row-model, docs, launcher, and test changes were left in the working tree.

## Self-review and risks

The time cell formats the existing UTC timestamp as a time and date on separate lines. Trace and agent text use the shared HTML escaping helper, and AI result labels continue to use the shared result color and value formatting helpers.

The compact status mapping treats `ok` and `success` as Success and `error` and `failed` as Error; any other source status remains visible verbatim and escaped. No integration or live trace tests were run because this task changes only table presentation.
