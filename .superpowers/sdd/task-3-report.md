# Task 3 report: red-team pipeline topology

## Completed

- `red_team()` now owns the complete `tracing_session(..., trace_type='redteam')` and `orq.redteam.pipeline` span.
- Added `RedTeamRunMetrics`; dynamic/hybrid and static runners return `(report, metrics)` for outer metric attribution.
- Moved the run ID from the dynamic inner owner to the outer tracing-session context.
- Added pipeline child spans for optional recommendations and executive-summary generation.
- Removed the dynamic runner's local tracing lifecycle, parent-context capture, and pipeline span; removed the prior trailing flush-only work-in-progress because the session owns flushing.
- Updated runner/lifecycle mocks and E2E tracing seams for the new handoff and session model.
- Added dynamic and static whole-run topology coverage, including report-child parentage and static dispatch under the pipeline span.

## Validation

- `.venv/bin/pytest tests/redteam/test_runner.py tests/redteam/test_hooks_lifecycle.py tests/redteam/test_tracing.py tests/redteam/test_tracing_spans.py tests/redteam/e2e/test_dynamic_pipeline.py tests/redteam/e2e/test_hybrid_pipeline.py tests/redteam/e2e/test_pipeline_options.py -q` — 112 passed.
- `ruff check src/evaluatorq/redteam/runner.py` — passed.
- `.venv/bin/basedpyright tests/redteam/test_tracing_spans.py src/evaluatorq/redteam/runner.py` — 0 errors.
- `git diff --check` — passed.

## Review

- Spec review found that the topology test initially identified the recommendations child only by parent. It now identifies it by both expected name and parent ID.
- Standards review found type-check issues in the new tracing test. Those are resolved; basedpyright passes.

## Scope

No static job/scorer span work was performed; that remains Task 4 scope.
