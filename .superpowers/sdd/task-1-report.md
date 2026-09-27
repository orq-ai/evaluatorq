# Task 1: Compact AI strip and one classification results surface

## Files and commit

Changed `src/evaluatorq/dashboard/trace_finder/views.py` to add a compact `/traces` command strip, submit review mode and settings-backed limit/parallelism values, and render progress plus classifier/filter details without the legacy dot matrix or matched-only table. The strip retains `finder-query-form`, `scope`, CSRF, query, `#finder-body`, and the separate `#explorer-results` target. `/find` continues to use its separate `search_views` page builder.

Changed `tests/dashboard/test_finder.py` with regression coverage for the command strip, payload defaults, result surfaces, compiling progress, and `/find` separation. Added this report at `.superpowers/sdd/task-1-report.md`.

The implementation, regression tests, and report are included in the Task 1 conventional commit `feat: compact AI trace strip and results surface`.

## TDD evidence

RED: added the `/traces` and `/find` behavior tests, then ran `uv run pytest tests/dashboard/test_finder.py::test_traces_page_uses_compact_ai_strip_and_one_classification_surface tests/dashboard/test_finder.py::test_find_keeps_legacy_search_hero_separate_from_traces -q`. The new `/traces` test failed at the missing `Search` control; the separate legacy `/find` assertion passed.

GREEN: after implementation, ran `uv run pytest tests/dashboard/test_finder.py::test_traces_page_uses_compact_ai_strip_and_one_classification_surface tests/dashboard/test_finder.py::test_find_keeps_legacy_search_hero_separate_from_traces tests/dashboard/test_finder.py::test_find_idle_page_is_standalone_legacy_search_and_reuses_facets -q`; result: `3 passed`.

The required focused suites ran with `uv run pytest tests/dashboard/test_finder.py tests/dashboard/test_explorer.py -q`; result: `100 passed, 2 failed`. Both failures are existing `/find` tests that POST `/find/run` without `surface=search`: `test_async_controls_and_trace_drawer_have_request_feedback` expects a legacy trace drawer link in the default `/traces` fragment, and `test_find_run_starts_polling_and_completed_poll_shows_matches` expects included rows there. The route selects the search surface from the submitted `surface` field; the production `/find` HTMX page supplies it through its enclosing `hx-vals`. The Task 1 changes do not modify the routes or legacy search builder.

Lint ran with `uv run ruff check src/evaluatorq/dashboard/trace_finder/views.py src/evaluatorq/dashboard/trace_finder/routes.py`; result: `All checks passed!`. `git diff --check` passed for the two changed source/test files.

## Self-review

The gear link is named `AI configuration`, has a minimum 42px target, and follows the scope toggle before `Search` in markup order. The hidden review mode remains attached to the query form; `/find/start` review behavior and the editable awaiting-review panel remain in place. The default limit and parallelism are submitted as hidden values. Classified, failed, cancelled, and compiling `/traces` states show progress without rendering `field(snapshot)` or `table(snapshot)`; available classifier and filter details remain visible.

## Risks and limits

The two full-suite failures described above remain outside this task's code changes and prevent a completely green focused suite. The new command strip uses class hooks for its final styling; Task 3 is expected to provide the shared stylesheet treatment. No route change was needed for the shared explorer-results target.
