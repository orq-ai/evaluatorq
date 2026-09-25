# Task 11 report: `eq insights` CLI

Status: DONE.

Implemented `src/evaluatorq/insights/cli.py` and registered `eq insights` in `src/evaluatorq/cli.py`. The CLI accepts the finder population query, facet, numeric, window, limit, and parallelism options; supports repeatable label presets and JSON `LabelSpec` files; validates dimensions, bounds, query/export conflicts, and labels before running; prints Rich cluster and label summaries with failed-trace counts, stage failures, and the stored run path; writes an optional JSON copy; and exits 1 for an error-status run or 2 for invalid input. The finder facet parser and `common.cli_errors.emit_error` are reused.

Added CLI tests for required help options, resolving a preset with a custom JSON spec, error-run exit status, and `--from-finder`/`--query` conflict.

Commands and final output:

- `uv run pytest tests/insights/test_cli.py -q` → `4 passed in 0.52s`.
- `uv run ruff check src` → `All checks passed!`.
- `uv run ruff format --check src` → `261 files already formatted`.
- `uv run basedpyright` → `0 errors, 0 warnings, 0 notes`.
- `uv run pytest -m 'not integration' tests/insights tests/test_reuse_guardrails.py` → `127 passed, 2 warnings in 17.34s`.
- `git diff --check` → no output, exit 0.
- The two pytest warnings are UMAP's expected notice that setting `random_state` sets `n_jobs=1`.

The command help specifies JSON label-spec files; YAML label files are not supported because PyYAML is not declared as an application runtime dependency. No other unresolved concerns.

## Review fix: validate finder exports before pipeline execution

The CLI now validates `--from-finder` input with `trace_finder.export.RunExport.model_validate_json` before constructing the Insights population. Missing files, malformed JSON, and invalid or unsupported export schemas now emit the shared CLI error and exit 2 without calling the Insights pipeline. Valid exports continue through the existing `InsightsPopulation.from_finder_export` path.

Added CLI regression coverage for a missing export, malformed JSON, unsupported schema version, pipeline non-invocation for invalid inputs, and valid export acceptance.

Commands and output:

- `uv run pytest tests/insights/test_cli.py -q` → `6 passed in 0.36s`.
- `uv run ruff check src` → `All checks passed!`.
- `uv run ruff format --check src` → `261 files already formatted`.
- `uv run basedpyright` → `0 errors, 0 warnings, 0 notes`.
- `uv run pytest -m 'not integration' tests/insights tests/test_reuse_guardrails.py` → `129 passed, 2 warnings in 15.81s`.
- `git diff --check` → no output, exit 0.
- The two pytest warnings are UMAP's expected notice that setting `random_state` sets `n_jobs=1`.

No new concerns; custom label files remain JSON-only as noted above.
