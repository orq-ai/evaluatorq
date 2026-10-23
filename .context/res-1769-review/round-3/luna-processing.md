# Round 3 — Luna disposition

Disposition: adopted the P2 finding that an empty saved choice-criteria dictionary hides observed values in the dashboard.

Evidence: the accepted JSON probe validates through `InsightsRun.model_validate_json` and `build_review_payload` with `criteria={}`, `counts={"Hallucination": 1}`, and a recorded `Hallucination` answer. In `lvals`, `Object.keys({})` returned an empty list, so the rendered bar and legend were empty and the saved-value filter could not be decoded. After adding the regression assertion and before changing production code, `node tests/dashboard/insights_review_failure_taxonomy.cjs .context/res-1769-review/round-3/empty-criteria-probe.json` failed because `labelBlock` lacked `data-v="Hallucination"` and rendered empty bar and legend containers.

Change: when choice criteria contain keys, `lvals` keeps those keys and their declared order. When criteria are missing, malformed, or empty, it uses only the recorded count keys. The regression covers rendering and filtering the recorded value, confirms that an absent count produces no choices, and retains the existing nonempty-criteria and no-invented-clean-choice assertions. No saved fields, score behavior, or unrelated malformed-result behavior changed.

Other round-three disposition: no additional action. The independent standards audit reported no findings; the specification review reported only the adopted P2; `reviews.json`, `inline-comments.json`, and review threads contain no formal review entries or unresolved inline threads.

Checks after the change:

- `umask 022 && node tests/dashboard/insights_review_failure_taxonomy.cjs .context/res-1769-review/round-3/empty-criteria-probe.json` — passed.
- `umask 022 && uv run pytest tests/dashboard/test_insights_page.py::test_insights_review_renders_legacy_counts_warning -q` — 1 passed.
- `umask 022 && uv run ruff check src` — passed.
- `umask 022 && uv run ruff format --check src` — passed; 358 files already formatted.
- `umask 022 && uv run ty check` — passed.

Commit: this disposition is included with the production change and regression in one conventional commit.
