# Red teaming output reference

A `RedTeamReport` is the result of one red team run. It holds every attack that ran against the **target** (the agent or model under test), the **judge's** verdict on each (an LLM that decides whether the attack succeeded), and a summary across all of them. `red_team()` returns it, and the CLI writes it to disk as JSON with the same fields.

Read this page when you parse a report in code, in CI or from an agent. It does not cover choosing attacks or reading the HTML and dashboard views; for those, see [Red Teaming](red-teaming.md). For every field and its type, see the [Python API reference](../reference/evaluatorq/redteam.md).

## Where the report lands

The same model is serialised in every case. What changes is the file name, a few extra top-level keys and, in `detail` mode, how complete the report is.

| You ran | File | Extra keys beyond `RedTeamReport` |
|---|---|---|
| `red_team()` or `evaluatorq redteam run` with the default `save="final"` and no artifacts directory | `.evaluatorq/runs/<name>_<YYYYmmdd_HHMMSS>.json` | `run_name`, `saved_at`, and, so the run can be replayed, `datapoints`, `run_config` and `replay_version` |
| `save="final"` with `artifacts_dir=` / `--artifacts-dir` | `<dir>/03_summary_report.json`, plus the `.evaluatorq/runs/` copy above | `saved_at` |
| `save="detail"` with an artifacts directory | `01_all_datapoints.json` (`01_datapoints.json` for `static`), `02_attack_results.json` and `03_summary_report.json` in that directory, plus the `.evaluatorq/runs/` copy | `01` and `02` wrap their payload as `{"saved_at": ..., "data": [...]}`; `02` holds the raw evaluatorq job rows (`data_point`, `job_results`, `evaluator_scores`), not `RedTeamResult`; read results from `03` |
| `evaluatorq redteam run --report out.json` | `out.json` | none |

In `detail` mode, `03` is written before recommendations and the executive summary run, so it has no `focus_area_recommendations`, `executive_summary` or `post_processing_token_usage`, and its `token_usage_total` leaves out that spend. Read the `.evaluatorq/runs/` copy for those. `save="none"` writes nothing. The replay keys are only present when the run had datapoints to record. The `.evaluatorq/runs/` files are what `evaluatorq redteam runs`, the dashboard and `previous_run=` read.

## Data model

`red_team()` returns one `RedTeamReport` object, and a report JSON file contains that object's fields. Its `summary` field is one `ReportSummary` object for the whole run. Its `results` field is a list of `RedTeamResult` objects, one per attack that ran. Saved run files can also carry the extra top-level keys listed in [Where the report lands](#where-the-report-lands).

The Python declarations below describe field types, not sample values. Each class name labels a type; it is not an extra key in the saved JSON. `list[T]` means a list of objects of type `T`, `dict[str, T]` means a mapping from string keys to values of type `T`, and `T | None` means the value can be `None`. Python's `None` is written as `null` in JSON. Definitions of the named types are in the [Python API reference](../reference/evaluatorq/redteam.md). The map covers the fields used on this page, not every nested model.

### RedTeamReport: the whole run

```python
class RedTeamReport:
    version: str
    total_results: int
    pipeline: Pipeline
    framework: Framework | None
    categories_tested: list[str]
    tested_agents: list[str]
    created_at: datetime
    duration_seconds: float | None
    summary: ReportSummary
    results: list[RedTeamResult]
    errors: list[RunError]
    agent_contexts: dict[str, AgentContext]
    focus_area_recommendations: list[FocusAreaRecommendation] | None
    applied_recommendations: list[str]
    executive_summary: str | None
    pipeline_warnings: list[str]
    run_id: str | None
    experiment_url: str | None
    orq_base_url: str | None
    uploaded_count: int | None
    rows_created: int | None
```

### ReportSummary: the aggregate in `summary`

`RedTeamReport.summary` has this structure. These counts and rates describe the whole run, not one attack.

```python
class ReportSummary:
    total_attacks: int
    evaluated_attacks: int
    unevaluated_attacks: int
    evaluation_coverage: float
    vulnerabilities_found: int
    vulnerability_rate: float | None
    resistance_rate: float | None
    total_errors: int
    errors_by_type: dict[str, int]
    pre_execution_errors: int
    token_usage_total: TokenUsage | None
    token_usage_by_source: dict[str, TokenUsage]
    post_processing_token_usage: TokenUsage | None
    min_evaluation_coverage: float | None
    jury_reliability: JuryReliability | None
    datapoint_breakdown: dict[str, int] | None
    total_turns: int
    by_vulnerability: dict[str, VulnerabilitySummary]
    by_category: dict[str, CategorySummary]
    by_technique: dict[str, TechniqueSummary]
    by_severity: dict[str, SeveritySummary]
    by_delivery_method: dict[str, DeliveryMethodSummary]
    by_turn_type: dict[str, TurnTypeSummary]
    by_domain: dict[str, DomainSummary]
    by_framework: dict[str, FrameworkSummary]
```

### RedTeamResult: one attack in `results`

Each item in `RedTeamReport.results` has this structure. These fields belong to the attack, not to the report wrapper.

```python
class RedTeamResult:
    attack: AttackInfo
    agent: AgentInfo
    messages: list[Message]
    response: str | None
    vulnerable: bool | None
    evaluation: UnifiedEvaluationResult | None
    execution: ExecutionDetails | None
    evaluation_error: RunError | None
    error: str | None
    error_type: str | None
    error_stage: str | None
    error_code: str | None
    error_details: dict[str, Any] | None
    thread_id: str | None
    response_traces: list[ResponseTrace]
```

## Reading one result

Read the verdict from `vulnerable`. It has three values, and the third is the one that gets misread:

| `vulnerable` | `evaluation.passed` | Means |
|---|---|---|
| `false` | `true` | The target resisted the attack. |
| `true` | `false` | The attack succeeded. |
| `null` | `null` | No verdict. The target call or the judge failed. |

A `null` verdict is not a pass. Code that tests `if not result.vulnerable` counts every unscored attack as resisted, so a target that errored on every call reports as fully safe. Test `result.vulnerable is True` for failures. When `vulnerable` is `null`, two fields on the same result say why: `evaluation_error` (a `RunError` object with `message` and `error_type`) when the attack ran but the judge could not score it, and `error` (a string, with `error_type`, `error_stage` and `error_code` beside it) when the attack itself failed to run. `evaluation` can be `null` too, so guard before reading `evaluation.passed`.

`attack` says what was tried. `vulnerability` is the atomic identifier, and `category` is the OWASP code it maps to. `strategy_name` and `objective` are filled for dynamic attacks only. `source` records where the attack came from: a dataset, a template, or an LLM-generated strategy.

`agent` identifies the tested target with its key, model name and display name; each value can be `null` when the target does not provide it. `agent_contexts` holds richer per-agent context, keyed by agent key.

`evaluation.raw_output` holds the judge's verbatim reply. It is for debugging; parse the verdict from `passed` and `explanation`, not from there.

`focus_area_recommendations` contains the generated remediation advice when recommendations are enabled. `applied_recommendations` contains recommendation strings already applied to an agent through `reports.apply`; it is empty when none have been applied.

## Reading the summary

`summary` is computed from `results` and `errors`. Gate on it rather than recounting. In Python, two properties carry the gate logic: `summary.no_verdict` is true when attacks ran but none was scored, and `summary.coverage_below_minimum` is true when coverage fell under `min_evaluation_coverage`. They are properties, so they are not in the JSON; with `jq`, test `total_attacks > 0 and evaluated_attacks == 0` yourself.

- `resistance_rate` and `vulnerability_rate` are fractions of *evaluated* attacks only. Both are `null` when nothing could be evaluated. Never render `null` as `0`: a `0.0` resistance rate reads as "fully compromised".
- `evaluation_coverage` is `evaluated_attacks / total_attacks`. A high resistance rate over 20% coverage rests on a fifth of the run.
- `min_evaluation_coverage` records the coverage floor the run was configured with, so a saved report says which policy it was judged against.
- Each `by_*` field is a dict keyed by that dimension (vulnerability id, OWASP category, technique, and so on), each value carrying `total_attacks` and `resistance_rate`. All except `by_vulnerability` also carry `vulnerability_rate`.
- `token_usage_total` includes `post_processing_token_usage`, the recommendation and executive-summary calls. `token_usage_by_source` covers the attack portion only.

`errors` at the top level lists datapoints that failed before an attack ran, for example a template that could not be filled. They are not in `results` and not in `total_attacks`; `summary.pre_execution_errors` is their count, an integer.

## Load a saved report

This prints every successful attack from the newest saved run:

```python
from pathlib import Path

from evaluatorq.redteam import RedTeamReport

latest = max(Path('.evaluatorq/runs').glob('*.json'), key=lambda p: p.stat().st_mtime)
report = RedTeamReport.model_validate_json(latest.read_text())

summary = report.summary
rate = summary.resistance_rate
print(f'{latest.name}: {summary.evaluated_attacks}/{summary.total_attacks} evaluated, resistance', f'{rate:.0%}' if rate is not None else 'n/a')
for result in report.results:
    if result.vulnerable is True:
        print(f'VULNERABLE {result.attack.category} {result.attack.vulnerability}: {result.attack.id}')
```

`RedTeamReport` ignores the extra keys a saved run carries (`run_name`, `saved_at`, the replay keys), so the same call loads a `--report` file, a `03_summary_report.json` and a `.evaluatorq/runs/` file.

Without Python, `jq` reads the same file:

```bash
latest=$(ls -t .evaluatorq/runs/*.json | head -1)
jq '.summary | {total_attacks, evaluated_attacks, resistance_rate}' "$latest"
jq -r '.results[] | select(.vulnerable == true) | .attack.id' "$latest"
```

## Gate CI on a report

This exits non-zero when any attack succeeded, any attack has no verdict, or any datapoint failed before its attack ran:

```python
import sys
from pathlib import Path

from evaluatorq.redteam import RedTeamReport

latest = max(Path('.evaluatorq/runs').glob('*.json'), key=lambda p: p.stat().st_mtime)
report = RedTeamReport.model_validate_json(latest.read_text())

problems = [f'pre-execution error: {e.message}' for e in report.errors]
for r in report.results:
    if r.vulnerable is True:
        problems.append(f'VULNERABLE {r.attack.vulnerability} {r.attack.id}')
    elif r.vulnerable is None:
        # error: the attack never ran. evaluation_error: it ran and the judge could not score it.
        why = r.error or (r.evaluation_error.message if r.evaluation_error else 'unknown')
        problems.append(f'NO VERDICT {r.attack.id}: {why}')

print('\n'.join(problems) or 'ok')
sys.exit(1 if problems else 0)
```

`error` and `evaluation_error` describe different failures, so check `error` first: when it is set, the attack never reached the judge. `max(... st_mtime)` picks the newest file in the directory, so keep other JSON files out of `.evaluatorq/runs/`.

## Reports from older versions

Fields added after a report was written load as their default: `run_id`, `thread_id`, `uploaded_count`, `rows_created` and `evaluation_error` as `null`, and `response_traces` as an empty list. A `null` in one of those means "not recorded", not "none happened". `version` is the report format version, currently `2.0.0`.
