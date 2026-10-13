# Agent simulation output reference

A `SimulationResult` is the record of one simulated conversation between a **persona** (an LLM playing a user with set traits) and the **target** (the agent under test), scored by a **judge** (an LLM that checks the conversation against the scenario). It holds the transcript, the judge's verdict, and per-turn scores. `simulate()` returns these as a list. A saved run wraps them in a `SimulationRun`, which adds the run's configuration, evaluator averages and the datapoints needed to replay it.

Read this page when you parse simulation output in code, in CI or from an agent. It does not cover writing personas and scenarios or reading the dashboard; for those, see [Agent Simulation](agent-simulation.md). For every field and its type, see the [Python API reference](../reference/evaluatorq/simulation.md).

A **scenario** is the situation and goal the persona brings. A **datapoint** is one persona paired with one scenario. A **criterion** is a behaviour the scenario says must happen or must not happen. An **evaluator** turns a finished conversation into a numeric score.


## Python imports

Import simulation data models from `evaluatorq.simulation.contracts`. The former `evaluatorq.simulation.types` module remains available for compatibility and emits a `DeprecationWarning`; update imports to the canonical path.

## Where the output lands

| You ran | What you get | Model |
|---|---|---|
| `simulate()` or `generate_and_simulate()` | the return value | `list[SimulationResult]` |
| the same with `save=True` | `.evaluatorq/sim-runs/<name>_<YYYYmmdd-HHMMSS>.json`, or `report_path=` when set | `SimulationRun` |
| `evaluatorq sim simulate` or `sim run` (saves by default; `--no-save` skips) | `.evaluatorq/sim-runs/<name>_<YYYYmmdd-HHMMSS>.json` | `SimulationRun` |
| the CLI with `--report out.json` | `out.json` | `SimulationRun` |
| the CLI with `--results out.jsonl` | one result per line | `SimulationResult` |

The SDK does not save by default and the CLI does. The `.evaluatorq/sim-runs/` files are what `evaluatorq sim runs`, the dashboard and `previous_run=` read.

## Data model

A saved JSON file contains one `SimulationRun` object. Its `results` field is a list of `SimulationResult` objects, one per conversation. `simulate()` returns that list directly, without the run wrapper; a JSONL export writes one `SimulationResult` per line.

The Python declarations below describe field types, not sample values. Each class name labels a type; it is not an extra key in the saved JSON. `list[T]` means a list of objects of type `T`, `dict[str, T]` means a mapping from string keys to values of type `T`, and `T | None` means the value can be `None`. Python's `None` is written as `null` in JSON. Definitions of the named types are in the [Python API reference](../reference/evaluatorq/simulation.md). The map covers the fields used on this page, not every nested model.

### SimulationRun: the whole saved run

```python
class SimulationRun:
    run_name: str
    created_at: datetime
    mode: str
    run_id: str | None
    experiment_url: str | None
    target_kind: str
    target: str | None
    target_model: str | None
    agent_info: AgentInfoSnapshot | None
    orq_base_url: str | None
    max_turns: int | None
    evaluator_names: list[str]
    total_results: int
    scorer_averages: dict[str, float]
    token_usage_total: TokenUsage | None
    executive_summary: str | None
    recommendations: list[SimulationRecommendation] | None
    applied_suggestions: list[str]
    datapoints: list[SimulationDatapoint] | None
    replay_version: int | None
    results: list[SimulationResult]
```

### SimulationResult: one conversation in `results`

Each item in `SimulationRun.results` has this structure. These fields belong to the conversation, not to the run wrapper.

```python
class SimulationResult:
    messages: list[Message]
    terminated_by: TerminatedBy
    reason: str
    turn_count: int
    goal_achieved: bool
    goal_completion_score: float
    criteria_results: dict[str, bool] | None
    criteria_verified: bool | None
    rules_broken: list[str]
    turn_metrics: list[TurnMetrics]
    token_usage: TokenUsage
    token_usage_known: bool
    evaluator_details: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    thread_id: str | None
    response_traces: list[ResponseTrace]
```

## Reading one result

`terminated_by` says why the conversation stopped:

| Value | Means |
|---|---|
| `judge` | The judge ended it: the goal was reached, a must-not-happen criterion occurred, or the judge's own reply could not be parsed and it stopped for safety. `reason` says which. |
| `max_turns` | It hit the turn cap without the judge ending it. The goal was not reached in the turns allowed, so `goal_achieved` is usually `false`; whether that fails your gate is your call. Criteria are still scored. |
| `timeout` | A target call, or the whole conversation, ran out of time. `reason` describes the failure; `metadata["error_type"]` identifies its exception type. |
| `error` | A target call failed after its retries, or the run itself raised (the simulated user, the judge, or setup before the first turn). `reason` describes the failure; `metadata["error_type"]` identifies its exception type. |

The criteria verdicts come in three places. `metadata["criteria_meta"]` is the one to read: one entry per criterion, keyed by a stable id (`criteria_0`, `criteria_1`, and so on), with its text, type, pass or fail, and the judge's evidence. `criteria_results` maps each criterion's *description* to pass or fail, so two criteria with the same description collapse into one key. `rules_broken` lists the ids of criteria that failed when `criteria_verified` is true; otherwise it is the judge's own free-text list, passed through.

`passed: true` means the criterion was met, whatever its type. For a must-happen criterion, the behaviour occurred. For a must-not-happen criterion, the behaviour did not occur.

Check `criteria_verified` before trusting `criteria_results`. When it is `false`, the judge never returned a per-criterion audit, and every verdict fell back to the free-text `rules_broken` list. That fallback cannot fail a must-happen criterion, so an agent that never did the required thing still shows it as passed. Treat those verdicts as unknown. The same applies to one criterion whose `criteria_meta` entry has `"audited": false` and `"passed": true`. `audited` is missing on runs saved before the field existed; that means "not recorded", not "unaudited".

`metadata` is a plain dict, so its keys are a convention rather than a schema. The keys evaluatorq writes:

| Key | Holds |
|---|---|
| `persona`, `scenario` | the names of the persona and scenario |
| `persona_traits` | the persona's patience, assertiveness, politeness, technical level, style and background |
| `scenario_goal`, `scenario_context` | the scenario's goal and context |
| `criteria_meta` | one entry per criterion: `id`, `description`, `type`, `passed`, `audited`, `evidence` |
| `datapoint_id` | the datapoint this conversation ran |
| `evaluator_scores` | evaluator name to numeric score, for this conversation |
| `evaluator_errors` | evaluator name to the reason its score was unusable, when one failed |
| `target_model` | the model the target used, when the client knows it |
| `error`, `error_type` | why the conversation failed, on `error` and `timeout` results |
| `timeout` | the time limit in seconds, when the conversation hit it |
| `token_usage_unknown` | `true` when a failed conversation's usage could not be collected |

`turn_metrics` has one entry per turn with the judge's `response_quality`, `hallucination_risk`, `tone_appropriateness` and `factual_accuracy` (each 0 to 1, or `null` when not scored), its reasoning, and that turn's token usage.

`token_usage_known` is `false` when usage could only be partly collected. Treat `token_usage` as unknown then, not as a cheap run.

## Reading the run

`scorer_averages` is the mean of each evaluator's `metadata["evaluator_scores"]` over the results that have one. A conversation where an evaluator failed is left out of that evaluator's average rather than counted as zero, so a run with many failures can still show a high average. Count `metadata["evaluator_errors"]` to see how many were left out.

`datapoints` holds the persona and scenario objects each conversation ran, which `results` keeps only by name. It is `null` on runs saved before replay existed, and those runs cannot be replayed.

`recommendations` lists suggested fixes for failed conversations, or is `null` when none were generated. Each entry points back to its result with `result_index`. `applied_suggestions` holds the suggestion strings already applied to the agent, so the dashboard marks them and a later apply skips them.

`agent_info` is a snapshot of an Orq agent's configuration at run time: `key`, `id`, `role`, `description`, `model` and `tools`, never its instructions. It is `null` for targets that are not Orq agents.

## Load a saved run

This prints each conversation in the newest saved run, and flags unverified criteria:

```python
from pathlib import Path

from evaluatorq.simulation import SimulationRun

latest = max(Path('.evaluatorq/sim-runs').glob('*.json'), key=lambda p: p.stat().st_mtime)
run = SimulationRun.model_validate_json(latest.read_text())

print(f'{run.run_name}: {run.total_results} conversations, averages {run.scorer_averages}')
for result in run.results:
    criteria = 'unverified' if result.criteria_verified is False else result.criteria_results
    print(f'{result.metadata.get("persona")} / {result.metadata.get("scenario")}: goal={result.goal_achieved} stop={result.terminated_by.value} criteria={criteria}')
```

To load a `--results out.jsonl` file from `evaluatorq sim simulate` instead, validate each line as a `SimulationResult`:

```python
from pathlib import Path

from evaluatorq.simulation import SimulationResult

results = [SimulationResult.model_validate_json(line) for line in Path('out.jsonl').read_text().splitlines() if line]
```

Without Python, `jq` reads a saved run:

```bash
latest=$(ls -t .evaluatorq/sim-runs/*.json | head -1)
jq '{run_name, total_results, scorer_averages}' "$latest"
jq -r '.results[] | select(.goal_achieved == false) | "\(.metadata.persona) / \(.metadata.scenario): \(.reason)"' "$latest"
```

## Gate CI on a run

This exits non-zero when any conversation errored or timed out, missed its goal, failed a criterion, or has a criterion the judge never audited:

```python
import sys
from pathlib import Path

from evaluatorq.simulation import SimulationRun

latest = max(Path('.evaluatorq/sim-runs').glob('*.json'), key=lambda p: p.stat().st_mtime)
run = SimulationRun.model_validate_json(latest.read_text())

problems = []
for i, r in enumerate(run.results):
    name = f'#{i} {r.metadata.get("persona")} / {r.metadata.get("scenario")}'
    if r.terminated_by.value in ('error', 'timeout'):
        problems.append(f'{name}: {r.terminated_by.value}: {r.reason}')
    elif not r.goal_achieved:
        problems.append(f'{name}: goal not achieved ({r.terminated_by.value}: {r.reason})')
    for c in r.metadata.get('criteria_meta') or []:
        if not c['passed']:
            problems.append(f'{name}: failed {c["id"]} {c["description"]}')
        elif c.get('audited') is False:
            problems.append(f'{name}: {c["id"]} never audited')
    if r.criteria_verified is not True:
        problems.append(f'{name}: criteria not verified')

print('\n'.join(problems) or 'ok')
sys.exit(1 if problems else 0)
```

`criteria_meta` is the complete list of failed criteria, so `rules_broken` adds nothing to this check. `criteria_verified is not True` also fails runs saved before the field existed, where it is `null`: a gate that must never pass silently should treat "not recorded" as unverified. Drop the `goal_achieved` branch if your scenarios are exploratory and a conversation that runs out of turns is acceptable; keep the `terminated_by` branch, or a dead target passes.

## Runs from older versions

Fields added after a run was saved load as their default: `criteria_verified`, `thread_id`, `response_traces`, `run_id`, `datapoints`, `token_usage_total` and `orq_base_url` are `null` or empty. A `null` there means "not recorded". `criteria_verified: null` in particular does not mean verified.
