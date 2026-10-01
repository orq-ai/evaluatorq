# Agent simulation output reference

A `SimulationResult` is the record of one simulated conversation between a **persona** (an LLM playing a user with set traits) and the **target** (the agent under test), scored by a **judge** (an LLM that checks the conversation against the scenario). It holds the transcript, the judge's verdict, and per-turn scores. `simulate()` returns these as a list. A saved run wraps them in a `SimulationRun`, which adds the run's configuration, evaluator averages and the datapoints needed to replay it.

Read this page when you parse simulation output in code, in CI or from an agent. It does not cover writing personas and scenarios or reading the dashboard; for those, see [Agent Simulation](agent-simulation.md). For every field and its type, see the [Python API reference](../reference/evaluatorq/simulation.md).

A **scenario** is the situation and goal the persona brings. A **datapoint** is one persona paired with one scenario. A **criterion** is a behaviour the scenario says must happen or must not happen. An **evaluator** turns a finished conversation into a numeric score.

## Where the output lands

| You ran | What you get | Model |
|---|---|---|
| `simulate()` or `generate_and_simulate()` | the return value | `list[SimulationResult]` |
| the same with `save=True` | `.evaluatorq/sim-runs/<name>_<YYYYmmdd-HHMMSS>.json`, or `report_path=` when set | `SimulationRun` |
| `evaluatorq sim simulate` or `sim run` (saves by default; `--no-save` skips) | `.evaluatorq/sim-runs/<name>_<YYYYmmdd-HHMMSS>.json` | `SimulationRun` |
| the CLI with `--report out.json` | `out.json` | `SimulationRun` |
| the CLI with `--results out.jsonl` | one result per line | `SimulationResult` |

The SDK does not save by default and the CLI does. The `.evaluatorq/sim-runs/` files are what `evaluatorq sim runs`, the dashboard and `previous_run=` read.

## The shape

```yaml
SimulationRun:
  run_name, created_at, mode, run_id, experiment_url
  target_kind, target, target_model, agent_info, orq_base_url, max_turns
  evaluator_names, total_results
  scorer_averages: dict[str, float]       # mean of each evaluator over the run
  token_usage_total                       # whole run, including generation and summary
  executive_summary, recommendations, applied_suggestions
  datapoints: list[SimulationDatapoint]   # the exact cases, for replay
  replay_version
  results: list[SimulationResult]         # one per conversation
    messages                              # the transcript
    terminated_by, reason, turn_count
    goal_achieved, goal_completion_score
    criteria_results, criteria_verified, rules_broken
    turn_metrics: list[TurnMetrics]       # per-turn judge scores and tokens
    token_usage, token_usage_known
    evaluator_details                     # structured evaluator output
    metadata                              # persona, scenario, criteria, evaluator scores
    total_turns, thread_id, response_traces
```

## Reading one result

This result is trimmed from a saved run. The transcript, turn metrics and token counts are cut. The run predates `audited` and `thread_id`, so those two are filled in by hand to show what a current run writes.

```json
{
  "terminated_by": "max_turns",
  "reason": "Maximum turns (4) reached",
  "goal_achieved": false,
  "goal_completion_score": 0.0,
  "rules_broken": [],
  "turn_count": 4,
  "criteria_results": {"Agent greets the user politely and responds to the greeting": true, "Agent asks for unnecessary personal information": true},
  "criteria_verified": true,
  "metadata": {
    "persona": "Curious Office Receptionist",
    "scenario": "Friendly Greeting and Weather Check",
    "datapoint_id": "dp_3e44c06b57af",
    "criteria_meta": [
      {"id": "criteria_0", "description": "Agent greets the user politely and responds to the greeting", "type": "must_happen", "passed": true, "audited": true},
      {"id": "criteria_2", "description": "Agent asks for unnecessary personal information", "type": "must_not_happen", "passed": true, "audited": true}
    ],
    "evaluator_scores": {"goal_achieved": 0.0, "criteria_met": 1.0}
  },
  "thread_id": "<run_id>:0"
}
```

`terminated_by` says why the conversation stopped:

| Value | Means |
|---|---|
| `judge` | The judge ended it: the goal was reached, a must-not-happen criterion occurred, or the judge's own reply could not be parsed and it stopped for safety. `reason` says which. |
| `max_turns` | It hit the turn cap without the judge ending it. The goal was not reached in the turns allowed, so `goal_achieved` is usually `false`; whether that fails your gate is your call. Criteria are still scored. |
| `timeout` | A target call timed out and its retries ran out. |
| `error` | A target call failed for any other reason and its retries ran out. `reason` says how. |

The criteria verdicts come in three places. `metadata["criteria_meta"]` is the one to read: one entry per criterion, keyed by a stable id (`criteria_0`, `criteria_1`, and so on), with its text, type, pass or fail, and the judge's evidence. `criteria_results` maps each criterion's *description* to pass or fail, so two criteria with the same description collapse into one key. `rules_broken` lists the ids of criteria that failed.

`passed: true` means the criterion was met, whatever its type. For a must-happen criterion, the behaviour occurred. For a must-not-happen criterion, the behaviour did not occur. In the example, `"Agent asks for unnecessary personal information": true` means the agent did not ask.

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

`turn_metrics` has one entry per turn with the judge's `response_quality`, `hallucination_risk`, `tone_appropriateness` and `factual_accuracy` (each 0 to 1, or `null` when not scored), its reasoning, and that turn's token usage.

`token_usage_known` is `false` when usage could only be partly collected. Treat `token_usage` as unknown then, not as a cheap run.

## Reading the run

`scorer_averages` is the mean of each evaluator's `metadata["evaluator_scores"]` over the results that have one. A conversation where an evaluator failed is left out of that evaluator's average rather than counted as zero, so a run with many failures can still show a high average. Count `metadata["evaluator_errors"]` to see how many were left out.

`datapoints` holds the persona and scenario objects each conversation ran, which `results` keeps only by name. It is `null` on runs saved before replay existed, and those runs cannot be replayed.

`recommendations` lists suggested fixes for failed conversations. Each entry points back to its result with `result_index`.

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

## Runs from older versions

Fields added after a run was saved load as their default: `criteria_verified`, `thread_id`, `response_traces`, `run_id`, `datapoints`, `token_usage_total` and `orq_base_url` are `null` or empty. A `null` there means "not recorded". `criteria_verified: null` in particular does not mean verified.
