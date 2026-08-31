# Simulation inside an evaluatorq run

`simulate()` runs conversations and hands you `SimulationResult` objects in memory. That is the right shape when simulation *is* the run. It is the wrong shape when a simulated conversation is one column of a larger evaluation — when you also score retrieval quality, or compare two prompts, or want the whole batch to land in Orq as one Experiment.

`wrap_simulation_agent()` is the seam between the two. It turns a simulation into an evaluatorq **job**: a callable `evaluatorq()` invokes once per `DataPoint`, returning the finished conversation in OpenResponses form. From there the conversation is an ordinary evaluatorq output — your evaluators score it, the results table prints it, and the run uploads as an Experiment with a URL you can send to someone.

## When you want this instead of `simulate()`

| Use `simulate()` | Use `wrap_simulation_agent()` + `evaluatorq()` |
|---|---|
| Simulation is the whole run | Simulation is one job among several |
| You want `SimulationResult` objects back | You want scored rows and an Experiment URL |
| The built-in simulation scorers are the scoring you need | You want your own evaluatorq evaluators over the transcript |
| Local, exploratory, no upload wanted | CI, shareable results, batch comparison |

Both drive the same `SimulationRunner` underneath, so the conversations themselves are identical. What differs is what surrounds them.

## Prerequisites

```bash
uv sync --all-extras --all-groups
python -c "import os; assert os.environ.get('ORQ_API_KEY') or os.environ.get('OPENAI_API_KEY'), 'set ORQ_API_KEY or OPENAI_API_KEY'; print('credentials OK')"
```

The user simulator and the judge are model calls, so one of those two keys has to be set. With `ORQ_API_KEY` set, the run also uploads an Experiment to your Orq workspace when it finishes — see [What gets uploaded](#what-gets-uploaded) before running this against a workspace you share with other people.

## A complete run

Save this as `sim_in_eval.py`. The target here is an ordinary async function, so nothing but the two model keys is required to run it.

```python
import asyncio

from evaluatorq import DataPoint, EvaluationResult, evaluatorq
from evaluatorq.simulation import Message, wrap_simulation_agent
from evaluatorq.simulation.types import CommunicationStyle, Criterion, Persona, Scenario


async def support_agent(messages: list[Message]) -> str:
    """The agent under test. Replace this with your own."""
    last = messages[-1].content if messages else ""
    if "order" in str(last).lower():
        return "Thanks — I have pulled up that order and started the refund."
    return "Sorry about that. Could you share your order number so I can look it up?"


async def criteria_scorer(params) -> EvaluationResult:
    """Score the fraction of the scenario's criteria the judge marked satisfied."""
    results = params["output"].get("metadata", {}).get("criteria_results") or {}
    if not results:
        return EvaluationResult(value=0.0, explanation="no criteria were judged")
    passed = sum(1 for ok in results.values() if ok)
    return EvaluationResult(
        value=passed / len(results),
        explanation=f"{passed}/{len(results)} criteria satisfied",
    )


async def main() -> None:
    persona = Persona(
        name="Impatient Customer",
        patience=0.2,
        assertiveness=0.8,
        politeness=0.4,
        technical_level=0.3,
        communication_style=CommunicationStyle.terse,
        background="Received the wrong item and wants a refund",
    )
    scenario = Scenario(
        name="Wrong Item Refund",
        goal="Get a refund for the wrong item",
        criteria=[
            Criterion(description="Agent asks for order details", type="must_happen"),
            Criterion(description="Agent blames the customer", type="must_not_happen"),
        ],
    )

    # One DataPoint per conversation. wrap_simulation_agent() reads
    # inputs["persona"] and inputs["scenario"] off each row.
    data = [
        DataPoint(inputs={"persona": persona.model_dump(), "scenario": scenario.model_dump()})
    ]

    job = wrap_simulation_agent(name="support-sim", target=support_agent, max_turns=3)
    try:
        results = await evaluatorq(
            "support-simulation",
            data=data,
            jobs=[job],
            evaluators=[{"name": "criteria", "scorer": criteria_scorer}],
        )
    finally:
        # Releases the wrapper's long-lived runner and its HTTP pool.
        await job.aclose()

    for row in results:
        for job_result in row.job_results or []:
            metadata = job_result.output.get("metadata", {})
            print("metadata keys:", sorted(metadata))
            print("criteria:", metadata.get("criteria_results"))


asyncio.run(main())
```

It prints the standard evaluatorq results table, then the two lines the script asks for:

```console
$ uv run python sim_in_eval.py
metadata keys: ['criteria_results', 'framework', 'goal_achieved', 'goal_completion_score', 'reason', 'rules_broken', 'terminated_by', 'turn_count', 'turn_metrics']
criteria: {'Agent asks for order details': True, 'Agent blames the customer': True}
```

Those keys are the contract, and the next section is a reference for them. The values behind two of them — `turn_count` and `goal_achieved` — are not printed here on purpose: the user simulator and the judge are both model calls, so how many turns the conversation takes and whether the judge calls the goal met genuinely vary between runs of this same script. Assert against the shape of `metadata`, never against a particular turn count.

## `goal_achieved` and `criteria_results` are different questions

`goal_achieved` and `criteria_results` answer different questions, and they can disagree in either direction. That surprises people, so it is worth being precise about which is which:

- `criteria_results` is the judge's per-criterion audit: for each `Criterion` on the scenario, did the thing it names occur, and does that occurrence mean pass or fail for its type? A `must_not_happen` criterion is `True` when the bad thing did **not** occur.
- `goal_achieved` is the judge's verdict on the scenario's `goal` as a whole — here, whether the customer actually got their refund.

The common disagreement is every criterion passing while `goal_achieved` is `False`, and the usual cause is the turn budget: a conversation that hits `max_turns` before the agent finishes the job stops with `terminated_by: max_turns`, and the goal is unmet even though the agent did every individual thing you thought to write down. Raising `max_turns` is the fix when the agent was on track; it is not the fix when the agent was stuck.

Score whichever one matches the question you are asking. Scoring both, as two separate evaluators, is usually what you want — a criteria score tells you which behaviour is missing, and the goal score tells you whether that mattered.

## What the job returns

The job's output is an OpenResponses response dict. The transcript is in `input` and `output` — user turns as `input_text`, agent turns as `output_text` — and everything the judge concluded is under `metadata`:

| `metadata` key | Type | What it holds |
|---|---|---|
| `framework` | `str` | Always `"simulation"`. |
| `goal_achieved` | `bool` | Judge's verdict on the scenario goal. |
| `goal_completion_score` | `float` | Graded version of the same verdict. |
| `terminated_by` | `str` | Why the conversation stopped: `judge`, `max_turns`, `error`. |
| `reason` | `str` | The termination reason in words. |
| `turn_count` | `int` | Turns actually run. |
| `rules_broken` | `list[str]` | Criteria the judge marked violated, **by id** (`criteria_0`, `criteria_1`). |
| `criteria_results` | `dict[str, bool]` | Per-criterion pass/fail, **by description**. Present only when the scenario had criteria. |
| `turn_metrics` | `list[dict]` | Per-turn latency and token detail. Present only when metrics were collected. |

Two of those keys are optional, so read them with a default rather than by subscript — a scenario with no criteria has no `criteria_results` at all, and an evaluator that assumes the key raises on the row instead of scoring it.

!!! warning "`rules_broken` and `criteria_results` are keyed differently"

    `rules_broken` holds criterion **ids** and `criteria_results` is keyed by criterion **description**. They do not join on a shared key, so a scorer that looks up a `rules_broken` entry in `criteria_results` finds nothing and silently scores every row as clean. Read one or the other, not both. Descriptions are also not unique: two criteria worded identically collapse into a single `criteria_results` entry, which is why the id-keyed form exists at all.

## Against an Orq deployment

If the agent under test is deployed on Orq rather than living in your Python process, pass `agent_key=` instead of `target=` and change nothing else. Internally the wrapper resolves the key through `from_orq_deployment()`, which builds the callback lazily — constructing the job issues no request, so a wrong key surfaces on the first conversation, not at import time.

```python
from evaluatorq.simulation import wrap_simulation_agent

job = wrap_simulation_agent(name="support-sim", agent_key="my-support-agent", max_turns=3)
print("target resolves to deployment:", job is not None)
```

In the script above, that one line replaces the `target=support_agent` line; the `DataPoint` list, the evaluators, the `evaluatorq()` call and the `aclose()` all stay as they are. Passing both `target=` and `agent_key=` is not an error — `target=` wins, and `agent_key=` is ignored — so pass exactly the one you mean.

## Always close the job

The callable `wrap_simulation_agent()` returns owns a long-lived `SimulationRunner` and the HTTP connection pool underneath it. Nothing closes that for you when `evaluatorq()` returns, so a script that skips `aclose()` leaks the pool until the process exits — survivable in a one-shot script, not survivable in a long-lived worker that builds a job per request.

```python
import asyncio

from evaluatorq.simulation import wrap_simulation_agent

async def main() -> None:
    job = wrap_simulation_agent(target=lambda messages: "hello", max_turns=1)
    try:
        pass  # await evaluatorq(...) goes here
    finally:
        await job.aclose()
    print("runner closed")

asyncio.run(main())
```

Put the `aclose()` in a `finally`, not after the `evaluatorq()` call. An exception raised mid-batch would otherwise skip it, which is exactly the run you least want leaking.

## Accepted input shapes

`wrap_simulation_agent()` reads one conversation per `DataPoint`, and accepts four spellings in `inputs`:

| `inputs` contains | Notes |
|---|---|
| `persona` + `scenario` | The usual form. Add `first_message` to fix the opening line instead of generating it. |
| `datapoint` | A full `SimulationDatapoint`, which must carry `persona`, `scenario` and `first_message`. |
| `datapoints` | A list of exactly one datapoint. More than one raises. |
| `personas` + `scenarios` | Lists of exactly one each. More than one raises. |

Nested objects are also accepted as JSON strings, because the Orq datasets API rejects nested objects in `inputs` — a dataset-backed row arrives with `persona` and `scenario` stringified, and the wrapper parses them back. The list forms exist for compatibility with datapoint files and cap at one element on purpose: one row is one conversation, so a row holding two personas is an ambiguity rather than a batch.

Anything else raises `ValueError` naming the four shapes.

## Scoring is not the wrapper's job

`wrap_simulation_agent()` used to take an `evaluators=` argument and no longer does. Passing it raises immediately rather than dropping your scoring on the floor:

```python
from evaluatorq.simulation import wrap_simulation_agent

try:
    wrap_simulation_agent(target=lambda messages: "hi", evaluators=[])
except TypeError as exc:
    print(exc)
```

Scoring belongs on the `evaluatorq()` call, as `evaluatorq(..., evaluators=[...])`. That is the whole reason to reach for this wrapper: the conversation becomes a normal evaluatorq output, and every evaluator you already have applies to it.

Constructing the job with neither `target=` nor `agent_key=` raises `ValueError` for the same reason — the wrapper has nothing to talk to.

## In CI

Three things change for a non-interactive run:

- `print_results=False` on the `evaluatorq()` call, so the results table does not go to a log nobody reads.
- `EVALUATORQ_DIR` pointed at a scratch directory, so the run store does not persist between jobs and one workflow cannot resolve another's runs.
- `ORQ_DISABLE_TRACING=1` if you do not want CI spans in your traces.

If you are arriving from `simulate()`, note what does and does not carry over. `simulate()` takes `exit_on_failure`, which defaults to `True` and raises `SimulationDroppedError` when a datapoint is *dropped* — a job raised and no result was cached. That is an infrastructure gate, not a quality gate: scorer verdicts are reporting only there too, so an agent that answered every question badly still exits 0. Moving to `wrap_simulation_agent()` costs you that infrastructure gate, because the parameter belongs to `simulate()` and there is no equivalent on `evaluatorq()`.

`evaluatorq()` does not fail the process for you either — it returns results and exits 0 whatever the scores are. So on this path the whole build verdict is yours to write, and it is a few lines:

```python
from evaluatorq import EvaluationResult


def gate(scores: list[float], threshold: float) -> int:
    """Return the exit code for a run: 0 when the mean score clears the threshold."""
    if not scores:
        return 1
    return 0 if sum(scores) / len(scores) >= threshold else 1


def scores_from(results, evaluator: str | None = None) -> list[float]:
    """Pull numeric evaluator scores out of an EvaluatorqResult.

    Pass `evaluator` to gate on one evaluator by name rather than on the mean of
    all of them. The name is `score.evaluator_name`, matching the `name` you gave
    the evaluator in the `evaluatorq()` call.
    """
    values = []
    for row in results:
        for job_result in row.job_results or []:
            for score in job_result.evaluator_scores or []:
                if evaluator is not None and score.evaluator_name != evaluator:
                    continue
                result = score.score
                value = result.value if isinstance(result, EvaluationResult) else result
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
    return values


print("empty run  ->", gate([], 0.8))
print("below bar  ->", gate([0.5, 0.5], 0.8))
print("clears bar ->", gate([1.0, 0.8], 0.8))
```

An empty score list returns 1 on purpose. A run where nothing was scored is a broken pipeline, and the one thing it must not do is look like a pass.

This threshold-on-mean-score gate suits an evaluator like `criteria_scorer` above, which returns partial credit rather than a bit. If your evaluators set `pass_` on their `EvaluationResult` instead, [`check_pass_failures`](../evaluation-reference.md#passfail-and-ci) covers the binary case in one call — import it as `from evaluatorq.evaluatorq import check_pass_failures`, which is the path that page uses, because it is not re-exported from the top-level package.

Pass it `treat_errors_as_failure=True` if you use it. Its default is `False`, and on that default an errored job contributes no evaluator scores and an errored evaluator leaves `pass_` unset, so a run where every judge call raised reports no failures and the build goes green. That is the same trap the empty-list case above returns 1 for.

Then `raise SystemExit(gate(scores_from(results), 0.8))` at the end of your script, and the workflow step fails when the agent regresses.

```yaml
name: Support agent simulation

on: pull_request

jobs:
  simulate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Install
        run: uv add "evaluatorq[simulation]"
      - name: Simulate the support agent
        env:
          ORQ_API_KEY: ${{ secrets.ORQ_API_KEY }}
          EVALUATORQ_DIR: ${{ runner.temp }}/evaluatorq
          ORQ_DISABLE_TRACING: "1"
        run: uv run python sim_in_eval.py
```

`ORQ_API_KEY` decides which workspace every PR's run uploads into, so point the secret at a workspace you are willing to fill with one Experiment per pull request — see [What gets uploaded](#what-gets-uploaded).

## What gets uploaded

With `ORQ_API_KEY` set, a finished `evaluatorq()` run sends its results to Orq and logs the Experiment URL:

```console
$ uv run python sim_in_eval.py
INFO  Results sent to Orq: support-simulation (1 rows created)
INFO  View your evaluation at: https://my.orq.ai/<workspace>/experiments/<experiment_id>?runId=<run_id>
```

Which workspace that is depends on `ORQ_API_KEY` alone. `ORQ_WORKSPACE` does not route anything — it is a display setting the dashboard reads to build trace deep-links.

!!! warning "On this path there is no flag that turns the upload off"

    `evaluatorq()` has no `upload_results` parameter. Every run of the script above, with `ORQ_API_KEY` set, adds a row to a table your whole team can see, and there is no argument you can pass to stop it. Iterating on a persona twenty times means twenty Experiments in that workspace.

    `upload_results=False` belongs to `simulate()` and `generate_and_simulate()`, which are a different entry point — reaching for it here does nothing, because there is no such parameter to pass. Even on those calls it only suppresses the Experiment upload: it is not an offline mode, and it does not stop the model-catalogue pricing lookup, so a run with the flag set still reaches Orq.

    So on this path the only real control is which workspace the key belongs to. If you are iterating rather than publishing a result, point `ORQ_API_KEY` at a workspace you keep for that, and move to the shared one when the cases have settled. Deleting a stray Experiment is a manual job in the Orq UI; neither the SDK nor the CLI exposes a delete.

Every conversation also emits OTel spans under `orq.job`, `orq.simulation.run` and `orq.simulation.turn`, so a run is inspectable turn by turn in the trace UI. Set `ORQ_DISABLE_TRACING=1` to turn that off.

## Where to next

- [Agent Simulation](agent-simulation.md) — personas, scenarios, criteria, and the built-in scorers, in depth.
- [Evaluation Reference](../evaluation-reference.md) — the `evaluatorq()` call itself, and what else can go in `jobs` and `evaluators`.
- [Dashboard](../dashboard.md) — reading a finished run locally.
- [Tracing](../tracing.md) — the span layout and the `ORQ_OTEL_*` batching knobs.
