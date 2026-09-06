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
```

The user simulator and the judge are model calls, so one of those two keys has to be set. With `ORQ_API_KEY` set, the run also uploads an Experiment to your Orq workspace when it finishes — see [What gets uploaded](#what-gets-uploaded) before running this against a workspace you share with other people.

## A complete run

Save this as `sim_in_eval.py`. The target here is an ordinary async function, so nothing but the two model keys is required to run it.

```python
import asyncio

from evaluatorq import DataPoint, EvaluationResult, evaluatorq
from evaluatorq.simulation import (
    CommunicationStyle,
    Criterion,
    Message,
    Persona,
    Scenario,
    wrap_simulation_agent,
)


async def support_agent(messages: list[Message]) -> str:
    """The agent under test. Replace this with your own."""
    last = messages[-1].content if messages else ""
    if "order" in str(last).lower():
        return "Thanks — I have pulled up that order and started the refund."
    return "Sorry about that. Could you share your order number so I can look it up?"


CRITERIA_THRESHOLD = 0.8


async def criteria_scorer(params) -> EvaluationResult:
    """Score the fraction of the scenario's criteria the judge marked satisfied.

    Sets `pass_` as well as `value`, so `check_pass_failures()` can gate a build
    on it without anyone re-deriving the threshold at the call site.
    """
    metadata = params["output"].get("metadata", {})
    terminated_by = metadata.get("terminated_by")
    if terminated_by in {"error", "timeout"}:
        # The conversation never reached the judge, so there is no verdict. The
        # value stays non-numeric because a 0.0 here would read as "the agent
        # scored zero" rather than "nothing was measured" — but pass_ is still
        # False, because an unmeasured run must never look like a passing one.
        return EvaluationResult(
            value="unevaluated",
            pass_=False,
            explanation=f"no verdict: run ended in {terminated_by}",
        )
    results = metadata.get("criteria_results") or {}
    if not results:
        return EvaluationResult(value=0.0, pass_=False, explanation="no criteria were judged")
    passed = sum(1 for ok in results.values() if ok)
    score = passed / len(results)
    return EvaluationResult(
        value=score,
        pass_=score >= CRITERIA_THRESHOLD,
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

    # CommunicationStyle: formal, casual, terse, verbose.
    # A Scenario also takes starting_emotion (neutral, frustrated, confused,
    # happy, urgent) and a Persona takes emotional_arc (stable, escalating,
    # de_escalating, volatile, manipulative, hostile).

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
            # A run the runner ended in error or timeout lands in job_result.error
            # and in the table's "Failed Jobs"; this exits on it instead of reading
            # a partial transcript as a result.
            if job_result.error:
                raise SystemExit(f"simulation did not run: {job_result.error}")


asyncio.run(main())
```

It prints the standard evaluatorq results table, then the two lines the script asks for:

```console
$ uv run python sim_in_eval.py
metadata keys: ['criteria_results', 'framework', 'goal_achieved', 'goal_completion_score', 'reason', 'rules_broken', 'terminated_by', 'turn_count', 'turn_metrics']
criteria: {'Agent asks for order details': True, 'Agent blames the customer': True}
```

The user simulator and the judge are model calls, so `turn_count` and `goal_achieved` vary between runs of this same script. Assert on the shape of `metadata`, never on a turn count.

## The parameters

| Parameter | Default | What it does |
|---|---|---|
| `name` | `"simulation"` | Names the job in the results table and in `job_result.name`. |
| `target` | — | The agent under test, as a callable taking `list[Message]`. |
| `agent_key` | — | An Orq deployment key, used instead of `target`. Exactly one of the two is required; if both are given, `target` wins. |
| `max_turns` | `10` | Turn budget per conversation. Hitting it stops the run with `terminated_by: max_turns`. |
| `model` | `openai/gpt-5.6-luna` | The model for the **user simulator and the judge** — not for the agent under test, which is whatever `target` or `agent_key` resolves to. |
| `user_simulator` | built-in | A custom `BaseAgent` to play the user. |
| `judge` | built-in | A custom `BaseAgent` to score the conversation. |

`model` is the one worth reading twice: the simulated user and the judge are themselves model calls, and this is the only knob for them. Changing it does not touch the agent you are testing.

## `goal_achieved` and `criteria_results` are different questions

- `criteria_results` is the judge's per-criterion audit: for each `Criterion` on the scenario, did the thing it names occur, and does that occurrence mean pass or fail for its type? A `must_not_happen` criterion is `True` when the bad thing did **not** occur.
- `goal_achieved` is the judge's verdict on the scenario's `goal` as a whole — here, whether the customer actually got their refund.

The common disagreement is every criterion passing while `goal_achieved` is `False`, and the usual cause is the turn budget: a conversation that hits `max_turns` before the agent finishes the job stops with `terminated_by: max_turns`, and the goal is unmet even though the agent did every individual thing you thought to write down. Raising `max_turns` is the fix when the agent was on track; it is not the fix when the agent was stuck.

Scoring both, as two separate evaluators, is usually what you want — criteria tell you which behaviour is missing, the goal score tells you whether it mattered.

## What the job returns

The job's output is an OpenResponses response dict. The transcript is in `input` and `output` — user turns as `input_text`, agent turns as `output_text` — and everything the judge concluded is under `metadata`:

| `metadata` key | Type | What it holds |
|---|---|---|
| `framework` | `str` | Always `"simulation"`. |
| `goal_achieved` | `bool` | Judge's verdict on the scenario goal. |
| `goal_completion_score` | `float` | Graded version of the same verdict. |
| `terminated_by` | `str` | Why the conversation stopped: `judge`, `max_turns`, `error`, `timeout`. |
| `reason` | `str` | The termination reason in words. |
| `turn_count` | `int` | Turns actually run. |
| `rules_broken` | `list[str]` | Criteria the judge marked violated, **by id** (`criteria_0`, `criteria_1`). |
| `criteria_results` | `dict[str, bool]` | Per-criterion pass/fail, **by description**. Present only when the judge produced at least one criterion verdict. |
| `criteria_verified` | `bool \| None` | Whether the criteria result has a per-criterion judge audit. Always present; `None` means the result predates this field. |
| `criteria_meta` | `list[dict] \| None` | Per-criterion audit records, including `id`, `type`, `passed`, `audited`, and `evidence`. Always present; `None` means no records were available. |
| `criteria_errors` | `list \| None` | Malformed criteria metadata recorded during scoring. Always present; it is `None` on this job path because conversion happens before scoring. |
| `scorer_errors` | `dict \| None` | Scorer failures recorded after conversion. Always present; it is `None` on this job path because conversion happens before scoring. |
| `datapoint_id` | `str \| None` | The simulation datapoint identifier. Always present; `None` when the result carries no datapoint id. |
| `turn_metrics` | `list[dict]` | Per-turn latency and token detail. Present only when metrics were collected. |

`criteria_results` and `turn_metrics` are optional, so read them with a default rather than by subscript. The audit and error keys listed as always present are different: their `None` value carries meaning, so preserve it instead of treating it as an empty collection.

`error` and `timeout` mean the conversation never reached the judge. Those runs are *unevaluated*, not failed, and scoring them as zero reports a broken pipeline as a bad agent — which is why `criteria_scorer` above checks `terminated_by` before it looks at anything else.

!!! note "The conversion preserves audit and usage state"

    `to_open_responses()` always includes `criteria_verified`, `criteria_meta`, `criteria_errors`, `scorer_errors`, and `datapoint_id` in `metadata`, plus `token_usage_known` beside the top-level `usage`. On the evaluatorq job path, conversion happens before scoring, so `criteria_errors` and `scorer_errors` are `null`; that means those errors were not known yet, not that scoring found none. A metadata value that cannot be serialized is published as its `repr()` and logged with a warning.

!!! warning "`rules_broken` and `criteria_results` are keyed differently"

    `rules_broken` normally holds criterion **ids** (`criteria_0`, `criteria_1`) and `criteria_results` is keyed by criterion **description**. When the judge's per-criterion audit did not arrive, `rules_broken` degrades to the judge's free-text strings instead, and on this path nothing tells you which form you got. They do not join on a shared key, so a scorer that looks up a `rules_broken` entry in `criteria_results` finds nothing and silently scores every row as clean. Read one or the other, not both. Descriptions are also not unique: two criteria worded identically collapse into a single `criteria_results` entry, which is why the id-keyed form exists at all.

## Against an Orq deployment

If the agent under test is deployed on Orq rather than living in your Python process, pass `agent_key=` instead of `target=` and change nothing else. Internally the wrapper resolves the key through `from_orq_deployment()`, which builds the callback lazily — constructing the job issues no request, so a wrong key surfaces on the first conversation, not at import time.

```python
from evaluatorq.simulation import wrap_simulation_agent

job = wrap_simulation_agent(name="support-sim", agent_key="my-support-agent", max_turns=3)
```

In the script above, that one line replaces the `target=support_agent` line; the `DataPoint` list, the evaluators, the `evaluatorq()` call and the `aclose()` all stay as they are. Passing both `target=` and `agent_key=` is not an error — `target=` wins, and `agent_key=` is ignored — so pass exactly the one you mean.

## Always close the job

The callable `wrap_simulation_agent()` returns owns a long-lived `SimulationRunner` and the HTTP connection pool underneath it. Nothing closes that for you when `evaluatorq()` returns, so a script that skips `aclose()` leaks the pool until the process exits — survivable in a one-shot script, not survivable in a long-lived worker that builds a job per request.

Put the `aclose()` in a `finally`, as the complete run above does, not after the `evaluatorq()` call. An exception raised mid-batch would otherwise skip it, which is exactly the run you least want leaking.

## Accepted input shapes

`wrap_simulation_agent()` reads one conversation per `DataPoint`, and accepts four spellings in `inputs`:

The table is in the order the code checks them, which is not the order you are most likely to use them in:

| `inputs` contains | Notes |
|---|---|
| `datapoint` | A full `SimulationDatapoint`. Every field is required: `id`, `persona`, `scenario`, `user_system_prompt` and `first_message`. The shape check only looks for the middle three, so a dict missing `id` or `user_system_prompt` gets past it and fails in validation instead. |
| `datapoints` | A list of exactly one datapoint, each with the five fields above. More than one raises. |
| `persona` + `scenario` | The usual form, and the one the complete run above uses. Add `first_message` to fix the opening line instead of generating it. |
| `personas` + `scenarios` | Lists of exactly one each. More than one raises. |

The first matching shape in the order listed wins; a row carrying both `datapoint` and `persona` + `scenario` silently uses the `datapoint`.

`persona`, `scenario`, `datapoint` and `datapoints` are also accepted as JSON strings, because the Orq datasets API rejects nested objects in `inputs` — a dataset-backed row arrives with `persona` and `scenario` stringified, and the wrapper parses them back. The `personas` + `scenarios` form is **not** coerced: pass real lists there, or it raises `Expected 'personas' and 'scenarios' to be arrays`. The list forms exist for compatibility with datapoint files and cap at one element on purpose: one row is one conversation, so a row holding two personas is an ambiguity rather than a batch.

Anything else raises `ValueError` naming the four shapes.

## Scoring is not the wrapper's job

`wrap_simulation_agent()` used to take an `evaluators=` argument and no longer does. Passing it raises `TypeError: wrap_simulation_agent() no longer accepts 'evaluators='` immediately, rather than dropping your scoring on the floor.

Scoring belongs on the `evaluatorq()` call, as `evaluatorq(..., evaluators=[...])`. That is the whole reason to reach for this wrapper: the conversation becomes a normal evaluatorq output, and every evaluator you already have applies to it.

The built-in simulation scorers are the exception. `goal_achieved_scorer`, `criteria_met_scorer` and their siblings take a `SimulationResult`, and the job hands `evaluatorq()` an OpenResponses dict — the `SimulationResult` does not survive the conversion. `criteria_scorer` in the run above is a hand-rolled stand-in for `criteria_met_scorer` working from the reduced metadata.

Constructing the job with neither `target=` nor `agent_key=` raises `ValueError` for the same reason — the wrapper has nothing to talk to.

## In CI

Three things change for a non-interactive run:

- `print_results=False` on the `evaluatorq()` call, so the results table does not go to a log nobody reads.
- `EVALUATORQ_DIR` pointed at a scratch directory, so the run store does not persist between jobs and one workflow cannot resolve another's runs.
- `ORQ_DISABLE_TRACING=1` if you do not want CI spans in your traces.

If you are arriving from `simulate()`, note what does and does not carry over. `simulate()` takes `exit_on_failure`, which defaults to `True` and raises `SimulationDroppedError` when a datapoint produced no conversation — *dropped* (a job raised and no result was cached) or ended in `error`/`timeout`. That is an infrastructure gate, not a quality gate: scorer verdicts are reporting only there too, so an agent that answered every question badly still exits 0. Moving to `wrap_simulation_agent()` costs you that infrastructure gate, because the parameter belongs to `simulate()` and there is no equivalent on `evaluatorq()` — gate on `job_result.error` yourself, as the example above does.

`evaluatorq()` does not fail the process for you either — it returns results and exits 0 whatever the scores are. So on this path the build verdict is yours to write, and it is one line:

```python
from evaluatorq.evaluatorq import check_pass_failures

# At the end of main(), after the evaluatorq() call has returned `results`:
#     if check_pass_failures(results, treat_errors_as_failure=True):
#         raise SystemExit(1)
print("gate helper imported:", check_pass_failures.__name__)
```

Import it from `evaluatorq.evaluatorq`, not the top-level package — it is not re-exported.

`check_pass_failures` reads `pass_` off each evaluator score, which is why `criteria_scorer` above sets `pass_` alongside `value`. Keeping the threshold inside the scorer is the point: the score and the verdict on that score stay in one place, so a CI step cannot gate on a different bar than the one the evaluator reports against, and the scorer's three branches — a real verdict, no criteria judged, and no verdict at all — each state their own `pass_` rather than leaving a caller to infer one from a number.

**Always pass `treat_errors_as_failure=True`.** The default is `False`, and on that default an errored job contributes no evaluator scores and an errored evaluator leaves `pass_` unset, so a run where every judge call raised reports no failures and the build goes green. A run that measured nothing is the one thing that must not look like a pass.

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
        run: uv add "evaluatorq[simulation,otel]"
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

    `evaluatorq()` exposes no public `upload_results` parameter. Every run of the script above, with `ORQ_API_KEY` set, adds a row to a table your whole team can see. Iterating on a persona twenty times means twenty Experiments in that workspace.

    `upload_results=False` belongs to `simulate()` and `generate_and_simulate()`, which are a different entry point — reaching for it here does nothing, because there is no such parameter to pass. Even on those calls it only suppresses the Experiment upload: it is not an offline mode, and it does not stop the model-catalogue pricing lookup, so a run with the flag set still reaches Orq.

    So on this path the only real control is which workspace the key belongs to. If you are iterating rather than publishing a result, point `ORQ_API_KEY` at a workspace you keep for that, and move to the shared one when the cases have settled. Deleting a stray Experiment is a manual job in the Orq UI; neither the SDK nor the CLI exposes a delete.

With the `otel` extra installed (`evaluatorq[simulation,otel]` — the `simulation` extra alone ships no OpenTelemetry), every conversation also emits spans under `orq.job`, `orq.simulation.run` and `orq.simulation.turn`, so a run is inspectable turn by turn in the trace UI. Set `ORQ_DISABLE_TRACING=1` to turn that off.

## Where to next

- [Agent Simulation](agent-simulation.md) — personas, scenarios, criteria, and the built-in scorers, in depth.
- [Evaluation Reference](../evaluation-reference.md) — the `evaluatorq()` call itself, and what else can go in `jobs` and `evaluators`.
- [Dashboard](../dashboard.md) — reading a finished run locally.
- [Tracing](../tracing.md) — the span layout and the `ORQ_OTEL_*` batching knobs.
