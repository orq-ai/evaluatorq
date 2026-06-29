# evaluatorq

Run LLM evaluations, red-team agents, and simulate multi-turn conversations
in Python — against any agent, with the Orq AI platform as optional infrastructure.

[Get Started](guides/getting-started.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/orq-ai/evaluatorq){ .md-button }

## Install

<!-- termynal -->

```console
$ pip install evaluatorq
---> 100%
Successfully installed evaluatorq
```

!!! tip "Optional extras"
    `pip install "evaluatorq[redteam]"` adds adversarial red teaming ·
    `pip install "evaluatorq[simulation]"` adds multi-turn agent simulation.

## What it does

<div class="grid cards" markdown>

-   :material-check-decagram:{ .lg .middle } __Evaluations__

    ---

    Run jobs over inline data or Orq datasets in parallel; score with custom or
    built-in evaluators; gate CI on pass/fail.

    [:octicons-arrow-right-24: Getting Started](guides/getting-started.md)

-   :material-account-voice:{ .lg .middle } __Agent simulation__

    ---

    A user-simulator LLM drives your agent across multi-turn conversations while
    a judge LLM scores whether it met its goals.

    [:octicons-arrow-right-24: Agent simulation](guides/agent-simulation.md)

-   :material-shield-sword:{ .lg .middle } __Red teaming__

    ---

    Adaptive adversarial attacks mapped to the OWASP LLM Top 10 and Agentic
    Security Initiative, with auto-discovered tool and memory attack surfaces.

    [:octicons-arrow-right-24: Red teaming](guides/red-teaming.md)

</div>

Works with LangGraph, OpenAI Agents SDK, PydanticAI, CrewAI, a plain async
function, or an Orq deployment. The Orq platform is optional: it stores results
and, when `ORQ_API_KEY` is set, routes the attacker and judge LLMs by default —
but you can bring your own and run entirely on OpenAI.

## Quick look

```python
import asyncio

from evaluatorq import (
    DataPoint,
    evaluatorq,
    job,
    string_contains_evaluator,
)


@job("greet")
async def greet_job(data: DataPoint, _row: int) -> str:
    name = str(data.inputs.get("name", ""))
    return f"Hello, {name}!"


async def main():
    data = [
        DataPoint(inputs={"name": "Ada"}, expected_output="Hello, Ada!"),
        DataPoint(inputs={"name": "Lin"}, expected_output="Hello, Lin!"),
    ]
    await evaluatorq(
        "smoke-test",
        data=data,
        jobs=[greet_job],
        evaluators=[string_contains_evaluator()],
        print_results=True,
    )


asyncio.run(main())
```

`print_results=True` renders a summary and a per-evaluator score panel:

```text
EVALUATION RESULTS

Summary:
╭──────────────────────┬───────╮
│ Metric               │ Value │
├──────────────────────┼───────┤
│ Total Data Points    │ 2     │
│ Failed Data Points   │ 0     │
│ Total Jobs           │ 2     │
│ Failed Jobs          │ 0     │
│ Success Rate         │ 100%  │
╰──────────────────────┴───────╯

Detailed Results:
╭──────────────────┬───────╮
│ Evaluators       │ greet │
├──────────────────┼───────┤
│ string-contains  │ 1.00  │
╰──────────────────┴───────╯
```

## Where to next

- **[Getting Started](guides/getting-started.md)** — your first evaluation in five minutes.
- **[Examples](examples/index.md)** — runnable scripts across every capability.
- **[Custom Evaluators & Frameworks](custom-evaluators-and-frameworks.md)** — extend the registries.
- **[API Reference](reference/evaluatorq.md)** — the full public API.
- **[Roadmap](roadmap.md)** — what's planned next.
