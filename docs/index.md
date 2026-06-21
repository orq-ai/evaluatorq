# evaluatorq

Run LLM evaluations, red-team agents, and simulate multi-turn conversations
in Python — against any agent, with the Orq AI platform as optional infrastructure.

## Install

```bash
pip install evaluatorq                # core evaluation loop
pip install "evaluatorq[redteam]"     # + red teaming / simulation
```

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
function, or an Orq deployment. The Orq platform is optional — for result
storage, not a requirement.

## Quick look

```python
import asyncio

from evaluatorq import DataPoint, evaluatorq, job, string_contains_evaluator


@job("uppercase-converter")
async def uppercase_job(data: DataPoint, _row: int) -> str:
    return str(data.inputs.get("text", "")).upper()


async def main():
    await evaluatorq(
        "smoke-test",
        data=[
            DataPoint(inputs={"text": "hello world"}, expected_output="HELLO"),
            DataPoint(inputs={"text": "evaluatorq rocks"}, expected_output="EVALUATORQ"),
        ],
        jobs=[uppercase_job],
        evaluators=[string_contains_evaluator()],
        print_results=True,
    )


asyncio.run(main())
```

## Where to next

- **[Getting Started](guides/getting-started.md)** — your first evaluation in five minutes.
- **[Examples](examples/index.md)** — runnable scripts across every capability.
- **[Custom Evaluators & Frameworks](custom-evaluators-and-frameworks.md)** — extend the registries.
- **[API Reference](reference/evaluatorq.md)** — the full public API.
- **[Roadmap](roadmap.md)** — what's planned next.
