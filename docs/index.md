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
function, or an Orq deployment. The Orq platform is optional — for result
storage, not a requirement.

## Quick look

```python
import asyncio

from evaluatorq import (
    DataPoint,
    evaluatorq,
    job,
    string_contains_evaluator,
)


@job("uppercase")
async def uppercase_job(data: DataPoint, _row: int) -> str:
    text = str(data.inputs.get("text", ""))
    return text.upper()


async def main():
    data = [
        DataPoint(inputs={"text": "hello"}, expected_output="HELLO"),
        DataPoint(inputs={"text": "world"}, expected_output="WORLD"),
    ]
    await evaluatorq(
        "smoke-test",
        data=data,
        jobs=[uppercase_job],
        evaluators=[string_contains_evaluator()],
        print_results=True,
    )


asyncio.run(main())
```

Running it prints a pass/fail table:

```text
       smoke-test — uppercase
┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ input ┃ output  ┃ expected ┃ score  ┃
┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ hello │ HELLO   │ HELLO    │ ✓ pass │
│ world │ WORLD   │ WORLD    │ ✓ pass │
└───────┴─────────┴──────────┴────────┘
        2/2 passed (100%)
```

## Where to next

- **[Getting Started](guides/getting-started.md)** — your first evaluation in five minutes.
- **[Examples](examples/index.md)** — runnable scripts across every capability.
- **[Custom Evaluators & Frameworks](custom-evaluators-and-frameworks.md)** — extend the registries.
- **[API Reference](reference/evaluatorq.md)** — the full public API.
- **[Roadmap](roadmap.md)** — what's planned next.
