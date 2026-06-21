# evaluatorq

Run LLM evaluations, red-team agents, and simulate multi-turn conversations
in Python — against any agent, with the Orq AI platform as optional infrastructure.

## Install

```bash
pip install evaluatorq                # core evaluation loop
pip install "evaluatorq[redteam]"     # + red teaming / simulation
```

## What it does

| | |
|---|---|
| **Evaluations** | Run jobs over inline data or Orq datasets in parallel; score with custom or built-in evaluators; gate CI on pass/fail. |
| **Agent simulation** | A user-simulator LLM drives your agent across multi-turn conversations while a judge LLM scores whether it met its goals. |
| **Red teaming** | Adaptive adversarial attacks mapped to the OWASP LLM Top 10 and Agentic Security Initiative, with auto-discovered tool and memory attack surfaces. |

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
- **[Agent Simulation](guides/agent-simulation.md)** — drive multi-turn conversations with a simulated user.
- **[Red Teaming](guides/red-teaming.md)** — OWASP security testing for agents.
- **[Custom Evaluators & Frameworks](custom-evaluators-and-frameworks.md)** — extend the registries.
- **[Examples](examples/index.md)** — runnable scripts across every capability.
- **[API Reference](reference/evaluatorq.md)** — the full public API.
- **[Roadmap](roadmap.md)** — what's planned next.
```
