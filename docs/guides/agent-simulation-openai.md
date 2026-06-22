# Simulate an OpenAI agent

No Orq deployment needed. `simulate()` accepts a `target_callback=` — any async
function that takes the conversation history and returns your agent's reply. Here
the agent *is* an OpenAI model, and the simulator/judge run on OpenAI too via
`sim_model=`.

```bash
pip install "evaluatorq[simulation]" openai
export OPENAI_API_KEY=sk-...
```

```python
import asyncio

from openai import AsyncOpenAI

from evaluatorq.contracts import Message
from evaluatorq.simulation import simulate
from evaluatorq.simulation.types import Criterion, Persona, Scenario

client = AsyncOpenAI()

SYSTEM = "You are a customer support agent for Acme Corp. Be concise and helpful."


async def openai_agent(messages: list[Message]) -> str:
    """Your agent under test — a raw OpenAI model."""
    history = [{"role": "system", "content": SYSTEM}]
    history += [{"role": m.role, "content": m.content or ""} for m in messages]
    resp = await client.chat.completions.create(model="gpt-4o-mini", messages=history)
    return resp.choices[0].message.content or ""


async def main():
    persona = Persona(name="Impatient Customer", patience=0.2, assertiveness=0.8)
    scenario = Scenario(
        name="Wrong Item Refund",
        goal="Get a full refund for the wrong item received",
        criteria=[
            Criterion(description="Agent asks for order details", type="must_happen"),
        ],
    )

    results = await simulate(
        evaluation_name="openai-agent-simulation",
        target_callback=openai_agent,        # your OpenAI agent
        personas=[persona],
        scenarios=[scenario],
        sim_model="gpt-4o-mini",             # simulator + judge on OpenAI directly
        max_turns=6,
        evaluator_names=["goal_achieved", "criteria_met"],
        upload_results=False,                # local-only run, no Orq experiment
    )

    result = results[0]
    score = result.goal_completion_score or 0.0
    print(f"Goal achieved: {result.goal_achieved}  score={score:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
```

The `target_callback` is the only difference from the [Orq agent
guide](agent-simulation.md) — personas, scenarios, criteria, and the result
shape are identical. Swap the callback body for any HTTP/LLM agent.
