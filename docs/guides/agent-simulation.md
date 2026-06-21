# Agent Simulation

Drive your agent through realistic multi-turn conversations without writing test
transcripts by hand. Three LLMs are in play:

- **Your agent** — the target under test (a callback, or an Orq deployment).
- **User simulator** — plays a **persona** pursuing a **scenario** goal, turn by turn.
- **Judge** — scores whether the goal was met and whether any rules were broken.

Requires the simulation extra and an `ORQ_API_KEY` (the simulator and judge are LLM calls):

```bash
pip install "evaluatorq[simulation]"
```

## Persona + scenario + judge

A **persona** is *who* is talking (patience, assertiveness, tone). A **scenario**
is *what they want* plus the **criteria** the agent must (or must not) satisfy.

```python
import asyncio

from evaluatorq.contracts import Message
from evaluatorq.simulation import simulate
from evaluatorq.simulation.types import (
    CommunicationStyle, Criterion, EmotionalArc, Persona, Scenario, StartingEmotion,
)


async def support_agent(messages: list[Message]) -> str:
    """Your agent. Replace with a real LLM/HTTP call."""
    last = (messages[-1].content or "").lower() if messages else ""
    if "refund" in last:
        return "I can help with that. Could you share your order number?"
    return "Thanks for reaching out. How can I assist you today?"


async def main():
    persona = Persona(
        name="Impatient Customer",
        patience=0.2, assertiveness=0.8, politeness=0.4, technical_level=0.3,
        communication_style=CommunicationStyle.terse,
        background="Received the wrong item and wants a refund urgently",
        emotional_arc=EmotionalArc.escalating,
    )
    scenario = Scenario(
        name="Wrong Item Refund",
        goal="Get a full refund for the wrong item received",
        context="Ordered headphones but received a phone case instead",
        starting_emotion=StartingEmotion.frustrated,
        criteria=[
            Criterion(description="Agent asks for order details", type="must_happen"),
            Criterion(description="Agent acknowledges the mistake", type="must_happen"),
            Criterion(description="Agent blames the customer", type="must_not_happen"),
        ],
    )

    results = await simulate(
        evaluation_name="basic-simulation-example",
        target_callback=support_agent,
        personas=[persona],
        scenarios=[scenario],
        max_turns=6,
        evaluator_names=["goal_achieved", "criteria_met"],
    )

    result = results[0]
    print(f"Goal achieved: {result.goal_achieved}  score={result.goal_completion_score:.2f}")
    for msg in result.messages:
        print(f"{'User' if msg.role == 'user' else 'Agent'}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
```

One persona × one scenario yields one `SimulationResult` with `goal_achieved`,
`goal_completion_score`, `turn_count`, `rules_broken`, and the full message
transcript.

## From a mock to a live deployment

`target_callback=` accepts any async function. To test an Orq deployment
instead, pass `agent_key=` (see
[`02_orq_deployment_simulation.py`](../examples/agent_simulation/02_orq_deployment_simulation.md)).
The simulator/judge model defaults to `openai/gpt-5.4-mini`; override with
`sim_model=`.

## Where to next

- **[Examples › Agent Simulation](../examples/index.md)** — tool simulation, hardening loops, LangGraph / CrewAI / OpenAI Agents targets.
- **[Red Teaming](red-teaming.md)** — adversarial, attack-driven testing.
