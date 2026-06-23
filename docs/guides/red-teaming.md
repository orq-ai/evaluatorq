# Red Teaming

Probe an agent or model with adversarial attacks mapped to the OWASP **LLM Top
10** and **Agentic Security Initiative (ASI)** frameworks, then read off a
resistance rate.

```bash
pip install "evaluatorq[redteam]"
export ORQ_API_KEY=...   # targets your Orq agent + routes the attacker LLM
```

## Modes

- **dynamic** — an LLM generates fresh attacks per run against the categories you pick.
- **static** — replays a fixed dataset of known attacks. Deterministic, cheap, good for CI.
- **hybrid** — static seeds plus dynamic expansion.

## Red-team an Orq agent

Point `red_team()` at an Orq agent by key (`"agent:<key>"`, from AI Studio →
Agents). The attack runs through the Orq platform — no model wiring on your side.

```python
import asyncio

from evaluatorq.redteam import red_team


async def main():
    report = await red_team(
        "agent:your-agent-key",            # Orq agent, routed via ORQ_API_KEY
        mode="dynamic",
        categories=["LLM01", "LLM07"],     # prompt injection, system-prompt leakage
        max_dynamic_datapoints=5,
        max_turns=2,
        generate_strategies=False,
    )

    print(f"Resistance rate: {report.summary.resistance_rate:.0%}")
    print(f"Vulnerabilities: {report.summary.vulnerabilities_found}/{report.summary.total_attacks}")


if __name__ == "__main__":
    asyncio.run(main())
```

!!! tip "Targeting a raw OpenAI model instead?"
    To red-team an OpenAI model directly (no Orq agent), see
    [Red-team an OpenAI model](red-teaming-openai.md).

## Categories vs vulnerabilities

`categories=` takes OWASP framework codes — `LLM01` (prompt injection), `LLM07`
(system-prompt leakage), `ASI01`, and so on. Each category maps to one or more
underlying vulnerabilities, the atomic unit that strategies and evaluators bind
to. Scope a run by category; the report breaks results down per vulnerability.

A higher `resistance_rate` is better — it's the fraction of attacks the target
withstood.

## In CI

For a fast gate, use the smoke example
([`08_quick_smoke_test.py`](../examples/redteam/08_quick_smoke_test.md)) — a
small fixed run you can assert a minimum resistance rate against.

## Where to next

- **[Examples › Red Teaming](../examples/index.md)** — static datasets, category filtering, custom clients, multi-target, report inspection, custom hooks.
- **[Custom Evaluators & Frameworks](../custom-evaluators-and-frameworks.md)** — add your own vulnerabilities and attack strategies.
```
