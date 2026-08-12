<p align="center">
  <img src="docs/assets/evaluatorq-splash.svg" alt="evaluatorq — LLM evals, red teaming, agent simulation" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/evaluatorq/"><img src="https://img.shields.io/pypi/v/evaluatorq.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/evaluatorq/"><img src="https://img.shields.io/pypi/pyversions/evaluatorq.svg" alt="Python versions"></a>
  <a href="https://github.com/orq-ai/evaluatorq/actions/workflows/ci.yml"><img src="https://github.com/orq-ai/evaluatorq/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://htmlpreview.github.io/?https://github.com/orq-ai/evaluatorq/blob/python-coverage-comment-action-data/htmlcov/index.html"><img src="https://raw.githubusercontent.com/orq-ai/evaluatorq/python-coverage-comment-action-data/badge.svg" alt="Coverage"></a>
  <a href="https://orq-ai.github.io/evaluatorq/"><img src="https://img.shields.io/badge/Docs-Live%20Site-0A7B83" alt="Docs"></a>
  <a href="https://github.com/orq-ai/evaluatorq/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <b>Find out how your AI agent breaks — before your users do.</b>
</p>

<p align="center">
  <a href="https://orq-ai.github.io/evaluatorq/">Documentation</a> ·
  <a href="https://orq-ai.github.io/evaluatorq/guides/getting-started/">Get Started</a> ·
  <a href="https://orq-ai.github.io/evaluatorq/guides/red-teaming/">Red Teaming</a> ·
  <a href="https://orq-ai.github.io/evaluatorq/guides/agent-simulation/">Agent Simulation</a> ·
  <a href="https://orq-ai.github.io/evaluatorq/dashboard/">Dashboard</a>
</p>

Shipping an agent means answering three questions no test suite answers: does it give good answers, can it be talked into doing something it shouldn't, and does it hold up over a real conversation with an impatient human? evaluatorq answers all three from Python. It scores your agent's outputs against your data, attacks it the way a bad actor would — jailbreaks, prompt injection, tool abuse, data exfiltration — and puts a simulated user in front of it for a few dozen turns. Then it hands you a report naming what broke and what to do about it.

It runs locally against any agent — LangChain, LangGraph, OpenAI Agents SDK, PydanticAI, CrewAI, a plain async function, or an Orq-hosted agent. Nothing leaves your machine unless you opt into the [Orq](https://orq.ai) platform.

![Red team report: attack success rate, vulnerabilities found, and per-finding remediation guidance](docs/assets/readme-hero-redteam.png)

## Install

```bash
uv add evaluatorq                     # core evaluation
uv add "evaluatorq[redteam]"          # + adversarial red teaming
uv add "evaluatorq[simulation]"       # + multi-turn agent simulation
uv add "evaluatorq[all]"              # everything, including the dashboard
```

New here? Take the first line — it and the quick start below need no API key and no account (set `ORQ_API_KEY` and results also upload to Orq). On pip: `python -m pip install evaluatorq`.

## Quick start

Two versions of a support agent, the same questions, one table telling you which one to ship:

```python
import asyncio

from evaluatorq import DataPoint, evaluatorq, job, string_contains_evaluator

POLICY = {
    "refund": "Refunds are available within 30 days of delivery.",
    "ship": "Orders ship within 2 business days.",
    "warranty": "Every device carries a 12 months warranty.",
}


@job("agent-v1")
async def agent_v1(data: DataPoint, _row: int) -> str:
    """Answers from memory — so it only really knows about refunds."""
    question = str(data.inputs["question"]).lower()
    if "refund" in question:
        return "Sure — you can request a refund within 30 days of delivery."
    return "Our support team is happy to help with that."


@job("agent-v2")
async def agent_v2(data: DataPoint, _row: int) -> str:
    """Looks the answer up in the support policy first."""
    question = str(data.inputs["question"]).lower()
    for topic, answer in POLICY.items():
        if topic in question:
            return answer
    return "Our support team is happy to help with that."


async def main():
    data = [
        DataPoint(inputs={"question": "How do I get a refund?"}, expected_output="30 days"),
        DataPoint(inputs={"question": "When will my order ship?"}, expected_output="2 business days"),
        DataPoint(inputs={"question": "How long is the warranty?"}, expected_output="12 months"),
    ]
    await evaluatorq(
        "support-agent-eval",
        data=data,
        jobs=[agent_v1, agent_v2],
        evaluators=[string_contains_evaluator()],
        parallelism=3,
    )


asyncio.run(main())
```

```bash
uv run support_agent_eval.py
```

<img src="docs/assets/readme-eval-terminal.svg" alt="Terminal output: summary table and a Detailed Results table scoring agent-v1 at 0.33 against agent-v2 at 1.00 on the string-contains evaluator" width="720">

Every job runs against every data point, so adding a variant adds a column. Swap the two function bodies for real model or agent calls and nothing else changes. Any evaluator that returns `pass_=False` exits the process non-zero, so the same script gates CI — which is why this run ends with status 1.

This is the repo's [`examples/lib/basics/support_agent_eval.py`](examples/lib/basics/support_agent_eval.py), minus its `__main__` guard.

→ [Getting Started](https://orq-ai.github.io/evaluatorq/guides/getting-started/) ·
[Evaluation reference](https://orq-ai.github.io/evaluatorq/evaluation-reference/) ·
[Structured scores](https://orq-ai.github.io/evaluatorq/structured-results/) ·
[LLM as a jury](https://orq-ai.github.io/evaluatorq/llm-as-a-jury/)

## Red teaming

**19 OWASP categories · 18 vulnerabilities · 45 curated attack strategies · 16 delivery methods · 18 LLM judges.** evaluatorq inspects the target, picks attack strategies per vulnerability, generates the prompts, runs them (single- or multi-turn), and judges each response with an evaluator written for that specific vulnerability.

| OWASP Agentic Security Initiative | OWASP LLM Top 10 |
|---|---|
| ASI01 Agent Goal Hijacking | LLM01 Prompt Injection |
| ASI02 Tool Misuse & Exploitation | LLM02 Sensitive Information Disclosure |
| ASI03 Identity & Privilege Abuse | LLM03 Supply Chain Vulnerabilities |
| ASI04 Supply Chain Vulnerabilities | LLM04 Data and Model Poisoning |
| ASI05 Unexpected Code Execution | LLM05 Improper Output Handling |
| ASI06 Memory & Context Poisoning | LLM06 Excessive Agency |
| ASI07 Insecure Inter-Agent Communication | LLM07 System Prompt Leakage |
| ASI08 Cascading Failures | LLM08 Vector and Embedding Weaknesses |
| ASI09 Human-Agent Trust Exploitation | LLM09 Misinformation |
| ASI10 Rogue Agents | |

Each category maps to a vulnerability with its own judge. Categories without curated strategies get them generated per-run against the target's actual tools and system prompt — see the [strategy coverage table](https://orq-ai.github.io/evaluatorq/guides/red-teaming/#coverage).

```python
import asyncio

from evaluatorq.redteam import red_team


async def main():
    report = await red_team(
        "agent:my-agent-key",
        categories=["LLM01", "ASI01", "ASI02"],
        max_dynamic_datapoints=5,
        max_turns=3,
    )
    print(f"Resistance rate: {report.summary.resistance_rate:.0%}")
    print(f"Vulnerabilities: {report.summary.vulnerabilities_found}")


asyncio.run(main())
```

Targets can be an Orq agent (`"agent:<key>"`), an Orq deployment (`"deployment:<key>"`), a raw model (`OpenAIModelTarget("openai/gpt-5.4-mini")`), or an agent from an external framework. Every attack, response and verdict is browsable afterwards:

![Conversation viewer showing an indirect prompt injection and the judge's verdict](docs/assets/readme-redteam-transcript.png)

### What a run costs

Measured from real runs against an Orq-hosted agent, judged by `gpt-5-mini` (numbers are that workspace's billed cost and wall clock, not estimates):

| Run | Wall clock | Tokens | Cost |
|---|---|---|---|
| Hybrid, 4 attacks against one agent | 3m 53s | 51k | $0.063 |
| Same shape, second run | 3m 47s | 48k | $0.065 |
| Short dynamic runs (1–2 categories) | 9s – 1m 26s | 6k – 26k | $0.004 – $0.031 |

Across those runs that works out to a cent or two, and roughly a minute, per attack — dominated by the judge's reasoning tokens. It is a handful of runs, not a benchmark, so treat it as an order of magnitude. Attacks run concurrently (`parallelism`), so a 40-attack sweep is minutes, not hours. Attacker and judge models are both configurable — point them at a cheaper model and the whole run gets cheaper.

→ [Red teaming guide](https://orq-ai.github.io/evaluatorq/guides/red-teaming/) ·
[Intro notebook](examples/red_teaming_intro.ipynb) ·
[Example scripts](examples/redteam/)

## Agent simulation

The non-adversarial counterpart: a user-simulator LLM plays a persona pursuing a goal across a multi-turn conversation, and a judge LLM scores each run against your criteria. Fix what the transcripts show you, re-run the frozen set, and the difference is the point:

![58% of scenarios passed before, 88% after re-running the frozen set against the improved agent](docs/assets/readme-sim-payoff.png)

From the [webinar demo](examples/agent_simulation/webinar_demo/) against Sterling, a credit-card support agent: every one of the 10 original failures showed up past turn 4, where single-prompt testing never looks.

```mermaid
flowchart LR
    P["Persona<br/>impatient, terse"] --> U["User simulator LLM"]
    S["Scenario<br/>goal + criteria"] --> U
    U <--> A["Your agent"]
    U --> J["Judge LLM"]
    A --> J
    J --> R["goal_achieved<br/>criteria_met<br/>rules_broken"]
```

```python
from evaluatorq.simulation import simulate

results = await simulate(
    evaluation_name="support-agent-sim",
    target="agent:my-support-agent",   # or any local async callable
    personas=[persona],
    scenarios=[scenario],
    max_turns=8,
)
print(results[0].goal_achieved, results[0].goal_completion_score)
```

Runs exit non-zero on failure by default (`exit_on_failure=True`), so they drop straight into CI. Recordings of simulations driving real agents:
[OpenAI Agents SDK](docs/assets/sim-openai-agents.mp4) ·
[LangGraph](docs/assets/sim-langgraph.mp4) ·
[CrewAI](docs/assets/sim-crewai.mp4) ·
[PydanticAI](docs/assets/sim-pydantic-ai.mp4)

→ [Agent simulation guide](https://orq-ai.github.io/evaluatorq/guides/agent-simulation/) ·
[Intro notebook](examples/agent_simulation_intro.ipynb) ·
[Example scripts](examples/agent_simulation/)

## Dashboard

Every red team and simulation run is saved locally. `eq dashboard` serves them all — filter findings, read transcripts, compare runs, export HTML/CSV/JSON.

```bash
eq dashboard
```

![Dashboard overview: runs by type, attack resistance, findings by severity, token usage](docs/assets/readme-dashboard.png)

→ [Dashboard guide](https://orq-ai.github.io/evaluatorq/dashboard/)

## CLI

The package installs `eq` (and its longer alias `evaluatorq`):

```bash
eq redteam run -t agent:my-agent     # red team an agent
eq sim run --target agent:my-agent   # generate personas/scenarios and simulate
eq dashboard                         # browse saved runs
eq --help
```

→ [CLI reference](https://orq-ai.github.io/evaluatorq/cli-reference/overview/)

## Configuration

Everything is environment variables; none are required for local evaluation. `ORQ_API_KEY` unlocks Orq datasets, result upload and automatic tracing; `OPENAI_API_KEY` backs red teaming and simulation without Orq.

→ [Configuration](https://orq-ai.github.io/evaluatorq/configuration/) ·
[Tracing](https://orq-ai.github.io/evaluatorq/tracing/)

## Development

`uv sync --all-extras --all-groups`, then `uv run pytest -m 'not integration'`. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.
