# Demo — Red-Teaming a Refund Agent

Runnable demo for the 45-minute red-teaming webinar (2026-05-19).

A customer-service refund agent with three tools (`lookup_order`, `issue_refund`, `get_policy`) is deployed to the orq platform in two variants — `vulnerable` and `fixed`. `evaluatorq.red_team` drives policy-bypass attacks against both, scores them with an LLM evaluator, and produces a comparison report.

Attack class: **policy bypass via social engineering** (OWASP Agentic ASI01 + ASI04). The vulnerable variant treats chat content as authority; the fixed variant rejects authority claims and re-fetches policy each turn.

## Slides

Self-contained Reveal.js deck at `presentation.html` (open directly in browser, no dependencies). PDF export at `presentation.pdf`.

## Prereqs

- `ORQ_API_KEY` exported in env
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installed

## Setup

```bash
cd packages/evaluatorq-py/examples/redteam/refund_agent_demo
uv sync
export ORQ_API_KEY=...
```

## Run

```bash
# Create KB + tools + both agent variants in orq (idempotent)
uv run python agent_build/build_agent.py

# Red-team the vulnerable variant
uv run python agent_build/run_redteam.py --variant vulnerable

# Red-team the fixed variant
uv run python agent_build/run_redteam.py --variant fixed

# Side-by-side run
uv run python agent_build/run_redteam.py --variant both --max-per-category 20 --parallelism 10
```

Reports are written to `agent_build/reports/report_<variant>.json`.

To inspect a run interactively:

```bash
uv run evaluatorq redteam ui agent_build/.evaluatorq/runs/<run-file>.json
```

## Tests

```bash
uv run pytest
```

## Files

- `agent_build/build_agent.py` — idempotent setup: creates KB, tools, vulnerable + fixed agent variants
- `agent_build/run_redteam.py` — drives `red_team()` against one or both variants
- `agent_build/refund_target.py` — custom `AgentTarget` that handles function-tool callbacks in-process
- `agent_build/handlers.py` — pure-function tool implementations (`lookup_order`, `issue_refund`, `get_policy`)
- `agent_build/prompts.py` — system prompts for vulnerable + fixed variants
- `agent_build/demo_data.py` — fictional orders / refunds fixtures
- `agent_build/policy_kb/` — refund-policy markdown docs ingested into the orq KB
- `agent_build/tests/` — unit tests for handlers, demo data, refund target
- `presentation.html` / `presentation.pdf` — webinar deck
- `assets/`, `imgs/` — slide media (Reveal.js, fonts, screenshots, brand assets)

## Why a custom AgentTarget?

`evaluatorq`'s built-in orq backend stubs all pending tool calls with an error result, so it can't be used for agents with real function tools. `RefundAgentTarget` implements the `AgentTarget` protocol with a tool-call loop that dispatches to local Python handlers — this is the recommended pattern when the demoed agent needs deterministic, scriptable tool behaviour.
