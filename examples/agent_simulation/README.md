# Agent Simulation Examples

Python examples for simulating user interactions against AI agents - testing goal completion, criteria adherence, and iterative instruction improvement.

## Prerequisites

```bash
uv pip install evaluatorq
```

> **Examples 03 and 04** (`03_tool_simulation.py`, `04_hardening_loop.py`) are **preview**: they depend on the
> `agent-simulation` research package, which is not on PyPI and not yet part of the production SDK. Install from source:
>
> ```bash
> uv pip install "agent-simulation @ git+https://github.com/orq-ai/research.git#subdirectory=projects/agent-simulation"
> ```
>
> Examples 01, 02, and 05 only need `evaluatorq` and are the recommended starting points.

## Where to start

- **No agent yet?** Run `01_basic_simulation.py` — exercises the loop against a local mock callback.
- **Have an orq.ai deployment or A2A agent?** Run `02_orq_deployment_simulation.py --deployment <key>`.
- **Wiring simulations into a CI/CD pipeline?** See `05_wrap_and_experiment.py` for the production pattern.
- **Tool-using agents / instruction hardening?** See `03` and `04` (preview, see note above).

## Quick Start

```bash
export ORQ_API_KEY="orq-..."

# Simplest example - runs against a local mock agent, no deployment needed
uv run python examples/agent_simulation/01_basic_simulation.py

# Against a live orq.ai deployment
uv run python examples/agent_simulation/02_orq_deployment_simulation.py --deployment my-support-agent
```

### Expected output

A successful run prints per-simulation pass/fail lines and a final pass rate:

```
INFO | Pass rate: 8/12 (67%)
INFO |   [PASS] score=0.85 turns=4
INFO |          terminated_by=judge rules_broken=[]
INFO |   [FAIL] score=0.30 turns=8
INFO |          terminated_by=max_turns rules_broken=['no-pii-disclosure']
```

When `ORQ_API_KEY` is set, the run also prints an Experiment URL on `my.orq.ai`. If you see no URL, the upload was skipped (e.g. `upload_results=False` or missing key) — the run itself still succeeded.

## Examples

| File | What it shows |
|------|--------------|
| `01_basic_simulation.py` | Core simulation loop with a mock agent - no deployment needed |
| `02_orq_deployment_simulation.py` | Batch simulation against a live orq.ai deployment or A2A agent |
| `03_tool_simulation.py` (preview) | Testing tool-calling agents with `MockToolRegistry` — needs `agent-simulation` |
| `04_hardening_loop.py` (preview) | Iterative instruction improvement with `HardeningLoop` — needs `agent-simulation` |
| `05_wrap_and_experiment.py` | Production pattern: `wrap_simulation_agent()` + `evaluatorq()` for Experiment upload and CI gating |

## Environment

Copy `.env.example` and fill in your key:

```bash
cp .env.example .env
# add ORQ_API_KEY=orq-...
```
