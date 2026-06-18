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
| `06_langgraph_simulation.py` | Simulating a LangGraph `StateGraph` via `LangGraphTarget` — needs `evaluatorq[langgraph]` |
| `07_openai_agents_simulation.py` | Simulating an OpenAI Agents SDK `Agent` via `OpenAIAgentTarget` — needs `evaluatorq[openai-agents]` |
| `08_pydantic_ai_simulation.py` | Simulating a Pydantic AI `Agent` via `PydanticAITarget` — needs `evaluatorq[pydantic-ai]` |
| `09_crewai_simulation.py` | Simulating a CrewAI `Crew` via `CrewAITarget` — needs `evaluatorq[crewai]` |

## External agent frameworks

Agent Simulation is framework-agnostic: any agent that implements the unified
`AgentTarget` protocol (`async respond(messages) -> AgentResponse`) runs through
the same three-part loop (user simulator -> agent under test -> judge). Pass the
target via `target=` to `simulate()` / `generate_and_simulate()`.

Examples 06-09 show the four supported frameworks. Each adapter handles that
framework's quirks:

| Framework | Adapter | Quirks handled |
|-----------|---------|----------------|
| LangGraph | `LangGraphTarget` | Owns thread state (forwards only the latest user turn); tool calls preserved; tokens via callback |
| OpenAI Agents SDK | `OpenAIAgentTarget` | Stateless per run (renders full transcript to Responses-API items); tool calls + results round-trip |
| Pydantic AI | `PydanticAITarget` | Threads typed `message_history` internally; tool calls extracted from message parts; usage has no total (derived) |
| CrewAI | `CrewAITarget` | Sync `kickoff` run off-thread; transcript flattened to one `{conversation}` input; final crew output is "the response" |

> **Python 3.10 note:** the `crewai` extra is gated to Python >= 3.11 (its
> `onnxruntime` dependency ships no 3.10 wheels), so `evaluatorq[crewai]` is
> silently unavailable on 3.10. The other three frameworks support 3.10+.

```python
from evaluatorq.integrations.langgraph_integration import LangGraphTarget
from evaluatorq.simulation import simulate

results = await simulate(target=LangGraphTarget(graph), personas=[...], scenarios=[...])
```

Custom framework not listed? Wrap any `fn(messages) -> str` with
`CallableTarget` (`evaluatorq.integrations.callable_integration`), or pass a
plain `target_callback`.

## Environment

Copy `.env.example` and fill in your key:

```bash
cp .env.example .env
# add ORQ_API_KEY=orq-...
```
