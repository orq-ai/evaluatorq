# Red Teaming

Adaptive red teaming for AI agents and LLMs — automatically discover security vulnerabilities using multi-turn adversarial attacks mapped to industry security frameworks.

## What it does

This subpackage probes AI agents and LLMs for security weaknesses using a multi-stage pipeline:

1. **Context retrieval** — Fetches the agent's configuration (tools, memory stores, knowledge bases, system prompt) from the ORQ platform, or accepts a system prompt for direct LLM targets.
2. **Capability classification** — An LLM semantically tags each tool (e.g. `file_system`, `code_execution`, `email`, `payment`) so attacks target real capabilities.
3. **Strategy planning** — Selects from 35+ hardcoded attack strategies filtered by agent capabilities, plus generates novel strategies tailored to the specific target via LLM.
4. **Multi-turn orchestration** — An adversarial LLM runs an adaptive attack loop: it observes the target's responses each turn and adjusts its approach.
5. **Evaluation** — Each attack result is scored against the relevant vulnerability evaluator to determine resistance or vulnerability.

> Internals (pipeline architecture, design decisions, the OpenResponses backend) live in [ARCHITECTURE.md](ARCHITECTURE.md).

## Vulnerability-first design

The atomic primitive is the **vulnerability**, not the framework category. Each vulnerability has a stable, framework-agnostic ID (e.g. `prompt_injection`, `goal_hijacking`) that maps to one or more framework categories. You can target attacks at either level:

```python
from evaluatorq.redteam import OpenAIModelTarget, red_team

target = OpenAIModelTarget("openai/gpt-5.4-mini", system_prompt="You are helpful.")

# Target specific vulnerabilities
report = await red_team(target, vulnerabilities=["prompt_injection", "goal_hijacking"])

# Or filter by framework category
report = await red_team(target, categories=["LLM01", "ASI01"])
```

> Model IDs route through the ORQ router by default, so use the provider-prefixed form (`openai/gpt-5.4-mini`). If you target OpenAI directly (only `OPENAI_API_KEY` set), drop the prefix: `gpt-5.4-mini`.

### Supported vulnerabilities

18 vulnerability types across three domains:

| Domain | Vulnerability | Description |
|--------|--------------|-------------|
| **Agent** | `goal_hijacking` | Redirect the agent away from its intended objective |
| **Agent** | `tool_misuse` | Trick the agent into misusing its tools |
| **Agent** | `identity_privilege_abuse` | Exploit identity or permissions |
| **Agent** | `supply_chain` | Compromise via third-party dependencies |
| **Agent** | `code_execution` | Trigger unauthorized code execution |
| **Agent** | `memory_poisoning` | Inject malicious data into agent memory |
| **Agent** | `inter_agent_comms` | Exploit insecure agent-to-agent communication |
| **Agent** | `cascading_failures` | Trigger failure cascades across agents |
| **Agent** | `trust_exploitation` | Exploit human-agent trust boundaries |
| **Agent** | `rogue_agents` | Cause agents to act outside their boundaries |
| **Agent** | `excessive_agency` | Exploit overly broad agent permissions |
| **Model** | `prompt_injection` | Override instructions via injected prompts |
| **Model** | `sensitive_info_disclosure` | Extract confidential information |
| **Model** | `improper_output` | Produce harmful or unvalidated output |
| **Model** | `system_prompt_leakage` | Leak the system prompt contents |
| **Model** | `misinformation` | Generate false or misleading information |
| **Data** | `data_poisoning` | Corrupt training data or model weights |
| **Data** | `vector_embedding_weakness` | Exploit vector/embedding retrieval |

### Supported frameworks

Vulnerabilities map to industry security frameworks via `framework_mappings`:

| Framework | Categories | Description |
|-----------|-----------|-------------|
| **OWASP ASI** | ASI01–ASI10 | Agentic Security Initiative — agent-layer risks |
| **OWASP LLM Top 10** | LLM01–LLM09 | LLM-layer risks (prompt injection, data leakage, etc.) |

Some vulnerabilities map to multiple frameworks (e.g. `supply_chain` → ASI04 + LLM03).

## Modes

| Mode | Description |
|------|-------------|
| `dynamic` | Generates adversarial attacks using LLM-based strategy planning and multi-turn orchestration |
| `static` | Runs a pre-built OWASP dataset for reproducible regression testing |
| `hybrid` | Combines dynamic generation with a static dataset in a single run |

## `red_team()` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | `str \| AgentTarget \| list` | **required** | Target(s): `"agent:<key>"`, `"deployment:<key>"`, or an `AgentTarget` such as `OpenAIModelTarget(...)` |
| `mode` | `str` | `"dynamic"` | `"dynamic"`, `"static"`, or `"hybrid"` |
| `categories` | `list[str] \| None` | all | OWASP categories (e.g. `["ASI01", "LLM07"]`) |
| `vulnerabilities` | `list[str] \| None` | `None` | Specific vulnerability IDs to test |
| `max_turns` | `int` | `5` | Max conversation turns per attack |
| `max_dynamic_datapoints` | `int \| None` | `None` | Cap generated attack datapoints |
| `llm_config` | `LLMConfig \| None` | `None` | Per-role attacker/evaluator model config |
| `parallelism` | `int` | `10` | Max concurrent jobs |
| `name` | `str \| None` | `None` | Experiment name (defaults to `"red-team"`) |
| `llm_client` | `AsyncOpenAI \| None` | `None` | Custom LLM client |
| `dataset` | `Path \| str \| None` | `None` | Path to local static dataset |

### LLM client configuration

Red teaming needs an OpenAI-compatible LLM for attack generation and evaluation. `ORQ_*` variables take priority over `OPENAI_*`:

| Priority | Variables | Description |
|----------|-----------|-------------|
| 1st | `ORQ_API_KEY` + `ORQ_BASE_URL` (optional) | ORQ router |
| 2nd | `OPENAI_API_KEY` + `OPENAI_BASE_URL` (optional) | Direct OpenAI or any compatible endpoint |

Or pass a custom client: `red_team(..., llm_client=AsyncOpenAI(api_key="sk-..."))`.

## External agent frameworks

Agents built with external frameworks are wrapped into a target the pipeline can attack. Install the matching extra (`pip install evaluatorq[langgraph]`, `evaluatorq[openai-agents]`, or `evaluatorq[all]`).

### LangGraph

Wrap any compiled LangGraph state graph. The graph must use `MessagesState` (or a state with a `messages` key).

```python
from langgraph.prebuilt import create_react_agent
from evaluatorq.integrations.langgraph_integration import LangGraphTarget
from evaluatorq.redteam import red_team

graph = create_react_agent(model, tools=[...])
target = LangGraphTarget(graph)  # or LangGraphTarget(graph, config={"recursion_limit": 50})
report = await red_team(target=target)
```

Each attack gets a fresh thread; `clone()` creates independent copies for parallel attacks.

### LangChain agents

- Agents built with `create_react_agent` / `StateGraph` run on LangGraph → use `LangGraphTarget` directly.
- Custom chains or legacy `AgentExecutor` → wrap with `CallableTarget` (see below).

### OpenAI Agents SDK

```python
from agents import Agent
from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget
from evaluatorq.redteam import red_team

agent = Agent(name="my-agent", instructions="You are a helpful assistant.")
target = OpenAIAgentTarget(agent)  # or OpenAIAgentTarget(agent, run_kwargs={"max_turns": 10})
report = await red_team(target=target)
```

### Custom callable (escape hatch)

Wrap any function that takes the conversation and returns a response. The callable receives the full transcript as typed `Message` objects, so it works for single- and multi-turn attacks even when stateless:

```python
from evaluatorq.contracts import Message
from evaluatorq.integrations.callable_integration import CallableTarget
from evaluatorq.redteam import red_team

async def my_agent(messages: list[Message]) -> str:
    result = await some_framework.run(messages)
    return result.text

target = CallableTarget(my_agent)
report = await red_team(target=target)
```

Sync functions are run in a thread automatically. If the callable holds state, pass `reset_fn` to clear it between attacks.

## CLI

`eq redteam` exposes the same capability. It accepts `agent:` / `deployment:` targets only — for an OpenAI model use `OpenAIModelTarget` in the Python API.

```bash
uv tool install "evaluatorq[redteam]"
eq redteam run -t "agent:my-agent-key" -c LLM01 -c LLM07 --max-turns 2 -y
eq redteam run --help   # all options
```

For the full flag reference (multi-target, report export, `eq redteam runs`, etc.), see [`examples/redteam/README.md`](../../../examples/redteam/README.md#cli-reference).

## Tracing & PII

Red teaming emits OpenTelemetry spans for every LLM call (attacker generation, target response, evaluation). By default these spans include the **message content** — the adversarial prompts sent and the target's responses — so the Orq dashboard can render input/output panels.

Because attack transcripts can contain sensitive target data, content capture is gated by an env var:

- `EVALUATORQ_CAPTURE_MESSAGE_CONTENT` — **default `true`**. Set to `false` (or `0`) to suppress message text on spans (both inputs and outputs) while still recording token usage, model, finish reason, and latency. Use this when exporting traces to a third-party backend or when prompts/responses may contain PII.
- `EVALUATORQ_SPAN_MAX_TEXT_CHARS` — max characters of message text (inputs and outputs) per span attribute before truncation (`... [truncated]` marker). **Defaults to capturing all content (no truncation).** Set a positive integer (e.g. `8192`) to cap; `-1` / `0` / unset all mean capture all.

These are shared with agent simulation — the same tracing layer (`evaluatorq.common.tracing`) backs both.

## Convention

Throughout this package: `passed=True` / `vulnerable=False` means the agent **resisted** the attack (good). `passed=False` / `vulnerable=True` means the agent was **vulnerable** (bad).
