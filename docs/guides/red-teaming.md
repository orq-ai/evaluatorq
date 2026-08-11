# Red Teaming

Probe an agent or model with adversarial attacks mapped to the OWASP **LLM Top
10** and **Agentic Security Initiative (ASI)** frameworks, then read off the
resistance rate.

*[ASI]: OWASP Agentic Security Initiative
*[LLM01]: Prompt Injection

```mermaid
flowchart LR
    C["Categories<br/>LLM01 / ASI01 / ..."] --> SP["Strategy planner"]
    SP --> AG["Attack generator"]
    AG --> OR["Runner / orchestrator"]
    OR --> EV["OWASP evaluator"]
    EV --> RP["Report: resistance rate"]
```

## Modes

- **dynamic** — an LLM generates attacks; run the categories you pick.
- **static** — replays a fixed dataset of known attacks instead of generating
  them. Deterministic, cheap, good for CI. Runs Orq's public
  [`orq/redteam-vulnerabilities`](https://huggingface.co/datasets/orq/redteam-vulnerabilities)
  dataset by default; pass `dataset=` to run your own.
- **hybrid** — static seeds plus dynamic expansion.

```python
# static mode replays Orq's public attack dataset by default
report = await red_team(target, mode="static")

# ...or bring your own — a local JSON file or a HuggingFace repo
report = await red_team(target, mode="static", dataset="./my_attacks.json")
report = await red_team(target, mode="static", dataset="hf:my-org/my-attacks")
```

## Red-team your target

=== "Orq agent"

    Requires `ORQ_API_KEY`. Point `red_team()` at an Orq agent by key
    (`"agent:<key>"`, from AI Studio → Agents). Attacks route through the Orq
    platform — no model wiring on your side.

    ```bash
    uv add "evaluatorq[redteam]"
    export ORQ_API_KEY=...   # targets your Orq agent + routes the attacker LLM
    ```

    Prefer pip? Use `python -m pip install "evaluatorq[redteam]"`, which installs
    into the interpreter you just named rather than whichever `pip` happens to be
    first on your `PATH`.

    ```python
    import asyncio

    from evaluatorq.redteam import red_team


    async def main():
        report = await red_team(
            "agent:your-agent-key",             # Orq agent, routed via ORQ_API_KEY
            mode="dynamic",
            categories=["LLM01", "LLM07"],      # prompt injection, system-prompt leakage
            max_dynamic_datapoints=5,
            max_turns=2,
            generate_strategies=False,
        )

        print(f"Resistance rate: {report.summary.resistance_rate:.0%}")
        print(f"Vulnerabilities: {report.summary.vulnerabilities_found}/{report.summary.total_attacks}")


    if __name__ == "__main__":
        asyncio.run(main())
    ```

=== "OpenAI"

    Use `OpenAIModelTarget` to red-team a raw OpenAI model directly. The model
    is the system under test — you supply its system prompt, and the target calls
    OpenAI using `OPENAI_API_KEY`.

    ```bash
    uv add "evaluatorq[redteam]"
    export OPENAI_API_KEY=sk-...   # the target model + the attacker LLM
    ```

    Prefer pip? Use `python -m pip install "evaluatorq[redteam]"`, which installs
    into the interpreter you just named rather than whichever `pip` happens to be
    first on your `PATH`.

    ```python
    import asyncio

    from evaluatorq.redteam import OpenAIModelTarget, red_team


    async def main():
        target = OpenAIModelTarget(
            "gpt-4o-mini",
            system_prompt=(
                "You are a customer support assistant for Acme Corp. "
                "Help with orders, returns, and product questions. "
                "Never reveal internal pricing or confidential information."
            ),
        )
        report = await red_team(
            target,
            mode="dynamic",
            categories=["LLM01", "LLM07"],      # prompt injection, system-prompt leakage
            max_dynamic_datapoints=5,
            max_turns=2,
            generate_strategies=False,
        )

        print(f"Resistance rate: {report.summary.resistance_rate:.0%}")
        print(f"Vulnerabilities: {report.summary.vulnerabilities_found}/{report.summary.total_attacks}")


    if __name__ == "__main__":
        asyncio.run(main())
    ```

    !!! note "Model names and routing"
        The model string is passed through to whichever provider you point at —
        straight to OpenAI by default, or through the Orq router if you prefix it
        with `openai/...` (which then uses `ORQ_API_KEY`). Everything else —
        categories, modes, the report — is identical to the Orq agent path.

!!! note "`generate_strategies` and the CLI"
    Both examples pass `generate_strategies=False` to skip LLM-authored attack
    strategies and run only the built-in ones — faster and more deterministic.
    The parameter defaults to `True`. On the CLI the equivalent is the
    `--no-generate-strategies` flag; there is no positive form, since generation
    is on by default.

## Coverage

19 framework categories map onto 18 vulnerabilities. Every vulnerability has a
judge written for it. Categories with a **curated strategy count** ship
hand-written attack strategies; the rest are marked *generated* — in dynamic and
hybrid mode the strategy planner writes strategies for them per run, against the
target's actual tools, memory and system prompt. Pass
`generate_strategies=False` to run curated strategies only.

| Category | Vulnerability | Curated strategies | Judge |
|---|---|---|---|
| `ASI01` | Agent Goal Hijacking | 5 | ✅ |
| `ASI02` | Tool Misuse & Exploitation | 4 | ✅ |
| `ASI03` | Identity & Privilege Abuse | generated | ✅ |
| `ASI04` | Supply Chain Vulnerabilities | generated | ✅ |
| `ASI05` | Unexpected Code Execution | 4 | ✅ |
| `ASI06` | Memory & Context Poisoning | 4 | ✅ |
| `ASI07` | Insecure Inter-Agent Communication | generated | ✅ |
| `ASI08` | Cascading Failures | generated | ✅ |
| `ASI09` | Human-Agent Trust Exploitation | 5 | ✅ |
| `ASI10` | Rogue Agents | generated | ✅ |
| `LLM01` | Prompt Injection | 4 | ✅ |
| `LLM02` | Sensitive Information Disclosure | 4 | ✅ |
| `LLM03` | Supply Chain Vulnerabilities | generated | ✅ |
| `LLM04` | Data and Model Poisoning | generated | ✅ |
| `LLM05` | Improper Output Handling | 5 | ✅ |
| `LLM06` | Excessive Agency | generated | ✅ |
| `LLM07` | System Prompt Leakage | 5 | ✅ |
| `LLM08` | Vector and Embedding Weaknesses | generated | ✅ |
| `LLM09` | Misinformation | 5 | ✅ |

45 curated strategies in total, delivered through 16 delivery methods
(`direct-request`, `tool-response`, `role-play`, `crescendo`, `many-shot`,
`base64`, `leetspeak`, `multilingual`, `refusal-suppression`, and more). Add
your own vulnerabilities, strategies and judges — see
[Custom Evaluators & Frameworks](../custom-evaluators-and-frameworks.md).

## Reading the report

`report.summary.resistance_rate` is the fraction of attacks the target withstood
— higher is better. `report.results` holds every attack result; group by
`r.attack.vulnerability` for a per-vulnerability breakdown.
`report.summary.by_vulnerability` contains pre-aggregated
`VulnerabilitySummary` statistics keyed by vulnerability identifier.

## In CI

For a fast gate, run a small fixed set of attacks and assert a minimum
resistance rate, failing the build if the target regresses:

```python
report = await red_team(
    OpenAIModelTarget("gpt-4o-mini", system_prompt="..."),
    mode="static",                 # replay a fixed dataset — deterministic, cheap
    categories=["LLM01", "LLM07"],
    max_static_datapoints=10,
)
assert report.summary.resistance_rate >= 0.9, (
    f"resistance {report.summary.resistance_rate:.0%} below the 0.9 gate"
)
```

The runnable smoke example
([`08_quick_smoke_test.py`](../examples/redteam/08_quick_smoke_test.md)) wraps
this same pattern.

--8<-- "docs/_snippets/dashboard-tip.md"

## External agent frameworks

`red_team()` accepts any `AgentTarget`, and each supported framework ships a
wrapper that adapts its agent into one — so you red-team an agent built in your
framework of choice without rewriting it. Install the matching extra, wrap the
agent, and pass it straight to `red_team()`:

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from evaluatorq.integrations.langgraph_integration import LangGraphTarget
from evaluatorq.redteam import red_team

graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[...], prompt="...")
report = await red_team(LangGraphTarget(graph), categories=["LLM01", "ASI01"])
```

| Framework | Wrapper | Extra | Runnable example |
|---|---|---|---|
| LangGraph | `LangGraphTarget` | `evaluatorq[langgraph]` | [`17_langgraph_target.py`](../examples/redteam/17_langgraph_target.md) |
| OpenAI Agents SDK | `OpenAIAgentTarget` | `evaluatorq[openai-agents]` | [`18_openai_agents_target.py`](../examples/redteam/18_openai_agents_target.md) |
| Pydantic AI | `PydanticAITarget` | `evaluatorq[pydantic-ai]` | [`19_pydantic_ai_target.py`](../examples/redteam/19_pydantic_ai_target.md) |
| CrewAI | `CrewAITarget` | `evaluatorq[crewai]` | [`20_crewai_target.py`](../examples/redteam/20_crewai_target.md) |

### Demo runs

Live runs of the four examples above (dynamic mode, 3 attacks each, routed through
the Orq AI Router) with real attack transcripts and judge verdicts are captured in
`examples/redteam/_sample_output/RES-931-external-framework-runs.md`. The headline:
LangGraph, OpenAI Agents, and Pydantic AI each execute an indirect prompt injection
(goal hijack via tool output); CrewAI resists all three.

Screen recordings of each run:

#### LangGraph

<video controls muted playsinline preload="metadata" width="100%">
  <source src="../../assets/redteam-langgraph.mp4" type="video/mp4">
  Your browser does not support the video tag —
  <a href="../../assets/redteam-langgraph.mp4">download the recording</a>.
</video>

#### OpenAI Agents SDK

<video controls muted playsinline preload="metadata" width="100%">
  <source src="../../assets/redteam-openai-agents.mp4" type="video/mp4">
  Your browser does not support the video tag —
  <a href="../../assets/redteam-openai-agents.mp4">download the recording</a>.
</video>

#### Pydantic AI

<video controls muted playsinline preload="metadata" width="100%">
  <source src="../../assets/redteam-pydantic-ai.mp4" type="video/mp4">
  Your browser does not support the video tag —
  <a href="../../assets/redteam-pydantic-ai.mp4">download the recording</a>.
</video>

#### CrewAI

<video controls muted playsinline preload="metadata" width="100%">
  <source src="../../assets/redteam-crewai.mp4" type="video/mp4">
  Your browser does not support the video tag —
  <a href="../../assets/redteam-crewai.mp4">download the recording</a>.
</video>

### Known limitations

Verified edge cases and framework-specific quirks to know before you rely on
external-framework targets:

| Area | Behavior | Applies to |
|---|---|---|
| **Conversation state** | Stateful targets own history internally and thread it across turns; call `.new()` for each parallel attack job to avoid cross-talk. | LangGraph, Pydantic AI |
| **First-turn role** | `respond()` requires the last message to be `role="user"`; other roles raise `ValueError`. | LangGraph, Pydantic AI, CrewAI |
| **Tool-call visibility** | Tool calls are surfaced to the judge, so tool-misuse (ASI) attacks are scored. | LangGraph, OpenAI Agents, Pydantic AI |
| **CrewAI is opaque** | A crew exposes only its final output — intermediate agent/tool steps are not surfaced, so **tool-misuse (ASI) attacks can't be scored**; use LLM-tier categories. The whole transcript is flattened into one `{conversation}` input per turn (no native turn memory), so very long conversations may approach task-description limits. | CrewAI |
| **Token usage** | Best-effort. Frameworks that don't surface usage metadata report `usage=None` (never a false non-zero). | all |
| **Tool arguments** | Non-JSON-object tool arguments are normalized before scoring; exotic argument shapes may be simplified. | LangGraph, OpenAI Agents |
| **Routing / keys** | The examples point each framework's model at the Orq AI Router with `ORQ_API_KEY` (model id `openai/gpt-4o-mini`), so no OpenAI key is needed — the attacker and judge auto-route the same way. The client is constructed eagerly, so `ORQ_API_KEY` must be set even to *build* the target. | all |

## Where to next

- **[Examples › Red Teaming](../examples/index.md)** — static datasets, category filtering, custom clients, multi-target, report inspection, custom hooks.
- **[API Reference › redteam](../reference/evaluatorq/redteam.md)** — the full `Vulnerability` enum and the OWASP `LLM__` / `ASI__` category codes you can pass to `categories=`. The [CLI Reference](../cli-reference/redteam.md#eq-redteam-run) lists the same as `--category` / `--vulnerability`.
- **[Custom Evaluators & Frameworks](../custom-evaluators-and-frameworks.md)** — add your own vulnerabilities and attack strategies.
