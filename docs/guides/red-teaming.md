# Red Teaming

Probe an agent or model with adversarial attacks mapped to the OWASP **LLM Top
10** and **Agentic Top 10 (ASI)** frameworks, then read off the
resistance rate.

*[ASI]: OWASP Agentic Top 10
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

        rate = report.summary.resistance_rate    # None when nothing could be evaluated
        print(f"Resistance rate: {rate:.0%}" if rate is not None else "Resistance rate: no verdict")
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

        rate = report.summary.resistance_rate    # None when nothing could be evaluated
        print(f"Resistance rate: {rate:.0%}" if rate is not None else "Resistance rate: no verdict")
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

`report.summary.resistance_rate` is the fraction of *evaluated* attacks the
target withstood — higher is better. It is `None` when no attack could be
evaluated at all (every judge call failed, e.g. a gateway guardrail rejecting
them): there is no verdict to report, and a `0.0` there would read as "fully
compromised" when in fact nothing was tested. Check
`report.summary.evaluated_attacks` against `total_attacks` before trusting a
rate. Individual results follow the same rule: `r.vulnerable` is `None`, not
`False`, when that attack could not be evaluated.

`report.results` holds every attack result; group by
`r.attack.vulnerability` for a per-vulnerability breakdown.
`report.summary.by_vulnerability` contains pre-aggregated
`VulnerabilitySummary` statistics keyed by vulnerability identifier.

When a judge fails to return a verdict, the reason is captured on
`result.evaluation_error` (a `RunError` with a `code` like `timeout`, `parse`,
`api_connection`, `api_status`, or `unknown`). It is deliberately separate from
`result.error`: `error` means the attack itself never ran, `evaluation_error`
means the attack ran and the transcript exists but no judge could score it. Both
roll up into `report.summary.errors_by_type`, where judge failures appear under
`evaluation/<code>` keys (execution failures use the bare code) — so a
systematically blocked judge shows up as one named cause (`evaluation/api_status:
40 attacks`) instead of vanishing into forty individual results.

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
rate = report.summary.resistance_rate
assert rate is not None, "no attack could be evaluated — the target was not tested"
assert rate >= 0.9, f"resistance {rate:.0%} below the 0.9 gate"
```

The `is not None` check is the part people forget: without it, a run where every
judge call was rejected produces no rate at all and the gate would crash (or, in
older versions, silently pass a `0.0`). `eq redteam run` applies the same rule —
it exits `1` when attacks ran but not one of them could be scored
(`report.summary.no_verdict`: `total_attacks > 0` and `evaluated_attacks == 0`),
so a CLI-driven gate fails loudly too. A run with zero attacks (an empty
category filter, say) is not this condition and does not trigger it.

!!! warning "Coverage gate — a run under 80% evaluated now fails, not just warns"
    `EvaluatorConfig.min_evaluation_coverage` (default **`0.8`**) is a run-level
    floor on top of `no_verdict`: even when *some* attacks got a verdict,
    `eq redteam run` exits `1` if fewer than 80% of attacks did
    (`report.summary.coverage_below_minimum`). **This is a behaviour change** —
    a run that finished at, say, 79% evaluation coverage used to exit `0` with a
    warning; it now exits `1`. If you wire `eq redteam run` into CI, a flaky
    judge/gateway that drops just over a fifth of verdicts will now fail the
    build. Pass `--min-evaluation-coverage 0` (or set
    `min_evaluation_coverage=None` on `EvaluatorConfig`) to restore the old
    warn-only behaviour. Zero coverage (`no_verdict`) always fails regardless
    of this setting — it isn't a case the floor can raise or lower.

    This is distinct from `EvaluatorConfig.min_successful_judges` (default `1`),
    which is a **per-attack** quorum: it decides whether one attack's jury panel
    produced enough decisive votes to reach *that attack's* verdict, and a
    quorum miss is exactly what produces an unevaluated attack.
    `min_evaluation_coverage` is the **run-level** floor on how many such
    unevaluated attacks the whole run can tolerate before its rates are
    considered untrustworthy. Tightening `min_successful_judges` makes more
    individual attacks fall through as unevaluated; tightening
    `min_evaluation_coverage` makes the run less tolerant of however many do.

    Set it from either surface: `eq redteam run --min-evaluation-coverage 0.5`
    on the CLI (pass `0` for warn-only), or
    `red_team(llm_config=LLMConfig(evaluator=EvaluatorConfig(min_evaluation_coverage=...)))`
    in Python (`None` there is warn-only).
    The Python API itself does not raise on this condition — `red_team()` only
    appends a `pipeline_warnings` entry and logs; only the `eq redteam run` CLI
    command turns it into a nonzero exit. A caller using the Python API for its
    own CI gate should check `report.summary.coverage_below_minimum` explicitly,
    the same way it already checks `resistance_rate is not None` above.

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
