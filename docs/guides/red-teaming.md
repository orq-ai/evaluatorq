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
  dataset by default; pass `dataset=` to run your own. The
  [static dataset cookbook](../examples/redteam/02_static_dataset.md) shows the
  reproducible version end to end.
- **hybrid** — static seeds plus dynamic expansion; see the
  [hybrid mode cookbook](../examples/redteam/03_hybrid_mode.md) when you want
  both known attacks and generated coverage.

```python
# static mode replays Orq's public attack dataset by default
report = await red_team(target=target, mode="static")

# ...or bring your own — a local JSON file or a HuggingFace repo
report = await red_team(target=target, mode="static", dataset="./my_attacks.json")
report = await red_team(target=target, mode="static", dataset="hf:my-org/my-attacks")
```

## Red-team your target

!!! warning "Use a sandbox or test agent"
    Red-team attacks run the target's real tools. Do not point them at production
    credentials or an agent that can send messages, move money, modify data, or
    run commands unless those side effects are isolated and intentional.

### Fastest first run

If you prefer the CLI, start with a small static run against a test agent. Static
mode uses the built-in attack dataset, so it is a predictable way to verify your
setup before exploring dynamic or hybrid runs.

```bash
export ORQ_API_KEY=...
uv add "evaluatorq[redteam]"
eq redteam run \
  --target agent:your-agent-key \
  --mode static \
  --category LLM01 \
  --max-static-datapoints 5 \
  --no-executive-summary \
  --no-recommendations \
  --report redteam-report.json \
  --yes
```

The command writes a JSON report and exits non-zero if no attacks receive a
verdict or evaluation coverage falls below the configured floor. See the
[CLI reference](../cli-reference/redteam.md) for the other output formats and
run options.

### Choose a cookbook

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Start with a smoke test__

    ---

    Run a small, CI-friendly check with an explicit exit-code gate.

    [:octicons-arrow-right-24: `08_quick_smoke_test.py`](../examples/redteam/08_quick_smoke_test.md)

-   :material-database-check:{ .lg .middle } __Make it reproducible__

    ---

    Replay a fixed dataset when you need stable attacks across agent versions.

    [:octicons-arrow-right-24: `02_static_dataset.py`](../examples/redteam/02_static_dataset.md)

-   :material-tune-variant:{ .lg .middle } __Aim attacks at your domain__

    ---

    Add domain context so generated attacks reflect your agent and threat model.

    [:octicons-arrow-right-24: `13_attacker_instructions.py`](../examples/redteam/13_attacker_instructions.md)

-   :material-file-chart-outline:{ .lg .middle } __Inspect what happened__

    ---

    Filter results, inspect verdicts, and export evidence from Python.

    [:octicons-arrow-right-24: `07_report_inspection.py`](../examples/redteam/07_report_inspection.md)

</div>

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
            target="agent:your-agent-key",      # Orq agent, routed via ORQ_API_KEY
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
            model="gpt-4o-mini",
            system_prompt=(
                "You are a customer support assistant for Acme Corp. "
                "Help with orders, returns, and product questions. "
                "Never reveal internal pricing or confidential information."
            ),
        )
        report = await red_team(
            target=target,
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

## Inspect results in Python

For the same numbers rendered as a browsable report, see
[Reading a run in the dashboard](#reading-a-run-in-the-dashboard) below.

The report fields most users need are:

- `report.summary.resistance_rate`: the fraction of evaluated attacks resisted;
  higher is better. `None` means no attack received a verdict.
- `report.summary.evaluated_attacks` and `total_attacks`: check both before
  trusting a rate. `evaluation_coverage` and `coverage_below_minimum` expose the
  same check for CI.
- `report.results`: the per-attack evidence. `result.vulnerable is None` means
  the attack was not evaluated, not that it was resisted.
- `report.summary.by_vulnerability`: the pre-aggregated vulnerability breakdown.

For failures, `result.error` means the attack did not run; `result.evaluation_error`
means it ran but the judge could not return a verdict. Both roll up into
`report.summary.errors_by_type`.

### What a run costs

`report.summary.token_usage_total` covers usage recorded for attack generation,
the target, and the judge. It includes `calls` and `priced_calls` alongside token
counts and the dollar figure. If `priced_calls < calls`, some calls reported
usage without a provider price, so the displayed cost is a lower bound. Use the
calculator below to include the fixed setup calls in a planning estimate.

#### Ballpark the cost

Use this calculator for a quick planning estimate. It prices real tokens at
published per-1k list prices, not a flat per-call guess. Pick a model or enter
your own prices.

<form class="cost-calculator" data-cost-calculator>
  <div class="cost-calculator__grid">
    <label>
      Fixed calls
      <input type="number" name="fixed-calls" min="0" step="1" value="0">
    </label>
    <label>
      Number of attacks
      <input type="number" name="attacks" min="0" step="1" value="10">
    </label>
    <label>
      Turns per attack
      <input type="number" name="turns" min="0" step="1" value="1">
    </label>
    <label>
      LLM calls per turn
      <input type="number" name="calls-per-turn" min="0" step="1" value="2">
    </label>
    <label>
      Judge calls per attack
      <input type="number" name="judge-calls" min="0" step="1" value="1">
    </label>
    <label>
      Model
      <select name="model">
        <option value="claude-opus-5">Claude Opus 5 — $5 / $25 per 1M</option>
        <option value="claude-sonnet-5" selected>Claude Sonnet 5 — $3 / $15 per 1M</option>
        <option value="claude-haiku-4-5">Claude Haiku 4.5 — $1 / $5 per 1M</option>
        <option value="gpt-5-mini">gpt-5-mini — $0.25 / $2 per 1M</option>
        <option value="custom">Custom</option>
      </select>
    </label>
    <label>
      Input price (USD per 1k tokens)
      <input type="number" name="input-price" min="0" step="any" value="0.003">
    </label>
    <label>
      Output price (USD per 1k tokens)
      <input type="number" name="output-price" min="0" step="any" value="0.015">
    </label>
    <label>
      Tokens per turn, per side
      <input type="number" name="tokens-per-side" min="0" step="100" value="1000">
    </label>
    <label>
      Cache hit rate (%)
      <input type="number" name="cache-hit" min="0" max="100" step="1" value="90">
    </label>
  </div>
  <div class="cost-calculator__result" aria-live="polite">
    <strong>Estimated cost: <span data-cost-total>$0.4842</span></strong>
    <span data-cost-breakdown>30 calls · 60,000 input tokens (90% cached) · 30,000 output tokens</span>
  </div>
</form>

Call count is **fixed calls + attacks × (turns × calls-per-turn + judge calls)**.

- **Calls per turn** is 2 for dynamic and hybrid attacks — one adversarial
  generation and one target call. Static attacks replay a fixed prompt, so set
  it to 1 (target only).
- **Judge calls per attack** is 1 by default. The judge runs once after the turn
  loop, not per turn. A jury multiplies this by panel size × repetitions.

The token model behind the dollar figure makes three assumptions, all editable
above:

- **1,000 tokens per turn, per side.** Every call emits one block and the
  transcript grows by one block per call, so input cost is quadratic in turns —
  which is why long multi-turn attacks cost more than the call count suggests.
- **90% cache hit rate** on input tokens, billed at **0.1× the base input
  price**. That is the standard cached-read rate for the models listed. Runs
  that never repeat a prefix should set this to 0.
- **Published list prices, snapshotted 2026-08-18.** Verify against your
  provider before quoting a number to anyone; at runtime evaluatorq prices calls
  from the live Orq `/v2/models` catalogue instead.

Content-filter retries on the attacker turn are real billed calls and are not in
this estimate. Actual spend varies with prompt and completion length.

If you want separate attacker and evaluator model settings rather than one
default model, see the [`11_redteam_config.py` cookbook](../examples/redteam/11_redteam_config.md).

There is no single fixed-call count for every run. Use these baselines when
setting the first field:

- **Static and replay runs: 0.** A replay selects nothing, so capability
  classification is skipped entirely.
- **Dynamic and hybrid runs:** 1 resource-inference call per target, plus 1
  tool-classification call per target when the target exposes tools. Both are
  skipped when no LLM client or credentials can be resolved.
- **Strategy generation** (when enabled): roughly 1 planning call per selected
  vulnerability or unresolved category, batched — more than eight objectives
  split across several calls rather than one.
- **Executive summary** (on by default): 1 call, best-effort — skipped silently
  without credentials.
- **Recommendations:** 1 call per *selected top* focus area, not per failed area,
  plus an occasional trace-condensing call for oversized attacks.

The CLI quickstart above disables strategy generation, recommendations, and the
executive summary, so its baseline is 0.

#### What the totals do not include

`token_usage_total` records the calls the run makes on the attack path. Setup and
post-processing calls are real spend that lands outside it:

- Capability classification (resource inference, tool classification) and
  blackbox target classification.
- Strategy and objective generation.
- Structured-output retries and the `json_object` fallback re-request.
- Recommendations, trace condensing, and the executive summary.
- Target-side usage, when the backend does not report it back.

If `priced_calls < calls` in the summary, some counted calls had no provider
price either, so the dollar figure is a lower bound on a subset.

## In CI

For a fast gate, run a small fixed set of attacks and assert a minimum
resistance rate, failing the build if the target regresses:

```python
report = await red_team(
    target=OpenAIModelTarget(model="gpt-4o-mini", system_prompt="..."),
    mode="static",                 # replay a fixed dataset — deterministic, cheap
    categories=["LLM01", "LLM07"],
    max_static_datapoints=10,
)
rate = report.summary.resistance_rate
assert rate is not None, "no attack could be evaluated — the target was not tested"
assert rate >= 0.9, f"resistance {rate:.0%} below the 0.9 gate"
```

The `is not None` check matters: a run where every judge call failed has no
honest rate. `eq redteam run` exits `1` when attacks ran but none could be
scored (`report.summary.no_verdict`).

!!! warning "Gate on coverage as well as resistance"
    `EvaluatorConfig.min_evaluation_coverage` defaults to **0.8**. The CLI exits
    `1` when `report.summary.coverage_below_minimum` is true. The Python API
    returns the report, so a Python CI gate should check both
    `summary.no_verdict` / `summary.coverage_below_minimum` and
    `summary.resistance_rate` before accepting the run.

The runnable smoke example
([`08_quick_smoke_test.py`](../examples/redteam/08_quick_smoke_test.md)) wraps
this same pattern; the [report inspection cookbook](../examples/redteam/07_report_inspection.md)
shows how to consume the resulting JSON in Python.

## Reading a run in the dashboard

Runs are saved to `.evaluatorq/runs/<name>_<timestamp>.json` by default
(`--save none` skips the file; `--save detail` also keeps per-stage artifacts
in `--artifacts-dir`), and `eq dashboard` browses those files locally — no
external service:

```bash
uv add "evaluatorq[dashboard]"

eq dashboard                                              # browse every saved run
eq dashboard .evaluatorq/runs/red-team_<timestamp>.json   # deep-link to one report
```

Land on the cross-surface overview, pick the run from **Red Team**, then work
down the report tabs from headline to evidence:

| Tab | What it answers |
|---|---|
| **Overview** | Executive summary, resistance rate and ASR, severity split |
| **Agents** | Per-target ASR and discovered tools, skills and knowledge |
| **Focus areas** | Fixes ranked by `success rate × avg severity`, with remediations |
| **Breakdowns** | Attack success per framework category, worst first |
| **Attacks** | Every attack, expandable to the judge verdict and transcript |
| **Usage** | Tokens and API calls per agent |
| **Config** | What was tested — and what was *not* |

Two traps worth knowing before you read a number: the resistance rate covers
only the categories this run attacked (check **Config** for the skipped ones),
and it is measured over *evaluated* attacks, so failed judge calls drop out of
the denominator rather than counting as resisted.

The tab-by-tab walkthrough, with screenshots, lives in the Dashboard reference:
[Reading a red-team run](../dashboard.md#reading-a-red-team-run). See also
[filters](../dashboard.md#filters), [trace links](../dashboard.md#orq-trace-links)
and [downloads](../dashboard.md#downloads).

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

graph = create_react_agent(model=ChatOpenAI(model="gpt-4o-mini"), tools=[...], prompt="...")
report = await red_team(
    target=LangGraphTarget(graph=graph),
    categories=["LLM01", "ASI01"],
)
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
