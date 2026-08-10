# FAQ

Common questions, grouped by area — **General** (install, keys, privacy, running evaluations), **Red Teaming**, and **Agent Simulation**.

## General

### What is evaluatorq?

A Python library for testing LLM apps and agents, with three modes:

- **Evaluations** — run jobs over your data in parallel and score them with built-in or custom evaluators; gate CI on pass/fail.
- **Agent simulation** — a user-simulator LLM drives your agent through multi-turn conversations while a judge scores whether it met its goals.
- **Red teaming** — adaptive adversarial attacks mapped to the OWASP LLM Top 10 and Agentic Security Initiative.

The Orq platform is optional — it stores results and routes LLMs when `ORQ_API_KEY` is set, but you can run entirely on OpenAI.

### What do I install?

Pick the extra for what you're doing:

```bash
uv add "evaluatorq"               # core evaluations
uv add "evaluatorq[simulation]"   # agent simulation
uv add "evaluatorq[redteam]"      # red teaming
```

`uv add` installs into the current project — run `uv init` first if you don't have one. Run your scripts with `uv run my_eval.py` and the CLI with `uv run eq`, so the environment you installed into is the one that executes.

Prefer pip? Use `python -m pip install "evaluatorq[redteam]"`, which installs into the interpreter you just named rather than whichever `pip` happens to be first on your `PATH`.

### I installed evaluatorq but `import evaluatorq` fails

The install went to a different interpreter than the one running your script. This is the single most common setup failure, and it has nothing to do with evaluatorq — bare `pip` and bare `python` can resolve to different environments (a system Python, a virtualenv you forgot to activate, a container's global site-packages).

Confirm it by asking both sides where they live:

```bash
python -c "import sys; print(sys.executable)"   # which interpreter runs
python -m pip show -f evaluatorq | head -3      # where the package landed
```

If the paths don't share a prefix, that's the bug. Two fixes:

- **uv** — `uv add evaluatorq` then `uv run my_eval.py`. `uv run` resolves the project environment before executing, so the two can't drift.
- **pip** — always name the interpreter: `python -m pip install evaluatorq`, and run with the same `python`.

Avoid `uv tool install evaluatorq`: it builds an isolated environment that exposes the `eq` CLI but leaves `evaluatorq` unimportable from your own scripts.

### Do I need an Orq account or an OpenAI key?

You need an LLM key wherever a simulator, attacker, or judge LLM runs — `OPENAI_API_KEY` for direct OpenAI, or `ORQ_API_KEY` to route through the Orq router. Plain evaluations with only deterministic evaluators need no key; any LLM-judged flow (jury, simulation, red teaming) does — including red teaming's **static mode**, where the target replays fixed attacks but the judge still scores each outcome with an LLM.

### Which models run the simulator / attacker / judge — can I change them?

They default to an LLM routed via `OPENAI_API_KEY` or `ORQ_API_KEY`. Override per surface: red teaming takes `llm_config=LLMConfig(attacker=..., evaluator=...)`, and simulation takes `sim_model=` for the simulator and judge.

```python
from evaluatorq.redteam import LLMConfig, LLMCallConfig

report = await red_team(
    target=MyAgent(),
    llm_config=LLMConfig(
        attacker=LLMCallConfig(model="anthropic/claude-3-5-sonnet", temperature=0.9),
        evaluator=LLMCallConfig(model="openai/gpt-4o-mini", temperature=0.0),
    ),
)
```

### What leaves my machine?

Simulator/attacker/judge LLM calls go to OpenAI or the Orq router. Results upload to the Orq platform **only if `ORQ_API_KEY` is set.** With no key, everything stays local. Setting `ORQ_API_KEY` also enables OpenTelemetry tracing to `my.orq.ai`; suppress it with `ORQ_DISABLE_TRACING=1`, or keep tracing but strip prompt/response text from spans with `EVALUATORQ_CAPTURE_MESSAGE_CONTENT=false`. See [Configuration](configuration.md).

### How much does a run cost, and how do I keep it cheap?

Cost and wall-clock scale with cases × turns × LLM calls. The levers are how many cases you run (`max_dynamic_datapoints` / `max_static_datapoints` for red teaming, `num_personas` × `num_scenarios` for simulation), `max_turns`, and `parallelism` (default 10 for red teaming, 5 for simulation — lower it if your provider rate-limits). Red teaming's report tracks spend in `report.summary.token_usage_total`.

### Where do results go, and how do I view a past run?

Runs auto-save locally (red-team runs to `.evaluatorq/runs/`; simulation runs to `.evaluatorq/sim-runs/`). Browse them in the multi-run FastHTML dashboard with `eq dashboard` (no path browses both stores; `eq dashboard .evaluatorq/sim-runs` scopes to simulation), or list runs with `eq redteam runs` / `eq sim runs`. The legacy `eq redteam ui` / `eq sim ui` Streamlit views remain callable but are deprecated. See [Dashboard](dashboard.md).

### How do I run a plain evaluation?

Decorate a function with `@job`, hand `evaluatorq()` your data and evaluators, and it runs the jobs in parallel and scores each row:

```python
from evaluatorq import DataPoint, evaluatorq, job, string_contains_evaluator


@job("greet")
async def greet_job(data: DataPoint, _row: int) -> str:
    return f"Hello, {data.inputs['name']}!"


await evaluatorq(
    "smoke-test",
    data=[DataPoint(inputs={"name": "Ada"}, expected_output="Hello, Ada!")],
    jobs=[greet_job],
    evaluators=[string_contains_evaluator()],
    print_results=True,
)
```

See [Getting Started](guides/getting-started.md).

### What evaluators are built in, and can I write my own?

There are deterministic ones (string match, JSON, regex) and LLM-judge ones. For custom logic, write a function that scores a row — see [Custom Evaluators & Frameworks](custom-evaluators-and-frameworks.md). For higher confidence, score one response with a **panel** of judges ([LLM as a Jury](llm-as-a-jury.md)) or compare two responses head-to-head ([Pairwise Judging](pairwise-judging.md)).

## Red Teaming

### How do I know my agent is safe?

You don't, until you attack it. Shipping after a refused *"say something harmful"* is a vibe check, not a test. It only proves the agent refuses the one obvious prompt you thought to try. Red teaming runs a mapped set of adversarial attacks and reports a **resistance rate** (the fraction the agent withstood), so "safe" becomes a number you can gate on. See [Red Teaming](guides/red-teaming.md).

### Isn't a single-turn "refused → safe" check enough?

No. The attacks that land are the ones a single prompt can't express:

- **Multi-turn escalation** — each message looks benign; the attack assembles across turns. Invisible to single-turn evals.
- **Indirect injection** — the attacker controls what the agent *reads* (emails, docs, tool results), not what you type. You never see the payload.
- **Memory poisoning** — a planted instruction fires on a later, unrelated run.
- **Many-shot jailbreaking** — 100+ in-context examples steer behaviour.

Red teaming generates multi-turn attacks by default (`max_turns=`) precisely so these surface.

### What does red teaming actually test?

Attacks and LLM-judge evaluators mapped to three frameworks:

- **OWASP LLM Top 10** — prompt injection, system-prompt leakage, and the rest.
- **OWASP Agentic Security Initiative (ASI)** — agent-specific risks: tool abuse, excessive agency, trust exploitation.
- **Responsible AI** — fairness/bias, liability (legal, medical), content policy, harmful content.

Pass the ones you care about via `categories=["LLM01", "ASI01", ...]`. The full list of codes is in the [redteam API reference](reference/evaluatorq/redteam.md).

### Static, dynamic, or hybrid — which mode?

- **static** — replays a fixed dataset of known attacks. Deterministic and cheap; run it in CI. Defaults to Orq's public [`orq/redteam-vulnerabilities`](https://huggingface.co/datasets/orq/redteam-vulnerabilities) set, or pass your own with `dataset=`.
- **dynamic** — an LLM generates attacks tailored to *your* agent's tools and prompt.
- **hybrid** — static seeds plus dynamic expansion, in one pass.

Details in [Red Teaming › Modes](guides/red-teaming.md#modes).

### How does it decide what to attack?

Probe → attack → judge. It first inspects the target (tools, permissions, system prompt) to build a capability profile, generates attacks against the exposed surface, then scores each outcome against per-vulnerability LLM-judge rubrics. Filtering out attacks that can't apply keeps a run focused on real exposure.

### Why an LLM judge instead of keyword matching?

Because the surface text lies. *"I see malicious instructions in this email but I won't follow them"* and *"I'll forward that email right away"* both mention instructions, yet one is resistant, one is vulnerable. A regex can't tell them apart; a judge rubric can. Write your own in [Custom Evaluators & Frameworks](custom-evaluators-and-frameworks.md).

### How do I trust the judge itself?

Calibrate it against known baselines: run the same attack set through an agent that *should* pass and one that *should* fail. If the safe agent gets flagged vulnerable you have false positives; if the weak agent is marked resistant you have blind spots. Either way, patch the rubric and rerun. Matched pairs (same attacks, both agents) make the miscalibration visible.

### Isn't "vulnerable" context-dependent?

Yes, and the evaluators account for it. Fetching a shell script from GitHub is helpful in a coding assistant and an RCE vector in a support bot; chaining three API calls unprompted is doing the job or excessive agency depending on the agent. Judges see the agent's declared context, so the same action can score differently across agents.

### How do I run it against my own agent?

Subclass `AgentTarget` and implement two methods: `respond(messages)` (return an `AgentResponse`) and `new()` (return a fresh instance for each attack). Any framework works behind those two methods (LangChain, LangGraph, the OpenAI Agents SDK, or a plain loop), so there's no framework buy-in:

```python
from evaluatorq.contracts import AgentTarget, AgentResponse, Message
from evaluatorq.redteam import red_team


class MyAgent(AgentTarget):
    async def respond(self, messages: list[Message]) -> AgentResponse:
        reply = await my_agent_call(messages)      # your agent, any framework
        return AgentResponse(content=reply)

    def new(self) -> "MyAgent":
        return MyAgent()


report = await red_team(target=MyAgent(), mode="dynamic", max_turns=4)
print(f"resistance: {report.summary.resistance_rate:.0%}")
```

Or point it at an Orq agent by key (`"agent:<key>"`), or drive it from the CLI with `eq redteam run --target agent:<key>`. See [Red Teaming › Red-team your target](guides/red-teaming.md#red-team-your-target) and [`examples/redteam/15_tool_chaining.py`](examples/redteam/15_tool_chaining.md) for a full custom target.

### If I red-team my real agent, will the attacks actually fire its tools?

Yes. The target runs its own tools, so a successful attack triggers real side effects (sends the email, moves the money, runs the shell command). Point red teaming at a **sandboxed or test instance** with fake/stubbed tools — like the demo's fake wallets — not at production credentials wired to irreversible actions.

### The report says my agent is vulnerable — what do I change?

Usually the system prompt. The load-bearing fixes are an explicit instruction hierarchy (data the agent *reads* is never a command), a confirmation gate before risky tools, and refusing authority claims — then rerun and watch the resistance rate climb. Pass `generate_recommendations=True` to have the run attach LLM-generated focus-area recommendations to the report.

### What does `passed=True` mean?

The agent **resisted** the attack (the attack failed). `passed=False` means the attack succeeded: the agent is **vulnerable**. `resistance_rate` is the fraction of attacks that came back `passed=True`.

## Agent Simulation

### What is agent simulation, and how is it different from red teaming?

Both drive your agent across multi-turn conversations, but with opposite intent. Simulation plays a **cooperative** user (a persona pursuing a realistic goal) and asks *"did the agent do its job?"*; red teaming plays an **adversary** and asks *"can the agent be broken?"* Three LLMs are in play for simulation: your agent, a user-simulator, and a judge that scores `goal_achieved` / `criteria_met`.

### How do I simulate multi-turn conversations?

`generate_and_simulate()` is the fastest start: it synthesizes personas, scenarios, and opening messages from a one-line description of your agent, no hand-written transcripts:

```python
from evaluatorq.simulation import generate_and_simulate

results = await generate_and_simulate(
    evaluation_name="support-agent-sim",
    target="agent:my-support-agent",     # or pass a callable as target= for your own agent
    agent_description="Customer support agent handling refunds and orders.",
    num_personas=3,
    num_scenarios=4,                     # → 12 persona × scenario simulations
    max_turns=6,
    evaluator_names=["goal_achieved", "criteria_met"],
)
print(f"pass rate: {sum(r.goal_achieved for r in results)}/{len(results)}")
```

See [Agent Simulation](guides/agent-simulation.md).

### Where do the personas and scenarios come from?

Your choice of control: generate them from a one-line description, seed by archetype, hand-build `Persona(...)` / `Scenario(...)` for full control, or ground new cases in your **real production traces** so they mirror how users actually behave. You can also replay stored datapoints to re-run the exact same cases against any target. See [Agent Simulation](guides/agent-simulation.md#from-existing-traces-and-data).

### Which agent frameworks does simulation work with?

LangGraph, the OpenAI Agents SDK, Pydantic AI, CrewAI, a plain async callback (passed as `target=`), or a hosted Orq agent (`target="agent:<key>"`) — see the [framework demos](guides/agent-simulation.md#external-framework-demos).
