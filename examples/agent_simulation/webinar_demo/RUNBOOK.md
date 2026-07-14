# Runbook — Agent Simulation Webinar

The live demo flow: ordered steps, exact commands, and what each one answers.
Spine agent: **Sterling**, the **Bank of Holland credit-card Q&A agent** (`make provision` recreates it in
the banking / CustomerDemo workspace). Platform ops (datasets, agents) use the
**`orq` CLI**; the simulation engine is **`evaluatorq sim`** (the `orq` CLI has no
simulation command — see Gaps §1).

> Status: validated end-to-end against the real provisioned agent
> (`sterling`). Set `ORQ_API_KEY` to the banking workspace key.

## Act 0 — Framing (open the talk, ~5 min)

Cue bullets for the opening — say these before touching a terminal. The room may
not know orq, the problem, or what "agent simulation" even means. (Fuller
versions: webinar-script.html §01 open + §02 diagram + Q&A bank.)

- **Who.** orq.ai = generative-AI collaboration platform: one control plane to
  **build** agents · **ship** via the model router · **optimize** with
  observability + evals. `evaluatorq` = the open-source eval engine (pip-install,
  runs anywhere) — the sim we're demoing lives here, no platform lock-in.
- **The problem.** Everyone here already evals — but the stack is *single-turn*:
  golden sets + LLM-as-judge on isolated responses. Agent failures are
  conversational — the user pivots, stalls, contradicts; it breaks at **turn 7,
  not turn 1**. And on day one there's no dataset that encodes those dynamics.
- **What agent simulation is.** Three LLMs in a loop: a **generator** writes
  personas × scenarios from just the agent's description → a **user-simulator**
  plays each persona multi-turn against **your agent** → a **judge** scores
  whether it hit the goal. (Point at webinar-script.html §02 diagram.)
- **Who this is for.** Anyone who's run a golden-set eval or shipped an agent.
  No prior sim knowledge assumed — no dataset, labels, or annotation budget to start.
- **Cooperative, not adversarial.** This is *not* red teaming. Red teaming is a
  hostile attacker hunting exploits; simulation plays **realistic users trying to
  get their job done**, and shows where the agent fails them.

## Audience questions this answers

| Question | Step |
|----------|------|
| Do I need data to start? | Act 1 (no — generate from a description) |
| Get started with an orq agent | Act 1 (`--target agent:sterling`) |
| Do it for a non-orq agent (CLI + SDK) | Act 2 |
| Start with data / re-run from a run | Act 3a (datapoints JSONL) |
| Extend existing data | Act 3b |
| Inspect the personas/scenarios created | Act 1, step 2 |

## Prereqs (run once, off-camera)

```bash
export ORQ_API_KEY="<banking / CustomerDemo key>"   # routes sim LLM + hosts the agent
# Run from the evaluatorq repository root (this checkout).
uv sync --all-extras

# Provision the Bank of Holland agent + its 2 code tools + FAQ knowledge base into Orq.
# Idempotent — safe to re-run. Uses distinct demo keys, never touches the
# customer's live entities. Creates agent key `sterling`.
cd examples/agent_simulation/webinar_demo && make provision
```

The agent is **Sterling**, the Bank of Holland credit-card support
bot: `openai/gpt-5.6-luna`, a 99-question Dutch/English FAQ knowledge base, and two
code tools (`get_card_info`, `get_transaction_details`). Definitions live in
`agent_build/orq_export/` + `agent_build/assets/boh_faq.txt`; `agent_build/provision.py`
recreates them via the Orq Python SDK. All demo commands below default to
`AGENT=sterling` (override on the `make` line).

---

## Act 1 — From scratch, no data ("do I need data? No.")

**1. Generate personas × scenarios from just the agent's description.**
No dataset, no examples — the Bank of Holland agent's own description seeds it.
*What happens:* a generator LLM writes the persona × scenario grid from the
description — the "coverage from nothing" step.

```bash
uv run evaluatorq sim generate \
  --target agent:sterling \
  --num-personas 3 --num-scenarios 3 \
  -o boh_datapoints.jsonl
```

(`--target` only fetches the description here; the agent is not called yet.
Swap for `--agent-description "..."` if you'd rather not hit the platform.)

**2. Inspect what was created** — the payoff slide. Open the JSONL: each row is a
persona (patience/assertiveness/style/emotional-arc/background) crossed with a
scenario (goal/context/criteria). This is the "coverage from nothing" moment.

```bash
uv run evaluatorq sim validate-dataset boh_datapoints.jsonl   # sanity check
python3 -m json.tool < <(head -1 boh_datapoints.jsonl)         # pretty-print one row
```

**3. Run the simulation against the live Bank of Holland agent** using the frozen datapoints
so we can re-run the *exact same* set later (Act 3a).
*What happens:* the user-simulator LLM drives each conversation turn-by-turn;
the judge LLM scores whether the agent met the goal + criteria.

```bash
uv run evaluatorq sim simulate \
  --target agent:sterling \
  -i boh_datapoints.jsonl \
  --name boh-from-scratch
```

**4. Show the results** — pass rate + a failing transcript in the run viewer.

```bash
uv run evaluatorq sim runs                      # list saved runs
uv run evaluatorq dashboard .evaluatorq/sim-runs    # multi-run viewer
```

---

## Act 2 — A non-orq agent (CLI *and* SDK)

Same simulation, agent that isn't hosted on orq. Show both surfaces.

**CLI** — point at any OpenAI-compatible model/endpoint or a Vercel AI SDK URL:

```bash
# OpenAI-compatible (vLLM / OpenRouter / local); OPENAI_API_KEY + OPENAI_BASE_URL
uv run evaluatorq sim run --openai-model gpt-4o-mini \
  --agent-description "Support agent for ..." --name boh-nonorq-cli

# or a deployed HTTP agent
uv run evaluatorq sim run --vercel-url https://my-agent.vercel.app/api/chat \
  --agent-description "..." --name boh-nonorq-http
```

**SDK** — wrap any `async fn(messages) -> str`, or a framework agent via the
`AgentTarget` protocol (LangGraph / OpenAI Agents SDK / Pydantic AI / CrewAI —
see `../06`–`../09`). Reference: `../01_basic_simulation.py`.

```python
from evaluatorq.simulation import simulate

async def my_agent(messages) -> str:
    ...  # call your model / graph / crew

results = await simulate(
    target=my_agent,                   # unified param — same as the CLI's --target
    personas=[...], scenarios=[...],   # or omit to auto-generate
    evaluator_names=["goal_achieved", "criteria_met"],
)
```

---

## Act 3 — Working with data

**3a. Re-run from a previous run's datapoints (no dataset needed).**
The frozen JSONL from Act 1 step 3 *is* your data. Re-run the identical set —
e.g. after tweaking the agent prompt — to compare like-for-like.

```bash
uv run evaluatorq sim simulate \
  --target agent:sterling \
  -i boh_datapoints.jsonl \
  --name boh-rerun
```

**3b. Extend the set.** Generate more personas/scenarios and append — your suite
grows without discarding what you have.

```bash
uv run evaluatorq sim generate --target agent:sterling \
  --num-personas 2 --num-scenarios 2 -o more.jsonl
cat more.jsonl >> boh_datapoints.jsonl
uv run evaluatorq sim validate-dataset boh_datapoints.jsonl
```

**3c. Publish the set as an orq dataset — and re-run from it.** New `eq sim`
commands close the round-trip (see Gaps §2/§3): upload a datapoints JSONL, then
run straight from the dataset.

```bash
# push the frozen set to a new orq dataset (persona/scenario are auto-stringified)
uv run evaluatorq sim upload-dataset -i boh_datapoints.jsonl -n "Bank of Holland Sim Set"
# -> Created dataset <id> ...  Uploaded N datapoint(s) -> dataset <id>

# extend an existing dataset instead of creating one
uv run evaluatorq sim upload-dataset -i more.jsonl --dataset-id <id>

# re-run directly from the orq dataset (CLI-native now)
uv run evaluatorq sim simulate --target agent:sterling --dataset-id <id> --name boh-from-dataset
```

`eq sim generate --dataset-format -o rows.jsonl` writes the dataset envelope up
front if you'd rather inspect/version the exact upload shape.

**The three-options slide** (in `webinar-deck.html`): **① from scratch**
(Act 1) · **② re-run from data** (Act 3a — local JSONL, or 3c — orq dataset) ·
**③ extend from data** (Act 3b append, or 3c `--dataset-id`).

---

## Act 4 — Inspect in the dashboard, then harden the agent

The payoff act: read the results, find *why* the agent failed, fix it, prove the fix.

**4a. Inspect the runs in the dashboard.** The dashboard scans this demo's run store
(`.evaluatorq/sim-runs/`) and shows every run together — browse runs, filter to the
failures, read the transcripts.

```bash
uv run evaluatorq dashboard .evaluatorq/sim-runs    # -> http://127.0.0.1:8080
```

Look for a *pattern*: a criterion that fails repeatedly, a persona the agent mishandles,
a turn where it over-promises or leaks something it shouldn't.

**4b. Apply the insight to the existing agent — with Claude + the `orq` CLI.**
Point Claude Code at the run and let it propose + apply a fix. Suggested prompts:

> **Investigate:** "Read the latest simulation run under `.evaluatorq/sim-runs/`.
> Summarise the failures — which criteria failed, for which personas/scenarios, and
> the common root cause in the agent's behaviour. Quote the transcript turns that
> show it."

> **Apply:** "Propose the *minimal* edit to the `sterling` agent's
> instructions that fixes the top failure mode without regressing the rest. Fetch the
> current instructions, append your fix, and apply it with the orq CLI:
> `orq agents update sterling --instructions \"<full updated instructions>\"`.
> Then tell me to re-run."

The working command (validated) is **`orq agents update <agent-key> --instructions "..."`**
— partial update, effective immediately. (`orqi agent update` 404s for these agents —
see Gaps §6.)

**4c. Re-run the *same* set and compare.** Because the datapoints are frozen, the only
variable is the instruction change — a clean before/after.

```bash
make rerun          # re-runs boh_datapoints.jsonl against the hardened agent
```

Show the pass rate move up (and the previously-failing transcript now pass) — the
inspect → harden → re-run loop, closed live.

---

## Gaps to call out honestly (the "it's new" story)

1. **No `orq sim` command.** Simulation lives in `evaluatorq sim`; the `orq` CLI
   covers platform CRUD (agents, datasets, evals) only. Expected split, worth stating.
2. **Run → orq dataset — now handled by `sim upload-dataset`.** The dataset API
   **rejects nested objects** (`inputs.*` must be scalar), so persona/scenario are
   stored **JSON-stringified**; the sim reader now accepts that, so the round-trip
   holds. Honest caveat to voice: in the orq UI those two columns show as JSON
   strings, not structured fields. (`sim generate --dataset-format` emits the same
   envelope for inspection.)
3. **`sim simulate --dataset-id` — now CLI-native** (was SDK-only). `simulate`
   takes exactly one of `-i`/`--input` (local JSONL) or `--dataset-id` (orq dataset).
4. **No native "seed from data + generate more."** `dataset_id` is still mutually
   exclusive with personas/scenarios — extension is JSONL append (3b) or
   `upload-dataset --dataset-id` (3c). No LLM augmentation *from* existing rows yet.
5. **`orq datasets create` needs a `path`** (e.g. `"Default"`); `--example` errors
   with no generated body, and `expected_output` must be a string (not `null`).
6. **Two CLIs, one works for agent updates.** `orq agents update <key> --instructions`
   applies cleanly; the other CLI's `orqi agent update` returns `deployment_not_found`
   for these agents (by key *and* by id). Use `orq agents update` in Act 4.

## Demo assets to capture (drop in `assets/`, then wire into `webinar-deck.html`)

- [ ] Screenshot: pretty-printed generated persona/scenario (Act 1.2)
- [ ] Screenshot/recording: `sim run` pass-rate output + a failing transcript in the dashboard
- [ ] Screenshot: the dashboard filtered to failures + a transcript (Act 4a)
- [ ] Screenshot: orq platform experiment view for a run
- [ ] The three-options diagram (from scratch / re-run / extend)
