# Runbook — Agent Simulation Webinar

The live demo flow: ordered steps, exact commands, and what each one answers.
Spine agent: the **ICS credit-card Q&A agent** (`make provision` recreates it in
the banking / CustomerDemo workspace). Platform ops (datasets, agents) use the
**`orq` CLI**; the simulation engine is **`evaluatorq sim`** (the `orq` CLI has no
simulation command — see Gaps §1).

> Status: validated end-to-end against the real provisioned agent
> (`ics_creditcard_agent_demo`). Set `ORQ_API_KEY` to the banking workspace key.

## Audience questions this answers

| Question | Step |
|----------|------|
| Do I need data to start? | Act 1 (no — generate from a description) |
| Get started with an orq agent | Act 1 (`--target agent:ics_creditcard_agent_demo`) |
| Do it for a non-orq agent (CLI + SDK) | Act 2 |
| Start with data / re-run from a run | Act 3a (datapoints JSONL) |
| Extend existing data | Act 3b |
| Inspect the personas/scenarios created | Act 1, step 2 |

## Prereqs (run once, off-camera)

```bash
export ORQ_API_KEY="<banking / CustomerDemo key>"   # routes sim LLM + hosts the agent
cd packages/evaluatorq-py
uv sync --all-extras

# Provision the ICS agent + its 2 code tools + FAQ knowledge base into Orq.
# Idempotent — safe to re-run. Uses distinct demo keys, never touches the
# customer's live entities. Creates agent key `ics_creditcard_agent_demo`.
cd examples/agent_simulation/webinar_demo && make provision
```

The agent is the ICS (International Card Services / ABN AMRO) credit-card support
bot: `azure/gpt-5-mini`, a 99-question Dutch/English FAQ knowledge base, and two
code tools (`get_card_info`, `get_transaction_details`). Definitions live in
`agent_build/orq_export/` + `agent_build/assets/ics_faq.txt`; `agent_build/provision.py`
recreates them via the Orq Python SDK. All demo commands below default to
`AGENT=ics_creditcard_agent_demo` (override on the `make` line).

---

## Act 1 — From scratch, no data ("do I need data? No.")

**1. Generate personas × scenarios from just the agent's description.**
No dataset, no examples — the ICS agent's own description seeds it.

```bash
uv run evaluatorq sim generate \
  --target agent:ics_creditcard_agent_demo \
  --num-personas 3 --num-scenarios 3 \
  -o ics_datapoints.jsonl
```
(`--target` only fetches the description here; the agent is not called yet.
Swap for `--agent-description "..."` if you'd rather not hit the platform.)

**2. Inspect what was created** — the payoff slide. Open the JSONL: each row is a
persona (patience/assertiveness/style/emotional-arc/background) crossed with a
scenario (goal/context/criteria). This is the "coverage from nothing" moment.

```bash
uv run evaluatorq sim validate-dataset ics_datapoints.jsonl   # sanity check
python -m json.tool < <(head -1 ics_datapoints.jsonl)          # pretty-print one row
```

**3. Run the simulation against the live ICS agent** and save the datapoints so we
can re-run the *exact same* set later (Act 3a).

```bash
uv run evaluatorq sim run \
  --target agent:ics_creditcard_agent_demo \
  --num-personas 3 --num-scenarios 3 \
  --save-datapoints ics_datapoints.jsonl \
  --name ics-from-scratch
```

**4. Show the results** — pass rate + a failing transcript in the run viewer.

```bash
uv run evaluatorq sim runs                      # list saved runs
uv run evaluatorq sim ui .evaluatorq/sim-runs/<run>.json
```

---

## Act 2 — A non-orq agent (CLI *and* SDK)

Same simulation, agent that isn't hosted on orq. Show both surfaces.

**CLI** — point at any OpenAI-compatible model/endpoint or a Vercel AI SDK URL:

```bash
# OpenAI-compatible (vLLM / OpenRouter / local); OPENAI_API_KEY + OPENAI_BASE_URL
uv run evaluatorq sim run --openai-model gpt-4o-mini \
  --agent-description "Support agent for ..." --name ics-nonorq-cli

# or a deployed HTTP agent
uv run evaluatorq sim run --vercel-url https://my-agent.vercel.app/api/chat \
  --agent-description "..." --name ics-nonorq-http
```

**SDK** — wrap any `async fn(messages) -> str`, or a framework agent via the
`AgentTarget` protocol (LangGraph / OpenAI Agents SDK / Pydantic AI / CrewAI —
see `../06`–`../09`). Reference: `../01_basic_simulation.py`.

```python
from evaluatorq.simulation import simulate

async def my_agent(messages) -> str:
    ...  # call your model / graph / crew

results = await simulate(
    target_callback=my_agent,
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
  --target agent:ics_creditcard_agent_demo \
  --datapoints ics_datapoints.jsonl \
  --name ics-rerun
```

**3b. Extend the set.** Generate more personas/scenarios and append — your suite
grows without discarding what you have.

```bash
uv run evaluatorq sim generate --target agent:ics_creditcard_agent_demo \
  --num-personas 2 --num-scenarios 2 -o more.jsonl
cat more.jsonl >> ics_datapoints.jsonl
uv run evaluatorq sim validate-dataset ics_datapoints.jsonl
```

**3c. Publish the set as an orq dataset — and re-run from it.** New `eq sim`
commands close the round-trip (see Gaps §2/§3): upload a datapoints JSONL, then
run straight from the dataset.

```bash
# push the frozen set to a new orq dataset (persona/scenario are auto-stringified)
uv run evaluatorq sim upload-dataset -i ics_datapoints.jsonl -n "ICS Sim Set"
# -> Created dataset <id> ...  Uploaded N datapoint(s) -> dataset <id>

# extend an existing dataset instead of creating one
uv run evaluatorq sim upload-dataset -i more.jsonl --dataset-id <id>

# re-run directly from the orq dataset (CLI-native now)
uv run evaluatorq sim simulate --target agent:ics_creditcard_agent_demo --dataset-id <id> --name ics-from-dataset
```

`eq sim generate --dataset-format -o rows.jsonl` writes the dataset envelope up
front if you'd rather inspect/version the exact upload shape.

**The three-options slide** (build in `presentation.qmd`): **① from scratch**
(Act 1) · **② re-run from data** (Act 3a — local JSONL, or 3c — orq dataset) ·
**③ extend from data** (Act 3b append, or 3c `--dataset-id`).

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
   takes exactly one of `--datapoints` (local JSONL) or `--dataset-id` (orq dataset).
4. **No native "seed from data + generate more."** `dataset_id` is still mutually
   exclusive with personas/scenarios — extension is JSONL append (3b) or
   `upload-dataset --dataset-id` (3c). No LLM augmentation *from* existing rows yet.
5. **`orq datasets create` needs a `path`** (e.g. `"Default"`); `--example` errors
   with no generated body, and `expected_output` must be a string (not `null`).

## Demo assets to capture (drop in `assets/`, wire into `presentation.qmd`)

- [ ] Screenshot: pretty-printed generated persona/scenario (Act 1.2)
- [ ] Screenshot/recording: `sim run` pass-rate output + a failing transcript in `sim ui`
- [ ] Screenshot: orq platform experiment view for a run
- [ ] The three-options diagram (from scratch / re-run / extend)
