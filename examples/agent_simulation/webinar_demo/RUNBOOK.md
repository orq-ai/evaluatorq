# Runbook — Agent Simulation Webinar

The live demo flow: ordered steps, exact commands, and what each one answers.
Spine agent: the **ICS credit-card Q&A agent** (orq `customer-demo` workspace).
Platform ops (datasets, agents) use the **`orq` CLI**; the simulation engine is
**`evaluatorq sim`** (the `orq` CLI has no simulation command — see Gaps §1).

> Status: draft flow, validated end-to-end against a stand-in agent. Fill in the
> real ICS agent key + `customer-demo` `ORQ_API_KEY` where marked `<...>`.

## Audience questions this answers

| Question | Step |
|----------|------|
| Do I need data to start? | Act 1 (no — generate from a description) |
| Get started with an orq agent | Act 1 (`--target agent:<ics-key>`) |
| Do it for a non-orq agent (CLI + SDK) | Act 2 |
| Start with data / re-run from a run | Act 3a (datapoints JSONL) |
| Extend existing data | Act 3b |
| Inspect the personas/scenarios created | Act 1, step 2 |

## Prereqs (run once, off-camera)

```bash
export ORQ_API_KEY="<customer-demo key>"     # routes sim LLM + reaches ICS agent
cd packages/evaluatorq-py
uv sync --all-extras
orq doctor                                    # confirm CLI auth + reachability
orq agents retrieve <ics-agent-key>           # confirm the ICS agent is visible
```

---

## Act 1 — From scratch, no data ("do I need data? No.")

**1. Generate personas × scenarios from just the agent's description.**
No dataset, no examples — the ICS agent's own description seeds it.

```bash
uv run evaluatorq sim generate \
  --target agent:<ics-agent-key> \
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
  --target agent:<ics-agent-key> \
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
  --target agent:<ics-agent-key> \
  --datapoints ics_datapoints.jsonl \
  --name ics-rerun
```

**3b. Extend the set.** Generate more personas/scenarios and append — your suite
grows without discarding what you have.

```bash
uv run evaluatorq sim generate --target agent:<ics-agent-key> \
  --num-personas 2 --num-scenarios 2 -o more.jsonl
cat more.jsonl >> ics_datapoints.jsonl
uv run evaluatorq sim validate-dataset ics_datapoints.jsonl
```

**The three-options slide** (build in `presentation.qmd`): **① from scratch**
(Act 1) · **② re-run from data** (Act 3a) · **③ extend from data** (Act 3b).

**Optional — publish the set as an orq dataset** (mention, don't dwell — see
Gaps §2/§3 for why the CLI round-trip is rough):

```bash
orq datasets create --stdin <<< '{"display_name":"ICS Sim Set","path":"Default"}'
# then push rows wrapped under `inputs` with persona/scenario JSON-stringified
```

---

## Gaps to call out honestly (the "it's new" story)

1. **No `orq sim` command.** Simulation lives in `evaluatorq sim`; the `orq` CLI
   covers platform CRUD (agents, datasets, evals) only. Expected split, worth stating.
2. **Run → orq dataset doesn't round-trip via the CLI.** `sim generate -o` writes
   datapoints at top level; an orq dataset row needs them under `inputs`, and the
   dataset API **rejects nested objects** (`inputs.*` must be scalar) — so
   persona/scenario must be JSON-stringified to upload, but the sim dataset-reader
   expects *objects*. No CLI flag emits dataset-shaped JSONL. → **For "re-use data",
   demo the local datapoints JSONL path (3a), not an orq dataset.**
3. **CLI has no `--dataset-id`.** Dataset-backed runs are SDK-only
   (`simulate(dataset_id="...")`). The CLI's data-reuse surface is `--datapoints`.
4. **No native "seed from data + generate more."** `dataset_id` is mutually
   exclusive with personas/scenarios/datapoints — extension is manual JSONL append (3b).
5. **`orq datasets create` needs a `path`** (e.g. `"Default"`); `--example` errors
   with no generated body, and `expected_output` must be a string (not `null`).

## Demo assets to capture (drop in `assets/`, wire into `presentation.qmd`)

- [ ] Screenshot: pretty-printed generated persona/scenario (Act 1.2)
- [ ] Screenshot/recording: `sim run` pass-rate output + a failing transcript in `sim ui`
- [ ] Screenshot: orq platform experiment view for a run
- [ ] The three-options diagram (from scratch / re-run / extend)
