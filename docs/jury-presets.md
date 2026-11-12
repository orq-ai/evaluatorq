# Jury Presets

A jury is only as good as its panel, and assembling one by hand means picking judges, checking they do not share a training lineage, keeping an odd number of them, and re-checking all of it every time a provider ships something new. A preset is that decision made once and kept current: `preset="balanced-trio"` seats three judges from three families, sets the aggregation rule, and requires a majority of them to return a verdict.

```python
from evaluatorq import evaluatorq, llm_jury

correctness = llm_jury(
    name="correctness",
    criteria="The answer is factually correct and directly answers the question.",
    preset="balanced-trio",
)
```

## The presets

Costs are USD per 1,000 pointwise items at 1,500 input and 500 output tokens, uncached, one call per judge per item. The token counts are a flat 3:1 assumption, the same for every seat whatever its reasoning effort, and the same ratio the seats are ranked on. Every figure is recomputed by a weekly check from the live model catalog's rates (seats priced from `MANUAL_MODEL_CARDS` use that register's dated rates), so a repricing shows up as a failing run rather than a quietly wrong table.

| Preset | Judges | Aggregation | $ / 1k | Reserve |
| --- | --- | --- | --- | --- |
| **`balanced-trio`** (default) | `baseten/deepseek-v4.1-flash`<br>`openai/gpt-6-luna`<br>`google-ai/gemini-3.8-flash` | majority | 4.45 | `deepseek/deepseek-flash` |
| **`strong-jury`** | `anthropic/claude-opus-5-5`<br>`openai/gpt-6.1-sol`<br>`google-ai/gemini-3.8-flash` | majority | 27.00 | `baseten/deepseek-v4.1-flash` |
| **`cheap-aggregate`** | `zai/glm-5.3-flash`<br>`google-ai/gemini-3.8-flash`<br>`baseten/deepseek-v4.1-flash` | majority | 4.29 | `mistral/mistral-large-4` |
| **`open-weight-portable`** | `baseten/deepseek-v4.1-flash`<br>`baseten/kimi-k3`<br>`zai/glm-5.3-flash` | majority | 13.29 | `greenpt/qwen3.6-35b-a3b` |
| **`eu-region`** | `aws/eu.anthropic.claude-haiku-5-5`<br>`google/eu.gemini-3.7-flash`<br>`azure/eu.gpt-5.6-luna` | majority | 4.64 | `google/eu.claude-sonnet-5` |
| **`single-provider-trio`** | `openai/gpt-6.1-sol`<br>`openai/gpt-6.1-terra`<br>`openai/gpt-6-luna` | majority | 12.40 | `openai/gpt-6-sol` |

`balanced-trio` and `single-provider-trio` use GPT-6 Luna at its `medium` default. On the same Artificial Analysis Intelligence Index v4.3.2, [GPT-6 Luna (Medium)](https://artificialanalysis.ai/models/gpt-6-luna-medium) scores 29.93 against [GPT-5.6 Luna (Medium)](https://artificialanalysis.ai/models/gpt-5-6-luna-medium)'s 25.04, while Orq's captured rates fall from $0.20/$1.20 to $0.10/$0.50 per million input/output tokens. Do not compare these scores with the older index values still present on some catalogue cards. The EU preset keeps its EU-hosted GPT-5.6 Luna seat; the US-hosted successor is not a residency-compatible replacement.

## Which one to pick

**`balanced-trio`** is the default and the right answer for most subjective evaluation. Three lineages means three different ways of being wrong, which is the whole reason to run a panel instead of one judge three times.

**`strong-jury`** costs about six times as much and is for verdicts that compound: customer-facing benchmarks, preference data you will train on, anything where a wrong label outlives the run that produced it.

**`cheap-aggregate`** is for volume: nightly reruns, regression suites and triage feeders. Its seats are derived rather than picked: the strongest model of each lineage whose 3:1 blended price is under $2.00 per million tokens, ranked at its default reasoning effort, or on its headline index where the card has no score at that effort. On the current catalog it shares two seats with `balanced-trio` and costs about the same; the difference is the third lineage (Zhipu instead of OpenAI). The `zai/glm-5.3-flash` seat is costed on its catalog card rate ($0.075/$0.25 per million); usage on the judge endpoint has been billed at twice that, which would put the panel at $4.53 rather than $4.29, and Portable at $13.53 rather than $13.29.

**`open-weight-portable`** buys independence from any closed frontier vendor, and a migration path if you later want to serve the judges yourself. It is not the cheap option: it costs about three times the default trio, almost all of it the Kimi K3 seat.

**`eu-region`** seats every judge on an EU endpoint, at about the price of `balanced-trio`. It is not `balanced-trio` relocated: the DeepSeek seat has no EU endpoint newer than v3.1, so the panel is Anthropic, Google and OpenAI, which is what the EU catalog can field at the current generation.

**`single-provider-trio`** exists for workspaces locked to one provider contract. It seats the three tiers of the newest OpenAI generation, with GPT-6 Luna standing in because no GPT-6.1 Luna is listed. It is the honest version of that constraint rather than a recommendation: three OpenAI tiers correlate, and an OpenAI-generated output faces a self-preference risk the panel cannot vote away.

## What a preset fills in

Naming a preset seats its judges, so passing `judges=` or `model=` alongside it is a contradiction rather than an override and raises. Two other things are defaults you can still overrule:

- `aggregator` becomes `"majority"`, a strict majority of the votes that actually arrived. On a trio where all three answer that is two of three; where only two answer it is both of them.
- `min_successful_judges` becomes a majority of the seats: two of three, three of five. Read it as a floor on how many judges have to answer at all, not as the threshold the aggregator applies. The two are different numbers on the same panel, and meeting the quorum does not guarantee a verdict: a trio that answers 1-1 clears a quorum of two and still returns nothing, because neither vote is more than half. Not one, because a preset publishes a cost and an agreement story about a panel and a lone surviving judge would keep the name while changing what it means. Not the whole panel either, or a single unreachable judge would void items the rest of the panel agreed on.

A run that does not reach that quorum is not a silent pass. `JuryResult` reports the panel it actually got: `judges_configured` against `judges_succeeded` shows a short panel, `tie` is set when the decisive votes split evenly and a caller tie policy broke them, and `inconclusive` is set when too few judges answered to conclude at all, with `stats` and `raw_agreement` left as `None`. A preset does not add verdict states of its own; it seats the panel and the jury runner reports what came back.

Each preset also names `reserve_judges`, the model that takes a seat when its occupant is retired. That is a maintenance record the freshness tests read, not a runtime failover: a panel whose judge errors on the day does not silently pull in a different lineage and change what the published cost and agreement figures describe.

Presets are pointwise panels, one call per judge per item, so `assignment="cyclic"` is rejected: a rotation scores each item with a single judge and leaves no panel to agree. Pairwise comparison is not part of any preset either; use `llm_jury_pairwise()` with an explicit judge list.

## What a preset does not carry

A preset sends no reasoning effort, so each seat runs at its provider's default. `JuryPreset.seated_efforts()` reads it from the live catalog (it needs `ORQ_API_KEY`), with `None` for a model whose catalog entry has no effort setting. Inside async code, `await load_garden()` from `evaluatorq.common.model_frontier` and pass the result as `garden=` instead:

```python
from evaluatorq import get_preset

get_preset("strong-jury").seated_efforts()
# {'anthropic/claude-opus-5-5': 'medium', 'openai/gpt-6.1-sol': 'medium', 'google-ai/gemini-3.8-flash': 'medium'}
```

Seats are ranked on their Artificial Analysis score at that default effort. Where a card has no score at its default, it ranks on its headline index, which carries no reasoning level: `baseten/deepseek-v4.1-flash` runs at `high` while Artificial Analysis measures it only at `max` and `none`, so it ranks on its `max` headline although it runs lower, and `mistral/mistral-large-4`'s card carries a headline with no rungs at all. `openai/gpt-6.1-terra`'s headline is on index v5.0.1 rather than the v4.3 every other card uses, so the in-family frontier check exempts GPT-6.1 Sol from it.

`llm_jury()` takes one `reasoning_effort` for the whole panel, so passing one overrides every seat's default. Per-judge call settings are a schema change and a separate ticket.

Two more limits worth knowing before you quote a number:

- Agreement between these panels and human raters is inherited from the literature, not measured on orq data. There is no human-agreement baseline behind any of these presets.
- The 500 output tokens are an assumption, not a measurement, and a floor rather than a typical figure: a judge that reasons without being asked writes more than that and costs more than the table.

## How a preset stays current

Seats are chosen in this package. `evaluatorq.common.model_frontier` reads the live `/v2/models` catalog, where every autorouter-eligible card carries Artificial Analysis intelligence scores per reasoning effort beside its rates, and ranks each model at its default effort on a 3:1 blended price. Compare seats within each lineage, because lineage diversity is the point of a panel: comparing seats against the whole catalog and choosing only the cheapest vendor would recreate the correlated errors the panel exists to cancel.

Nothing about a seat is committed except the decision to seat it. Rates and default efforts are read live, so there is no rate table to refresh; the exceptions are the seats the catalog does not list, which are priced from the dated `MANUAL_MODEL_CARDS` register. A weekly workflow (`.github/workflows/preset-drift-weekly.yml`) runs the integration checks in `tests/integration/test_preset_freshness.py` against the live catalog; a failure leaves that run red in Actions and opens a pull request (see below). Each failure names the seat and the register to update:

- a seat the catalog no longer serves
- a repricing that moves a published figure in the table above
- a newer model in a seated lineage that nobody has reviewed (`REVIEWED_SUCCESSORS`)
- a same-lineage model that beats a seat on both intelligence and price
- a seat older than 180 days whose lineage has stopped shipping, unless it has a written waiver in `STALE_SEAT_WAIVERS`

When the check fails, the workflow runs `python -m evaluatorq.preset_drift`, which lists what moved and the options the arithmetic allows: the reserve that can replace a retired seat, the same-lineage models that beat a seat, the newer models nobody has reviewed. A Pi agent, launched through the orq CLI with the workspace key, then proposes the register changes. It must confirm on the vendor's own pages that a candidate really is newer or superseded, because the catalog knows listing dates and not release notes, and it records each source beside the change. The workflow re-runs the live check on the result and always opens a real pull request: a fix when the check passes again, otherwise one labelled `needs-human-attention` that carries the report and the agent's summary. Nothing merges automatically. The pull request is opened with the repository's default token, so CI does not start on it until a person pushes to the branch or closes and reopens it.

Read the catalog with a workspace-scoped key. A project-scoped key lists only what that project has enabled, so the checks would run on a truncated catalog; the suite fails on purpose when it sees one. Routes the router serves but the catalog omits are recorded in `MANUAL_MODEL_CARDS`, each with its source and date.

A preset that is not in `PRESETS` still owes you a reason. `DROPPED_PRESETS` records one that shipped and was then retired, `WITHHELD_PRESETS` one that is derived and costed but not seated because a seat cannot do the job yet, and asking for either by name raises an error carrying that reason rather than a bare "unknown preset".

`value-trio` is the one entry in `DROPPED_PRESETS`. It was a budget panel whose measured cost ran far above its published figure, because two of its judges wrote long prose verdicts. Asking for it by name raises an error carrying that reason rather than a bare "unknown preset".

## Router pools

`evaluatorq.router_pools` derives candidate pools for the autorouter from the same live garden. A router calls one model per request rather than every member, so a pool is chosen for value under a constraint rather than for independent errors: each pool is three models on the Pareto frontier of what the constraint allows, with adjacent price tiers at least 2.75x apart. Membership is derived on first use, never committed.

```python
from evaluatorq.router_pools import get_pools, get_underivable_pools

pools = get_pools()  # reads /v2/models; needs ORQ_API_KEY
pools["EU Region Only"].models
get_underivable_pools()  # constraint name -> why no pool could be seated today
```

A pool publishes a cost range rather than a figure, at the same 1,500 input and 500 output tokens: what a request costs depends on the routing policy that picks within the pool. `DROPPED_POOLS` records the shapes that were retired and why, and `REVIEWED_UNDERIVABLE_POOLS` the ones the current catalog cannot seat.
