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

Hosting in brackets is the region the catalog lists for each endpoint on 2026-11-17. `cheap-aggregate` and `open-weight-portable` seat `tencent/deepseek-v4.1-flash`, which is served only from Singapore, and `strong-jury` keeps it in reserve. `balanced-trio`, the default, seats the same model on Baseten instead at twice the rate, so the default never routes to Singapore unasked. Tencent doubles its rate on weekdays from 01:00 to 04:00 and from 06:00 to 10:00 UTC, which adds about $0.53 per 1,000 items for each Tencent seat, so the 4.05 and 8.70 figures are best-case costs and run to about 4.58 and 9.23 inside those hours.

| Preset | Judges (hosting) | Aggregation | $ / 1k | Reserve (hosting) |
| --- | --- | --- | --- | --- |
| **`balanced-trio`** (default) | `baseten/deepseek-v4.1-flash` (Global)<br>`openai/gpt-6-luna` (US)<br>`google-ai/gemini-3.8-flash` (Global) | majority | 4.45 | `deepseek/deepseek-flash` (Global) |
| **`strong-jury`** | `anthropic/claude-opus-5-5` (Global)<br>`openai/gpt-6.1-sol` (US)<br>`google-ai/gemini-3.8-flash` (Global) | majority | 27.00 | `tencent/deepseek-v4.1-flash` (Singapore) |
| **`cheap-aggregate`** | `google-ai/gemini-3.8-flash` (Global)<br>`tencent/deepseek-v4.1-flash` (Singapore)<br>`wafer/MiniMax-M3.1-Flash` (Global) | majority | 4.05 | `wafer/GLM-5.3-Flash` (Global) |
| **`open-weight-portable`** | `tencent/deepseek-v4.1-flash` (Singapore)<br>`moonshotai/kimi-k3.1-pro` (Global)<br>`wafer/GLM-5.3-Flash` (Global) | majority | 8.70 | `alibaba/qwen3.6-35b-a3b` (US) |
| **`eu-region`** | `aws/eu.anthropic.claude-haiku-5-5` (EU)<br>`google/eu.gemini-3.7-flash` (EU)<br>`openai/eu.gpt-6-luna` (EU) | majority | 4.18 | `google/eu.claude-sonnet-5` (EU) |
| **`single-provider-trio`** | `openai/gpt-6.1-sol` (US)<br>`openai/gpt-6.1-terra` (US)<br>`openai/gpt-6-luna` (US) | majority | 12.40 | `openai/gpt-6-sol` (US) |

`balanced-trio` and `single-provider-trio` use GPT-6 Luna at its `medium` default, and `eu-region` seats the same model on the OpenAI EU endpoint (`openai/eu.gpt-6-luna`, $0.11/$0.55 per million input/output tokens). Every score on this page is the Artificial Analysis Intelligence Index v5 as Orq's model catalog carried it on 2026-11-17; v5 runs lower than v4.3, so do not compare these numbers with older v4.3 figures.

## Which one to pick

**`balanced-trio`** is the default and the right answer for most subjective evaluation. Three lineages means three different ways of being wrong, which is the whole reason to run a panel instead of one judge three times.

**`strong-jury`** costs about six times as much and is for verdicts that compound: customer-facing benchmarks, preference data you will train on, anything where a wrong label outlives the run that produced it.

**`cheap-aggregate`** is for volume: nightly reruns, regression suites and triage feeders. Its seats are derived rather than picked: the strongest model of each lineage whose 3:1 blended price is under $2.00 per million tokens, ranked at its default reasoning effort, or on its headline index where the card has no score at that effort. On the current catalog it seats the same Gemini model and the same DeepSeek weights as `balanced-trio` (DeepSeek on Tencent instead of Baseten) and costs a little less; the difference is the third lineage (MiniMax instead of OpenAI).

**`open-weight-portable`** buys independence from any closed frontier vendor, and a migration path if you later want to serve the judges yourself. It is not the cheap option: it costs about twice the default trio, almost all of it the Kimi K3.1 Pro seat.

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

A preset sends no reasoning effort, so each seat runs at its provider's default. `JuryPreset.seated_efforts()` reads it from the live public catalog (no key needed), with `None` for a model whose catalog entry has no effort setting. Inside async code, `await load_garden()` from `evaluatorq.common.model_frontier` and pass the result as `garden=` instead:

```python
from evaluatorq import get_preset

get_preset("strong-jury").seated_efforts()
# {'anthropic/claude-opus-5-5': 'medium', 'openai/gpt-6.1-sol': 'medium', 'google-ai/gemini-3.8-flash': 'medium'}
```

Seats are ranked on their Artificial Analysis score at that default effort. Where a card has no score at its default, it ranks on its headline index, which carries no reasoning level: `tencent/deepseek-v4.1-flash` runs at `high` while Artificial Analysis measures it only at `max` and `none`, so it ranks on its `max` headline although it runs lower, and `wafer/GLM-5.3-Flash`'s card carries a headline with no rungs at all. `openai/gpt-6.1-terra` has no score in the catalog, so it is priced and ranked from its `MANUAL_MODEL_CARDS` entry. Gemini 4 Hydrogen is listed by Google as a stable Gemini 4 model, but stays out of the panel until its live card carries an Artificial Analysis score; Gemini 4 Carbon is scored and waits on a reviewer decision about which preset takes it.

`llm_jury()` takes one `reasoning_effort` for the whole panel, so passing one overrides every seat's default. Per-judge call settings are a schema change and a separate ticket.

Two more limits worth knowing before you quote a number:

- Agreement between these panels and human raters is inherited from the literature, not measured on orq data. There is no human-agreement baseline behind any of these presets.
- The 500 output tokens are an assumption, not a measurement, and a floor rather than a typical figure: a judge that reasons without being asked writes more than that and costs more than the table.

## How a preset stays current

Seats are chosen in this package. `evaluatorq.common.model_frontier` reads the keyless public `/v2/model-catalog`, where scored cards carry Artificial Analysis intelligence scores per reasoning effort beside their rates, and ranks each model at its default effort on a 3:1 blended price. Compare seats within each lineage, because lineage diversity is the point of a panel: comparing seats against the whole catalog and choosing only the cheapest vendor would recreate the correlated errors the panel exists to cancel.

Nothing about a seat is committed except the decision to seat it. Rates and default efforts are read live, so there is no rate table to refresh; the exceptions are seats the catalog lists without a score, which are ranked from the dated `MANUAL_MODEL_CARDS` register. A weekly workflow (`.github/workflows/preset-drift-weekly.yml`) runs the integration checks in `tests/integration/test_preset_freshness.py` against the live catalog; a failure leaves that run red in Actions and opens a pull request (see below). Each failure names the seat and the register to update:

- a seat the catalog no longer serves
- a repricing that moves a published figure in the table above
- a newer model in a seated lineage that nobody has reviewed (`REVIEWED_SUCCESSORS`)
- a same-lineage model that beats a seat on both intelligence and price
- a seat older than 180 days whose lineage has stopped shipping, unless it has a written waiver in `STALE_SEAT_WAIVERS`
- a seat whose card has no score, so no upgrade check can run against it, unless it has a written waiver in `UNSCORED_SEAT_WAIVERS` (a reserve kept for cost, for example)

When the check fails, the workflow runs `python -m evaluatorq.preset_drift`, which lists what moved and the options the arithmetic allows: the reserve that can replace a retired seat, the same-lineage models that beat a seat, the newer models nobody has reviewed. A Pi agent, launched through the orq CLI with the router key, then proposes the register changes. It must confirm on the vendor's own pages that a candidate really is newer or superseded, because the catalog knows listing dates and not release notes, and it records each source beside the change. The workflow re-runs the live check on the result and always opens a real pull request: a fix when the check passes again, otherwise one labelled `needs-human-attention` that carries the report and the agent's summary. Nothing merges automatically. The pull request is opened with the repository's default token, which GitHub does not treat as a person: its CI runs may wait as `action_required` until a maintainer approves them, or until someone pushes to the branch.

The public catalog lists the whole platform catalog whatever key you hold, so the checks need no key; `/v2/models` would list only what one workspace or project enabled. Routes the router serves that the catalog does not price or score are recorded in `MANUAL_MODEL_CARDS`, each with its source and date.

A preset that is not in `PRESETS` still owes you a reason. `DROPPED_PRESETS` records one that shipped and was then retired, `WITHHELD_PRESETS` one that is derived and costed but not seated because a seat cannot do the job yet, and asking for either by name raises an error carrying that reason rather than a bare "unknown preset".

`value-trio` is the one entry in `DROPPED_PRESETS`. It was a budget panel whose measured cost ran far above its published figure, because two of its judges wrote long prose verdicts. Asking for it by name raises an error carrying that reason rather than a bare "unknown preset".

## Router pools

`evaluatorq.router_pools` derives candidate pools for the autorouter from the same live garden. A router calls one model per request rather than every member, so a pool is chosen for value under a constraint rather than for independent errors: each pool is three models on the Pareto frontier of what the constraint allows, with adjacent price tiers at least 2.75x apart. Membership is derived on first use, never committed.

```python
from evaluatorq.router_pools import get_pools, get_underivable_pools

pools = get_pools()  # reads the public /v2/model-catalog; no key needed
pools["EU Region Only"].models
get_underivable_pools()  # constraint name -> why no pool could be seated today
```

A pool publishes a cost range rather than a figure, at the same 1,500 input and 500 output tokens: what a request costs depends on the routing policy that picks within the pool. `DROPPED_POOLS` records the shapes that were retired and why, and `REVIEWED_UNDERIVABLE_POOLS` the ones the current catalog cannot seat.
