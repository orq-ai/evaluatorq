# Jury Presets

A jury is only as good as its panel, and assembling one by hand means picking judges, checking they do not share a training lineage, keeping an odd number of them, and re-checking all of it every time a provider ships something new. A preset is that decision made once and kept current: `preset="Balanced Trio"` seats three judges from three families, sets the aggregation rule, and requires a majority of them to return a verdict.

```python
from evaluatorq import evaluatorq, llm_jury

correctness = llm_jury(
    name="correctness",
    criteria="The answer is factually correct and directly answers the question.",
    preset="Balanced Trio",
)
```

## The presets

Costs are USD per 1,000 pointwise items at 1,500 input and 1,500 output tokens, uncached, one call per judge per item. The token counts are a flat assumption, the same for every seat whatever its reasoning effort. Every figure is recomputed from the committed model garden snapshot in the test suite, so the table below cannot drift away from what the code seats.

| Preset | Judges | Aggregation | $ / 1k | Reserve |
| --- | --- | --- | --- | --- |
| **Balanced Trio** (default) | `deepseek/deepseek-v4-pro`<br>`openai/gpt-6-luna`<br>`google/gemini-3.6-flash` | majority | 16.36 | `deepseek/deepseek-flash` |
| **Strong Jury** | `anthropic/claude-opus-5-5`<br>`openai/gpt-5.6-sol`<br>`google/gemini-3.6-flash` | majority | 85.50 | `deepseek/deepseek-v4-pro` |
| **Open-Weight / Portable** | `deepseek/deepseek-v4-pro`<br>`wafer/Kimi-K3`<br>`zai/glm-5.2` | majority | 34.28 | `minimax/MiniMax-M2.7` |
| **EU Region** | `aws/eu.anthropic.claude-haiku-4-5-20251001-v1:0`<br>`google/eu.gemini-3.5-flash`<br>`azure/eu.gpt-5.6-luna` | majority | 27.75 | `google/eu.claude-sonnet-5` |
| **Single-Provider Trio** | `openai/gpt-5.6-sol`<br>`openai/gpt-5.6-terra`<br>`openai/gpt-6-luna` | majority | 57.90 | `openai/gpt-5.4-nano` |

Balanced Trio and Single-Provider Trio use GPT-6 Luna at its `medium` default. On the same Artificial Analysis Intelligence Index v4.3.2, [GPT-6 Luna (Medium)](https://artificialanalysis.ai/models/gpt-6-luna-medium) scores 29.93 against [GPT-5.6 Luna (Medium)](https://artificialanalysis.ai/models/gpt-5-6-luna-medium)'s 25.04, while Orq's captured rates fall from $0.20/$1.20 to $0.10/$0.50 per million input/output tokens. Do not compare these scores with the older index values still present on some catalogue cards. The EU preset keeps its EU-hosted GPT-5.6 Luna seat; the US-hosted successor is not a residency-compatible replacement.

## Which one to pick

**Balanced Trio** is the default and the right answer for most subjective evaluation. Three lineages means three different ways of being wrong, which is the whole reason to run a panel instead of one judge three times.

**Strong Jury** costs nearly five times as much and is for verdicts that compound: customer-facing benchmarks, preference data you will train on, anything where a wrong label outlives the run that produced it.

**Open-Weight / Portable** buys independence from any closed frontier vendor, and a migration path if you later want to serve the judges yourself. It is not the cheap option: it costs twice the default trio, because the open-weight cards that would make it cheaper are dominated by models already seated elsewhere.

**EU Region** seats every judge on an EU endpoint. It is not Balanced Trio relocated: the DeepSeek seat has no EU endpoint newer than v3.1, so the panel is Anthropic, Google and OpenAI, which is what the EU catalog can field at the current generation.

**Single-Provider Trio** exists for workspaces locked to one provider contract. It is the honest version of that constraint rather than a recommendation: three OpenAI tiers correlate, and an OpenAI-generated output faces a self-preference risk the panel cannot vote away.

## What a preset fills in

Naming a preset seats its judges, so passing `judges=` or `model=` alongside it is a contradiction rather than an override and raises. Two other things are defaults you can still overrule:

- `aggregator` becomes `"majority"`, a strict majority of the votes that actually arrived. On a trio where all three answer that is two of three; where only two answer it is both of them.
- `min_successful_judges` becomes a majority of the seats: two of three, three of five. Read it as a floor on how many judges have to answer at all, not as the threshold the aggregator applies. The two are different numbers on the same panel, and meeting the quorum does not guarantee a verdict: a trio that answers 1-1 clears a quorum of two and still returns nothing, because neither vote is more than half. Not one, because a preset publishes a cost and an agreement story about a panel and a lone surviving judge would keep the name while changing what it means. Not the whole panel either, or a single unreachable judge would void items the rest of the panel agreed on.

A run that does not reach that quorum is not a silent pass. `JuryResult` reports the panel it actually got: `judges_configured` against `judges_succeeded` shows a short panel, `tie` is set when the decisive votes split evenly and a caller tie policy broke them, and `inconclusive` is set when too few judges answered to conclude at all, with `stats` and `raw_agreement` left as `None`. A preset does not add verdict states of its own; it seats the panel and the jury runner reports what came back.

Each preset also names `reserve_judges`, the model that takes a seat when its occupant is retired. That is a maintenance record the freshness tests read, not a runtime failover: a panel whose judge errors on the day does not silently pull in a different lineage and change what the published cost and agreement figures describe.

Presets are pointwise panels, one call per judge per item, so `assignment="cyclic"` is rejected: a rotation scores each item with a single judge and leaves no panel to agree. Pairwise comparison is not part of any preset either; use `llm_jury_pairwise()` with an explicit judge list.

## What a preset does not carry

A preset sends no reasoning effort, so each seat runs at its provider's default. `JuryPreset.seated_efforts()` reports it, with `None` for a model whose catalog entry has no effort setting:

```python
from evaluatorq import get_preset

get_preset("Strong Jury").seated_efforts()
# {'anthropic/claude-opus-5-5': 'medium', 'openai/gpt-5.6-sol': 'medium', 'google/gemini-3.6-flash': 'medium'}
```

Open-Weight / Portable uses `wafer/Kimi-K3` at Wafer's default effort, `none`, instead of `baseten/kimi-k3` at `max`. Wafer bills $3.00 input and $12.75 output per million tokens, versus Baseten's $3.00/$15.00. This is an accepted host-and-effort trade, not a claim that non-reasoning and maximum-reasoning benchmark scores are equal. Both hosts passed a two-answer factual-judging smoke check; that check does not establish equal accuracy on your dataset.

`llm_jury()` takes one `reasoning_effort` for the whole panel, so passing one overrides every seat's default. Per-judge call settings are a schema change and a separate ticket.

Two more limits worth knowing before you quote a number:

- Agreement between these panels and human raters is inherited from the literature, not measured on orq data. There is no human-agreement baseline behind any of these presets.
- The 1,500 output tokens are an assumption, not a measurement. A judge that reasons less than that on your items costs less than the table, and one that reasons more costs more.

## How a preset stays current

Seats are chosen in this package. Compare Artificial Analysis intelligence at the model's default reasoning effort against its price within each lineage, because the default is the effort a preset call runs at. Explicitly reviewed host-and-effort trades can retain a seat without a benchmark at its new default; the Wafer Kimi seat is such a trade, not a measured dominance claim. Lineage diversity is the point of a panel: comparing seats against the whole catalog and choosing only the cheapest vendor would recreate the correlated errors the panel exists to cancel.

`common/data/jury_judge_rates.json` records what each seated judge bills, the default reasoning effort it runs at, and when both were captured. Every published figure is recomputed from those rates on every test run, so a repricing fails CI instead of quietly making the docs wrong. An opt-in integration test (`ORQ_API_KEY` plus `RUN_PRICING_DRIFT=1`) compares the table against the live catalog, which is how a repricing or a changed default gets noticed in the first place.

Waiting for a test run to notice a repricing means waiting for someone to push. `scripts/refresh_jury_judge_rates.py` re-reads each seat's rates and default reasoning effort from the live catalog and rewrites the table, and a monthly workflow runs it and opens a pull request when anything moved. A repricing, a changed default and a seat missing from the catalog are each printed as a decision for a human, and the script refuses to report a whole table of retirements when what actually failed was the key.

A preset that is not in `PRESETS` still owes you a reason. `DROPPED_PRESETS` records one that shipped and was then retired, `WITHHELD_PRESETS` one that is derived and costed but not seated because a seat cannot do the job yet, and asking for either by name raises an error carrying that reason rather than a bare "unknown preset".

Cheap Aggregate is the entry in `WITHHELD_PRESETS`. It is a five-judge volume panel, but `minimax/MiniMax-M2.7` answers a `json_schema` response format with prose, and the fallback that would resend the schema as instructions only fires when a provider rejects the request outright, not when a 200 comes back the wrong shape. Four voting seats is an even panel, which is the one shape the validator exists to reject, and it slips through only because the validator counts declared seats rather than voting ones. Seating the named reserve does not rescue it either: `zai/glm-5.2` costs $8.70 per 1k for that seat alone, which defeats a budget panel. It ships when the judge path falls back on an unparseable 200 as well as on a rejected request.

Value Trio is the one entry in `DROPPED_PRESETS`. It was a budget panel whose measured cost ran far above its published figure, because two of its judges wrote long prose verdicts. Asking for it by name raises an error carrying that reason rather than a bare "unknown preset".
