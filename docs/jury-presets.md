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

Costs are USD per 1,000 pointwise items at 1,500 input and 150 output tokens, uncached, one call per judge per item. Every figure is recomputed from the committed model garden snapshot in the test suite, so the table below cannot drift away from what the code seats.

| Preset | Judges | Aggregation | $ / 1k | Reserve |
| --- | --- | --- | --- | --- |
| **Balanced Trio** (default) | `deepseek/deepseek-v4-pro`<br>`openai/gpt-5.6-luna`<br>`google/gemini-3.6-flash` | majority | 4.64 | `deepseek/deepseek-v4-flash` |
| **Strong Jury** | `anthropic/claude-opus-5`<br>`openai/gpt-5.6-sol`<br>`google/gemini-3.6-flash` | majority | 23.63 | `deepseek/deepseek-v4-pro` |
| **Open-Weight / Portable** | `deepseek/deepseek-v4-pro`<br>`baseten/kimi-k3`<br>`zai/glm-5.2` | majority | 10.29 | `minimax/MiniMax-M2.7` |
| **EU Region** | `aws/eu.anthropic.claude-haiku-4-5-20251001-v1:0`<br>`google/eu.gemini-3.5-flash`<br>`azure/eu.gpt-5.6-luna` | majority | 6.56 | `google/eu.claude-sonnet-5` |
| **Single-Provider Trio** | `openai/gpt-5.6-sol`<br>`openai/gpt-5.6-terra`<br>`openai/gpt-5.6-luna` | majority | 14.28 | `openai/gpt-5.4-nano` |

## Which one to pick

**Balanced Trio** is the default and the right answer for most subjective evaluation. Three lineages means three different ways of being wrong, which is the whole reason to run a panel instead of one judge three times.

**Strong Jury** costs five times as much and is for verdicts that compound: customer-facing benchmarks, preference data you will train on, anything where a wrong label outlives the run that produced it.

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

Each seat is ranked and costed at a particular reasoning effort, which `JuryPreset.seated_efforts()` reports:

```python
from evaluatorq import get_preset

get_preset("Strong Jury").seated_efforts()
# {'anthropic/claude-opus-5': 'high', 'openai/gpt-5.6-sol': 'max', 'google/gemini-3.6-flash': 'high'}
```

Those are the rungs the cards were scored at rather than values to send. A card that only distinguishes thinking from not thinking is scored at `reasoning` or `none`, and no provider accepts either as a `reasoning_effort`, so read the mapping as a report of where a panel was ranked and priced. `llm_jury()` takes one `reasoning_effort` for the whole panel, so a preset whose seats disagree cannot express itself through it. Per-judge call settings are a schema change and a separate ticket. Until then a panel run at the provider defaults is being run at an operating point it was not costed at, which is why the published figures are a floor rather than an estimate.

Three more limits worth knowing before you quote a number:

- Agreement between these panels and human raters is inherited from the literature, not measured on orq data. There is no human-agreement baseline behind any of these presets.
- Some seats are costed at a cheaper rung than the one they are seated at, which `JuryPreset.priced_below_seated_effort()` lists. The captured blend for those judges is the price at the rung the probe measured, so the published $/1k understates them until someone re-probes the panel at its seated effort. Disclosed rather than corrected, because correcting it is a re-probe of every panel and not an arithmetic fix.
- Measured spend runs above the table when a judge reasons without being asked to. Gemini has been observed spending several hundred unrequested reasoning tokens on a one-sentence verdict, and a probe put Balanced Trio 25% over its own figure.

## How a preset stays current

Seats are derived rather than hand-listed. Each one is the best buy within its own lineage on the two axes every autorouter-eligible model card carries, an Artificial Analysis intelligence index and a 3:1 blended price. Lineage diversity is the point of a panel, so seats are compared in-family: judged against the whole garden, most seats look dominated, and acting on that would collapse every panel onto whichever vendor is cheapest this month and recreate exactly the correlated errors a panel exists to cancel.

The derivation is not in this package. It reads a capture of the whole model garden and is rerun, reviewed and argued in the research repo that owns the capture; what ships here is its output, the panels and the rates they were costed at, in `common/data/jury_judge_rates.json`. That file records what each seated judge bills, the reasoning rung it was ranked at, and when the rates were captured.

What this package does hold you to is that the published table matches the code: every figure in it is recomputed from those captured rates on every test run, so a repricing fails CI instead of quietly making the docs wrong. An opt-in integration test (`ORQ_API_KEY` plus `RUN_PRICING_DRIFT=1`) goes further and compares the captured rates against the live catalog, which is how a repricing gets noticed in the first place. Judging whether a seat has been overtaken, retired or left behind by its lineage is a question about the whole garden, so it is answered where the whole garden lives.

A preset that is not in `PRESETS` still owes you a reason. `DROPPED_PRESETS` records one that shipped and was then retired, `WITHHELD_PRESETS` one that is derived and costed but not seated because a seat cannot do the job yet, and asking for either by name raises an error carrying that reason rather than a bare "unknown preset".

Cheap Aggregate is the entry in `WITHHELD_PRESETS`. It is the five-judge volume panel at $2.56 per 1k and the derivation stands, but `minimax/MiniMax-M2.7` answers a `json_schema` response format with prose, and the fallback that would resend the schema as instructions only fires when a provider rejects the request outright, not when a 200 comes back the wrong shape. Four voting seats is an even panel, which is the one shape the validator exists to reject, and it slips through only because the validator counts declared seats rather than voting ones. Seating the named reserve does not rescue it either: `zai/glm-5.2` costs $2.76 per 1k for that seat alone and takes the panel past the default trio, which is the reason the panel exists gone. It ships when the judge path falls back on an unparseable 200.

Value Trio is the one entry in `DROPPED_PRESETS` so far. It shipped as the budget panel, and was retired when a repricing left Cheap Aggregate cheaper on paper with five judges to its three, while a probe measured Value Trio at 83% over its own published figure. Asking for it by name raises an error carrying that reason rather than a bare "unknown preset".
