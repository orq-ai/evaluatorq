# Jury Presets

A jury is only as good as its panel, and assembling one by hand means picking judges, checking they do not share a training lineage, keeping an odd number of them, and re-checking all of it every time a provider ships something new. A preset is that decision made once and kept current: `preset="Balanced Trio"` seats three judges from three families, sets the aggregation rule, and requires the whole panel to vote.

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
| **Strong Jury** | `anthropic/claude-opus-5`<br>`openai/gpt-5.6-sol`<br>`google/gemini-3.6-flash` | majority | 23.62 | `deepseek/deepseek-v4-pro` |
| **Cheap Aggregate** | `openai/gpt-5.6-luna`<br>`google/gemini-3.5-flash-lite`<br>`xai/grok-4-1-fast`<br>`deepseek/deepseek-v4-flash`<br>`minimax/MiniMax-M2.7` | 3 of 5 majority | 2.56 | `zai/glm-5.2` |
| **Open-Weight / Portable** | `deepseek/deepseek-v4-pro`<br>`baseten/kimi-k3`<br>`zai/glm-5.2` | majority | 10.29 | `minimax/MiniMax-M2.7` |
| **EU Region** | `aws/eu.anthropic.claude-haiku-4-5-20251001-v1:0`<br>`google/eu.gemini-3.5-flash`<br>`azure/eu.gpt-5.6-luna` | majority | 6.55 | `google/eu.claude-sonnet-5` |
| **Single-Provider Trio** | `openai/gpt-5.6-sol`<br>`openai/gpt-5.6-terra`<br>`openai/gpt-5.6-luna` | majority | 14.28 | `openai/gpt-5.4-nano` |

## Which one to pick

**Balanced Trio** is the default and the right answer for most subjective evaluation. Three lineages means three different ways of being wrong, which is the whole reason to run a panel instead of one judge three times.

**Strong Jury** costs five times as much and is for verdicts that compound: customer-facing benchmarks, preference data you will train on, anything where a wrong label outlives the run that produced it.

**Cheap Aggregate** is the volume option. Five judges across five lineages, concluding on three of five, at half the price of the default trio. Use it for nightly regression runs and for feeding a human-review queue, where the panel's job is to sort rather than to decide.

**Open-Weight / Portable** buys independence from any closed frontier vendor, and a migration path if you later want to serve the judges yourself. It is not the cheap option: it costs twice the default trio, because the open-weight cards that would make it cheaper are dominated by models already seated elsewhere.

**EU Region** seats every judge on an EU endpoint. It is not Balanced Trio relocated: the DeepSeek seat has no EU endpoint newer than v3.1, so the panel is Anthropic, Google and OpenAI, which is what the EU catalog can field at the current generation.

**Single-Provider Trio** exists for workspaces locked to one provider contract. It is the honest version of that constraint rather than a recommendation: three OpenAI tiers correlate, and an OpenAI-generated output faces a self-preference risk the panel cannot vote away.

## What a preset fills in

Naming a preset seats its judges, so passing `judges=` or `model=` alongside it is a contradiction rather than an override and raises. Two other things are defaults you can still overrule:

- `aggregator` becomes `"majority"`, a strict majority of the decisive votes. On the five-seat panel that is exactly three of five.
- `min_successful_judges` becomes a majority of the seats, the same count the aggregation rule needs to be decisive: two of three, three of five. Not one, because a preset publishes a cost and an agreement story about a panel and a lone surviving judge would keep the name while changing what it means. Not the whole panel either, or a single unreachable judge would void items the rest of the panel agreed on.

Presets are pointwise panels, one call per judge per item, so `assignment="cyclic"` is rejected: a rotation scores each item with a single judge and leaves no panel to agree. Pairwise comparison is not part of any preset either; use `llm_jury_pairwise()` with an explicit judge list.

## What a preset does not carry

Each seat is ranked and costed at a particular reasoning effort, which `JuryPreset.seated_efforts()` reports:

```python
from evaluatorq import get_preset

get_preset("Strong Jury").seated_efforts()
# {'anthropic/claude-opus-5': 'high', 'openai/gpt-5.6-sol': 'max', 'google/gemini-3.6-flash': 'high'}
```

`llm_jury()` takes one `reasoning_effort` for the whole panel, so a preset whose seats disagree cannot express itself through it. Per-judge call settings are a schema change and a separate ticket. Until then a panel run at the provider defaults is being run at an operating point it was not costed at, which is why the published figures are a floor rather than an estimate.

Two more limits worth knowing before you quote a number:

- Agreement between these panels and human raters is inherited from the literature, not measured on orq data. There is no human-agreement baseline behind any of these presets.
- Measured spend runs above the table when a judge reasons without being asked to. Gemini has been observed spending several hundred unrequested reasoning tokens on a one-sentence verdict, and a probe put Balanced Trio 25% over its own figure.

## How a preset stays current

Seats are derived rather than hand-listed. Each one is the best buy within its own lineage on the two axes every autorouter-eligible model card carries, an Artificial Analysis intelligence index and a 3:1 blended price, read from a committed snapshot of the model garden. Lineage diversity is the point of a panel, so seats are compared in-family: judged against the whole garden, most seats look dominated, and acting on that would collapse every panel onto whichever vendor is cheapest this month and recreate exactly the correlated errors a panel exists to cancel.

Four ways a preset goes stale without anyone editing it, and the test that catches each:

1. A judge is retired, caught by asserting every seat against the snapshot's served list.
2. A provider reprices, caught by recomputing every published cost from the captured rates, with an opt-in integration test that compares those rates against the live catalog.
3. A newer sibling ships, caught by the in-family upgrade check.
4. The lineage stops shipping, which none of the first three can see because they all need something new to exist first. A seat older than 180 days fails unless the code carries a written reason for keeping it.

Anything the arithmetic cannot settle is written down rather than passing quietly. `REVIEWED_SUCCESSORS` records a newer model deliberately not seated, `AGED_SEATS` records a seat knowingly held past the age limit, and `DROPPED_PRESETS` records a preset that shipped and was then retired. An entry that stops being true fails the suite.

Value Trio is the one entry in that last register so far. It shipped as the budget panel, and was retired when a repricing left Cheap Aggregate cheaper on paper with five judges to its three, while a probe measured Value Trio at 83% over its own published figure. Asking for it by name raises an error carrying that reason rather than a bare "unknown preset".
