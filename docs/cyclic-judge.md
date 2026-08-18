# Cyclic judge assignment (CyclicJudge)

A jury normally runs every judge on every datapoint. That buys a per-item
consensus, and it costs `len(panel)` judge calls per item. If what you actually
report is a run-level number — a benchmark mean, a pass rate, a regression delta
between two prompts — you are paying for per-item confidence you never read.

`assignment="cyclic"` implements CyclicJudge
([arXiv:2603.01865](https://arxiv.org/abs/2603.01865)): give each datapoint one
judge, and rotate through the panel so every judge covers an equal share of the
run. Individual judge bias still cancels across the dataset, because each judge
scored the same fraction of it — but the run costs the same as a single-judge
evaluation.

```python
from evaluatorq import llm_jury

jury = llm_jury(
    name="quality",
    criteria="Is the answer helpful and correct?",
    judges=["openai/gpt-5.4-mini", "openai/gpt-5.4-nano", "deepseek/deepseek-v4-flash"],
    assignment="cyclic",
)
```

Everything else about the jury is unchanged — `criteria`, verdict modes,
aggregators, replacements. `assignment` only decides *who scores what*, so it
composes with the vote-level settings rather than competing with them.

## Which to pick

| | `assignment="all"` (default) | `assignment="cyclic"` |
| --- | --- | --- |
| Judge calls per item | `len(panel)` | 1 |
| Per-item verdict | panel consensus | one judge's opinion |
| Per-item `stats` / `raw_agreement` | populated | `None` |
| Run-level score | panel mean | panel mean, in expectation |
| Use it for | reviewing individual rows, gating a release on one verdict | benchmark means, pass rates, A/B deltas over many rows |

The rule of thumb: **if you would act on a single row's verdict, use `"all"`.**
If you would only ever act on the aggregate, `"cyclic"` gets you the same
aggregate for a third of the money on a three-judge panel.

Cyclic does not remove bias the whole panel shares. It cancels the offsets
*between* your judges; if every judge in the panel is lenient about the same
thing, so is the rotation. That still comes down to picking a diverse panel.

## How items are assigned

Inside `evaluatorq()` the assignment is keyed on the **dataset row**: datapoint
`i` goes to judge `i % len(panel)`. Datapoint 0 gets judge 0, datapoint 1 gets
judge 1, wrapping around.

Keying on the row rather than on call order matters. The runner evaluates
datapoints concurrently, so "the next judge in the rotation" would otherwise mean
"whichever judge call happened to arrive first" — non-reproducible between two
identical runs. Because the key is the row:

- the mapping is identical at `datapoint_parallelism=1` and `datapoint_parallelism=32`,
- re-running the same dataset assigns the same judges again,
- reusing one `llm_jury(...)` object across several runs does not shift it.

In a multi-job run every job sees the same row index, so datapoint `i` is scored
by the same judge under each job. That is deliberate: when you compare job A
against job B, judge identity is not a confound.

!!! note "Shuffle first if row order is meaningful"

    Rotation over a dataset sorted by category lines judge 0 up with category 0.
    Shuffle before evaluating so the cycle cannot align with a latent grouping —
    this is the paper's own caveat.

### The arrival-order fallback

Calling a scorer directly, outside `evaluatorq()`, provides no row index. Those
calls fall back to a cursor that lives on the evaluator object and hands out
judges in arrival order. There, only the equal-share balance is guaranteed, not
which judge sees which item, and a reused evaluator continues the rotation where
the previous run left off.

`PairwiseComparator.compare` has no dataset row at all, so it is always
arrival-ordered. See [Pairwise judging](pairwise-judging.md).

## Auditing the rotation

Under `assignment="cyclic"`, every result carries the full jury record —
per-judge votes, model IDs, verdicts — under `raw_output["jury"]`. It is the only
record of which judge scored which item, so it is what you reach for to check the
rotation actually balanced, or to find a judge that has gone off the rails.

This is cyclic-only. Under `"all"` the panel itself is the record, so the payload
would be redundant: `raw_output` stays `None` and results keep the shape they had
before cyclic existed.

```python
from collections import Counter

from evaluatorq.contracts import JURY_RAW_OUTPUT_KEY

shares = Counter()
for result in results:
    for job_result in result.job_results or []:
        for score in job_result.evaluator_scores or []:
            jury_record = (score.score.raw_output or {}).get(JURY_RAW_OUTPUT_KEY, {})
            shares.update(vote["model"] for vote in jury_record.get("votes", []))

print(shares)  # each judge should hold roughly 1/len(panel) of the run
```

For pairwise runs, `build_report` already rolls per-judge rates up for you into
`PairwiseReport.per_judge`.

## What changes per item

- **`stats` and `raw_agreement` are `None`.** One vote has no cross-judge
  agreement to report, and emitting `1.0` / `std=0.0` would be indistinguishable
  from a genuinely unanimous panel. Summaries render `raw agreement n/a`.
- **`repetitions` still applies to the assigned judge.** `repetitions=3` means
  three calls to *one* judge, smoothing that judge's own call noise — never three
  judges. It multiplies the cost accordingly.
- **Rotation runs over the deduplicated panel.** `judges=["a", "a", "b"]` gives
  `a` two votes per item under `"all"`, but an equal share under `"cyclic"`.
  Listing a judge twice to up-weight it only works under `"all"`.

## Failures

`min_successful_judges` must stay `1` — only one judge runs per item, so a higher
floor could never be met, and passing one raises `ValueError`.

When the assigned judge fails mechanically:

- If `replacement_judges` is set, a stand-in is promoted and casts a real vote.
- Otherwise the item comes back **inconclusive**. The rest of the run still has
  redundancy, since other items are scored by other judges, so one item's outage
  does not kill the run.
- The exception is a single-model cyclic panel with no stand-ins. There is no
  redundancy left to fall back on, so the error propagates loudly rather than
  quietly marking every item inconclusive.

One caveat worth knowing: a judge that is *systematically* broken — a bad key for
one provider, a prompt its API rejects — shows up under cyclic as scattered
inconclusive items across `1/len(panel)` of the run, rather than as the loud
collapse in agreement you would see under `"all"`. Check the share counts from
the audit snippet above if a run comes back with more inconclusive rows than you
expect.

## Full example

[Cyclic Jury](examples/lib/basics/cyclic_jury.md) scores one dataset twice with
the same three-judge panel — once with `"all"`, once with `"cyclic"` — and prints
the item-to-judge mapping and total judge calls for each.

## See also

- [LLM as a jury](llm-as-a-jury.md) — panel configuration, verdict modes, aggregators
- [Pairwise judging](pairwise-judging.md) — `assignment` on `llm_jury_pairwise`
