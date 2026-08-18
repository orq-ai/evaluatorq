# Pairwise (Preference) Judging

Some questions are easier to answer by comparison than in isolation. Instead of
asking "is this answer good?" you ask "is A better than B?". Pairwise judging
runs a panel of judges over two responses and reconciles their picks into one
winner, correcting for the position bias that makes a judge favour whichever
response it happens to see first.

It is a sibling of [LLM as a Jury](llm-as-a-jury.md): same panel machinery, same
judge models, but the verdict is a preference (`A` / `B` / `tie`) rather than a
pass or a score.

## When to use it

- You are comparing two systems, prompts, or model versions and want a direct
  A-vs-B preference rather than two separate absolute scores.
- Absolute grading is hard to calibrate but "which one is better" is clear.
- You want the position bias measured and corrected instead of hoping it washes
  out.

## Quick start

`llm_jury_pairwise()` builds a reusable comparator. Call `compare()` once per
A/B pair:

```python
import asyncio

from evaluatorq import build_report, llm_jury_pairwise

comparator = llm_jury_pairwise(
    criteria="The answer is accurate, complete, and directly addresses the question.",
    judges=[
        "anthropic/claude-sonnet-4-6",
        "google/gemini-2.5-pro",
        "openai/gpt-5.4-mini",
    ],
)


async def main() -> None:
    comparison = await comparator.compare(
        question="What is the capital of France?",
        response_a="The capital of France is Paris.",
        response_b="The capital of France is Berlin.",
    )
    print(comparison.winner)  # "A"

    # Roll many comparisons up into headline and per-judge metrics.
    report = build_report([comparison])
    print(report.a_win_rate, report.inconclusive_rate)


asyncio.run(main())
```

`llm_jury_pairwise(model="x")` is the single-judge shorthand for `judges=["x"]`.
Pass two or more `judges` to get a panel.

--8<-- "docs/_snippets/panel-tip.md"

## Swap and reconcile: how a vote is decided

A judge shown response A first and response B second may lean toward the first
slot regardless of content. Pairwise judging controls for this by running every
judge **twice**: once as (A, B) and once as (B, A). The second ordering is
un-swapped back into the canonical A/B frame, and the two verdicts are
reconciled into a single vote:

| First ordering | Second ordering (un-swapped) | Reconciled vote | Flipped |
| --- | --- | --- | --- |
| `A` | `A` | `A` | no |
| `tie` | `tie` | `tie` | no |
| `A` | `B` | abstains (`None`) | **yes** |
| `A` | `tie` | abstains (`None`) | **yes** |
| `A` | missing / failed | abstains (`None`) | no |

A judge that agrees with itself across both orderings casts that vote. A judge
that contradicts itself has no real preference: it **flips**, abstains from the
tally, and the flip is recorded as position bias. The comparison winner is the
plurality of the reconciled votes, or `inconclusive` when no side reaches a
plurality or too few judges cast a decisive vote.

Both orderings run concurrently, so swapping does not add wall-clock latency,
only cost. Set `swap=False` to run a single ordering when you have already
controlled for position another way; the position-bias metric is then
unavailable (no second ordering to disagree with).

## Panel configuration

`llm_jury_pairwise()` mirrors [`llm_jury()`](llm-as-a-jury.md#panel-configuration):

| Argument | Default | What it does |
| --- | --- | --- |
| `judges` | — | Judge model IDs. Two or more makes it a panel. Mutually exclusive with `model`. |
| `model` | — | Single-judge shorthand for `judges=[model]`. |
| `criteria` | a general quality rubric | What "better" means. An empty string falls back to the default rubric. |
| `swap` | `True` | Run both orderings and reconcile. Turn off to skip position-bias correction. |
| `repetitions` | `1` | How many times each judge is asked per ordering; the judge takes its own majority first. |
| `assignment` | `"all"` | `"cyclic"` gives each comparison exactly one judge, rotating through the panel ([CyclicJudge](cyclic-judge.md)). Judge bias cancels in expectation across the run at single-judge cost; the assigned judge still runs both orderings when `swap` is on. Rotation is over the deduplicated panel, `repetitions` still applies to the one assigned judge, and the cursor lives on the comparator: a reused comparator continues where the previous run stopped. |
| `replacement_judges` | `None` | Stand-ins for judges that fail mechanically. Promoted per pair and run in **both** orderings, so a stand-in casts a real reconciled vote. |
| `min_successful_judges` | `1` | Minimum decisive reconciled votes, otherwise the comparison is **inconclusive**. Must not exceed the panel size. |
| `max_concurrency` | `None` | Cap on total in-flight judge LLM calls across all concurrently running `compare()` calls (each pair fans out judges × orderings × repetitions). Unbounded when unset. |

## Reading a comparison

`compare()` returns a `PairwiseComparison`:

```python
comparison.winner        # "A" | "B" | "tie" | "inconclusive"
comparison.token_usage   # summed across both orderings and any replacements

for vote in comparison.votes:
    vote.model        # judge model ID
    vote.vote         # reconciled "A" | "B" | "tie" | None (abstained)
    vote.flipped      # True if the judge contradicted itself across orderings
    vote.completed    # True if both orderings were decisive, so a flip was possible
    vote.replacement  # True if this judge stood in for a failed one
    vote.explanation  # rationale from the ordering that produced the vote
```

## Rolling up many comparisons

`build_report()` aggregates a list of comparisons into a `PairwiseReport`:

```python
report = build_report(comparisons)

report.comparisons        # how many went in
report.a_win_rate         # A consensus wins over comparisons decided A or B
report.b_win_rate         # B consensus wins over comparisons decided A or B
report.tie_rate           # consensus ties over all comparisons
report.inconclusive_rate  # comparisons the panel could not decide, over all comparisons
report.mean_agreement     # mean inter-judge agreement (comparisons with >=2 decisive votes)

for judge in report.per_judge:
    judge.model          # judge model ID
    judge.a_rate         # share of its decisive picks that went to A
    judge.b_rate         # share of its decisive picks that went to B
    judge.position_bias  # flips over pairs where a flip was possible
    judge.tie_rate       # ties over all comparisons the judge saw
```

!!! note "Watch `inconclusive_rate` alongside the win rates"
    The win rates are computed over decided comparisons only, so a run that was
    mostly noise can still show a high `a_win_rate`. Read it together with
    `inconclusive_rate`: a healthy result is a high win rate **and** a low
    inconclusive rate. `mean_agreement` ignores comparisons with a single
    decisive vote, since one lone voter always "agrees" with itself and would
    otherwise flatter a degraded panel.

## Reliability-weighted aggregation (BT-sigma)

The default consensus treats every judge's vote equally. When your panel mixes
judges of different quality (say a frontier model next to a small open-weight
one), uniform plurality lets the noisy judges outvote the sharp one. BT-sigma
(from "Who can we trust? LLM-as-a-jury for Comparative Assessment",
[arXiv:2602.16610](https://arxiv.org/abs/2602.16610)) fixes this without any
labels: it fits a Bradley-Terry model with a per-judge discriminator over the
run's own reconciled votes, learning which judges are internally consistent and
down-weighting the rest.

```python
from evaluatorq import build_report

report = build_report(comparisons, aggregation="bt-sigma")

report.bt_sigma.p_a_beats_b     # fitted global probability that A beats B
report.bt_sigma.judge_sigmas    # per-judge discriminator, smaller = more reliable
report.bt_sigma.winners         # reliability-weighted winner per comparison
report.bt_sigma.a_win_rate      # weighted rollup, next to the plurality one
```

The headline plurality rates in the report are unchanged, so the two
aggregations stay directly comparable, and each `JudgeStats` entry gains its
fitted `sigma`. The fit is a regularized, unsupervised maximum-likelihood fit on
the votes the run already collected: no extra LLM calls or training data.
Identical votes are collapsed into weighted counts before the fit (at most
three distinct judgements per judge in the A/B setting), so the cost stays flat
no matter how many comparisons the run holds. The report exposes fit warnings.
When the optimizer does not converge it falls back to uniform plurality **only
without repetition weights**; on a repetition run (below) the winners stay
consistency-weighted and only the pooled `p_a_beats_b` headline degrades to
neutral. Do not treat a capped fit as a reliability estimate.

Notes worth knowing:

- Reconciliation already symmetrises position bias (every judge votes in both
  orderings), which is a requirement of the model, so votes feed the fit as-is.
- With a single judge the discriminator is unidentifiable; the fit falls back
  to plain Bradley-Terry and says so in `fit_warnings`.
- A perfectly split panel stays inconclusive rather than letting numerical
  noise crown one judge reliable.
- A judge whose decisive votes are unanimous (always A, or always B) is
  excluded from the sigma weighting. With only two items such a judge's sigma
  measures one-sidedness, not reliability, and `1/sigma` would hand the most
  degenerate judge on the panel an unbounded weight - the exact shape a
  position- or verbosity-biased judge takes. On the pooled-fit path it votes
  with a neutral (median) weight instead; on the repetition path every weight
  comes from consistency, so that neutral assignment is replaced by the judge's
  own consistency weight. Either way `fit_warnings` names it.
- Check `bt_sigma.converged` (and `fit_warnings`) before trusting sigmas: a
  fit that stopped at the iteration cap still produces numbers.
- Like all unsupervised aggregation, BT-sigma rewards internal consistency. A
  majority of judges sharing the same systematic bias will still carry the
  vote; it protects against noisy judges, not coordinated ones.

### Repetition-aware reliability

Run each comparison more than once (`repetitions=2` or more) and the reliability
weights change source. Instead of the global two-item fit, they come from how
often each judge **agrees with itself** on repeated passes of the same prompt.

Two things must be true, and if either is missing the run falls back to the
two-item fit and says so in `fit_warnings`:

- **Both orderings must run (`swap=True`, the default).** Self-agreement is
  measured inside one ordering, so a single-ordering run (`swap=False`) never
  reaches this path, even at `repetitions=2`.
- **At least two judges must have repeated decisive passes.** With only one, the
  fallback weight for every other judge is just that one judge's number, so we
  do not use it.

Every pass is kept on `PairwiseVote.observations`, normalized so a swapped-order
'B' is stored as 'A' and the two orderings line up; abstained and failed passes
are kept as `None`.

**Why self-agreement, not the global fit.** With one pass per judge, the
two-item fit cannot tell a noisy judge apart from a hard batch of questions.
Repeating the *same* prompt removes the ambiguity: any disagreement is the judge
being inconsistent, nothing else. So a judge's consistency is scored on each
prompt, counted once per judge per datapoint, then averaged. Because of that:

- Different questions are never compared against each other, so a hard batch
  cannot look like an unreliable judge.
- Position bias is not mistaken for inconsistency, since each ordering is its
  own group.
- Extra repeats sharpen a judge's score but never add weight: each judge still
  casts one weighted vote per comparison.

**Two numbers, and which to read.** `bt_sigma.repetition_consistency` is the
reliability **weight**. It is pulled toward the panel average (so one lucky
2-pass agreement cannot take over the run) and lowered when passes fail, so even
a perfectly steady judge reads a little under 1.0 unless the whole panel is
steady. `bt_sigma.repetition_consistency_raw` (and the `Consistency (raw)` report
column) is the **plain** self-agreement, with no adjustment: 1.0 means the judge
always agreed with itself. Read the raw number to compare judges inside one run;
do not compare the weight across two runs with different panels, where the same
judge can move without changing. `p_a_beats_b` is still the pooled run-level
headline. A judge with no repeats gets the median of the measured weights and is
named in `fit_warnings`. Runs saved before this feature existed still load and
behave as before.

**What consistency is not.** It is self-agreement under fixed conditions, not
task difficulty, not judge quality (a judge can be steadily wrong), and not
accuracy against a ground truth. A clean abstention does not count against a
judge: `['A', <abstention>, 'A']` scores 1.0 in both numbers. A pass that errored
or came back off-contract is a failure, not a free abstention, and is counted in
`repetition_failures`. The judge still agreed with itself on the passes it
finished, so the **raw** number for `['A', None, 'A']` stays **1.0**; only the
**weight** drops, to **2/3**, because a flaky judge is less dependable even when
it agrees with itself. The two `None`s look identical in the vote list; only
`repetition_failures` tells a clean abstention from a broken pass.

**Cost.** Calls scale linearly with judges x orderings x repetitions, so `R=2`
doubles the calls per comparison. Wall-clock barely moves, since the passes run
in parallel, and each pass adds one small record. Use `R=2` when reliability
weighting matters: it is the smallest R that produces any consistency evidence.
`R=3` only refines the score, for 50% more cost.

For ranking more than two candidates, use `evaluatorq.ranking.fit_bt()`
directly: it takes item pairs from any number of judges and returns skills, a
ranking, and per-judge reliability, with `cycle_rate()` as the matching
consistency diagnostic. `cycle_rate` does not apply to the two-item case above,
since two fixed items have no cycles to measure.

## Saving a run and viewing it in the dashboard

`build_report()` gives you the numbers in memory. To keep a run and read it in
the dashboard, collect the comparisons into a `PairwiseRun` and save it:

```python
from evaluatorq.pairwise_run import new_run

run = new_run(
    run_name="prompt-v2 vs prompt-v3",
    label_a="prompt-v2",
    label_b="prompt-v3",
    judges=["anthropic/claude-sonnet-4-6", "google/gemini-2.5-pro"],
    criteria="The answer is accurate, complete, and directly addresses the question.",
)

for question, response_a, response_b in my_pairs:
    comparison = await comparator.compare(
        question=question, response_a=response_a, response_b=response_b
    )
    run.add(comparison, question=question, response_a=response_a, response_b=response_b)

run.save()  # -> .evaluatorq/pairwise-runs/<timestamp>_prompt-v2-vs-prompt-v3.json
```

`save()` rolls the comparisons up with `build_report()` and stores the result on
the run, so the dashboard never recomputes it. Pass a path to choose the file
yourself; the default lands in the pairwise run store, where `eq dashboard`
discovers it.

`label_a` and `label_b` name the two systems being compared. They default to
`"A"` and `"B"`, but nothing in the judging data records what was in each slot,
so a reader of the dashboard cannot tell what "A won" means. Set them.

A run also records `swap`. Position bias is only meaningful when both orderings
ran, so a run saved with `swap=False` shows that column as unavailable rather
than as a flattering `0.00`.

The dashboard renders the run as three sections: the consensus win rates for
each side, a per-judge table (win rates, tie rate, position bias), and the
comparison list, where each row expands to show the two responses side by side
with every judge's vote and rationale.

## The lower-level core

`run_pairwise()` is the ordering-independent engine underneath the comparator.
It takes any async `judge_fn(first, second, model)` rather than building LLM
calls itself, so you can drive the swap-and-reconcile logic with your own judge.
`reconcile_pair()` and `pairwise_consensus()` are exposed for the same reason.
Most callers want `llm_jury_pairwise()`; reach for `run_pairwise()` when you are
plugging in a non-LLM judge or testing the reconciliation directly.

All three live in `evaluatorq.pairwise` — not `evaluatorq.pairwise_run`, which
holds the run-persistence helpers used above:

```python
from evaluatorq.pairwise import pairwise_consensus, reconcile_pair, run_pairwise
```

`run_pairwise()` is also re-exported at the top level as `evaluatorq.run_pairwise`;
`reconcile_pair()` and `pairwise_consensus()` are not. `run_jury()` is internal to
`evaluatorq.common.jury` and is not part of the public top-level API.

Both `run_pairwise()` and the shared `run_jury()` accept `max_concurrency` as an
int or an existing `asyncio.Semaphore`; pass the same semaphore to several runs
to bound their combined fan-out with one budget.

## Where to next

- **[LLM as a Jury](llm-as-a-jury.md)** — score a single response with a judge panel.
- **[Custom Evaluators & Frameworks](custom-evaluators-and-frameworks.md)** — define your own evaluators.
- **[Red Teaming](guides/red-teaming.md)** — the red-team workflow judging plugs into.
