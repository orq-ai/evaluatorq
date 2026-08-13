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
no matter how many comparisons the run holds. The report exposes fit warnings
and falls back to uniform plurality when the optimizer does not converge; do
not treat a capped fit as a reliability estimate.

Notes worth knowing:

- Reconciliation already symmetrises position bias (every judge votes in both
  orderings), which is a requirement of the model, so votes feed the fit as-is.
- With a single judge the discriminator is unidentifiable; the fit falls back
  to plain Bradley-Terry and says so in `fit_warnings`.
- A perfectly split panel stays inconclusive rather than letting numerical
  noise crown one judge reliable.
- A judge whose decisive votes are unanimous (always A, or always B) is
  excluded from the weighting. With only two items such a judge's sigma
  measures one-sidedness, not reliability, and `1/sigma` would hand the most
  degenerate judge on the panel an unbounded weight - the exact shape a
  position- or verbosity-biased judge takes. It votes with a neutral weight
  instead, and `fit_warnings` names it.
- Check `bt_sigma.converged` (and `fit_warnings`) before trusting sigmas: a
  fit that stopped at the iteration cap still produces numbers.
- Like all unsupervised aggregation, BT-sigma rewards internal consistency. A
  majority of judges sharing the same systematic bias will still carry the
  vote; it protects against noisy judges, not coordinated ones.

### Repetition-aware reliability

Run comparisons with `repetitions=2` (or more) and the reliability weights stop
coming from the global two-item fit and start coming from **within-datapoint
repetition consistency**: how often a judge agrees with itself on repeated
passes of the same prompt. Every raw pass is preserved and canonicalized on
`PairwiseVote.observations` (a swapped-ordering 'B' is recorded as 'A', so
entries are comparable across orderings), including abstained and failed
passes as `None`.

Why this exists: with a single pass per judge, the fitted sigma of the global
two-item collapse mixes judge noise with datapoint heterogeneity when the run
spans different questions or response pairs. Repeated passes of the same
prompt are the one setting where disagreement is unambiguously judge noise, so
consistency is computed per (judge, comparison, ordering) group, averaged to
at most ONE observation per judge per datapoint, then averaged across
datapoints. Consequences by construction:

- Different datapoints are never compared to each other, so heterogeneity
  cannot masquerade as unreliability.
- Position bias does not read as inconsistency (orderings are separate
  groups); the swap/reconcile machinery already handles bias.
- Extra repetitions refine a judge's weight but never multiply its panel
  weight: each judge still casts exactly one weighted vote per comparison.

Read `bt_sigma.repetition_consistency` for the per-judge values (1.0 = always
agrees with itself). When it is non-empty, the winner weights came from these;
`p_a_beats_b` still comes from the pooled fit (a run-level headline, not a
per-judge reliability). Judges without repeated decisive passes vote with the
median weight and are named in `fit_warnings`. Legacy runs saved before
repetition capture load fine and keep the previous global-fit behaviour.

What consistency estimates - and what it must not be read as: it measures a
judge's self-agreement under fixed conditions. It is NOT task difficulty, NOT
overall judge quality (a judge can be consistently wrong), and NOT accuracy
against any ground truth. Abstained or failed passes are excluded from
consistency, so a judge that abstains often is not penalized: `['A', None,
'A']` scores 1.0.

Cost: repetitions multiply judge calls linearly (calls = judges x orderings x
R), so R=2 doubles spend per comparison; wall-clock barely moves because
passes fan out concurrently. Storage adds one small observation record per
pass. R=2 is the recommended default when reliability weighting matters - it
is the minimum that produces consistency evidence; R=3 only refines the
per-group agreement scale and costs 50% more.

For ranking more than two candidates (leaderboard-style), the underlying
`evaluatorq.ranking.fit_bt()` accepts arbitrary item pairs from any number of
judges and returns skills, a ranking, and per-judge reliability; `cycle_rate()`
gives the matching consistency diagnostic. `cycle_rate` is deliberately not
part of the two-item aggregation above: with two fixed items there are no
3-cycles to rate, so it only means something on the multi-item ranking path.

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
