# LLM as a Jury

A single judge model is a single point of failure. It can be noisy from one call to the next, and it can be biased toward outputs from its own provider family. The jury (or panel of judges) replaces that one judge with several, runs them together, aggregates their verdicts into one decision, and reports how much they agreed.

You can use a jury two ways: as a general evaluator in `evaluatorq()` through `llm_jury()`, or inside red teaming through `EvaluatorConfig`. Both share the same panel machinery.

## When to use it

- The evaluation is high stakes and you want a verdict that does not rest on one model's opinion.
- You are judging outputs from the same provider as your usual judge and want to avoid a judge grading its own family.
- You want a quantitative signal for how much your judges actually agree, so you know when a verdict is solid and when it is contested.

A single judge is cheaper and faster. Reach for a jury when the cost of a wrong verdict outweighs the extra calls. A single-judge panel runs with no aggregation overhead, so a jury is purely additive.

## Quick start

`llm_jury()` builds an evaluator you drop into the `evaluators=[...]` list of `evaluatorq()`. Give it two or more `judges` and it becomes a jury:

```python
import asyncio

from evaluatorq import DataPoint, evaluatorq, llm_jury

correctness = llm_jury(
    name="correctness",
    criteria="The answer is factually correct and directly answers the question.",
    judges=[
        "anthropic/claude-sonnet-4-6",
        "google/gemini-2.5-pro",
        "mistral/mistral-large-2411",
    ],
)


async def answer(data: DataPoint, _row: int) -> dict:
    # Your system under test produces the output to be judged.
    return {"name": "qa", "output": "Paris is the capital of France."}


async def main() -> None:
    await evaluatorq(
        "qa-eval",
        data=[DataPoint(inputs={"question": "What is the capital of France?"})],
        jobs=[answer],
        evaluators=[correctness],
    )


asyncio.run(main())
```

`llm_jury(model="x")` is shorthand for `judges=["x"]` — a single judge, the classic LLM-as-a-judge. Pass `judges=[...]` with two or more models to turn it into a jury.

--8<-- "docs/_snippets/panel-tip.md"

## Verdict modes

`verdict_kind` (with `labels`) decides what each judge returns and how `passed` is set. It is not inferred from `labels` — pick the mode explicitly.

| Mode | Configure it with | Judge returns | `passed` is |
| --- | --- | --- | --- |
| **Boolean** (default) | `verdict_kind="categorical"`, no `labels` | `true` / `false` | the boolean itself |
| **Labeled** | `verdict_kind="categorical"`, `labels=[...]` | one of `labels` | `verdict in passing_labels` (`None` if `passing_labels` omitted) |
| **Numeric** | `verdict_kind="numeric"` | a float in `score_range` | `score >= threshold` |

```python
# Labeled: a fixed rubric, only some labels pass
tone = llm_jury(
    name="tone",
    criteria="Rate the tone of the reply.",
    judges=["anthropic/claude-sonnet-4-6", "google/gemini-2.5-pro"],
    labels=["rude", "neutral", "friendly"],
    passing_labels=["neutral", "friendly"],
)

# Numeric: a 0-1 score with a pass threshold
helpfulness = llm_jury(
    name="helpfulness",
    criteria="Score how helpful the answer is, 0 to 1.",
    judges=["anthropic/claude-sonnet-4-6", "google/gemini-2.5-pro"],
    verdict_kind="numeric",
    threshold=0.7,
)
```

`labels`/`passing_labels` are valid only for `categorical`; passing them with `numeric` raises `ValueError`. In labeled mode `passing_labels` must be a subset of `labels`; omit it and the verdict is still recorded but `passed` is `None`.

## Panel configuration

| Argument | Default | What it does |
| --- | --- | --- |
| `judges` | — | Judge model IDs. Two or more makes it a jury. Mutually exclusive with `model`. |
| `model` | — | Single-judge shorthand for `judges=[model]`. |
| `repetitions` | `1` | How many times each judge is asked. The judge takes its own majority before the panel votes, which smooths per-call noise. |
| `assignment` | `"all"` | How judges are allocated across datapoints. `"all"` runs every judge on every datapoint. `"cyclic"` runs exactly one judge per datapoint, rotating through the panel (see below). |
| `replacement_judges` | `None` | Stand-in models called only when a configured judge fails mechanically. |
| `min_successful_judges` | `1` | Minimum decisive judges required, otherwise the verdict is **inconclusive**. Must not exceed the panel size. |
| `threshold` | `0.5` | Numeric mode: `passed` when `score >= threshold`. |
| `structured_output` | `True` | Use the provider's structured-output API; falls back to a schema-injected `json_object` call for models that reject it. |

### Cyclic assignment (CyclicJudge)

`assignment="cyclic"` gives each datapoint exactly one judge, rotating through the panel so every judge covers an equal share of the run. Judge bias still cancels in expectation across the dataset, but the run costs the same as a single-judge evaluation instead of `len(panel)` times as much.

```python
jury = llm_jury(
    name="quality",
    criteria="Is the answer helpful and correct?",
    judges=["openai/gpt-5.4-mini", "openai/gpt-5.4-nano", "deepseek/deepseek-v4-flash"],
    assignment="cyclic",
)
```

Use it for run-level scores (a benchmark mean, a pass rate); keep the default `"all"` when an individual verdict has to stand on its own, since each per-item verdict under `"cyclic"` is one judge's opinion and `stats`/`raw_agreement` come back `None`.

See [Cyclic judge assignment](cyclic-judge.md) for how items map to judges, auditing the rotation via `raw_output["jury"]`, and the failure semantics.

## How the verdict is decided

1. **Each judge votes.** With `repetitions > 1` a judge is asked several times and reduces its own passes to one vote first (plurality for categorical, mean or median for numeric).
2. **Failures pull in replacements.** For every configured judge that fails mechanically, one model from `replacement_judges` stands in, up to the number of failures.
3. **The panel aggregates.** Categorical verdicts are decided by plurality vote; numeric verdicts by mean or median.
4. **Thresholds and ties apply.** If fewer than `min_successful_judges` return a usable verdict, the result is **inconclusive**.

A judge can also **abstain**: it returns cleanly but declines to choose. An abstention is not a failure and does not trigger a replacement, but it is excluded from the decisive tally.

## Reading the output

`llm_jury()` returns a standard evaluator, so each result carries the aggregated verdict in `value`, the pass/fail in `passed`, and a human-readable panel breakdown (who voted what, how close it was) appended to `explanation`:

```python
results = await evaluatorq(..., evaluators=[correctness])
for r in results:
    for job in r.job_results:
        for score in job.evaluator_scores:
            print(score.evaluator_name, score.score.value, score.score.pass_)
            print(score.score.explanation)  # includes the per-judge jury summary
```

## In red teaming

Red teaming reaches the same panel through `EvaluatorConfig`, where the verdict is the categorical RESISTANT/VULNERABLE case (`passed=True` means RESISTANT):

```python
from evaluatorq.redteam import EvaluatorConfig, LLMConfig, OpenAIModelTarget, red_team

report = await red_team(
    target=OpenAIModelTarget(model="gpt-5.6-luna"),
    llm_config=LLMConfig(
        evaluator=EvaluatorConfig(
            judges=[
                "anthropic/claude-sonnet-4-6",
                "google/gemini-2.5-pro",
                "mistral/mistral-large-2411",
            ],
            min_successful_judges=2,
            strict_panel=True,  # refuse a judge that shares the target's family
        ),
    ),
    mode="dynamic",
    categories=["LLM01"],
    max_dynamic_datapoints=3,
)
```

!!! note "`strict_panel` only fires on a same-family judge"
    `strict_panel=True` raises `ValueError` when any judge shares the target's provider family — same-family self-judging can bias the verdict toward the target's own provider. The panel above is entirely cross-family against an OpenAI target, so the guard passes silently (the healthy case). It would raise only if you added an in-family judge such as `"openai/gpt-5.6-luna"`.

!!! warning "`min_successful_judges` vs. `min_evaluation_coverage` — two different levels"
    `min_successful_judges` above is a **per-attack** quorum: it decides whether *this one* jury panel produced enough decisive votes to reach a verdict for *this one* attack. `EvaluatorConfig` also has `min_evaluation_coverage` (default `0.8`), which is a separate, **run-level** floor: the fraction of *all* attacks in the run that must get any verdict at all. A `min_successful_judges` miss on one attack is exactly what produces one of the unevaluated attacks that `min_evaluation_coverage` counts against. Missing the run-level floor makes `eq redteam run` exit `1` — see [Red Teaming › In CI](guides/red-teaming.md#in-ci).

`EvaluatorConfig` adds `strict_panel` (turn panel-composition warnings into hard errors) and surfaces a per-attack `jury` breakdown plus a run-level reliability statistic:

```python
for result in report.results:
    jury = result.evaluation.jury if result.evaluation else None
    if jury is None:
        continue
    print(f"{jury.judges_succeeded}/{jury.judges_configured} judges, agreement {jury.raw_agreement}")
    for vote in jury.votes:
        print(vote.model, vote.value, vote.abstained, vote.error)

reliability = report.summary.jury_reliability
if reliability:
    print(reliability.krippendorff_alpha)  # 1.0 = perfect, ~0 = chance, <0 = systematic disagreement
```

## Reliability, in short

For red-team runs, `raw_agreement` tells you how lopsided one vote was, and Krippendorff's alpha on the run tells you whether your judges agree more than they would by chance:

- `1.0` is perfect agreement.
- around `0` is chance level, so the panel is not adding signal.
- below `0` is systematic disagreement, which usually means the judges are reading the rubric differently and the prompt or panel needs another look.

It is `None` when undefined, for example a single-judge run or fewer than two multi-judge samples.

## Endpoint and retries

`llm_jury()` and `PairwiseComparator` always send judge calls to the Orq router's `responses` endpoint (`api='responses'`) — the endpoint the router prices, so a jury verdict records cost the same way the call it judges does. There is no config knob to opt out; `run_judge` handles the cases that cannot use it on its own, falling back to Chat Completions for a client that does not route through the Orq router, a model the catalogue does not qualify for Responses, or a model that 400s on the endpoint.

Retries happen at the `run_judge` layer, not the SDK's: with a default budget of one retry (`LLMCallConfig.retry_count=1`, i.e. up to two requests per judge call), `run_judge` disarms the SDK-level retry budget (`max_retries=0`) on whichever client it is given — the one either helper builds for itself when no `client=` is passed in, or a client you pass in yourself — so the two retry layers never stack. There is no way to keep an injected client's own SDK retries active alongside `run_judge`'s.

## Reasoning effort on a jury

`llm_jury()`, `llm_jury_pairwise()` and `PairwiseComparator` take a `reasoning_effort=` argument that pins the effort on the *judge* model:

```python
from evaluatorq import llm_jury

jury = llm_jury(
    name='helpfulness',
    criteria='Is the answer helpful and correct?',
    judges=['openai/gpt-5.6-luna', 'anthropic/claude-sonnet-5'],
    reasoning_effort='high',
)
```

Because these judges send to the `responses` endpoint, the effort renders as a `reasoning` block — unless `run_judge` falls back to Chat Completions for one of the reasons above, in which case it renders as a flat `reasoning_effort` field. Either way the provider is the authority on which values it accepts.

This is a distinct knob from red teaming's `target_reasoning_effort` (the agent *under test*), from `LLMCallConfig.reasoning_effort` on red teaming's own `attacker=` / `evaluator=` roles, and from `EVALUATORQ_REASONING_EFFORT` (the simulator's user-simulator and judge). See [Tuning](tuning.md) for the full disambiguation.

## Provider options

`llm_jury()`, `llm_jury_pairwise()` and `PairwiseComparator` take both injection seams, and they are not interchangeable:

```python
jury = llm_jury(
    name='helpfulness',
    criteria='Is the answer helpful?',
    judges=['openai/gpt-5.6-luna'],
    extra_kwargs={'top_p': 0.9},          # top-level call argument
    extra_body={'my_router_field': 'x'},  # request body field
)
```

`extra_kwargs` sets arguments on the SDK call and **replaces** the key. `extra_body` adds fields to the request body and is **merged** per key, so router-owned body fields survive alongside yours.

!!! warning "`extra_body` inside `extra_kwargs` is rejected"
    `extra_body` is one of the structural fields the call site owns, so passing it through `extra_kwargs` raises — and because a judge failure is caught and turned into a verdict, the symptom is a judge that always fails rather than a crash. Use the `extra_body=` parameter.

## Full example

A complete red-teaming script covering repetitions, replacements, the `min_successful_judges` threshold, `strict_panel`, and reading the per-result and run-level output lives at [`examples/redteam/16_llm_as_a_jury.py`](https://github.com/orq-ai/evaluatorq/blob/main/examples/redteam/16_llm_as_a_jury.py).

## Where to next

- **[Pairwise Judging](pairwise-judging.md)** — compare two responses instead of scoring one.
- **[Custom Evaluators & Frameworks](custom-evaluators-and-frameworks.md)** — define your own evaluators.
- **[Red Teaming](guides/red-teaming.md)** — use a jury as the red-team verdict.
