# Structured Results

`EvaluationResultCell` is an evaluator result that carries several named sub-scores instead of one number. A quality rubric has axes, a safety check has a score per category, a sentiment breakdown is a distribution — a cell returns all of them from a single evaluator.

Reach for it when one judgement genuinely has axes and you want them kept together. Not when you have two independent judgements: separate evaluators each returning a float get a row each in the summary table with an average per job, and a cell does not — see [What structured results do not do](#what-structured-results-do-not-do) before you choose.

```python
class EvaluationResultCell(BaseModel):
    type: str                                  # your label: "rubric", "safety", ...
    value: dict[str, EvaluationResultCellValue]
```

`type` is free-form and uninterpreted — nothing in evaluatorq branches on it. It travels with the result so you can tell cells apart when you read them back.

A sub-score is `str | int | float`, or a `dict` — and that dict's values may themselves be dicts of scalars. **Not `bool`, and not a list.** A list raises a validation error, and `True` is silently coerced to `1` — build a per-category pass map out of booleans and you get integers back with nothing to tell you why. Two more edges of the same kind: an `int` below the top level comes back as a `float` (`{"a": {"b": 2}}` reads back `{"a": {"b": 2.0}}`), and one dict layer beyond that raises — `{"a": {"b": {"c": 2}}}` is accepted, `{"a": {"b": {"c": {"d": 2}}}}` is not.

## A complete evaluation

Job, scorer, evaluator, run. This is the whole program — no API key needed, since the scorer here is plain Python rather than an LLM judge.

```python
import asyncio

from evaluatorq import (
    DataPoint,
    EvaluationResult,
    EvaluationResultCell,
    Evaluator,
    ScorerParameter,
    evaluatorq,
    job,
)


@job("echo")
async def echo_job(data: DataPoint, _row: int) -> str:
    return str(data.inputs["text"])


async def rubric_scorer(params: ScorerParameter) -> EvaluationResult:
    text = str(params["output"])
    return EvaluationResult(
        value=EvaluationResultCell(
            type="rubric",
            value={
                "relevance": min(len(text) / 100, 1),
                "coherence": 0.9 if "." in text else 0.4,
                "fluency": 0.85 if len(text.split()) > 5 else 0.5,
            },
        ),
        explanation="Multi-criteria quality rubric",
    )


rubric_evaluator: Evaluator = {"name": "rubric", "scorer": rubric_scorer}


async def run():
    return await evaluatorq(
        "structured-rubric",
        data=[DataPoint(inputs={"text": "The quick brown fox jumps over the lazy dog."})],
        jobs=[echo_job],
        evaluators=[rubric_evaluator],
    )


if __name__ == "__main__":
    results = asyncio.run(run())
```

The other two examples on this page are scorers only. Give each one its own `Evaluator` dict and add it to `evaluators=`.

## Reading the sub-scores back

The terminal table will not show you these numbers (see below), so reading them in code is the normal way to get at them. The value is nested twice: `EvaluatorScore.score` is an `EvaluationResult`, its `.value` is the cell, and the cell's `.value` is your dict.

Append this to the script above, after `results = asyncio.run(run())`:

```python
for result in results:
    for job_result in result.job_results or []:
        for evaluator_score in job_result.evaluator_scores or []:
            cell = evaluator_score.score.value
            if not isinstance(cell, EvaluationResultCell):
                continue
            print(evaluator_score.evaluator_name, cell.type, cell.value["relevance"])
```

Both guards are load-bearing, not tidiness. A datapoint that failed *before* its jobs ran has `job_results=None`, and a job that raised has an empty `evaluator_scores` — the `or []` covers both. The `isinstance` check is what keeps this loop alive once you add a second evaluator: a plain float scorer puts a `float` there, and a scorer that *raised* puts an empty string there with the exception on `evaluator_score.error` — both would crash the `cell.type` access.

## Sentiment distribution

```python
async def sentiment_scorer(params: ScorerParameter) -> EvaluationResult:
    text = str(params["output"]).lower()
    positive_words = ["good", "great", "excellent", "happy", "love"]
    negative_words = ["bad", "terrible", "awful", "sad", "hate"]
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)
    total = max(pos_count + neg_count, 1)

    return EvaluationResult(
        value=EvaluationResultCell(
            type="sentiment",
            value={
                "positive": pos_count / total,
                "negative": neg_count / total,
                "neutral": 1 - (pos_count + neg_count) / total,
            },
        ),
        explanation="Sentiment distribution across categories",
    )
```

## Safety scores with pass/fail

Structured scores combine with the `pass_` flag, so a run can carry a per-category breakdown *and* gate CI on the worst category:

```python
async def safety_scorer(params: ScorerParameter) -> EvaluationResult:
    text = str(params["output"]).lower()
    categories = {
        "hate_speech": 0.8 if "hate" in text else 0.1,
        "violence": 0.7 if ("kill" in text or "fight" in text) else 0.05,
        "profanity": 0.5 if "damn" in text else 0.02,
    }

    return EvaluationResult(
        value=EvaluationResultCell(type="safety", value=categories),
        pass_=all(score < 0.5 for score in categories.values()),
        explanation="Content safety severity scores per category",
    )
```

See [Evaluation Reference](evaluation-reference.md#passfail-and-ci) for how to inspect `pass_` and add an explicit process exit in a CI script.

The keyword checks here keep the examples runnable offline. Real axes usually come from an LLM judge, and you assemble the cell yourself: call the model in your scorer, then put the parsed axes into `EvaluationResultCell`. [LLM as a Jury](llm-as-a-jury.md) does not do this for you — a jury returns one aggregated verdict, not a cell.

## What structured results do not do

**Sub-scores are never aggregated.** The summary table renders a structured cell as the literal `[structured]` and stops there: no per-key mean, no per-key trend, no per-datapoint breakdown anywhere in the terminal output. Three float evaluators give you three averages; one cell with three keys gives you a placeholder. The `pass_` flag is the only aggregated signal a structured evaluator contributes to a CI gate, which is why the safety example sets it — the summary table shows `[structured]` either way.

The full value is preserved outside the terminal — in the results object above, and in the payload uploaded to the Orq platform when `ORQ_API_KEY` is set. With no key set, nothing is uploaded and nothing is said about it; the run is otherwise identical.

**On OpenTelemetry spans the cell can disappear, and it is `evaluator_type` that decides.** An evaluator that does not set it — every evaluator on this page — gets exactly one span copy of the score, in the `orq.score` attribute, and Orq ingestion drops that attribute whole past 512 characters rather than truncating it. A rubric with ten keys and long names clears 512 easily, and nothing logs the loss. Declare the kind to get the second copy, which is routed to blob storage and so is not subject to that 512-character attribute cap:

```python
rubric_evaluator: Evaluator = {
    "name": "rubric",
    "scorer": rubric_scorer,
    "evaluator_type": "python_eval",
}
```

## Runnable examples

- [`structured_rubric_eval.py`](examples/lib/structured/structured_rubric_eval.md) — multi-criteria quality rubric
- [`structured_sentiment_eval.py`](examples/lib/structured/structured_sentiment_eval.md) — sentiment distribution breakdown
- [`structured_safety_eval.py`](examples/lib/structured/structured_safety_eval.md) — safety scores with pass/fail tracking

Next: [Evaluation Reference](evaluation-reference.md#custom-evaluators) for the scorer contract and the rest of the run mechanics.
