# Structured Results

Some evaluators do not produce one number. A quality rubric has several axes, a
safety check has a score per category, a sentiment breakdown is a distribution.
`EvaluationResultCell` lets an evaluator return all of them from a single call
instead of splitting one judgement across several evaluators.

```python
class EvaluationResultCell(BaseModel):
    type: str                                  # your label: "rubric", "safety", ...
    value: dict[str, EvaluationResultCellValue]
```

## Multi-criteria rubric

```python
from evaluatorq import DataPoint, EvaluationResult, EvaluationResultCell, evaluatorq, job


@job("echo")
async def echo_job(data: DataPoint, row: int):
    return data.inputs["text"]


async def rubric_scorer(params):
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
```

## Sentiment distribution

```python
async def sentiment_scorer(params):
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

Structured scores combine with the `pass_` flag, so a run can carry a
per-category breakdown *and* gate CI on the worst category:

```python
async def safety_scorer(params):
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

See [Evaluation Reference](evaluation-reference.md#passfail-and-ci) for how
`pass_` drives the process exit code.

!!! note "Display vs. storage"
    Structured results render as `[structured]` in the terminal table — the
    breakdown does not fit a cell. The full value is preserved in what is sent
    to the Orq platform and written to OpenTelemetry spans.

## Runnable examples

- [`structured_rubric_eval.py`](https://github.com/orq-ai/evaluatorq/blob/main/examples/lib/structured/structured_rubric_eval.py) — multi-criteria quality rubric
- [`structured_sentiment_eval.py`](https://github.com/orq-ai/evaluatorq/blob/main/examples/lib/structured/structured_sentiment_eval.py) — sentiment distribution breakdown
- [`structured_safety_eval.py`](https://github.com/orq-ai/evaluatorq/blob/main/examples/lib/structured/structured_safety_eval.py) — safety scores with pass/fail tracking
