# Decision model judges

A **decision-model judge** is a jury seat held by a decision model: a model served by the Orq router's `/decisions` endpoint, which is never prompted. It scores structured state by answering a typed question instead of reading a rendered prompt and generating a verdict in text.

Use one when you want a fast, inexpensive vote for a yes/no, fixed-label, ordered-scale or pairwise decision. Use a prompted LLM judge when you need model-written reasoning or a verdict outside those four shapes.

If you want typed answers without a panel around them, call the endpoint directly — see [the `/decisions` endpoint](decisions-endpoint.md).

## How a decision model fits into a jury

An evaluatorq **jury** is a panel of judge models whose votes are aggregated into one result. A decision-model judge occupies an ordinary seat beside prompted LLM judges, but the request and response differ:

| | Prompted LLM judge | Decision-model judge |
| --- | --- | --- |
| Endpoint | Responses or Chat Completions | Orq router `POST /v3/router/decisions` |
| Input | Rendered system and user prompts | Structured state plus one typed question |
| Output | Schema-constrained verdict and written explanation | Probability, choice or score |
| Explanation | Written by the model | Synthesised by evaluatorq from the returned numbers |
| Sampling settings | May use temperature, token budget and reasoning effort | Does not generate text, so those settings do not apply |

The current decision model is `typesafe/jev-latest`. You still name judges with strings; evaluatorq reads the Orq model catalogue at call time and routes models whose `supports_classify` flag (the router's name for it) is exactly `true` to the `/decisions` endpoint. `EVALUATORQ_DECISION_MODELS` overrides the catalogue, see [Configuration](configuration.md).

## Configure a decision-model seat

The example below constructs a mixed jury without making a network call. Pass `correctness` in `evaluators=[correctness]` as shown in the [jury quick start](llm-as-a-jury.md#quick-start); with the default client, the scoring call needs `ORQ_API_KEY` because the `/decisions` endpoint exists only on the Orq router. An injected client that already routes through Orq works too.

```python
from evaluatorq import llm_jury

correctness = llm_jury(
    name="correctness",
    criteria="The answer is factually correct and directly answers the question.",
    judges=["openai/gpt-5.6-luna", "typesafe/jev-latest"],
    labels={
        "correct": "every claim is accurate and relevant",
        "incorrect": "at least one claim is wrong, unsupported, or irrelevant",
    },
    passing_labels=["correct"],
    state_fields=["input.all_messages", "output.response", "input.expected_output"],
)

assert correctness["name"] == "correctness"
```

`criteria` becomes the question the decision model answers. The label descriptions become its choices; the prompted judge receives the same descriptions through its verdict schema and system prompt. `state_fields` selects the material being judged.

## Choose the question shape

The jury configuration determines which decision question evaluatorq builds:

| Jury configuration | Decision question | Result |
| --- | --- | --- |
| `verdict_kind="categorical"`, no `labels` | `noul` probability | `True` when `probability >= threshold`; otherwise `False` |
| `verdict_kind="categorical"`, `labels={...}` | `choice` over the labels | Chosen label; `passing_labels` maps it to pass/fail |
| `verdict_kind="numeric"`, `levels=[...]` | `score` over ordered levels | Raw level-index score divided by `len(levels) - 1`, producing `0.0` to `1.0` |
| `llm_jury_pairwise(...)` | `choice` over `A`, `B` and `tie` | Pairwise winner, reconciled across swapped orderings |

Numeric panels that seat a decision model require 2 to 10 `levels` and keep `score_range=(0.0, 1.0)`. Boolean panels that seat one require `threshold` within that same range. These constraints are checked again when a model is first discovered through the catalogue, so a catalogue-only decision model cannot bypass validation.

## What becomes the decision state

The literal prompt is never sent to the `/decisions` endpoint. By default, evaluatorq extracts every `{{placeholder}}` from the jury's prompt template, resolves those values for the current datapoint and sends them as structured state. It excludes `criteria` because that text is already the decision question.

This means `prompt` still matters even though the decision model never reads it: its placeholders select the default state fields. Set `state_fields` when you want that selection to be explicit or narrower.

Values keep their types. A message list remains a list rather than becoming an escaped JSON string, and nested values win over the flat strings used when rendering a prompted judge's template. A named path that does not resolve is skipped. If none resolve, evaluatorq warns once per evaluator because the decision model is receiving a question with no material to judge.

## Settings that do not apply

A decision-model judge does not read `system_prompt`, the literal `prompt`, `temperature`, `structured_output`, `max_tokens`, `reasoning_effort`, `extra_kwargs` or `extra_body`. evaluatorq omits those fields from the `/decisions` call and names any non-default values in one warning. On a mixed panel they still apply to any prompted judges.

`max_tokens` is intentionally omitted: the decision model returns numbers rather than generated text, so there is no output-token budget to set.

Decision-model verdicts are deterministic for the same state and question. `repetitions > 1` therefore warns because it bills duplicate calls. Pairwise `swap=True` is different: it runs two distinct A/B orderings, so swapping remains useful and does not trigger that warning.

## Read the result

The decision model's vote has the same public shape as any other jury vote. evaluatorq synthesises a compact explanation such as `noul=0.5000001 (threshold 0.5)`, `choice='correct' (confidence 0.96)` or `score=1.70/2 → 0.85 (confidence 0.81)`. Values near a decision boundary retain enough decimal places to show which side produced the verdict.

Each successful decision-model repetition keeps its complete validated answer in `JuryRepetition.raw_output`, including provider-added fields. Fields the provider did not report are omitted; prompted judges have `raw_output=None`. These dictionaries survive in returned results and local saved result artifacts, separately for every repetition.

This complete example scores one answer and reads the typed result. The default client needs `ORQ_API_KEY` because the `/decisions` endpoint is on the Orq router:

```python
import asyncio

from evaluatorq import DataPoint, llm_jury
from evaluatorq.contracts import JURY_RAW_OUTPUT_KEY, JuryResult
from evaluatorq.types import ScorerParameter

correctness = llm_jury(
    name="correctness",
    criteria="The answer is factually correct and directly answers the question.",
    judges=["openai/gpt-5.6-luna", "typesafe/jev-latest"],
    labels={
        "correct": "every claim is accurate and relevant",
        "incorrect": "at least one claim is wrong, unsupported, or irrelevant",
    },
    passing_labels=["correct"],
    state_fields=["input.all_messages", "output.response", "input.expected_output"],
)


async def main() -> None:
    result = await correctness["scorer"](
        ScorerParameter(
            data=DataPoint(
                inputs={"question": "What is the capital of France?"},
                expected_output="Paris",
            ),
            output="Paris is the capital of France.",
        )
    )
    assert result.raw_output is not None
    jury = JuryResult.model_validate(result.raw_output[JURY_RAW_OUTPUT_KEY])
    for vote in jury.votes:
        for repetition in vote.repetitions:
            decision_answer = repetition.raw_output
            if decision_answer is not None:
                print(vote.model, decision_answer)  # complete answer, including provider-added fields
                for field in ("choice", "confidence", "probabilities"):
                    if field in decision_answer:
                        print(field, decision_answer[field])


asyncio.run(main())
```

`send_results_to_orq()` strips `raw_output` before upload. The hosted Orq experiment view does not receive these decision-model details; read them from the returned result or a local saved artifact.

Tracing is a second observability surface: the judge span records `judge.confidence` and `judge.probabilities` (a JSON string) when reported. Token usage and provider-reported cost contribute to the jury total. When the response has no readable usage block, evaluatorq records one unpriced call instead of making the call disappear from coverage.

## Failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| A decision-model seat fails before its first `/decisions` request | It has no `criteria`, numeric mode has no `levels`, or its threshold/range is invalid. Known decision-model ids raise `ValueError` at construction; catalogue-only ids fail when first scored | Supply the missing fields or remove the decision-model judge |
| The model is sent as an ordinary chat judge and fails | The client does not route through Orq | Set `ORQ_API_KEY` or use only prompted judges |
| The decision model warns that state is empty | None of the selected paths exists on the datapoint | Correct `state_fields` or add the needed placeholders to the prompt template |
| A custom sampling setting is named as ignored | The decision model cannot use text-generation settings | Keep the setting for prompted seats or remove it on a panel of decision models only |
| A decision-model seat drops out of the tally and nothing is recorded as an error | The model answered `refusal`. That abstains rather than failing, so the seat is excluded from the decisive tally, gets no replacement judge, and a panel where every seat abstained is `inconclusive` with no error payload. The refusal is recorded on the outcome's `error_message`, and the raw reply is logged at warning level | Read the vote's `explanation` (`decision model refused to answer the <kind> question`), then rephrase the question's `instructions` or move the question to a decision model that answers it |
| Pairwise evaluation makes more calls than expected | `repetitions` duplicates each deterministic ordering; `swap` creates the separate reverse ordering | Keep `swap` for bias correction and leave `repetitions=1` |

## Routing and fallbacks

The fetched model catalogue is authoritative per call, after `EVALUATORQ_DECISION_MODELS`. A catalogue entry with `supports_classify=false` keeps even a built-in model on the prompted path. If a model has no catalogue entry because the catalogue is empty or unavailable, evaluatorq has a built-in fallback only for `typesafe/jev-latest` and logs that fallback once.

You do not need to register a model that the fetched catalogue already marks as a decision model. `register_model()` remains an override for missing or incorrect catalogue metadata and an early construction-time hint; it is not the normal setup path.

For the pre-implementation design exploration and original endpoint diagrams, see the [archived decision-model judge design explainer](assets/classify-judge-design-explainer.html). It is intentionally absent from the site navigation because it is a historical design artifact, not the current user contract.
