# Classify judges

A **classify judge** scores structured state by answering a typed question instead of reading a rendered prompt and generating a verdict in text.

Use one when you want a fast, inexpensive vote for a yes/no, fixed-label, ordered-scale or pairwise decision. Use a prompted LLM judge when you need model-written reasoning or a verdict outside those four shapes.

## How a classifier fits into a jury

An evaluatorq **jury** is a panel of judge models whose votes are aggregated into one result. A classify judge occupies an ordinary seat beside prompted LLM judges, but the request and response differ:

| | Prompted LLM judge | Classify judge |
| --- | --- | --- |
| Endpoint | Responses or Chat Completions | Orq router `POST /v3/router/classify` |
| Input | Rendered system and user prompts | Structured state plus one typed question |
| Output | Schema-constrained verdict and written explanation | Probability, choice or score |
| Explanation | Written by the model | Synthesised by evaluatorq from the returned numbers |
| Sampling settings | May use temperature, token budget and reasoning effort | Does not generate text, so those settings do not apply |

The current classify model is `typesafe/jev-latest`. You still name judges with strings; evaluatorq reads the Orq model catalogue at call time and routes models whose `supports_classify` flag is exactly `true` to the classify endpoint.

## Configure a classify seat

The example below constructs a mixed jury without making a network call. Pass `correctness` in `evaluators=[correctness]` as shown in the [jury quick start](llm-as-a-jury.md#quick-start); with the default client, the scoring call needs `ORQ_API_KEY` because the classify endpoint exists only on the Orq router. An injected client that already routes through Orq works too.

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

`criteria` becomes the question the classifier answers. The label descriptions become its choices; the prompted judge receives the same descriptions through its verdict schema and system prompt. `state_fields` selects the material being judged.

## Choose the question shape

The jury configuration determines which classify question evaluatorq builds:

| Jury configuration | Classify question | Result |
| --- | --- | --- |
| `verdict_kind="categorical"`, no `labels` | `noul` probability | `True` when `probability >= threshold`; otherwise `False` |
| `verdict_kind="categorical"`, `labels={...}` | `choice` over the labels | Chosen label; `passing_labels` maps it to pass/fail |
| `verdict_kind="numeric"`, `levels=[...]` | `score` over ordered levels | Raw level-index score divided by `len(levels) - 1`, producing `0.0` to `1.0` |
| `llm_jury_pairwise(...)` | `choice` over `A`, `B` and `tie` | Pairwise winner, reconciled across swapped orderings |

Numeric classify panels require 2 to 10 `levels` and keep `score_range=(0.0, 1.0)`. Boolean classify panels require `threshold` within that same range. These constraints are checked again when a model is first discovered through the catalogue, so a catalogue-only classifier cannot bypass validation.

## What becomes classifier state

The literal prompt is never sent to the classify endpoint. By default, evaluatorq extracts every `{{placeholder}}` from the jury's prompt template, resolves those values for the current datapoint and sends them as structured state. It excludes `criteria` because that text is already the classify question.

This means `prompt` still matters even though the classifier never reads it: its placeholders select the default state fields. Set `state_fields` when you want that selection to be explicit or narrower.

Values keep their types. A message list remains a list rather than becoming an escaped JSON string, and nested values win over the flat strings used when rendering a prompted judge's template. A named path that does not resolve is skipped. If none resolve, evaluatorq warns once per evaluator because the classifier is receiving a question with no material to judge.

## Settings that do not apply

A classify judge does not read `system_prompt`, the literal `prompt`, `temperature`, `structured_output`, `max_tokens`, `reasoning_effort`, `extra_kwargs` or `extra_body`. evaluatorq omits those fields from the classify call and names any non-default values in one warning. On a mixed panel they still apply to any prompted judges.

`max_tokens` is intentionally omitted: the classifier returns numbers rather than generated text, so there is no output-token budget to set.

Classify verdicts are deterministic for the same state and question. `repetitions > 1` therefore warns because it bills duplicate calls. Pairwise `swap=True` is different: it runs two distinct A/B orderings, so swapping remains useful and does not trigger that warning.

## Read the result

The classifier's vote has the same public shape as any other jury vote. evaluatorq synthesises a compact explanation such as `noul=0.5000001 (threshold 0.5)`, `choice='correct' (confidence 0.96)` or `score=1.70/2 → 0.85 (confidence 0.81)`. Values near a decision boundary retain enough decimal places to show which side produced the verdict.

The full probability distribution and confidence are recorded on the judge span as `judge.probabilities` and `judge.confidence`; they are not copied into the vote. Token usage and provider-reported cost contribute to the jury total. When the response has no readable usage block, evaluatorq records one unpriced call instead of making the call disappear from coverage.

## Failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| A classify seat fails before its first classify request | It has no `criteria`, numeric mode has no `levels`, or its threshold/range is invalid. Known classify ids raise `ValueError` at construction; catalogue-only ids fail when first scored | Supply the missing classify fields or remove the classify judge |
| The model is sent as an ordinary chat judge and fails | The client does not route through Orq | Set `ORQ_API_KEY` or use only prompted judges |
| The classifier warns that state is empty | None of the selected paths exists on the datapoint | Correct `state_fields` or add the needed placeholders to the prompt template |
| A custom sampling setting is named as ignored | The classifier cannot use text-generation settings | Keep the setting for prompted seats or remove it on a classifier-only panel |
| Pairwise evaluation makes more calls than expected | `repetitions` duplicates each deterministic ordering; `swap` creates the separate reverse ordering | Keep `swap` for bias correction and leave `repetitions=1` |

## Routing and fallbacks

The fetched model catalogue is authoritative per call. A catalogue entry with `supports_classify=false` keeps even a built-in model on the prompted path. If a model has no catalogue entry because the catalogue is empty or unavailable, evaluatorq has a built-in fallback only for `typesafe/jev-latest` and logs that fallback once.

You do not need to register a model that the fetched catalogue already marks as classify-capable. `register_model()` remains an override for missing or incorrect catalogue metadata and an early construction-time hint; it is not the normal setup path.

For the pre-implementation design exploration and original endpoint diagrams, see the [archived classify-judge design explainer](assets/classify-judge-design-explainer.html). It is intentionally absent from the site navigation because it is a historical design artifact, not the current user contract.
