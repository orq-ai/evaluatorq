# The classify endpoint

`run_classify()` sends one piece of state and several typed questions about it to the Orq router's Decisions endpoint, `POST /v3/router/decisions`, and returns every answer from one round trip. The older `/classify` route is a deprecated alias of it with the same request and response, and evaluatorq no longer calls it.

"Classify" is the name of the operation in evaluatorq's API and in the Orq catalogue and API-key permission; `/decisions` is the route it calls.

Reach for it when you want typed labels over state you already have — a trace, a transcript, a row of a dataset — and you do not want a judge panel around them. If you want a verdict aggregated over several models, with repetitions and abstentions, you want a [jury](llm-as-a-jury.md), and [classify judges](classify-judges.md) is how a classify model takes a seat on one.

`run_classify`, `ClassifyQuestion`, `ClassifyRequest`, `ClassifyOutcome` and `llm_concurrency_limit` import from top-level `evaluatorq` and are the supported surface. Five names on this page do not: `with_retry`, `models_by_provider`, `classify_request_body`, `ClassifyResponse` and `JudgeError` are reachable only under `evaluatorq.common.*`, which carries no stability promise. `ClassifyAnswer` is in the same position — you receive one, so you rarely need to import it.

## Three question kinds

A classify model is never prompted. You hand it the material to judge and a question about it, and `kind` picks the shape of the answer.

| `kind` | `criteria` | Answer field | Range | Use it when |
|---|---|---|---|---|
| `noul` | omitted, or `{"true": ..., "false": ...}` | `noul` | `0.0` to `1.0` | The question is yes or no, and you want the confidence rather than the coin flip. |
| `choice` | `{label: description}`, non-empty | `choice` | one of your labels | The answer is one of a fixed, unordered set. |
| `score` | an ordered list of 2 to 10 level descriptions | `score` | `0.0` to `len(criteria) - 1` | The levels have an order, and the mean between two of them is meaningful. |

A `criteria` shape that disagrees with `kind` raises `pydantic.ValidationError` at construction, before the call is paid for.

## Ask three questions in one call

`/decisions` lives only on the Orq router, so the client's `base_url` has to be the router and `ORQ_API_KEY` has to be set. The key needs the `classify` permission, which the Decisions rename did not change and the standard gateway key preset grants. The gateway has to be an Orq release that serves Decisions (4.16 or later); an older or self-hosted gateway answers 404 and evaluatorq does not fall back to `/classify`.

```bash
export ORQ_API_KEY=...
```

`state` takes a dict, a string or a list, so a transcript goes in as it already is. Every question is answered against the request's single `state`, and comes back under the key you gave it — the keys are yours to choose, not a fixed vocabulary:

```python
import asyncio
import os

from openai import AsyncOpenAI

from evaluatorq import ClassifyQuestion, ClassifyRequest, run_classify
from evaluatorq.contracts import LLMCallConfig

client = AsyncOpenAI(
    api_key=os.environ["ORQ_API_KEY"],
    base_url=f"{os.environ.get('ORQ_BASE_URL', 'https://my.orq.ai')}/v3/router",
)

transcript = [
    {"role": "user", "content": "My order never arrived and I want a refund."},
    {"role": "assistant", "content": "I have issued the refund. It lands in 3 working days."},
    {"role": "user", "content": "Great, thank you."},
]

request = ClassifyRequest(
    state=transcript,
    questions={
        "resolved": ClassifyQuestion(
            kind="noul",
            instructions="The user's problem was resolved.",
            state={},
        ),
        "outcome": ClassifyQuestion(
            kind="choice",
            instructions="How did the conversation end?",
            state={},
            criteria={
                "refunded": "money was returned",
                "escalated": "handed to someone else",
                "abandoned": "the user stopped replying",
            },
        ),
        "politeness": ClassifyQuestion(
            kind="score",
            instructions="How polite was the assistant?",
            state={},
            criteria=["rude", "neutral", "warm"],
        ),
    },
)

outcome = asyncio.run(
    run_classify(
        client=client,
        model="typesafe/jev-latest",
        cfg=LLMCallConfig(timeout_ms=60_000),
        request=request,
    )
)

assert outcome.error_kind is None, outcome.error_message
answers = outcome.response.answers

resolved = answers["resolved"].noul
politeness = answers["politeness"].score / 2  # three levels, so divide by 2 for a 0-to-1 reading

print("resolved:", f"{resolved:.1f}", "->", resolved >= 0.5)
print("outcome:", answers["outcome"].choice)
print("politeness:", f"{politeness:.1f}")
print("score answer carries:", sorted(answers["politeness"].model_dump(exclude_none=True)))
print("priced:", outcome.token_usage.total_cost is not None)
```

```console
resolved: 0.9 -> True
outcome: refunded
politeness: 0.5
score answer carries: ['confidence', 'legend', 'probabilities', 'score', 'type']
priced: True
```

The example rounds and thresholds on purpose. The chosen label and the level a score lands on hold from call to call, but a probability can move by a hundredth between identical calls — `resolved` comes back as `0.91`, `0.92` or `0.93` — so read the label, or compare against a cutoff, and never assert on an exact value.

Three things in that snippet are not obvious and bite later if you change them:

- **`state={}` on every question.** The field is required, and on a `ClassifyRequest` it is also discarded — only the request's `state` is sent. [What this path does not honour](#what-this-path-does-not-honour) shows the wire body.
- **`assert outcome.error_kind is None` rather than a `try`.** `run_classify` returns provider failures instead of raising them, so a `try` around the call would never fire — the `assert` is what turns a failure into a stop.
- **The client carries no `max_retries=0`, and does not need one.** `run_classify` clones it with the SDK budget disarmed before the first attempt, and your own client is never mutated, so the value only governs the *other* calls you make on the same client. [Tuning › One retry layer](tuning.md#pipeline-call-reliability) is where that rule lives.

`ORQ_BASE_URL` is the host, so the snippet appends the router suffix itself. If your `ORQ_BASE_URL` already ends in `/v3/router`, drop the `f`-string and pass it unchanged — appending twice gives a 404.

### What comes back

| Field | What it holds |
|---|---|
| `response.answers` | One `ClassifyAnswer` per question. The keys are the provider's echo of yours; nothing checks parity, so index with `.get()` when it matters. `response` itself is `None` on any failure. |
| `response.model` | Whatever the reply carried, passed through unchanged — nothing normalises it. The current Orq router echoes the qualified id you sent, such as `typesafe/jev-latest`, but that is an observation and not a contract: the library's own test fixtures use a bare `jev-latest`. Do not split it or compare it to a qualified string, and do not read it as proof against a substitution. |
| `token_usage` | Tokens and cost for the one call — but `None` on a `timeout`, `api_connection`, `api_status` or `unknown` failure, and a `calls=1, priced_calls=0` marker with `total_cost=None` when the reply carried no readable usage block. |
| `raw_content` | The validated response as JSON on success, and the raw wire payload on a `parse` failure, which is the case where you most need it. |
| `error_kind` | `None` on success; otherwise one of `timeout`, `parse`, `api_connection`, `api_status`, `unknown`. |
| `error_message` | The provider's text, or the validation error. |
| `error_exc` | The original exception, which is what the retry section below re-raises. `None` on a `parse` failure. |

`ClassifyAnswer` fills only the field matching its `type`, and `probabilities`, `confidence` and `legend` are present only when the provider reported them — a `noul` answer typically carries none of the three. A field evaluatorq does not know is kept rather than dropped.

### Reading each kind

A `noul` answer is a probability and nothing else: no verdict is computed for you, so pick your own cutoff and compare. The library's own jury path defaults to `0.5` (`ClassifyQuestion.noul_threshold`), which is a starting point rather than a measured recommendation.

A `score` answer is on the **level-index scale**, not `0.0` to `1.0`: three levels give `0.0` to `2.0`, so a raw `1.01` means "neutral" and a raw `1.69` means "between neutral and warm, nearer warm". Divide by `len(criteria) - 1` for a `0.0` to `1.0` reading, which is what the jury path stores and what the example above prints.

A `choice` answer's `choice` is one of the labels you supplied, and `probabilities` is keyed by those same labels.

## Which model to send it to

A model serves `/decisions` when its Orq catalogue entry sets `supports_classify` to exactly `true`; the catalogue flag keeps its old name. Being a chat model neither qualifies nor disqualifies it — several models serve both. Routing, the built-in fallback for `typesafe/jev-latest` and the `register_model()` override are covered at [Classify judges › Routing and fallbacks](classify-judges.md#routing-and-fallbacks), and the classifier model role is at [Configuration › Models](configuration.md#models).

To see the current list, which reads the live catalogue and so needs `ORQ_API_KEY` too:

```python
import asyncio

from evaluatorq.common.model_catalogue import models_by_provider

by_provider = asyncio.run(models_by_provider(kind="classify"))
ids = sorted(model_id for ids in by_provider.values() for model_id in ids)

# print(len(ids), ids) for the whole list — its size depends on your own key
print("typesafe/jev-latest available:", "typesafe/jev-latest" in ids)
```

```console
typesafe/jev-latest available: True
```

**The list is not a fixed set.** It is whatever your key's workspace can see, so the count moves between keys — two keys running the call above on the same day returned 24 ids and 59. Check for the id you want rather than relying on a number.

`models_by_provider` returns `{}` rather than raising when the catalogue is unavailable, so flatten it as above instead of indexing a provider key. `typesafe/jev-latest` is the model evaluatorq's own classify judges default to, and the one this page uses.

## What this path does not honour

Six things here do less than they look like they do, and none of them raises. The first two are visible in the body that actually goes over the wire, which you can print without making a call:

```python
from evaluatorq import ClassifyQuestion
from evaluatorq.common.llm_call import classify_request_body

question = ClassifyQuestion(
    kind="noul",
    instructions="The statement is true.",
    state={"ignored": "the question's own state"},
    noul_threshold=0.9,
)

body = classify_request_body("typesafe/jev-latest", {"sent": "the request's state"}, {"verdict": question})
print(body)
assert body["state"] == {"sent": "the request's state"}
assert "noul_threshold" not in body["questions"]["verdict"]
assert "state" not in body["questions"]["verdict"]
```

```console
{'model': 'typesafe/jev-latest', 'state': {'sent': "the request's state"}, 'questions': {'verdict': {'type': 'noul', 'instructions': 'The statement is true.'}}}
```

**`noul_threshold` is not sent, and `run_classify` never reads it.** It belongs to the jury path, where it turns a probability into the boolean a panel counts. A threshold of `0.9` on a question answered `noul=0.6` produces no verdict, no warning and no error — just the `0.6`.

**A question's own `state` is dropped.** `ClassifyQuestion.state` is a required field because the type is shared with the single-question judge path, but on a `ClassifyRequest` only the request's `state` is sent. Pass `state={}` on each question, as the examples here do.

**An answer's `type` is not checked against the question's `kind`.** `run_classify` validates the response against the wire schema and stops there. The jury path rejects a mismatch; this one hands it to you as sent, with the field for the kind you asked about left `None`. Read `answer.type` before you read a value. A native OpenAI model such as `openai/gpt-6-luna` may answer `{"type": "refusal"}` with no other field; `run_classify` returns it as sent, and the jury path records it as an abstention and logs the raw reply at error level.

**`run_classify` never reads `retry_count`.** It makes exactly one attempt. The field is nonetheless in the set the "unread config field" warning counts as read, because its sibling `run_classify_judges` does honour it — so a `retry_count=5` on a direct `run_classify` call gets you neither a retry nor a warning. Set a field neither function reads, such as `temperature`, and you do get one: `run_classify ignores llm_config temperature — this call site owns those. It reads retry_count, timeout_ms.` That warning is deduplicated per process and per field set, so a loop logs it once, not once per call.

**`model` and `client` on the config are dropped in complete silence.** The warning above exempts both, because every call site takes them as required keyword arguments instead — so `LLMCallConfig(model='openai/gpt-6-sol')` alongside `model='typesafe/jev-latest'` is not a contradiction the call reports, it is a field nothing looks at.

**Only an answer keyed `verdict` gets the per-answer span attributes.** `run_classify` reads `answers['verdict']` to set `judge.confidence` and `judge.probabilities` on the span, so a request with several named questions records the call but none of those two attributes. If you are sending one question and want the tracing, name it `verdict`.

The first and third need no network to check, and nor does the claim above that an unknown field survives rather than being dropped:

```python
import inspect

from evaluatorq import run_classify
from evaluatorq.common.judge import ClassifyResponse

assert "noul_threshold" not in inspect.getsource(run_classify)

mismatched = ClassifyResponse.model_validate(
    {
        "answers": {"quality": {"type": "choice", "choice": "good", "house_metric": 0.4}},
        "model": "typesafe/jev-latest",
    }
)
answer = mismatched.answers["quality"]
print("asked for a score, got type:", answer.type, "| score:", answer.score)
print("unknown field kept:", answer.model_extra)
```

```console
asked for a score, got type: choice | score: None
unknown field kept: {'house_metric': 0.4}
```

## Retry, which is yours

`run_classify` **returns** provider failures instead of raising them, so a retry helper wrapped around it does nothing until you re-raise — and a naive wrapper then loses the no-raise contract that made the call safe. The pattern below is the one the library uses internally (`_classify_with_retry` in `evaluatorq/insights/labeling.py`): keep every attempt's outcome, re-raise only a transient one so `with_retry`'s classifier can see it, and hand back the last outcome when the budget is spent, so the function still never raises.

```python
import asyncio
import os

from openai import AsyncOpenAI

from evaluatorq import ClassifyQuestion, ClassifyRequest, run_classify
from evaluatorq.common.judge import JudgeError
from evaluatorq.common.retry import with_retry
from evaluatorq.contracts import LLMCallConfig

RETRYABLE = frozenset({JudgeError.TIMEOUT, JudgeError.API_CONNECTION, JudgeError.API_STATUS})

client = AsyncOpenAI(
    api_key=os.environ["ORQ_API_KEY"],
    base_url=f"{os.environ.get('ORQ_BASE_URL', 'https://my.orq.ai')}/v3/router",
)

request = ClassifyRequest(
    state="Paris is the capital of France.",
    questions={"verdict": ClassifyQuestion(kind="noul", instructions="The statement is true.", state={})},
)


async def classify_with_retry(req):
    attempts = []

    async def attempt():
        outcome = await run_classify(
            client=client,
            model="typesafe/jev-latest",
            cfg=LLMCallConfig(timeout_ms=60_000),
            request=req,
        )
        attempts.append(outcome)
        if outcome.error_kind in RETRYABLE and outcome.error_exc is not None:
            raise outcome.error_exc
        return outcome

    try:
        return await with_retry(attempt, max_attempts=3, label="classify")
    except Exception:
        if not attempts:
            raise
        return attempts[-1]


outcome = asyncio.run(classify_with_retry(request))
# `outcome.token_usage` prices the LAST attempt only; sum `attempts` if you bill retries.
if outcome.error_kind is not None:
    print("failed:", outcome.error_kind.value, "-", outcome.error_message)
else:
    print(outcome.response.answers["verdict"].noul)
```

```console
0.97
```

Three failures never reach a second attempt, and the shape above is why none of them crashes your row:

- **A timeout from `timeout_ms`.** That deadline is enforced with `asyncio.wait_for`, and `with_retry` re-raises `asyncio.TimeoutError` without retrying, so a `timeout_ms` that is too short retries nothing — raise the deadline rather than the attempt count. An SDK-level timeout is different: `openai.APITimeoutError` subclasses `APIConnectionError`, so it is retried as a connection error even though it lands under the same `timeout` error kind.
- **A 400**, such as a model that does not serve classify (the router's message says `does not support classify`). `with_retry`'s classifier sees it is not retryable and raises at once.
- **A `parse` failure**, which carries no `error_exc` at all and so is returned rather than raised. Classify verdicts for the same state and question are deterministic ([Classify judges](classify-judges.md#settings-that-do-not-apply)), so a reply that did not validate will not validate better on a second attempt.

What does retry is a 429, a 500 or above, and a connection error. The wait is `2s`, then `4s`, then `8s`, doubling to a `60s` ceiling, with up to 25% of each wait added as jitter; `with_retry` reads no `Retry-After` header, so a provider asking for a longer pause does not get one.

## Many rows at once

One `ClassifyRequest` carries one `state`, so labelling 2000 rows is 2000 calls — the saving is in the questions per call, not the rows per call. Fan them out yourself with `asyncio.gather` or a task group.

You do not need your own semaphore. `run_classify` takes an evaluatorq concurrency slot automatically, so the shared default of 10 concurrent requests already applies, and `llm_concurrency_limit(n)` around the fan-out changes it:

```python
from evaluatorq import llm_concurrency_limit

with llm_concurrency_limit(25):
    ...  # your asyncio.gather over run_classify calls
```

`timeout_ms` is the deadline for **one attempt**, and it starts after the slot is acquired, so time spent queueing behind other requests never counts against it. The slot is held only around the request itself, so a retry's backoff sleep does not sit on one.

Because the wrapper above hands back the last outcome instead of raising, a row that failed arrives beside the rows that did not. Collect them by `error_kind` rather than catching exceptions — `[row for row, outcome in zip(rows, outcomes) if outcome.error_kind is not None]` is the retry-later list, and `gather` needs no `return_exceptions=True` to survive a provider failure.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `error_kind` is `api_status`, message `Model '<id>' in model does not support classify` | The model's catalogue entry does not set `supports_classify`. `openai/gpt-6-sol` is one such id; being a chat model is not what decides it. | Check the id against the listing above, or use `typesafe/jev-latest`. |
| A 404, or an authentication error | The client's `base_url` is not the Orq router, only `OPENAI_API_KEY` is set, the router suffix was appended twice, or the gateway is older than Orq 4.16 and does not serve `/decisions`. | Set `ORQ_API_KEY` and make `base_url` exactly `<host>/v3/router`; on an older gateway, upgrade it. |
| `error_kind` is `timeout` and no retry happened | `with_retry` re-raises `asyncio.TimeoutError` without retrying. | Raise `timeout_ms`; a short deadline is not recovered by the retry budget. |
| `error_kind` is `parse` and `error_exc` is `None` | The reply did not validate. There is nothing to re-raise, so a retry wrapper sees success-shaped control flow. | Branch on `error_kind`, never on `error_exc` alone. Read `raw_content` for what arrived. |
| `pydantic.ValidationError` before any request | `criteria` disagrees with `kind` — a `choice` without a label dict, or a `score` without 2 to 10 levels. | Match the `criteria` column in the table above. |
| An answer's value field is `None` on a success | The provider answered a different `kind` than you asked. Nothing on this path rejects that. | Read `answer.type` first. |
| `KeyError` on `response.answers[...]` | The keys are the provider's echo of yours and are not checked for parity. | Use `.get()`, or compare the key sets before reading. |
| `total_cost` is `None` on a success | The reply carried no readable usage block, so the call is recorded as unpriced rather than free. | Treat `None` as unknown, not zero. |
| No `judge.confidence` on the span | Only an answer keyed `verdict` sets it. | Name a single question `verdict`, or read the confidence off the answer instead. |

## Where to next

- **[Classify judges](classify-judges.md)** — the same endpoint with a jury around it: aggregation, repetitions, routing, and `noul_threshold` turned into a verdict.
- **[Tuning](tuning.md#pipeline-call-reliability)** — timeouts, retry budgets and the one-retry-layer rule across every evaluatorq call.
- **[Configuration › Models](configuration.md#models)** — the classifier role, and the flags and environment variables that set it.
