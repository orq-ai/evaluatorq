# Fetching Traces

`fetch_traces()` reads recorded Orq traces and hands you each one as a normalized `Trace` object.

Reach for it when you want the traces themselves: to inspect a production conversation, to build your own rows, or to filter a population down before you spend anything on scoring. If all you want is to *evaluate* recorded traces, pass a `TraceInput` as `data=` to `evaluatorq()` instead and let it fetch for you — see [Data sources](evaluation-reference.md#data-sources). Core evaluation, red teaming and simulation all read traces through this importer, so a trace that imports here imports the same way there — though each surface then applies its own rules about what to keep. [Trace Insights](insights.md) and the [trace finder](trace-finder.md) do not: they load trace populations through their own path, with their own record shape and no `import_error`.

## Fetch a bounded batch

`TraceInput` describes what to select and `fetch_traces()` performs the selection. Both are async.

```python
import asyncio
from datetime import datetime, timedelta, timezone

from evaluatorq import TraceInput, fetch_traces


async def main() -> None:
    traces = await fetch_traces(
        source=TraceInput(limit=10, start_time=datetime.now(timezone.utc) - timedelta(days=7)),
    )
    print(f'fetched {len(traces)} trace(s)')
    for trace in traces:
        print(trace.trace_id, trace.message_format, len(trace.messages))


asyncio.run(main())
```

`ORQ_API_KEY` must be set, or the call raises `ValueError: Missing Orq API key: set ORQ_API_KEY or pass api_key=.` before any request leaves the process. Pass `api_key=` to override it for one call. The host comes from `ORQ_BASE_URL`, and `base_url=` overrides that for one call.

## The two selection modes

`TraceInput` is either a search or one exact trace, never both.

| Mode | Fields | Use it when |
|---|---|---|
| Query | `limit`, `start_time`, `end_time`, `search`, `filters` | You do not have a trace ID and want a bounded batch |
| Trace | `trace_id`, optionally `span_id` | You have the ID, usually from the Orq UI or `orq traces search` |

Mixing them raises at construction rather than at request time: `trace_id` with any search criterion or `limit` raises `trace_id cannot be combined with trace search criteria or limit.`, `span_id` without `trace_id` raises `span_id requires trace_id.`, and `start_time` after `end_time` raises `start_time must be before end_time.`. `limit` is rejected in trace mode because trace mode never reads it, and accepting `limit=50` beside a `trace_id` would answer with one trace and no warning.

```python
from evaluatorq import TraceInput

try:
    TraceInput(trace_id='trace_123', limit=50)
except ValueError as exc:
    print('rejected:', exc.errors()[0]['msg'])
```

The exception is pydantic's `ValidationError`, which subclasses `ValueError`; catch either.

`limit` is optional and defaults to 20 traces. Read the resolved number off `TraceInput.query_limit` rather than the field, because the field stays `None` when you did not set it. A naive `start_time` or `end_time` is read as UTC, not as the host's local time, so one query means one window wherever it runs.

Query mode bounds the *count* with `limit` and the *window* with the time fields. It requests no sort order, and there is no option that asks for one, so do not read "the 50 most recent" into `limit=50` — if you need a specific period, name it with `start_time` and `end_time`.

The two modes are not interchangeable in what they return. Trace mode has no list response behind it, so a trace fetched by ID carries no `trace_name`, none of your custom trace metadata, and no `expected_output`, even when the same trace fetched by query would have all three.

## Narrow the query

`search` takes free text. Reach for it first: evaluatorq passes it straight through and it needs no field vocabulary.

```python
import asyncio

from evaluatorq import TraceInput, fetch_traces


async def main() -> None:
    traces = await fetch_traces(source=TraceInput(limit=10, search='error'))
    print(f'{len(traces)} trace(s) matched the search')
    # Zero is a normal answer here, and it logs a warning rather than raising.


asyncio.run(main())
```

`filters` takes a list of field conditions that evaluatorq also passes through untouched, so the accepted fields and operators are the Orq trace API's vocabulary rather than anything this library defines or validates. A condition the API does not accept comes back as a `RuntimeError` wrapping the HTTP status, not as an empty result, so a filter is worth running on a small `limit` before you build a job on it.

```python
import asyncio

from evaluatorq import TraceInput, fetch_traces


async def main() -> None:
    source = TraceInput(limit=3, filters=[{'field': 'not.a.real.field', 'op': 'gt', 'values': ['0']}])
    try:
        await fetch_traces(source=source)
    except RuntimeError as exc:
        print('rejected by the API:', str(exc).splitlines()[0])


asyncio.run(main())
```

[Data sources](evaluation-reference.md#data-sources) carries a worked `filters` condition. Which field narrows a population to your own traffic depends on what your app records — session, actor, deployment key or a custom attribute — so take the field names from the trace as Orq shows it and confirm the count on a small `limit` first.

What you cannot express as a filter is the thing the next section is about: no field says "this trace has readable messages". Importability is not a query condition, so the approach is to over-fetch and partition.

## Two checks before you use a trace

A fetched batch contains more than conversations you can score, and neither problem raises. Check both of these before you spend anything on the rows.

### Did it import?

`fetch_traces()` returns one `Trace` per trace the list endpoint gave it, including the ones it could not read. A trace it could not read comes back with `import_error` set to the reason and **no messages at all**.

Nothing raises, so a failed import looks exactly like a conversation that happened to be empty. Filter on `import_error` first. Do not read the returned length as a confirmation either: a query can match fewer traces than `limit`, and a trace row the list endpoint returns without an ID is skipped.

### Is there an exchange to score?

`import_error is None` does not mean there is a reply. The importer selects a span that carries input **or** output messages, so a span holding only a request imports cleanly and leaves `output_messages` empty — you get the question with no answer. They pass the `import_error` check. On the supported scoring path such a row fails on its own and costs nothing; it is in your *own* pipeline that it reaches a judge and scores a blank answer as a real one.

Check both:

```python
import asyncio

from evaluatorq import TraceInput, fetch_traces


async def main() -> None:
    traces = await fetch_traces(source=TraceInput(limit=20))

    imported = [trace for trace in traces if trace.import_error is None]
    failed = [trace for trace in traces if trace.import_error is not None]
    answered = [trace for trace in imported if trace.output_messages]

    print(f'{len(traces)} fetched, {len(imported)} imported, {len(failed)} failed')
    print(f'{len(answered)} of the {len(imported)} imported traces carry an assistant reply')
    for trace in failed[:3]:
        print(f'  {trace.trace_id}: {trace.import_error}')


asyncio.run(main())
```

How many traces survive both checks depends entirely on what your workspace records. A trace whose latest non-evaluator span and its parents carry no LLM messages — a bare HTTP span, a tool call, a retrieval step — has nothing for the importer to read, and that is the usual reason for a failed import. In a workspace that records more than LLM calls, failed imports can be the majority of a batch rather than an edge case. Run the snippet above against your own traces before you size a job on the count, and do not assume a recent window is richer: a narrow `start_time` can return a batch in which nothing imports at all.

A third thing the checks cannot tell you: whether the trace is a *product* conversation. Spans that Orq marks as evaluator activity are excluded, but a judge, classifier or simulated-persona prompt recorded through an ordinary LLM call is an ordinary LLM call as far as the importer is concerned, and it imports as a normal trace. If your workspace evaluates as well as serves, filter on something that identifies your own traffic rather than trusting the batch.

You do not have to remember the first check to stay safe from a *half*-imported trace: a `Trace` carrying `import_error` together with messages cannot be constructed, because the model validator rejects it. What the check buys you is knowing the row is there at all.

`fetch_traces()`'s own docstring points at a `partition_traces` helper for this split, and it does exist — it takes the list and returns `(usable, failed)`, logging the failures once. It is not exported from the top-level package, so reaching it means `from evaluatorq.common.trace_input import partition_traces, load_traces`, an internal path with no stability promise. Hand-filtering as above uses only the public surface, which is why this page does it that way. Neither helper applies the second check.

### The log stream is loud

Every failed import is logged as a `WARNING` naming the trace and the reason, so a batch that shrank shows up in the logs even if your code never reads `import_error`.

Individual messages are dropped the same way, and this is the most common case where a trace imports with `import_error is None` and still loses content: a message mapping with no recognised message keys, an unknown role, or an unexpected shape is dropped with its own `WARNING` and the rest of the trace imports around it. Lines like `Trace message mapping carries no message keys (saw ['gen_ai']); dropping it.` are that, not an error. A malformed tool call or a non-string `name` goes the same way. OpenTelemetry reasoning parts are dropped at `DEBUG`, so those you will not see at all at the default level.

That is one line per failed trace and one per dropped message, which on a mixed population outnumbers your own output. Silence the library's logging if it is in the way, bearing in mind that this is also how you stop hearing about dropped content:

```python
import asyncio

from loguru import logger

from evaluatorq import TraceInput, fetch_traces

logger.disable('evaluatorq')


async def main() -> None:
    traces = await fetch_traces(source=TraceInput(limit=5))
    print(f'{len(traces)} fetched, with evaluatorq log records suppressed')


asyncio.run(main())
```

Loguru filters at log time, so the call works wherever you put it relative to the import.

### What each `import_error` means

| `import_error` says | What happened |
|---|---|
| `the trace has no spans.` | The span list came back empty — also what an unknown or mistyped trace ID looks like |
| `no non-evaluator span on the latest span path contains input or output messages.` | Spans exist, but the latest non-evaluator span and its parent chain carry none. An earlier sibling span holding messages is never examined |
| `every span belongs to an evaluator span or its subtree.` | The trace records only evaluator activity, which is excluded on purpose |
| `requested span '<id>' was not found.` | You passed a `span_id` that is not in this trace |
| `requested span '<id>' has no input or output messages.` | The exact span you named carries no exchange |
| `requested span '<id>' belongs to an evaluator span or its subtree.` | The span you named is inside evaluator activity |
| `could not fetch its spans: <error>` | That trace's span request failed after its retries |
| `its spans payload is <type>, not a list.` | The span response was not a list |
| `its spans payload contains a non-object span.` | One element of the span list was not an object |

One unreadable trace does not cost you the others: a span fetch that fails becomes an `import_error` on that trace alone. The trace *list* request is the exception — if it fails, there is nothing to partition and `fetch_traces()` raises `RuntimeError: Failed to list Orq traces: ...` for the whole call.

## What a `Trace` carries

| Field | What it holds |
|---|---|
| `trace_id` | The Orq trace ID |
| `input_messages` / `output_messages` | The exchange, with its source-side boundaries kept |
| `messages` | Input followed by output as one transcript, with the largest exact overlap removed |
| `message_format` | `chat_completions`, `responses`, `otel_genai`, or `mixed` |
| `tools_called` | Names of the tools called in the exchange |
| `retrievals` | Retrieved context, as recorded on the span |
| `query` | The text of the last user turn in the exchange |
| `session_id`, `actor_id`, `thread_id` | Conversation grouping, as recorded |
| `metadata` | Trace name and any custom trace metadata from the list response, plus the selected span's name, type, operation, provider, model and status |
| `expected_output` | A human `correction` annotation on the trace, when one exists — query mode only |
| `message_span_id`, `requested_span_id` | Which span the messages came from, and which one you asked for |
| `import_error` | The reason this trace could not be read, or `None` |

`query` is computed from the transcript, not from your search text, and it has no relationship to `retrievals` — pairing the two as question-and-context will mis-score any multi-turn trace whose last user turn is not what drove the retrieval.

`messages` is derived rather than stored. Use it when you want one conversation; use `input_messages` and `output_messages` when the boundary between what went in and what came back matters.

Each entry is a `Message` with `role` and `content`, so printing one trace's transcript is a loop:

```python
import asyncio

from evaluatorq import TraceInput, fetch_traces


async def main() -> None:
    traces = await fetch_traces(source=TraceInput(limit=20))
    answered = [trace for trace in traces if trace.import_error is None and trace.output_messages]
    if not answered:
        print('no trace in this batch has both an import and a reply')
        return
    trace = answered[0]
    print(trace.trace_id, trace.message_format)
    for message in trace.messages:
        print(f'  {message.role}: {message.content}')


asyncio.run(main())
```

`content` is `str | None`. An assistant turn that only called a tool has `content=None` and its action in `tool_calls`, so a loop that prints `content` alone renders an agent that acted as an agent that said nothing. Read `tool_calls` too, or render the list with `common.messages.messages_to_text`.

## Turn traces into evaluatorq rows

`Trace.to_datapoint()` converts one trace into evaluatorq's native row shape. The transcript and the source metadata go under `inputs`; `expected_output` is the one field that does not, because it becomes the row's own `expected_output` and is what makes the row scorable against a ground truth.

```python
import asyncio

from evaluatorq import TraceInput, fetch_traces


async def main() -> None:
    traces = await fetch_traces(source=TraceInput(limit=5))
    rows = [trace.to_datapoint() for trace in traces if trace.import_error is None and trace.output_messages]
    print('rows:', len(rows))
    if rows:
        print('input keys:', sorted(rows[0].inputs))
        print('expected_output set:', rows[0].expected_output is not None)


asyncio.run(main())
```

A failed trace converts too, and its row carries the reason under `inputs['trace_import_error']`. That is deliberate — it is what lets a surface keep the row and fail it individually instead of dropping it out of the denominator — but it means `to_datapoint()` is not a filter. Apply both checks before you convert.

`inputs['recorded_output']` holds the recorded reply as a list of JSON dicts, not a string, and it is empty for a trace that imported without one. An evaluator that expects text has to render it rather than compare against it.

For scoring these rows, passing the original `TraceInput` as `data=` is the supported path and reads the recorded output for you — see [Data sources](evaluation-reference.md#data-sources). On that path the two checks turn into errors you will actually read: a failed import raises `The source trace could not be imported: ...` on its row, and a trace with no reply raises `inference=False requires a non-empty 'recorded_output' for an imported trace ...`.

Surfaces that read traces answer a failed import differently, so do not carry an expectation from one to another:

| Surface | What it does with a failed import |
|---|---|
| `fetch_traces()` | Returns it, `import_error` set. Both checks are yours |
| `evaluatorq(data=TraceInput(...))` | Keeps it as a row so it fails on its own and stays in the denominator |
| [`redteam.datapoints_from_traces()`](guides/red-teaming.md) | Skips it, and raises if every trace failed |
| [Simulation's trace helpers](guides/agent-simulation.md) | Log a `WARNING` per dropped trace; `datapoints_from_traces()` raises when a `TraceInput` left nothing usable, `fetch_trace_conversations()` returns what is left |

## Reuse one HTTP client

Pass `http_client=` to share an `httpx.AsyncClient` across calls. `fetch_traces()` closes a client it created itself and leaves one you passed open.

```python
import asyncio

import httpx

from evaluatorq import TraceInput, fetch_traces


async def main() -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        first = await fetch_traces(source=TraceInput(limit=2), http_client=client)
        second = await fetch_traces(source=TraceInput(limit=2, search='error'), http_client=client)
    print(len(first), len(second))


asyncio.run(main())
```

Set the timeout yourself when you pass a client. The one `fetch_traces()` builds uses 60 seconds; httpx's own default is 5, which a slow trace with many spans will exceed.

Retries belong to `fetch_traces()` — the trace list and span requests retry internally, including the 429s and 5xx responses. Do not add a retrying client on top; the two layers multiply.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: Missing Orq API key` | `ORQ_API_KEY` unset and no `api_key=` passed | Set the variable or pass the key |
| Most traces have `import_error` set | The population records no LLM exchanges on readable spans | Expected on a mixed workspace. Over-fetch and partition — importability is not filterable |
| `RuntimeError: Failed to list Orq traces` | The list request failed: a rejected `filters` condition, a bad key, or an API error that survived retry | Check the filter on a small `limit`; otherwise read the HTTP status in the message |
| `trace_id cannot be combined with trace search criteria or limit.` | Query fields passed alongside `trace_id` | Pick one mode |
| A query returns nothing and logs a warning | Nothing matched the window or filters | Widen the time bounds or relax the filters. `limit` is a count cap and cannot help here |
| Your own pipeline judges a blank answer as a real one | Traces with no reply passed through `to_datapoint()` | Check `output_messages`, not just `import_error` |
| A trace fetched by ID has no `expected_output` or `trace_name` | Trace mode has no list response behind it | Fetch it in query mode if you need those fields |
