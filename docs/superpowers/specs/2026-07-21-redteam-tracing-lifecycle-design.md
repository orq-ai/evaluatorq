# Red team tracing: framework-owned session lifecycle + full-run pipeline span

Date: 2026-07-21

## Problem

Red team runs frequently produced **no `orq.redteam.pipeline` trace** in the target
workspace even though span export returned HTTP 200.

1. **Premature tracing shutdown.** The base `evaluatorq()` runner shuts tracing down
   at its own end (`evaluatorq.py:404-410`: `flush` → `sleep(2)` → `shutdown`). The
   red team pipeline calls `evaluatorq()` *nested* inside its run, so the global
   OpenTelemetry provider was torn down mid-run; every span emitted afterwards
   (pipeline root end, recommendations, executive summary) hit a dead provider and
   was dropped (`Shutdown called, ignoring Span`). The final design removes runtime
   provider shutdown entirely and uses a framework-owned flush-only session.

2. **No safe provider replacement after explicit shutdown.** OpenTelemetry installs
   one global provider per process. A replacement provider after shutdown cannot be
   installed reliably, so private test-only `_shutdown_tracing()` is terminal for
   that interpreter: it clears local references and prevents re-initialization.
   Runtime paths do not call it.

3. **Wrong pipeline-span scope.** Dynamic/hybrid opens `orq.redteam.pipeline` inside
   `_run_dynamic_or_hybrid()` and closes it **before** `red_team()` runs
   recommendations, executive summary, and save. Static (`_run_static()`) opens **no
   redteam span at all** — no root, and the outbound target call (attack payload +
   response) is untraced.

## Root-cause reframing

The lifecycle bug was not "we need smart ownership" — it was **calling provider
shutdown mid-process at all**. Shutdown belongs at process exit, and the
SDK's `TracerProvider` already registers that itself: `shutdown_on_exit=True`
(verified in the installed SDK — `TracerProvider.__init__` does
`atexit.register(self.shutdown)`). So the fix is: **flush per run, never shut down
mid-run, let the SDK's atexit hook tear down at process exit.** This removes the need
for any ownership/refcount machinery — nesting, sequential reuse, and concurrent
sibling runs all become correct for free because nothing tears the provider down
while work is in flight.

## Goals

- No mid-run provider shutdown. Each run flushes deterministically at its end.
- One framework-owned tracing lifecycle **object** used uniformly by `evaluatorq()`,
  `red_team()`, and `simulate()` — infrastructure, not a user hook.
- One `orq.redteam.pipeline` span wraps the **entire** `red_team()` run, both modes.
- Executive summary + recommendations captured as their own child spans.
- Static mode gets a pipeline root and a `target_call` span for the attack/response.

## Non-goals

- Surfacing OTEL export failures to the user (separate concern).
- Supporting `sk-orq` keys for ingestion (already works — verified live).
- Putting tracing into the user-facing `PipelineHooks`/`SimulationHooks` (wrong
  layer: user-overridable, per-subsystem, no run-start hook, misses the base runner).
- Multi-turn / adversarial-generation spans in static mode (static is single-shot).

## Design

### 1. Framework-owned tracing session (`tracing/`)

`tracing_session()` is an async context manager next to `TracingContext`
(`tracing/context.py` is the natural home; it already models tracing as an object).
It owns the flush-only lifecycle and yields the existing `TracingContext`:

```python
@asynccontextmanager
async def tracing_session(
    run_name: str, *, trace_type: str = 'evaluatorq'
) -> AsyncGenerator[TracingContext, None]:
    """Framework-owned tracing lifecycle. Initializes tracing on enter (idempotent),
    flushes on exit. NEVER shuts the provider down — the SDK TracerProvider's atexit
    hook (shutdown_on_exit=True) handles teardown at process exit. Safe to nest, to
    reuse sequentially, and to run concurrently: enter inits if needed, exit only
    flushes buffered spans."""
    enabled = await init_tracing_if_needed()
    ctx = TracingContext(
        run_id=generate_run_id(),
        run_name=run_name,
        enabled=enabled,
        parent_context=await capture_parent_context() if enabled else None,
        trace_type=trace_type,
    )
    try:
        yield ctx
    finally:
        if enabled:
            await flush_tracing()   # force_flush blocks until export; no sleep needed
```

Callers:
- `evaluatorq()` — wraps its body in `async with tracing_session(name, trace_type=_trace_type)
  as tracing_context:`. Replaces the manual `init_tracing_if_needed()` +
  `TracingContext(...)` construction (`evaluatorq.py:204-215`) **and** the
  legacy `flush → sleep(2) → shutdown` block (`404-410`).
- `red_team()` — wraps its whole body in `async with tracing_session(name):`.
- `simulation/api.py` (3 sites) — replace the `init_tracing_if_needed()` + trailing
  `flush_tracing()` pair with `async with tracing_session(...)`. This is now a clean,
  regression-free migration (simulation already never shut down).

`_shutdown_tracing()` is retained for explicit, terminal test teardown; the SDK atexit
hook handles process-exit teardown. It must keep `_initialization_attempted = True` so
a later run does not claim it installed a replacement provider. Tests requiring a fresh
provider must use a fresh interpreter. `tracing_session()` itself never calls shutdown.

Concurrency note: the module globals are unlocked, but `tracing_session()` needs no lock —
`init_tracing_if_needed()` is idempotent, and exit only flushes. asyncio is
single-threaded; concurrent `tracing_session()` enters each init idempotently and each
exit flushes independently. No provider is ever shut down while another run is live.

### 2. Pipeline span moves up into `red_team()`

`red_team()` opens the `orq.redteam.pipeline` span (with the run-level attributes
currently set in `_run_dynamic_or_hybrid`), inside the `tracing_session()` block, wrapping
the whole run: context retrieval, attacks, evaluation, recommendations, executive
summary, and save.

- `_run_dynamic_or_hybrid()` and `_run_static()` no longer open their own
  `orq.redteam.pipeline` span; their child spans nest under the ambient current
  context (the span `red_team()` opened). The now-redundant `init_tracing_if_needed()`
  + `capture_parent_context()` at `runner.py:1383-1384` are removed (the session +
  red_team's own `capture_parent_context()` cover them).
- **End-of-run attributes are returned up, not reached down** (chosen: threading over
  ambient `trace.get_current_span()`). The inner runners return their run-level
  metrics (`num_datapoints`, `num_categories`, `duration_seconds`) alongside the
  report; `red_team()` sets them on the span it owns via `set_span_attrs`. Where the
  values are already derivable from the returned `report` + red_team's own timing,
  `red_team()` computes them directly and no new return channel is added.
- Static gains an `orq.redteam.pipeline` root for the first time — intentional.

### 3. Executive-summary and recommendations spans

Wrap the exec-summary block (`runner.py:717`) in
`with_redteam_span('orq.redteam.executive_summary', {'orq.redteam.model': evaluator_model})`
and the recommendations block (`runner.py:695`) in
`with_redteam_span('orq.redteam.recommendations', {'orq.redteam.model': evaluator_model})`.
Both nest under the pipeline span (Decision 2). The existing best-effort try/except
that appends a `pipeline_warnings` entry is preserved (span records the exception via
`with_redteam_span`; the warning is complementary).

### 4. Static-mode spans

Static currently emits no `with_redteam_span`. After this change:
- Static runs under the `red_team()`-owned `orq.redteam.pipeline` root (Decision 2),
  so its base `orq.job` / `orq.evaluation` / LLM spans nest under it.
- **Target call**: the two async static job closures in `redteam/runtime/jobs.py`
  (`router_job` and `deployment_job`, both created by `create_model_job`) plus
  `_create_static_job_for_agent_target` wrap their outbound call in an
  `orq.redteam.target_call` span (attrs: `orq.redteam.llm_purpose='target'`, target
  key/model), mirroring the dynamic path. This is the request/response pair a user
  opens a static trace to inspect — the actual gap behind "add the missing static
  spans."
- **Evaluation**: the static scorer (`evaluatorq_bridge.py:251`, returned by
  `create_owasp_evaluator`) is wrapped in an `orq.redteam.security_evaluation` span
  (attr `orq.redteam.category`), matching dynamic.
- **Not added** (single-shot, no multi-turn): `orq.redteam.attack_turn`,
  `orq.redteam.adversarial_generation`. Documented as an intentional difference.

Update the `redteam/tracing.py` docstring hierarchy to show both dynamic and static
shapes and the new spans.

## Span hierarchy after change

```
dynamic/hybrid: pipeline → context/datapoint work → job → attack
                → target_call/attack_turn → evaluation
                → security_evaluation → recommendations/executive_summary
static:         pipeline → job → attack → target_call → evaluation
                → security_evaluation → recommendations/executive_summary
```

Dynamic/hybrid:
```
orq.redteam.pipeline                      [red_team()]  ← wraps whole run
  +-- orq.redteam.context_retrieval
  +-- orq.redteam.datapoint_generation
  +-- orq.job (framework) / orq.redteam.attack ...
  +-- orq.evaluation (framework)
  +-- orq.redteam.memory_cleanup
  +-- orq.redteam.recommendations         [new]
  +-- orq.redteam.executive_summary       [new]
```
Static:
```
orq.redteam.pipeline                      [red_team()]  ← new root for static
  +-- orq.job (framework)
  |   +-- orq.redteam.attack
  |       +-- orq.redteam.target_call     [new — attack payload + response]
  +-- orq.evaluation (framework)
  |   +-- orq.redteam.security_evaluation [new — the judge]
  +-- orq.redteam.recommendations         [new]
  +-- orq.redteam.executive_summary       [new]
```

## Error handling

- `tracing_session()` flushes in `finally`; `flush_tracing()` logs a warning when its
  timeout leaves spans unexported. No shutdown means no mid-run teardown failure mode.
- Exec-summary and recommendations keep their best-effort try/except; a summary
  failure never fails the run and the span records the exception.

## Testing

- Unit (`tracing_session()`): enter inits + yields a `TracingContext(enabled=...)`; exit
  flushes; **no** `_shutdown_tracing()` is called on exit (assert via mock). Nested
  entry does not re-shutdown. Two sequential `tracing_session()` blocks in one process
  both flush and both keep tracing live (guards the `_initialization_attempted`
  regression). Concurrent (`asyncio.gather`) entries: neither tears the provider down.
- Unit: `_shutdown_tracing()` remains terminal: a later
  `init_tracing_if_needed()` returns `False` without attempting a provider replacement.
- Keep the existing tracing test suite green (count via
  `pytest tests -k tracing --co -q | tail -1`, not a hardcoded number).
- Regression: an in-memory `SpanExporter` asserts that after `red_team()` the exported
  spans include `orq.redteam.pipeline` with `orq.redteam.executive_summary` **and**
  (static) `orq.redteam.target_call` as descendants.
- Live: re-run red team against a cookbooks agent with an `sk-orq` key in **both**
  dynamic and static modes; assert the static trace shows the root, `target_call`,
  and `security_evaluation` spans — not just "exit 0".

## Round-2 review resolutions (final, supersede above where conflicting)

Recommendations folded in:
- **R1.** `flush_tracing()` must not block the event loop. Change it to
  `await asyncio.to_thread(provider.force_flush, timeout_millis)` and inspect the
  return: `force_flush()` returns `False` on timeout — log a `loguru` warning on
  failure (a health signal, not a silent drop). Default `timeout_millis` ~5000.
- **R2.** `_shutdown_tracing()` (renamed, see D2) must add `_initialization_attempted`
  to its `global` statement AND set it `False`, or the reset is a silent no-op.
- **R3.** The tracing-session migration **deletes** `evaluatorq.py:204-215` (the inline
  `init_tracing_if_needed()` + `capture_parent_context()` + `TracingContext(...)`
  block) and replaces it with `async with tracing_session(...) as tracing_context:`.
  No wrapping-around — that would build two `TracingContext`s per run.
- **R4.** Static outbound-call inventory is **two** async job closures — `router_job`
  and `deployment_job`, both defined inside `create_model_job` in
  `redteam/runtime/jobs.py` — plus the agent-target static job
  (`_create_static_job_for_agent_target`). There is no third top-level job function.
- **R5.** The static `target_call` span reuses the **dynamic** convention exactly:
  attributes via `truncate_for_span` (`common/tracing.py`), wrapped in
  `conversation_thread` where the dynamic path does. Do **not** set
  `orq.redteam.llm_purpose='target'` on the outer `target_call` span — the backend's
  own child span (`with_llm_span` in `backends/orq.py` / `backends/openai.py`) sets
  it. Cross-check attrs against `redteam/adaptive/pipeline.py` target_call site.
- **R6.** Delete the `asyncio.sleep(2)` at `evaluatorq.py:408` — `force_flush` blocks
  until export (R1 keeps that guarantee off-thread), so the sleep is dead weight.
- **R7.** `orq.redteam.num_datapoints` keeps its **"attempted"** meaning: thread
  `len(all_datapoints)` (and `len(resolved_categories)`, duration) up from the inner
  runner to `red_team()`; do not recompute from the merged report (that would silently
  mean "scored"). Add these to the inner runner's return, set via `set_span_attrs`.
- **R9.** Migrate **all three** `simulation/api.py` sites (~214, ~442, ~558) to
  `tracing_session`, not one.

Decisions:
- **D1 = B (mitigate long-lived-process hazards).** (a) Make `BatchSpanProcessor`
  tuning env-configurable with larger defaults: `max_queue_size` (default 4096),
  `schedule_delay_millis`, `max_export_batch_size`, read from
  `ORQ_OTEL_MAX_QUEUE_SIZE` / `ORQ_OTEL_SCHEDULE_DELAY_MS` / `ORQ_OTEL_MAX_BATCH_SIZE`
  in `setup.py`. (b) The R1 flush-failure warning surfaces export stalls instead of
  silently dropping. (c) Remaining limitation — a rotated `ORQ_API_KEY` needs a
  process restart to take effect (the exporter bakes headers at init and we never
  rotate the provider) — is **documented** in the `tracing_session` docstring and the
  tracing module docstring, not mitigated (would require re-init machinery out of
  scope). SIGKILL span loss is inherent to any batch exporter and also documented.
- **D2 = A (privatize the footgun).** Rename `shutdown_tracing` →
  `_shutdown_tracing`, drop it from `tracing/__init__.py` `__all__`, update the (test-
  only) callers, and add a module-level comment: "Do not call during a run — process
  exit is handled by the SDK TracerProvider atexit hook; calling this mid-run
  black-holes all later spans." A regression test asserts no runtime path calls it.
- **D3 = B (symmetric static shape).** Static wraps each datapoint's work in an
  `orq.redteam.attack` span, with `orq.redteam.target_call` nested inside it, matching
  the dynamic depth (`orq.job → orq.redteam.attack → orq.redteam.target_call`). The
  static hierarchy in §4 gains the `orq.redteam.attack` intermediate level.

## Rollout / risk

Single PR (per decision). The lifecycle change is low-risk (strictly removes mid-run
shutdown). The span-topology change (Decision 2/4) is medium-risk — it touches both
inner runners' attribute handling and the static job bodies — mitigated by the
in-memory-exporter regression test and live verification in both modes.
