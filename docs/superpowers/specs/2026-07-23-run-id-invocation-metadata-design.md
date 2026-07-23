# Run-id invocation metadata — design

**Date:** 2026-07-23
**Status:** approved for planning

## Goal

Make every LLM chat-completion / responses invocation issued during a red-team or
agent-simulation run carry the run's identifier as request `metadata`, and expose the
same identifier on the run's root trace span. This lets an operator filter, in Orq's
trace UI, every model call that belongs to one `red_team()` / `simulate()` invocation.

No new identifier is minted: both pipelines already produce a per-run `run_id`. We reuse
it. No new dependency (no ULID library).

## Background — what already exists

- **`run_id` per run.** Red team mints it in `tracing_session` (`tracing/context.py`,
  `generate_run_id()` → `str(uuid4())`), surfaced as `tracing_context.run_id`.
  Simulation mints `uuid4().hex` at each outer entry in `simulation/api.py`.
- **A ContextVar metadata rail.** `common/thread_context.py` already carries the pipeline
  label (`red_teaming` / `agent_simulation`) on the `_pipeline` ContextVar and exposes it
  via `pipeline_metadata()` → `{'evaluatorq_pipeline': ...}`. That dict is merged into the
  request `metadata` at **all four** LLM/response call sites:
  - `common/llm_call.py` (`_apply_pipeline_metadata`)
  - `redteam/backends/openai.py`
  - `redteam/backends/orq.py` (via `pipeline_metadata_param()`, which wraps `pipeline_metadata()`)
  - `openresponses/target.py`
- **A span run-id attribute.** `tracing/spans.py` sets `orq.run_id` on `orq.job` and
  `orq.evaluation` spans (shared by the core `evaluate()` pipeline too).

Because all four call sites already splat `pipeline_metadata()`, adding a second key to
that function propagates to every invocation with **zero call-site edits**.

## Design

### 1. `common/thread_context.py` — carry run_id on the same rail

- Add `_run_id: ContextVar[str | None]` (mirror of `_pipeline`).
- Add `evaluatorq_run_id(run_id)` — a sync `@contextmanager` that sets/resets `_run_id`,
  a copy of the existing `evaluatorq_pipeline` CM.
- Add `run_metadata_scope(run_id, span=None)` — a sync `@contextmanager` helper that:
  - if `span is not None`, calls `span.set_attribute('orq.evaluatorq_run_id', run_id)`
    (duck-typed; no OTel import needed), then
  - enters `evaluatorq_run_id(run_id)`.
  This is the single shared entry helper both surfaces call (requested to prevent future
  drift when new entry points are added).
- Extend `pipeline_metadata()` to also emit `evaluatorq_run_id` when `_run_id` is set:
  returns `{'evaluatorq_pipeline': ..., 'evaluatorq_run_id': ...}` (each key present only
  when its ContextVar is set). `pipeline_metadata_param()` is unchanged (wraps the above).
- Export the two new names in `__all__`.

### 2. Set the scope at pipeline entry

- **Red team** (`redteam/runner.py`): after the root span opens
  (`with with_redteam_span(...) as pipeline_span:`), wrap the dispatch block in
  `with run_metadata_scope(tracing_context.run_id, pipeline_span):`. This sets the root-span
  attribute and binds the ContextVar for every child job task (which read `pipeline_metadata()`
  in the backends). Sits inside the existing `evaluatorq_pipeline('red_teaming')` scope.
- **Simulation** (`simulation/api.py`): at **each** outer entry, immediately after `run_id`
  is minted and with `pipeline_span` in scope, wrap the dispatch (`try:` block) in
  `with run_metadata_scope(run_id, pipeline_span):`. Wrapping at the outer entry — not inside
  `_simulate_core` — is deliberate: it also tags the **generate phase** LLM calls, so *all*
  run-related calls carry the metadata, per requirement.

### 3. Root-span attribute rename

Rename `orq.run_id` → `orq.evaluatorq_run_id` in `tracing/spans.py` (both `orq.job` and
`orq.evaluation` spans). The root spans gain `orq.evaluatorq_run_id` via
`run_metadata_scope`. Result: one consistent attribute name (`orq.evaluatorq_run_id`) across
root, job, and evaluation spans, plus `evaluatorq_run_id` in every invocation's `metadata`.

`orq.run_id` is dropped. In-repo it is written only in `spans.py` and documented in
`docs/tracing.md`; **no code, test, dashboard, or the orq SDK reads it**.

**Known risk (accepted):** `orq.run_id` is in the `orq.*` namespace that Orq's platform
ingests server-side. If Orq groups spans into a run by reading `orq.run_id`, renaming could
break that grouping in Orq's own UI — unverifiable from this repo. Rollback is trivial:
re-add the `orq.run_id` attribute alongside the new one.

## Naming

| Surface | Key |
|---|---|
| Invocation request `metadata` | `evaluatorq_run_id` |
| Span attribute (root / job / evaluation) | `orq.evaluatorq_run_id` |

## Files touched

- `src/evaluatorq/common/thread_context.py` — new ContextVar, `evaluatorq_run_id`,
  `run_metadata_scope`, extend `pipeline_metadata()`, `__all__`.
- `src/evaluatorq/tracing/spans.py` — rename attribute (2 lines).
- `src/evaluatorq/redteam/runner.py` — wrap dispatch in `run_metadata_scope`.
- `src/evaluatorq/simulation/api.py` — wrap each outer entry in `run_metadata_scope`.
- `docs/tracing.md` — update attribute name; note invocation-metadata propagation.
- `tests/unit/test_thread_context.py` — extend.

## Testing

- Unit: inside `evaluatorq_run_id('r1')`, `pipeline_metadata()` contains
  `evaluatorq_run_id == 'r1'`; outside the scope the key is absent. Combined with
  `evaluatorq_pipeline`, both keys present.
- Unit: `run_metadata_scope('r1', span=<fake>)` sets `orq.evaluatorq_run_id` on the fake
  span and binds the ContextVar; with `span=None` it no-ops the attribute and still binds.

## Out of scope

- Core `evaluate()` invocation metadata (request is red-team + simulation only). The span
  attribute rename does touch shared `orq.job` / `orq.evaluation` spans, which `evaluate()`
  also emits — that is a consistency rename, not new behavior.
- Changing thread-id composition or the manifest run_id.
