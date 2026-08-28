# Run-id invocation metadata — design

**Date:** 2026-07-23 **Status:** approved for planning (revised after hate-review)

## Goal

Make every LLM chat-completion / responses invocation issued during a red-team or agent-simulation run carry the run's identifier as request `metadata`, and expose that identifier on the run's **root** trace span. This lets an operator filter, in Orq's trace UI, every model call that belongs to one `red_team()` / `simulate()` invocation.

No new identifier is minted where one already exists: both pipelines already produce a per-run `run_id`, and we reuse it. No new dependency (no ULID library).

## Background — what already exists

- **`run_id` per run.** Red team mints it in `tracing_session` (`tracing/context.py`, `generate_run_id()` → `str(uuid4())`), surfaced as `tracing_context.run_id`. Simulation mints `uuid4().hex` at each outer entry in `simulation/api.py`.
- **A ContextVar metadata rail.** `common/thread_context.py` carries the pipeline label (`red_teaming` / `agent_simulation`) on the `_pipeline` ContextVar and exposes it two ways — `pipeline_metadata()` → `{'evaluatorq_pipeline': ...}` (native `metadata=` kwarg) and `pipeline_metadata_param()` → `{'metadata': {'evaluatorq_pipeline': ...}}` (extra_body form). **These two functions are independent implementations — `pipeline_metadata_param()` does NOT call `pipeline_metadata()`; it re-reads `_pipeline` directly.** Both must be extended.
- **Call sites that tag invocations with pipeline metadata (7, not 4):**
  - `common/llm_call.py` (`_apply_pipeline_metadata`) — `pipeline_metadata()`
  - `redteam/backends/openai.py` — `pipeline_metadata()`
  - `openresponses/target.py` — `pipeline_metadata()`
  - `redteam/backends/orq.py` (2 spots) — `pipeline_metadata_param()`
  - `redteam/runner.py` (executive-summary LLM call) — `pipeline_metadata_param()`
  - `redteam/reports/recommendations.py` — `pipeline_metadata_param()`
  - `redteam/adaptive/tool_chaining.py` — `pipeline_metadata_param()`

  Extending both `pipeline_metadata()` and `pipeline_metadata_param()` propagates `evaluatorq_run_id` to all seven with **zero call-site edits**.
- **Nested `evaluatorq()` is the execution engine.** Red team (`runner.py:2278`, `:2733`) and simulation (`api.py` via `_simulate_via_evaluatorq`) route their datapoints through a nested `evaluatorq()` call to inherit its auto-upload, OTel tracing, and result handling. Each nested `evaluatorq()` opens its **own** `tracing_session` and mints its **own** `run_id`, stamped on its `orq.job` / `orq.evaluation` spans as `orq.run_id`.

## Key architectural decisions (from review)

1. **Do not touch `orq.run_id`.** It is set only in `tracing/spans.py` (`with_job_span` / `with_evaluation_span`) — these are **evaluatorq-core** spans, emitted by `evaluate()` and by the nested `evaluatorq()` calls. Renaming it would break a documented public attribute for every `evaluate()` user, and would stamp the same attribute name with different values across a run (the nested runs each mint their own). Rule applied: attributes owned by evaluatorq-core stay; only red-teaming / agent-sim work changes. So `spans.py` and `docs/tracing.md` are **not** edited.

2. **Correlation flows via the ContextVar, not a new `evaluatorq()` parameter.** Binding `evaluatorq_run_id(run_id)` at the redteam/sim *outer* scope means the nested `evaluatorq()`'s LLM and evaluator invocations inherit the value automatically — a ContextVar set in an ancestor scope is visible in nested calls and is copied into child tasks spawned inside the scope. No signature change to `evaluatorq()`. The "same attr, different value" span problem is moot because decision (1) leaves `orq.run_id` alone; the new `orq.evaluatorq_run_id` lives only on the red-team / sim **root** span.

3. **Every sim entrypoint gets a unique run_id.** The `run_id=''` sentinel in `generate()` is a *manifest* no-save marker, unrelated to LLM tagging. A shared sim helper mints a real `run_id` and binds the ContextVar at every entrypoint, so all generation-phase and seeded-generator LLM calls are tagged. The manifest's `''` behavior is left untouched.

## Design

### 1. `common/thread_context.py` — carry run_id on the same rail

- Add `_run_id: ContextVar[str | None]` (mirror of `_pipeline`).
- Add `evaluatorq_run_id(run_id)` — a sync `@contextmanager` that does `token = _run_id.set(run_id)` / `try: yield ... finally: _run_id.reset(token)`, exactly mirroring the existing `evaluatorq_pipeline` CM (so exception-path reset and nested-scope restore-parent behavior match).
- Extend **both** metadata functions to include `evaluatorq_run_id` when `_run_id` is set (each key present only when its ContextVar is set):
  - `pipeline_metadata()` → `{'evaluatorq_pipeline': ..., 'evaluatorq_run_id': ...}`
  - `pipeline_metadata_param()` → `{'metadata': {'evaluatorq_pipeline': ..., 'evaluatorq_run_id': ...}}`
  - Refactor `pipeline_metadata_param()` to wrap `pipeline_metadata()` so they cannot drift
    again: `md = pipeline_metadata(); return {'metadata': md} if md else {}`.
- No `run_metadata_scope` helper. Setting the root-span attribute is a single guarded statement inlined at each root-span site (below); it is not scoped state, so it does not belong in a context manager.
- Export `evaluatorq_run_id` in `__all__`.

### 2. Bind the run_id at pipeline entry

- **Red team** (`redteam/runner.py`, single site): after the root span opens (`with with_redteam_span(...) as pipeline_span:`), inside the existing `evaluatorq_pipeline('red_teaming')` scope:
  ```python
  if pipeline_span is not None and tracing_context.run_id:
      pipeline_span.set_attribute('orq.evaluatorq_run_id', tracing_context.run_id)
  with evaluatorq_run_id(tracing_context.run_id):
      ... dispatch (_run_dynamic_or_hybrid / _run_static, incl. nested evaluatorq()) ...
  ```
  This binds the ContextVar for every child job task and the nested `evaluatorq()`.

- **Simulation** (`simulation/api.py`): add one small module-local helper and use it at every entrypoint (`simulate`, `generate_and_simulate`, `generate`, `generate_personas`, `generate_persona`, `generate_scenarios`, `generate_scenario`):
  ```python
  @contextmanager
  def _sim_run_scope(run_id: str, span):
      if span is not None and run_id:
          span.set_attribute('orq.evaluatorq_run_id', run_id)
      with evaluatorq_run_id(run_id):
          yield
  ```
  Each entrypoint mints `run_id = uuid4().hex` (entries that don't already) and wraps its dispatch/generation block in `with _sim_run_scope(run_id, pipeline_span):`. Wrapping at the outer entry — not inside `_simulate_core` — is deliberate: it tags the generate phase and the nested `evaluatorq()` too. The helper wraps the pure `evaluatorq_run_id` CM; it does not reinvent it.

### 3. Root-span attribute

`orq.evaluatorq_run_id` is set only on the red-team root span (`with_redteam_span`) and the sim root spans (`with_simulation_span` pipeline spans), guarded on a truthy `run_id` and a non-None span. `orq.run_id` on `orq.job` / `orq.evaluation` spans is untouched (decision 1).

## Naming

| Surface | Key |
|---|---|
| Invocation request `metadata` | `evaluatorq_run_id` |
| Root span attribute (red-team / sim only) | `orq.evaluatorq_run_id` |

## Files touched

- `src/evaluatorq/common/thread_context.py` — new ContextVar + `evaluatorq_run_id` CM; extend `pipeline_metadata()`; refactor `pipeline_metadata_param()` to wrap it; `__all__`.
- `src/evaluatorq/redteam/runner.py` — set root-span attr + bind `evaluatorq_run_id` around dispatch (single site).
- `src/evaluatorq/simulation/api.py` — `_sim_run_scope` helper + wrap every entrypoint; mint `run_id` where entrypoints lack one.
- `tests/unit/test_thread_context.py` — extend.
- (No `tracing/spans.py`, no `docs/tracing.md` edits — decision 1.)

## Testing

- Unit: inside `evaluatorq_run_id('r1')`, both `pipeline_metadata()` and `pipeline_metadata_param()` contain `evaluatorq_run_id == 'r1'`; absent outside the scope; combined with `evaluatorq_pipeline`, both keys present in both functions.
- Unit: concurrency isolation — mirror the existing `test_concurrent_tasks_are_isolated`; `asyncio.gather` two scopes with distinct run_ids, assert no leakage across tasks.
- Unit: exception inside `evaluatorq_run_id(...)` still resets `_run_id` (no leak).
- Integration-ish (mocked client): a red-team backend call and a sim call carry `evaluatorq_run_id` in the request `metadata`; a call issued from within a nested `evaluatorq()` inherits the same run_id.
- Unit: `_sim_run_scope(run_id, span)` sets `orq.evaluatorq_run_id` on a fake span and binds the ContextVar; with `span=None` or falsy `run_id` it no-ops the attribute and still binds.

## Out of scope

- Core `evaluate()` invocation metadata (request is red-team + simulation only).
- `orq.run_id` on job/evaluation spans, thread-id composition, and the manifest run_id (incl. `generate()`'s `''` no-save sentinel) — all unchanged.
