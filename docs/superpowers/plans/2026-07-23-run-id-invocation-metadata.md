# Run-id Invocation Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tag every LLM chat-completion / responses invocation issued during a red-team or agent-simulation run with the run's `run_id` (as request `metadata`), and stamp that id on the run's root trace span, so an operator can filter every model call of one run in Orq.

**Architecture:** Reuse each pipeline's existing per-run `run_id`. Carry it on the existing `thread_context` ContextVar rail (mirroring the `evaluatorq_pipeline` label). One guarded helper (`run_metadata_kwarg`) is the single source of truth for the `{'metadata': {...}}` payload. Call sites that already merge `pipeline_metadata()` get the run_id for free; sites that issue `create()` directly (the Responses-API agent path, report + generation calls) get an explicit guarded merge. The root red-team / simulation span gets an `orq.evaluatorq_run_id` attribute. The evaluatorq-core `orq.run_id` attribute on `orq.job` / `orq.evaluation` spans is left untouched.

**Tech Stack:** Python 3.10+, `contextvars`, OpenTelemetry, `openai` AsyncOpenAI, pytest / pytest-asyncio, `uv`.

## Global Constraints

- Python 3.10+ compatible; every module already uses `from __future__ import annotations`. Keep it.
- Package manager is `uv`. Run tests with `uv run pytest -m 'not integration'`.
- Lint: `uv run ruff check src`. Format: `uv run ruff format src`. Types: `uv run basedpyright`.
- Metadata key on invocations: `evaluatorq_run_id`. Root span attribute: `orq.evaluatorq_run_id`.
- Do NOT edit `src/evaluatorq/tracing/spans.py` or `docs/tracing.md` — `orq.run_id` stays (evaluatorq-core).
- The guarded merge must no-op when the client does not route through Orq (a plain OpenAI endpoint rejects unknown fields), reusing `client_routes_through_orq`.
- Commit after each task with a Conventional Commit message.

---

### Task 1: Carry `run_id` on the ContextVar rail

**Files:**
- Modify: `src/evaluatorq/common/thread_context.py`
- Test: `tests/unit/test_thread_context.py`

**Interfaces:**
- Consumes: existing `_pipeline` ContextVar + `pipeline_metadata()` in the same file.
- Produces:
  - `evaluatorq_run_id(run_id: str) -> Iterator[str]` — sync `@contextmanager`, binds/resets `_run_id`.
  - `pipeline_metadata() -> dict[str, str]` — now also emits `'evaluatorq_run_id'` when `_run_id` is set.
  - `pipeline_metadata_param() -> dict[str, dict[str, str]]` — now wraps `pipeline_metadata()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_thread_context.py`:

```python
import asyncio

from evaluatorq.common.thread_context import (
    evaluatorq_pipeline,
    evaluatorq_run_id,
    pipeline_metadata,
    pipeline_metadata_param,
)


def test_run_id_absent_when_unset() -> None:
    assert 'evaluatorq_run_id' not in pipeline_metadata()
    assert pipeline_metadata_param() == {} or 'evaluatorq_run_id' not in pipeline_metadata_param().get('metadata', {})


def test_run_id_present_in_both_metadata_forms() -> None:
    with evaluatorq_run_id('r1'):
        assert pipeline_metadata()['evaluatorq_run_id'] == 'r1'
        assert pipeline_metadata_param()['metadata']['evaluatorq_run_id'] == 'r1'
    # restored after scope
    assert 'evaluatorq_run_id' not in pipeline_metadata()


def test_run_id_and_pipeline_label_travel_together() -> None:
    with evaluatorq_pipeline('agent_simulation'), evaluatorq_run_id('r2'):
        md = pipeline_metadata()
        assert md['evaluatorq_pipeline'] == 'agent_simulation'
        assert md['evaluatorq_run_id'] == 'r2'
        assert pipeline_metadata_param()['metadata'] == md


def test_run_id_resets_on_exception() -> None:
    try:
        with evaluatorq_run_id('r3'):
            raise RuntimeError('boom')
    except RuntimeError:
        pass
    assert 'evaluatorq_run_id' not in pipeline_metadata()


def test_run_id_concurrent_tasks_are_isolated() -> None:
    async def _run() -> set[str]:
        seen: set[str] = set()

        async def worker(rid: str) -> None:
            with evaluatorq_run_id(rid):
                await asyncio.sleep(0)
                seen.add(pipeline_metadata()['evaluatorq_run_id'])

        await asyncio.gather(worker('a'), worker('b'), worker('c'))
        return seen

    assert asyncio.run(_run()) == {'a', 'b', 'c'}
    assert 'evaluatorq_run_id' not in pipeline_metadata()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_thread_context.py -k run_id -v` Expected: FAIL — `ImportError: cannot import name 'evaluatorq_run_id'`.

- [ ] **Step 3: Implement in `src/evaluatorq/common/thread_context.py`**

Add the ContextVar next to `_pipeline` (after line 35):

```python
# Per-run identifier ('run_id') for the active red-team / agent-simulation run.
# Carried on the same ContextVar rail as the pipeline label so it reaches the
# stateless, shared target/judge/generator instances that read pipeline_metadata().
_run_id: ContextVar[str | None] = ContextVar('evaluatorq_run_id', default=None)
```

Replace `pipeline_metadata()` (lines 86-94) with:

```python
def pipeline_metadata() -> dict[str, str]:
    """Return run-identifying invocation metadata for the active run, or ``{}``.

    Emits ``evaluatorq_pipeline`` (run surface) and ``evaluatorq_run_id`` (this run's
    id) when each is bound. Pass straight to the native ``metadata=`` kwarg on
    ``chat.completions.create`` / ``responses.create`` so the invocation's Orq trace
    is filterable by run type and by run.
    """
    md: dict[str, str] = {}
    label = _pipeline.get()
    if label:
        md['evaluatorq_pipeline'] = label
    run_id = _run_id.get()
    if run_id:
        md['evaluatorq_run_id'] = run_id
    return md
```

Replace `pipeline_metadata_param()` (lines 97-105) with a wrapper so the two forms cannot drift:

```python
def pipeline_metadata_param() -> dict[str, dict[str, str]]:
    """``extra_body``-ready ``{'metadata': {...}}`` form of :func:`pipeline_metadata`.

    Wraps :func:`pipeline_metadata` (single source of truth) so the two forms stay
    in sync. The SDK merges ``extra_body`` into the request body, yielding the same
    top-level ``metadata`` property.
    """
    md = pipeline_metadata()
    return {'metadata': md} if md else {}
```

Add the context manager after `evaluatorq_pipeline` (after line 121):

```python
@contextmanager
def evaluatorq_run_id(run_id: str) -> Iterator[str]:
    """Bind the run id for a red-team / agent-simulation run.

    Restores the previous value on exit so nested/sequential runs don't bleed
    (mirrors :func:`evaluatorq_pipeline`).

    Yields:
        The bound run id.
    """
    token = _run_id.set(run_id)
    try:
        yield run_id
    finally:
        _run_id.reset(token)
```

Add `'evaluatorq_run_id'` to `__all__` (keep alphabetical, after `'evaluatorq_pipeline'`):

```python
    'evaluatorq_pipeline',
    'evaluatorq_run_id',
    'pipeline_metadata',
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_thread_context.py -v` Expected: PASS (all, including pre-existing tests).

- [ ] **Step 5: Commit**

```bash
git add src/evaluatorq/common/thread_context.py tests/unit/test_thread_context.py
git commit -m "feat(tracing): carry run_id on the pipeline-metadata ContextVar rail"
```

---

### Task 2: One guarded `run_metadata_kwarg` helper

> **Correction, post-implementation:** the helper shipped as `_run_metadata_kwarg` (private). Every call site uses the public `apply_pipeline_metadata`, which is its only caller — a second public entry point with a different shape had no consumers. Names below reflect the plan as written.

**Files:**
- Modify: `src/evaluatorq/common/llm_call.py`
- Test: `tests/unit/test_run_metadata_kwarg.py` (create)

**Interfaces:**
- Consumes: `client_routes_through_orq` (already imported), `pipeline_metadata` (already imported).
- Produces: `run_metadata_kwarg(client: AsyncOpenAI | None) -> dict[str, dict[str, str]]` — returns `{'metadata': {...}}` when the client routes through Orq and metadata is bound, else `{}`. `_apply_pipeline_metadata` is refactored to delegate to it (unchanged external behavior).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_run_metadata_kwarg.py`:

```python
from __future__ import annotations

import evaluatorq.common.llm_call as llm_call
from evaluatorq.common.llm_call import run_metadata_kwarg
from evaluatorq.common.thread_context import evaluatorq_pipeline, evaluatorq_run_id


def test_empty_off_orq(monkeypatch) -> None:
    monkeypatch.setattr(llm_call, 'client_routes_through_orq', lambda _c: False)
    with evaluatorq_run_id('r1'):
        assert run_metadata_kwarg(object()) == {}


def test_empty_when_nothing_bound_on_orq(monkeypatch) -> None:
    monkeypatch.setattr(llm_call, 'client_routes_through_orq', lambda _c: True)
    assert run_metadata_kwarg(object()) == {}


def test_metadata_on_orq_when_bound(monkeypatch) -> None:
    monkeypatch.setattr(llm_call, 'client_routes_through_orq', lambda _c: True)
    with evaluatorq_pipeline('red_teaming'), evaluatorq_run_id('r1'):
        assert run_metadata_kwarg(object()) == {
            'metadata': {'evaluatorq_pipeline': 'red_teaming', 'evaluatorq_run_id': 'r1'}
        }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_run_metadata_kwarg.py -v` Expected: FAIL — `ImportError: cannot import name 'run_metadata_kwarg'`.

- [ ] **Step 3: Implement in `src/evaluatorq/common/llm_call.py`**

Add the public helper immediately above `_apply_pipeline_metadata` (before its `def` near line 95):

```python
def run_metadata_kwarg(client: AsyncOpenAI | None) -> dict[str, dict[str, str]]:
    """Guarded ``{'metadata': {...}}`` for splatting into a ``create()`` call.

    Returns ``{}`` off-Orq (a plain OpenAI endpoint rejects unknown fields) or when
    no run is bound. Single source of truth for tagging direct ``create()`` sites
    that don't go through :func:`execute_chat_completion`.
    """
    if not client_routes_through_orq(client):
        return {}
    md = pipeline_metadata()
    return {'metadata': md} if md else {}
```

Refactor `_apply_pipeline_metadata` (lines 95-106) to delegate:

```python
def _apply_pipeline_metadata(client: AsyncOpenAI, params: dict[str, Any]) -> None:
    """Tag the invocation with the active run surface + run id via ``metadata``.

    No-op off-Orq or when no run is bound. Caller-supplied metadata (via
    ``extra_kwargs``) wins on key conflict.
    """
    md = run_metadata_kwarg(client).get('metadata')
    if md:
        params['metadata'] = {**md, **(params.get('metadata') or {})}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_run_metadata_kwarg.py tests/unit/test_thread_context.py -v` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluatorq/common/llm_call.py tests/unit/test_run_metadata_kwarg.py
git commit -m "feat(tracing): add guarded run_metadata_kwarg helper"
```

---

### Task 3: Tag the direct-`create()` (Class-B) call sites

These four sites issue `create()` / `parse()` directly and currently carry no pipeline metadata. Each gets one guarded merge via `run_metadata_kwarg`.

**Files:**
- Modify: `src/evaluatorq/simulation/agents/base.py` (Responses-API agent path)
- Modify: `src/evaluatorq/common/reports/executive_summary.py`
- Modify: `src/evaluatorq/simulation/utils/structured_output.py`
- Modify: `src/evaluatorq/simulation/generators/first_message_generator.py`
- Test: `tests/simulation/test_run_metadata_propagation.py` (create)

**Interfaces:**
- Consumes: `run_metadata_kwarg` (Task 2), `evaluatorq_run_id` (Task 1).
- Produces: no new symbols; each `create()`/`parse()` now receives `metadata` when a run is bound and the client routes through Orq.

- [ ] **Step 1: Write the failing test**

Create `tests/simulation/test_run_metadata_propagation.py`:

```python
from __future__ import annotations

from typing import Any

import pytest

import evaluatorq.common.llm_call as llm_call
from evaluatorq.common.thread_context import evaluatorq_run_id


class _FakeChatCompletions:
    def __init__(self, sink: dict[str, Any]) -> None:
        self._sink = sink

    async def create(self, **kwargs: Any) -> Any:
        self._sink['create'] = kwargs
        return _FakeChatResponse()

    async def parse(self, **kwargs: Any) -> Any:
        self._sink['parse'] = kwargs
        return _FakeParseResponse()


class _FakeChatResponse:
    class _Msg:
        content = 'hi'
        tool_calls = None
        refusal = None
    class _Choice:
        message = _FakeChatResponse._Msg()
        finish_reason = 'stop'
    choices = [_Choice()]


class _FakeParseResponse:
    class _Msg:
        content = '{}'
        refusal = None
        parsed = {'ok': True}
    class _Choice:
        message = _FakeParseResponse._Msg()
    choices = [_Choice()]


class _FakeClient:
    def __init__(self, sink: dict[str, Any]) -> None:
        self.chat = type('C', (), {'completions': _FakeChatCompletions(sink)})()


@pytest.fixture(autouse=True)
def _force_orq(monkeypatch):
    monkeypatch.setattr(llm_call, 'client_routes_through_orq', lambda _c: True)


@pytest.mark.asyncio
async def test_first_message_generator_tags_run_id(monkeypatch) -> None:
    from evaluatorq.simulation.generators.first_message_generator import FirstMessageGenerator

    sink: dict[str, Any] = {}
    gen = FirstMessageGenerator(model='gpt-4o', client=_FakeClient(sink))
    with evaluatorq_run_id('r1'):
        await gen.generate(persona=_stub_persona(), scenario=_stub_scenario())
    assert sink['create']['metadata']['evaluatorq_run_id'] == 'r1'


@pytest.mark.asyncio
async def test_generate_structured_tags_run_id(monkeypatch) -> None:
    from evaluatorq.simulation.utils.structured_output import generate_structured

    sink: dict[str, Any] = {}
    with evaluatorq_run_id('r2'):
        await generate_structured(
            client=_FakeClient(sink),
            model='gpt-4o',
            messages=[{'role': 'user', 'content': 'x'}],
            response_model=dict,
            label='t',
        )
    assert sink['parse']['metadata']['evaluatorq_run_id'] == 'r2'
```

NOTE for the implementer: `_stub_persona()` / `_stub_scenario()` and the exact `generate_structured` / `FirstMessageGenerator.generate` signatures must be read from the source first; adapt the two call helpers to the real constructors (they may require minimal fields). If a real constructor is awkward to stub, drop that sub-test and instead assert on `run_metadata_kwarg` being present in the `extra` dict via a thinner seam — but keep at least one end-to-end assertion per file touched.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simulation/test_run_metadata_propagation.py -v` Expected: FAIL — `KeyError: 'metadata'` (metadata not yet added at the create sites).

- [ ] **Step 3: Implement the four merges**

**(a) `src/evaluatorq/simulation/agents/base.py`** — add `run_metadata_kwarg` to the llm_call import block (line 18-24):

```python
from evaluatorq.common.llm_call import (
    execute_chat_completion,
    run_metadata_kwarg,
    ...  # keep existing names
)
```

In `_call_responses`, inside `_do_call`, right after the thread block (after line 391 `call_kwargs['extra_body'] = {...}`):

```python
                call_kwargs.update(run_metadata_kwarg(self._client))
```

This carries metadata on both the initial and the reasoning-stripped retry (both reuse `call_kwargs`).

**(b) `src/evaluatorq/common/reports/executive_summary.py`** — add the import near the top with the other `evaluatorq.common` imports:

```python
from evaluatorq.common.llm_call import run_metadata_kwarg
```

Merge into `merged_kwargs` (after it is built, before the `create` call around line 120):

```python
        merged_kwargs.update(run_metadata_kwarg(llm_client))
```

**(c) `src/evaluatorq/simulation/utils/structured_output.py`** — add the import:

```python
from evaluatorq.common.llm_call import run_metadata_kwarg
```

Merge into `extra` right after it is built (after line 59 `extra: dict[str, Any] = {...}`), so both `parse()` (primary) and `create()` (fallback) inherit it:

```python
    extra.update(run_metadata_kwarg(client))
```

**(d) `src/evaluatorq/simulation/generators/first_message_generator.py`** — add the import:

```python
from evaluatorq.common.llm_call import run_metadata_kwarg
```

Merge into `extra` right after it is built (after line 133 `extra: dict[str, Any] = {...}`):

```python
                extra.update(run_metadata_kwarg(self._client))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/simulation/test_run_metadata_propagation.py -v` Expected: PASS. Then `uv run pytest -m 'not integration' -q` to confirm no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/evaluatorq/simulation/agents/base.py src/evaluatorq/common/reports/executive_summary.py src/evaluatorq/simulation/utils/structured_output.py src/evaluatorq/simulation/generators/first_message_generator.py tests/simulation/test_run_metadata_propagation.py
git commit -m "feat(tracing): tag direct-create LLM calls with run_id metadata"
```

---

### Task 4: Bind run_id + stamp root span in the red-team runner

**Files:**
- Modify: `src/evaluatorq/redteam/runner.py`
- Test: `tests/redteam/test_tracing_spans.py`

**Interfaces:**
- Consumes: `evaluatorq_run_id` (Task 1), `tracing_context.run_id`, existing `set_span_attrs`.
- Produces: the `'Orq Red Team'` root span carries `orq.evaluatorq_run_id`; all dispatch (incl. nested `evaluatorq()`, executive-summary + recommendations calls) runs inside `evaluatorq_run_id(...)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/redteam/test_tracing_spans.py` (reuse the `span_collector` fixture + `_find_span` + `_attrs` helpers already in the file; model this on `test_red_team_owns_whole_pipeline_span`):

```python
@pytest.mark.asyncio
async def test_root_span_carries_evaluatorq_run_id(
    span_collector: _CollectingExporter,
) -> None:
    # Reuse whatever minimal red_team(...) invocation test_red_team_owns_whole_pipeline_span
    # uses to produce a 'Orq Red Team' span. Copy that setup here.
    ...  # <run a minimal red_team() exactly as the sibling test does>
    pipeline_span = _find_span(span_collector, 'Orq Red Team')
    assert pipeline_span is not None
    attrs = _attrs(pipeline_span)
    assert attrs.get('orq.evaluatorq_run_id')  # non-empty run id present
```

The implementer copies the exact `red_team(...)` setup from `test_red_team_owns_whole_pipeline_span` (same fixture, targets, mode) into the `...` block.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/redteam/test_tracing_spans.py::test_root_span_carries_evaluatorq_run_id -v` Expected: FAIL — attribute absent.

- [ ] **Step 3: Implement in `src/evaluatorq/redteam/runner.py`**

Add `evaluatorq_run_id` to the import block (lines 28-33):

```python
from evaluatorq.common.thread_context import (
    build_static_thread_id,
    conversation_thread,
    evaluatorq_pipeline,
    evaluatorq_run_id,
    pipeline_metadata_param,
)
```

Wrap the root-span body in `evaluatorq_run_id`. Change lines 718-722 from:

```python
        async with with_redteam_span(
            'Orq Red Team',
            pipeline_attributes,
            parent_context=tracing_context.parent_context,
        ) as pipeline_span:
            if resolved_mode in (Pipeline.DYNAMIC, Pipeline.HYBRID):
```

to:

```python
        async with with_redteam_span(
            'Orq Red Team',
            pipeline_attributes,
            parent_context=tracing_context.parent_context,
        ) as pipeline_span, evaluatorq_run_id(tracing_context.run_id):
            if tracing_context.run_id:
                set_span_attrs(pipeline_span, {'orq.evaluatorq_run_id': tracing_context.run_id})
            if resolved_mode in (Pipeline.DYNAMIC, Pipeline.HYBRID):
```

(**Correction, post-implementation:** this step originally claimed `async with` accepts sync context managers. It does not — every item in an `async with` group must implement `__aenter__`/`__aexit__`, and a plain `@contextmanager` raises `TypeError` there. `evaluatorq_run_id` therefore returns `_RunIdScope`, which implements both protocols, so it can be bound in the runner's existing `async with (...)` tuple without indenting the pipeline body.) The existing `if resolved_mode ...` block and everything below it keeps its current indentation — it was already one level under `as pipeline_span:`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/redteam/test_tracing_spans.py -v` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluatorq/redteam/runner.py tests/redteam/test_tracing_spans.py
git commit -m "feat(redteam): bind run_id + stamp orq.evaluatorq_run_id on root span"
```

---

### Task 5: Bind run_id + stamp root span across every simulation entrypoint

**Files:**
- Modify: `src/evaluatorq/simulation/api.py`
- Test: `tests/simulation/test_run_metadata_propagation.py` (extend from Task 3)

**Interfaces:**
- Consumes: `evaluatorq_run_id` (Task 1), `uuid`, `with_simulation_span` pipeline spans.
- Produces: module-local `_sim_run_scope(run_id: str, span: Any | None) -> Iterator[None]` — sets `orq.evaluatorq_run_id` on a non-None span (guarded on truthy run_id) and binds `evaluatorq_run_id`. Every entrypoint (`simulate`, `generate_and_simulate`, `generate`, `generate_personas`, `generate_persona`, `generate_scenarios`, `generate_scenario`) runs its LLM work inside it, each with a unique minted `run_id`.

- [ ] **Step 1: Write the failing test**

Extend `tests/simulation/test_run_metadata_propagation.py`:

```python
@pytest.mark.asyncio
async def test_generate_personas_binds_a_run_id(monkeypatch) -> None:
    # generate_personas mints its own run_id internally; assert the create call
    # it issues carries a non-empty evaluatorq_run_id.
    import evaluatorq.simulation.api as sim_api

    sink: dict[str, Any] = {}

    # Force the PersonaGenerator to use our fake client by injecting generation_client.
    from evaluatorq.simulation.api import generate_personas

    personas = await generate_personas(
        ['angry customer'],
        agent_description='support bot',
        generation_client=_FakeClient(sink),
    )
    md = (sink.get('parse') or sink.get('create') or {}).get('metadata', {})
    assert md.get('evaluatorq_run_id')  # some unique id was bound
```

NOTE: `generate_personas` routes through `generate_structured` (Task 3(c)) which already merges `run_metadata_kwarg`. This test therefore verifies the *binding* added in this task, not the merge. Adapt `_FakeClient` if the persona generator uses `.parse`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simulation/test_run_metadata_propagation.py::test_generate_personas_binds_a_run_id -v` Expected: FAIL — no run_id bound (metadata absent), because no entrypoint binds `evaluatorq_run_id` yet.

- [ ] **Step 3: Implement in `src/evaluatorq/simulation/api.py`**

Add imports near the top (line 17 area already has `import uuid`; line 22 has the thread_context import):

```python
from contextlib import contextmanager  # if not already imported
```

Extend the thread_context import (line 22):

```python
from evaluatorq.common.thread_context import build_thread_id, evaluatorq_run_id
```

Add the module-local helper near the other private helpers (e.g. just below `_compose_sim_hooks`):

```python
@contextmanager
def _sim_run_scope(run_id: str, span: Any | None):
    """Bind ``run_id`` for the run and stamp it on the run's root span.

    Wraps the pure :func:`evaluatorq_run_id` CM (no reinvention). The span
    attribute is a one-shot stamp; the ContextVar bind is what reaches the
    nested ``evaluatorq()`` LLM/evaluator calls and the generation-phase calls.
    """
    if span is not None and run_id:
        span.set_attribute('orq.evaluatorq_run_id', run_id)
    with evaluatorq_run_id(run_id):
        yield
```

Then, at **each** entrypoint, mint a `run_id` (reuse the existing minted one where present) and wrap the LLM-issuing body in `_sim_run_scope`:

- `simulate` (line ~306): a `run_id` already exists. Wrap the dispatch (the `try:`/`return await _simulate_core(...)` block) in `with _sim_run_scope(run_id, pipeline_span):`.
- `generate_and_simulate` (line ~575): `run_id` exists. Wrap the generate→simulate body in `with _sim_run_scope(run_id, pipeline_span):`.
- `generate` (line ~726): replace the `run_id=''` no-save sentinel usage by minting a real id for the scope, keeping `''` for the manifest hook. Concretely, before the `with_simulation_span(...)` body, add `run_id = uuid.uuid4().hex`; keep `_compose_sim_hooks(..., run_id='')` unchanged; capture the span via `as pipeline_span:` on the existing `with_simulation_span('orq.simulation.generate', {...}) as pipeline_span:` and wrap the GENERATE body in `with _sim_run_scope(run_id, pipeline_span):`.
- `generate_personas` (line ~760) and `generate_scenarios` (line ~826): these open no span. Mint `run_id = uuid.uuid4().hex` at the top of the function body and wrap the `asyncio.gather(...)` block in `with _sim_run_scope(run_id, None):`.
- `generate_persona` (line ~804) and `generate_scenario` (line ~867): they delegate to the batch form, which now binds a run_id — no change needed (the batch call runs inside its own scope). Leave them as-is.

For each wrap, indent the enclosed body one level; do not change any other logic.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/simulation/test_run_metadata_propagation.py -v` Then: `uv run pytest -m 'not integration' -q` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluatorq/simulation/api.py tests/simulation/test_run_metadata_propagation.py
git commit -m "feat(simulation): bind unique run_id + stamp root span at every entrypoint"
```

---

### Task 6: Final verification

- [ ] **Step 1: Full unit suite**

Run: `uv run pytest -m 'not integration' -q` Expected: PASS.

- [ ] **Step 2: Lint + format + types**

Run:
```bash
uv run ruff check src
uv run ruff format --check src
uv run basedpyright
```
Expected: clean (basedpyright is lenient; fix any new errors introduced by these changes).

- [ ] **Step 3: Grep for the goal invariant**

Run: `rtk proxy grep -rn "run_metadata_kwarg\|evaluatorq_run_id" src/evaluatorq` Expected: helper referenced at the 4 Class-B sites + `_apply_pipeline_metadata`; `evaluatorq_run_id` bound in redteam runner + sim api + defined in thread_context.

- [ ] **Step 4: Commit any lint/type fixups**

```bash
git add -A && git commit -m "chore(tracing): lint/type fixups for run-id metadata"
```

---

## Self-Review

**Spec coverage:**
- Reuse existing `run_id`, no ULID/dep → Tasks 1, 4, 5. ✅
- Extend BOTH `pipeline_metadata()` and `pipeline_metadata_param()` (independent → wrap) → Task 1. ✅
- Cover all metadata-rail consumers for free → Task 1 (Class A). ✅
- Cover the direct-`create()` bypass sites (agent Responses path, exec summary, structured gen, first-message) → Task 3 (Class B, full-coverage decision). ✅
- Correlation flows into nested `evaluatorq()` via ContextVar, no new param → Tasks 4, 5 (wrap at outer scope). ✅
- Root span attr `orq.evaluatorq_run_id` on redteam + sim roots only; `orq.run_id` untouched → Tasks 4, 5; Global Constraints. ✅
- Every sim entrypoint gets a unique id; `generate()` `''` manifest sentinel left alone → Task 5. ✅
- Tests: concurrency isolation, exception reset, both metadata forms, nested/generation propagation, root-span attr → Tasks 1, 3, 4, 5. ✅

**Placeholder scan:** Two intentional "read the real signature and adapt the stub" notes in Tasks 3 & 5 tests — these are guidance for test-fixture wiring against real constructors, not implementation placeholders; the production edits are fully specified. The red-team root-span test reuses an existing sibling test's setup (referenced explicitly).

**Type consistency:** `run_metadata_kwarg(client) -> dict[str, dict[str, str]]` used identically in Tasks 2/3. `evaluatorq_run_id(run_id: str)` bound in Tasks 4/5 as defined in Task 1. `_sim_run_scope(run_id, span)` defined and used only within Task 5.
