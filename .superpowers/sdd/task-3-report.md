# Task 3 Report — Orq-deployment target: token usage + pipeline/thread tagging

Commit: `5740761` — `fix(simulation): track token usage and tag Orq-deployment target calls`

## Files changed

### `src/evaluatorq/deployment.py`
- Added runtime import `from evaluatorq.contracts import TokenUsage` (top-level, alongside existing imports; no `TYPE_CHECKING` split needed since it's used at runtime in `deployment()`).
- Added `usage: TokenUsage | None = None` field to `DeploymentResponse` dataclass, with a one-line docstring comment matching the `content`/`raw` style.
- In `deployment()`, after `content = _extract_content_from_response(completion)`, added `usage = TokenUsage.extract(getattr(completion, 'usage', None), calls=1)` and changed the return to `DeploymentResponse(content=content, raw=completion, usage=usage)`.
- Purely additive — `invoke()` (which just returns `response.content`) and all other existing callers are unaffected.
- Confirmed no circular import: `contracts.py` has no import of `deployment.py`.

### `src/evaluatorq/simulation/adapters.py`
- Added `AgentResponse` to the top-level `from evaluatorq.contracts import ...` line (alongside `content_to_text`).
- `from_orq_deployment`'s return type changed from `Callable[[list[Message]], Awaitable[str]]` to `Callable[[list[Message]], Awaitable[AgentResponse]]`.
- The inner `callback`:
  - Lazily imports `current_thread_id`, `pipeline_metadata` from `evaluatorq.common.thread_context`, and `ThreadConfig`, `deployment` from `evaluatorq.deployment` (kept lazy, matching the file's existing style of deferring the Orq-touching import to call time).
  - Builds `metadata: dict[str, object] | None = pipeline_metadata() or None` and `thread: ThreadConfig | None = {'id': tid} if tid else None` (`tid = current_thread_id()`) — these two explicit annotations were needed to satisfy basedpyright (`pipeline_metadata()` returns `dict[str, str]`, which is invariant-incompatible with `dict[str, object]`; a bare `{'id': tid}` dict literal isn't structurally recognized as `ThreadConfig` without an annotation).
  - Calls `deployment(agent_key, messages=[...], metadata=metadata, thread=thread)` instead of `invoke(...)`.
  - Returns `AgentResponse(text=resp.content, usage=resp.usage)`.
- `callback.deployment_key = agent_key` attribute assignment (with its pyright-ignore) is unchanged.
- `from_chat_completions` was left untouched, as instructed.

### `tests/simulation/test_adapters_deployment.py` (new)
Three async tests, patching `evaluatorq.deployment.deployment` with an `AsyncMock` returning a real `DeploymentResponse`:
1. `test_from_orq_deployment_returns_agent_response_with_usage` — asserts the callback returns an `AgentResponse` with `.text == 'hello'` and a populated `.usage` (not `None`).
2. `test_from_orq_deployment_tags_pipeline_and_thread` — under `evaluatorq_pipeline('agent_simulation')` + `conversation_thread('t-123')`, asserts `deployment(...)` was called with `metadata={'evaluatorq_pipeline': 'agent_simulation'}` and `thread={'id': 't-123'}` via `call_args.kwargs`.
3. `test_from_orq_deployment_no_metadata_without_bound_context` — without any bound pipeline/thread, asserts `metadata=None` and `thread=None`.

`# ruff: noqa: S101` added at module scope (matching the convention used in `tests/simulation/test_pipeline_metadata.py`) since these are plain-`assert` tests, not pytest.raises-only.

## Downstream-caller grep findings (return-type change str → AgentResponse)

Grepped for `from_orq_deployment` usages across `src/` and `tests/`:
- `src/evaluatorq/simulation/wrap_agent.py:74` — `resolved_target = from_orq_deployment(agent_key)`; the resulting callable is only ever invoked by `SimulationRunner`, which routes target output through `common/target_call.py::_coerce_to_agent_response` (`if isinstance(raw, AgentResponse): return raw`). Safe.
- `src/evaluatorq/simulation/cli.py:148` — `return from_orq_deployment(value)`; only returns the callback, no output inspection. Safe.
- `src/evaluatorq/simulation/api.py:1303` — `return from_orq_deployment(value), None, 'orq_deployment'`; same, only returns the callback. `api.py:1075` separately does `getattr(target_callable, 'deployment_key', None)` on the *function object* (unrelated to its return value). Safe.
- `tests/simulation/test_cli.py:320` and `tests/simulation/test_resolve_target.py:31` — both patch/call `from_orq_deployment` only to check the factory call / `.deployment_key`, never invoke and inspect the callback's return value as a `str`. Confirmed unaffected by re-running the full `tests/simulation/` suite (see below).

No caller relies on a bare `str` return from the callback.

## Test commands + results

- `uv run pytest tests/simulation/test_adapters_deployment.py -q` → `3 passed in 0.08s`
- `uv run pytest tests/simulation/ -q` → `663 passed, 2 skipped, 15 warnings in 817.76s` (skips/deprecation warnings are pre-existing/unrelated to this change — sync-hook deprecation warnings in `test_hooks.py`)
- `uv run ruff check src/evaluatorq/deployment.py src/evaluatorq/simulation/adapters.py tests/simulation/test_adapters_deployment.py` → `All checks passed!` (after adding `# ruff: noqa: S101` for the new test file's plain asserts)
- `uv run ruff format --diff <same 3 files>` → `3 files already formatted`
- `uv run basedpyright src/evaluatorq/deployment.py src/evaluatorq/simulation/adapters.py` → `0 errors, 0 warnings, 0 notes` (after adding explicit `dict[str, object] | None` / `ThreadConfig | None` annotations to fix two `reportArgumentType` errors on the first pass)

## Commit

Only staged and committed the 3 intended files (`git add src/evaluatorq/deployment.py src/evaluatorq/simulation/adapters.py tests/simulation/test_adapters_deployment.py`) — did not touch/stage `docs/tracing.md`, `src/evaluatorq/common/reports/executive_summary.py`, `src/evaluatorq/simulation/reports/executive_summary.py`, `tests/redteam/test_tracing_spans.py`, or `tests/simulation/test_tracing.py`, which remain as pre-existing uncommitted changes in the working tree.

## Concerns

None blocking. Two minor notes:
- `deployment.py` now imports `TokenUsage` from `evaluatorq.contracts` at module load (previously `deployment.py` had no dependency on `contracts.py`); verified no circular-import issue since `contracts.py` doesn't import `deployment.py`, and the full test suite / basedpyright both pass clean.
- The `metadata`/`thread` type annotations added in `adapters.py` are slightly more verbose than a one-liner would be, but were required to satisfy basedpyright's invariant `dict` value-type checking against `deployment()`'s parameter types (`dict[str, object]` / `ThreadConfig`); this matches how the redteam backend also has to be explicit about kwarg shapes when calling into typed SDK wrappers.

---

# Follow-up: fix(simulation) — widen target types for AgentResponse-returning deployment adapter

Fixes the `uv run basedpyright` regressions introduced by commit `5740761` (this task's own change above), where `from_orq_deployment`'s callback return type moved from `str` to `AgentResponse`.

## Files changed

- `src/evaluatorq/simulation/adapters.py`
- `src/evaluatorq/simulation/api.py`
- `src/evaluatorq/simulation/wrap_agent.py`
- `src/evaluatorq/simulation/runner/simulation.py`

## The 3 (reported) errors and how each was resolved

1. **`adapters.py:31`** — `metadata: dict[str, object] | None = pipeline_metadata() or None` failed because `pipeline_metadata()` returns `dict[str, str]` and `dict`'s value type is invariant, so `dict[str, str]` isn't assignable to the declared `dict[str, object]`.
   - Tried the "drop/loosen the local annotation" approach first (`dict[str, str] | None`), but that only moved the error one line down: the invariant mismatch then surfaced at the `deployment(..., metadata=metadata, ...)` call site itself (`deployment()`'s param is `dict[str, object] | None`), since a `dict[str, str]` value — regardless of how the local variable is annotated — is never assignable to a `dict[str, object]`-typed parameter.
   - Fixed properly by rebuilding the dict with the *declared* value type in mind: `metadata: dict[str, object] | None = dict(pipeline_metadata()) or None`. The declared `dict[str, object]` annotation flows into the `dict(...)` constructor call as expected-type context, so pyright infers `dict[str, object]` for the constructor's result instead of copying the invariant `dict[str, str]` type through. This is a real fix (rebuilds the object with the correct declared type), not a cast-based suppression.

2. **`api.py:1259`** (`_resolve_target`'s return type) — still declared `Callable[[list[Message]], str | Awaitable[str]] | None` in the returned tuple, but the `TargetKind.DEPLOYMENT` branch now returns `from_orq_deployment(value)`, which is `Callable[[list[Message]], Awaitable[AgentResponse]]`.
   - Widened `_resolve_target`'s return annotation to `Callable[[list[Message]], str | Awaitable[str] | Awaitable[AgentResponse]] | None` (added `AgentResponse` to the `TYPE_CHECKING` import block).
   - This surfaced two knock-on errors at the two call sites that receive `_resolve_target`'s output and pass it further down (`_build_simulation_job_and_cache` and `_simulate_via_evaluatorq`, both taking a `target: Callable[[list[Message]], str | Awaitable[str]] | None` parameter) — widened those two signatures identically for consistency, since they're the same value flowing through, not a separate design decision.

3. **`wrap_agent.py:81`** (passing `resolved_target` into `SimulationRunner.__init__`) and **`runner/simulation.py:245`** (`SimulationRunner.__init__`'s `target` parameter) — `resolved_target` in `wrap_agent.py` is inferred as a union including `from_orq_deployment`'s `Callable[[list[Message]], Awaitable[AgentResponse]]` arm, which `SimulationRunner.__init__`'s `target: Callable[[list[Message]], str | Awaitable[str]] | None` didn't accept.
   - No existing shared type alias for this specific "target callable" shape lives in `common/target_call.py` (that module only owns `TargetCallResult`/retry helpers). The existing `AgentCallable` alias in `evaluatorq.integrations.callable_integration.target` already spans `str | AgentResponse` (sync + async) and is used internally in the runner via `cast('AgentCallable', target)`, but the public-facing `target=` parameter types across `wrap_agent.py` / `api.py` / `runner/simulation.py` intentionally use an inline `Callable[...]` form rather than that alias, so I kept the same inline style and just added the missing arm rather than introducing a new alias or switching to `AgentCallable` (larger, out-of-scope change).
   - Widened both `wrap_simulation_agent`'s `target` parameter (`wrap_agent.py`) and `SimulationRunner.__init__`'s `target` parameter (`runner/simulation.py`) to `Callable[[list[Message]], str | Awaitable[str] | Awaitable[AgentResponse]] | None`, mirroring the existing `str | Awaitable[str]` style with one added arm. Added `AgentResponse` to each file's `TYPE_CHECKING` import block.

No runtime behavior changed — `common/target_call.py::_coerce_to_agent_response` already handles both `str` and `AgentResponse` raw returns at call time; only the static type surface was widened to match what already flows at runtime.

## Final basedpyright summary

```
$ uv run basedpyright
.context/observe_span.py:57:32 - error: Expected type arguments for generic class "list" (reportMissingTypeArgument)
1 error, 0 warnings, 0 notes
```

The one remaining error is in `.context/observe_span.py`, which is outside `src/` and unrelated to commit `5740761` or this fix — confirmed pre-existing by stashing all changes (including the original commit's diff wasn't stashed, since it's already on the branch) and independently verifying via `git show 5740761` that this file was untouched by that commit. Project-wide `basedpyright` on `src/evaluatorq/**` is clean.

## ruff

```
$ uv run ruff check src
All checks passed!

$ uv run ruff format src/evaluatorq/simulation/adapters.py src/evaluatorq/simulation/api.py src/evaluatorq/simulation/wrap_agent.py src/evaluatorq/simulation/runner/simulation.py
4 files left unchanged
```

## Tests

```
$ uv run pytest tests/simulation/test_adapters_deployment.py tests/simulation/test_pipeline_metadata.py -q
11 passed in 0.10s

$ uv run pytest tests/simulation/ -q
663 passed, 2 skipped in ~818s
```

(skips and sync-hook `DeprecationWarning`s in `test_hooks.py` are pre-existing/unrelated.)

## Commit

Staged only the 4 changed files explicitly (`git add src/evaluatorq/simulation/adapters.py src/evaluatorq/simulation/api.py src/evaluatorq/simulation/wrap_agent.py src/evaluatorq/simulation/runner/simulation.py`) — did not touch `docs/tracing.md`, `src/evaluatorq/common/reports/executive_summary.py`, `src/evaluatorq/simulation/reports/executive_summary.py`, `tests/redteam/test_tracing_spans.py`, or `tests/simulation/test_tracing.py`, which remain pre-existing uncommitted changes in the working tree.
