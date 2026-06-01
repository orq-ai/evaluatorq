# RES-897: Extract OpenResponses Runtime + Shared Models out of simulation/ — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple `redteam/` from `simulation/` by relocating the domain-generic OpenResponses runtime and shared helpers to leaf modules (`openresponses/`, `common/`), so both domains depend on a shared layer and never on each other.

**Architecture:** Pure import-relocation refactor with **one documented behavior delta** (`response_id` now populated). Moves + import rewrites + one new classmethod (`AgentResponse.from_openresponses`, collapsing a 3-way-split parse into one path) + name-collision aliasing. The duplicated *tracing* layer is explicitly OUT of scope (follow-up RES-899, blocked by this).

**Tech Stack:** Python 3.10+, pydantic, uv, basedpyright, pytest. Run from `packages/evaluatorq-py/`.

**Global invariant (every task):** After each task, BOTH must be green:
- `uv run basedpyright` → `0 errors, 0 warnings`
- `uv run pytest -m 'not integration' -q` → all pass (baseline: 1414 passed)

**Base:** branched off `origin/main` (res-540 already merged via PR #131). Branch: `bauke/res-897-refactorevaluatorq-py-extract-openresponses-runtime-shared`.

**Guiding rule (DRY home):** Anything used in >1 module goes to a shared home — never copy-pasted, never imported cross-domain. Generic helper → `common/`; cohesive around the OpenResponses wire format → `openresponses/`.

---

## File Structure (end state)

```
common/
  fields.py    ← NEW (from simulation/utils/fields.py)   exports get_field
  retry.py     ← NEW (from simulation/utils/retry.py)    exports with_retry, MAX_RETRY_ATTEMPTS, ...
openresponses/
  client.py    ← NEW (from simulation/_client.py)        exports build_simulation_client
  target.py    ← NEW (from simulation/target.py)         exports OrqResponsesTarget
contracts.py   ← AgentResponse.from_openresponses() added; TokenUsage stays the canonical home
simulation/
  _client.py   ← DELETED
  target.py    ← DELETED
  utils/fields.py ← DELETED
  utils/retry.py  ← DELETED
  types.py     ← TokenUsage no longer re-exported (importers repointed to contracts)
  tracing.py, judge, user_simulator, prompt_builders, runner, domain types ← UNTOUCHED (RES-899 scope)
```

**Task order is dependency-driven** — do not reorder. T4 (parse consolidation) must precede T5 (client move). T5 must precede T6 (target move).

---

### Task 1: Move `fields.py` → `common/fields.py`

Leaf helper, zero behavior change. `utils/__init__.py` does NOT re-export it, so only the 4 direct importers need repointing.

**Files:**
- Create: `src/evaluatorq/common/fields.py`
- Delete: `src/evaluatorq/simulation/utils/fields.py`
- Modify importers: `src/evaluatorq/simulation/tracing.py`, `src/evaluatorq/simulation/_client.py`, `src/evaluatorq/simulation/target.py`, `src/evaluatorq/openresponses/dataset.py`
- Tests: existing suite (no fields-specific unit test file).

- [ ] **Step 1: Move the file**

```bash
git mv src/evaluatorq/simulation/utils/fields.py src/evaluatorq/common/fields.py
```

The file content is unchanged (it has no internal evaluatorq imports):

```python
"""Shared field-accessor utility for dict/object dual-natured payloads."""

from __future__ import annotations

from typing import Any

def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Get a named field from a dict or object attribute."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
```

- [ ] **Step 2: Repoint all 4 importers**

In each file, replace `from evaluatorq.simulation.utils.fields import get_field` (note: some alias as `_get_field` / `_field` — preserve the alias) with `from evaluatorq.common.fields import get_field` (keep the same `as` alias). Exact current forms:

- `simulation/tracing.py`, `simulation/_client.py`, `simulation/target.py`: `from evaluatorq.simulation.utils.fields import get_field as _get_field` → `from evaluatorq.common.fields import get_field as _get_field`
- `openresponses/dataset.py`: check the exact alias used (`grep -n fields src/evaluatorq/openresponses/dataset.py`) and repoint preserving it. **This also removes the openresponses→simulation backwards dep noted in the ticket.**

- [ ] **Step 3: Verify no stale references**

```bash
grep -rn "simulation.utils.fields\|simulation/utils/fields" src/ tests/ | grep -v __pycache__
```
Expected: no output.

- [ ] **Step 4: Verify green**

```bash
uv run basedpyright 2>&1 | tail -1
uv run pytest -m 'not integration' -q 2>&1 | tail -3
```
Expected: `0 errors`; all tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor(evaluatorq-py): move fields helper to common/ (RES-897)"
```

---

### Task 2: Move `retry.py` → `common/retry.py`

Leaf helper, zero behavior change.

**Files:**
- Create: `src/evaluatorq/common/retry.py`
- Delete: `src/evaluatorq/simulation/utils/retry.py`
- Modify importers (src): `simulation/target.py`, `simulation/agents/base.py`, `simulation/utils/structured_output.py`, `simulation/generators/first_message_generator.py`
- Modify importers (tests): `tests/simulation/test_first_message_generator_errors.py`, `tests/simulation/test_orq_responses_target.py`

- [ ] **Step 1: Move the file**

```bash
git mv src/evaluatorq/simulation/utils/retry.py src/evaluatorq/common/retry.py
```
Content unchanged (imports are stdlib + `openai`, `loguru` only — no evaluatorq imports).

- [ ] **Step 2: Repoint all importers**

In each of the 4 src files and 2 test files, replace `from evaluatorq.simulation.utils.retry import ...` with `from evaluatorq.common.retry import ...` (preserve the imported names, e.g. `with_retry`, `MAX_RETRY_ATTEMPTS`).

- [ ] **Step 3: Verify no stale references**

```bash
grep -rn "simulation.utils.retry\|simulation/utils/retry" src/ tests/ | grep -v __pycache__
```
Expected: no output.

- [ ] **Step 4: Verify green** (same commands as Task 1 Step 4)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor(evaluatorq-py): move retry helper to common/ (RES-897)"
```

---

### Task 3: Kill `TokenUsage` re-export from `simulation/types.py`

`TokenUsage` is canonically defined in `contracts.py:189`. `simulation/types.py:12` imports it from contracts and other modules import it *via* types — an indirection. Repoint every external importer to `contracts`. `types.py` keeps its own import (it annotates `TokenUsage` internally at lines 282/303) — that is fine; the goal is that no *other* module relies on the re-export path.

**Files (repoint `TokenUsage` only; leave other names from `simulation.types` alone):**
- `src/evaluatorq/simulation/__init__.py`
- `src/evaluatorq/simulation/runner/simulation.py`
- `src/evaluatorq/simulation/_client.py`
- `src/evaluatorq/simulation/agents/base.py`
- `tests/simulation/test_convert.py`, `test_types.py`, `test_tracing.py`, `test_evaluators.py`, `test_wrap_agent.py`, `test_stamp_scores.py`, `test_simulate_injection.py`
- `tests/openresponses/test_backend_registry.py`

- [ ] **Step 1: Repoint each importer**

For each file, find the `from evaluatorq.simulation.types import ...` line containing `TokenUsage`. Remove `TokenUsage` from that line (keep the other names) and add a separate `from evaluatorq.contracts import TokenUsage`. If `TokenUsage` is the only name imported from `simulation.types` in that file, replace the whole line. Example for `agents/base.py:23`:

```python
# before
from evaluatorq.simulation.types import DEFAULT_MODEL, Message, TokenUsage
# after
from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.types import DEFAULT_MODEL, Message
```

Note: `simulation/__init__.py` may re-export `TokenUsage` in its `__all__`/imports for the public API — if it does, repoint the *source* of that import to `contracts` but keep it exported there (it is a legitimate public surface, just sourced from the canonical home). Verify with `grep -n TokenUsage src/evaluatorq/simulation/__init__.py`.

- [ ] **Step 2: Verify no module imports `TokenUsage` from `simulation.types`**

```bash
grep -rn "from evaluatorq.simulation.types import" src/ tests/ | grep -v __pycache__ | grep TokenUsage
```
Expected: no output (multi-name imports where TokenUsage was split out will no longer match).

- [ ] **Step 3: Verify green** (same commands as Task 1 Step 4)

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor(evaluatorq-py): import TokenUsage from contracts, not simulation.types (RES-897)"
```

---

### Task 4: Collapse 3-way parse into `AgentResponse.from_openresponses` (THE behavior delta: `response_id`)

Today OpenResponses→AgentResponse parsing is split across 3 sites: `extract_responses_output` (`_client.py`) returns only `(output_items, usage)`; then `target.py:146-165` and `agents/base.py:348` each separately pull `model`/`status`. `response_id` is captured by **neither** — currently dropped. This task adds a single pure-parse classmethod on `contracts.AgentResponse` that populates the full object, **restoring `response_id`** (the one intentional behavior change in this PR). Per-call-site `usage.calls` accounting (target sets `calls=1`, base.py does `+1`) stays at the call sites — the classmethod is pure parse.

`contracts.py` already imports `openresponses.convert_models` (line 67) and defines `TextOutputItem`/`ToolCallOutputItem` locally, so no new dependency or circular-import risk.

**Files:**
- Modify: `src/evaluatorq/contracts.py` (add classmethod to `AgentResponse`)
- Modify: `src/evaluatorq/simulation/target.py:146-165` (route through classmethod)
- Modify: `src/evaluatorq/simulation/agents/base.py:348-354` (route through classmethod)
- Modify: `src/evaluatorq/simulation/_client.py` (delete `extract_responses_output` + its `__all__` entry)
- Test: `tests/simulation/test_extract_responses_output.py` → rename/retarget to `tests/contracts/test_agent_response_from_openresponses.py` (or keep filename, repoint import + add response_id/model/finish_reason assertions). New test must assert the restored `response_id`.

- [ ] **Step 1: Write the failing test**

Create `tests/contracts/test_agent_response_from_openresponses.py` (port all cases from the existing `test_extract_responses_output.py` — message→TextOutputItem, function_call→ToolCallOutputItem, reasoning skipped, unknown-type warning, None-usage stays None, usage math — then add the new-coverage assertions):

```python
"""Unit tests for AgentResponse.from_openresponses (RES-897 parse consolidation)."""

from __future__ import annotations

from evaluatorq.contracts import (
    AgentResponse,
    TextOutputItem,
    ToolCallOutputItem,
)


def _resp(**kw):
    base = {"output": [], "usage": None, "model": None, "status": None, "id": None}
    base.update(kw)
    return base


def test_message_text_becomes_text_output_item():
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
    ))
    assert [type(i) for i in r.output] == [TextOutputItem]
    assert r.text == "hi"


def test_function_call_becomes_tool_call_output_item():
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "function_call", "name": "f", "arguments": "{}", "call_id": "c1"}],
    ))
    assert [type(i) for i in r.output] == [ToolCallOutputItem]
    assert r.tool_calls[0].name == "f"
    assert r.tool_calls[0].call_id == "c1"


def test_reasoning_item_skipped():
    r = AgentResponse.from_openresponses(_resp(output=[{"type": "reasoning"}]))
    assert r.output == []


def test_none_usage_stays_none():
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "x"}]}],
        usage=None,
    ))
    assert r.usage is None


def test_usage_parsed_without_calls_accounting():
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "x"}]}],
        usage={"input_tokens": 3, "output_tokens": 5},
    ))
    assert r.usage is not None
    assert r.usage.prompt_tokens == 3
    assert r.usage.completion_tokens == 5
    assert r.usage.total_tokens == 8
    assert r.usage.calls == 0  # classmethod is pure parse; call sites bump calls


def test_model_finish_reason_and_response_id_populated():
    # response_id is the RESTORED field — previously dropped on every parse path.
    r = AgentResponse.from_openresponses(_resp(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "x"}]}],
        model="azure/gpt-4o",
        status="completed",
        id="resp_abc123",
    ))
    assert r.model == "azure/gpt-4o"
    assert r.finish_reason == "completed"
    assert r.response_id == "resp_abc123"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/contracts/test_agent_response_from_openresponses.py -q
```
Expected: FAIL — `AttributeError: type object 'AgentResponse' has no attribute 'from_openresponses'`.

- [ ] **Step 3: Add the classmethod to `AgentResponse` in `contracts.py`**

Insert after the `tool_calls` property (around line 339). It reuses the exact parse logic from `extract_responses_output` plus model/status/id extraction. Use `get_field` from `common.fields` (Task 1) for dict/object duality:

```python
    @classmethod
    def from_openresponses(cls, response: Any) -> AgentResponse:
        """Build a full AgentResponse from a Responses API response object.

        Single parse path for the OpenResponses wire format. Populates
        ``output`` + ``usage`` + ``model`` + ``finish_reason`` + ``response_id``.

        ``usage`` is ``None`` when the response carries no ``usage`` block —
        callers must distinguish "no usage reported" from "zero tokens used" so
        cost reports stay honest. ``usage.calls`` is intentionally left at 0:
        this is a pure parse; per-call-site accounting (``calls=1`` / ``+1``) is
        applied by callers to the returned object's ``usage``.
        """
        import json as _json

        from evaluatorq.common.fields import get_field as _gf

        items: list[OutputMessage] = []
        for item in _gf(response, "output") or []:
            item_type = _gf(item, "type")
            if item_type == "message":
                for part in _gf(item, "content") or []:
                    if _gf(part, "type") == "output_text" and _gf(part, "text"):
                        items.append(
                            TextOutputItem(
                                type="output_text",
                                text=_gf(part, "text"),
                                annotations=[],
                                logprobs=[],
                            )
                        )
            elif item_type == "function_call":
                raw_args = _gf(item, "arguments") or "{}"
                call_id = _gf(item, "call_id") or _gf(item, "id") or ""
                result = _gf(item, "result")
                items.append(
                    ToolCallOutputItem(
                        type="function_call",
                        name=str(_gf(item, "name") or ""),
                        call_id=str(call_id),
                        arguments=raw_args if isinstance(raw_args, str) else _json.dumps(raw_args),
                        result=str(result) if result is not None else None,
                    )
                )
            elif item_type == "reasoning":
                pass  # o1/o3/o4-mini reasoning steps intentionally excluded
            else:
                logger.warning("AgentResponse.from_openresponses: skipping unknown item type={!r}", item_type)

        usage_obj = _gf(response, "usage")
        if usage_obj is None:
            logger.warning(
                "AgentResponse.from_openresponses: response.usage is None; usage left "
                "None so cost reports do not record fake-zero usage for billed calls"
            )
            usage = None
        else:
            input_toks = int(_gf(usage_obj, "input_tokens", 0) or 0)
            output_toks = int(_gf(usage_obj, "output_tokens", 0) or 0)
            usage = TokenUsage(
                prompt_tokens=input_toks,
                completion_tokens=output_toks,
                total_tokens=input_toks + output_toks,
            )

        model = _gf(response, "model")
        status = _gf(response, "status")
        response_id = _gf(response, "id")
        return cls(
            output=items,
            usage=usage,
            model=model if isinstance(model, str) else None,
            finish_reason=status if isinstance(status, str) else None,
            response_id=response_id if isinstance(response_id, str) else None,
        )
```

Confirm `logger` is already imported at module level in `contracts.py` (`from loguru import logger`); if not, add it.

- [ ] **Step 4: Run the new test to verify it passes**

```bash
uv run pytest tests/contracts/test_agent_response_from_openresponses.py -q
```
Expected: PASS (all cases incl. `response_id`).

- [ ] **Step 5: Route `target.py` through the classmethod**

In `simulation/target.py`, replace the `_do_call` body that calls `extract_responses_output` (lines 146-165). The target preserves: empty-output guard, `calls=1` accounting, and the `self.config.model` fallback (target falls back to its configured model when the response omits one):

```python
            agent_response = AgentResponse.from_openresponses(response)
            if not agent_response.output:
                raise RuntimeError(
                    f"OrqResponsesTarget: response contained no extractable "
                    f"output items (model={self.config.model}). This likely indicates "
                    f"an API error or unexpected response format."
                )
            usage = agent_response.usage
            if usage is not None:
                usage = usage.model_copy(update={"calls": 1})
            return AgentResponse(
                output=agent_response.output,
                usage=usage,
                model=agent_response.model or self.config.model,
                finish_reason=agent_response.finish_reason,
                response_id=agent_response.response_id,
            )
```

Remove the now-unused `extract_responses_output` from the import on line 10 (keep `build_simulation_client`). The `_get_field` import (line 16) may now be unused in `target.py` — if so, remove it; if still used elsewhere in the file, keep it. Verify with grep before removing.

- [ ] **Step 6: Route `agents/base.py` through the classmethod**

In `simulation/agents/base.py`, replace lines 348-354. It needs `output_items` + `usage` (not a full AgentResponse), so read them off the parsed object:

```python
                agent_response = AgentResponse.from_openresponses(response)
                output_items = agent_response.output
                usage = agent_response.usage

                record_llm_response(span, response)

                # Accumulate token usage (from_openresponses leaves calls=0, add 1)
                if usage is not None:
                    self._usage = self._usage + usage.model_copy(update={"calls": 1})
```

Update the import on line 16: `from evaluatorq.simulation._client import build_simulation_client` (drop `extract_responses_output`). Ensure `AgentResponse` is imported from `contracts` in this file (add to the existing `from evaluatorq.contracts import ...` if missing).

- [ ] **Step 7: Delete `extract_responses_output` from `_client.py`**

Remove the entire `extract_responses_output` function (lines 60-131) and drop it from `__all__` (line 134 → `__all__ = ["build_simulation_client"]`). Remove now-unused imports in `_client.py`: `json as _json`, the `TextOutputItem, ToolCallOutputItem` local import (was inside the deleted function), the `TokenUsage` TYPE_CHECKING import (line 16), and `get_field` if no longer used. Verify each is truly unused before removing.

- [ ] **Step 8: Verify no stale references to `extract_responses_output`**

```bash
grep -rn "extract_responses_output" src/ tests/ | grep -v __pycache__
```
Expected: no output (old test file retargeted in Step 1).

- [ ] **Step 9: Verify green**

```bash
uv run basedpyright 2>&1 | tail -1
uv run pytest -m 'not integration' -q 2>&1 | tail -3
```
Expected: `0 errors`; all pass.

- [ ] **Step 10: Commit**

```bash
git add -A && git commit -m "refactor(evaluatorq-py): collapse OpenResponses parse into AgentResponse.from_openresponses, restore response_id (RES-897)"
```

---

### Task 5: Move `_client.py` → `openresponses/client.py`

After Task 4, `_client.py` holds only `build_simulation_client`. Relocate it to its DRY home (cohesive around OpenResponses client construction).

**Files:**
- Create: `src/evaluatorq/openresponses/client.py`
- Delete: `src/evaluatorq/simulation/_client.py`
- Modify importers: `simulation/target.py`, `simulation/agents/base.py`, `simulation/runner/simulation.py` (if it imports `_client`)
- Tests: `tests/simulation/test_client_builder.py` (repoint import)

- [ ] **Step 1: Move the file**

```bash
git mv src/evaluatorq/simulation/_client.py src/evaluatorq/openresponses/client.py
```
Update the module docstring to drop the "and response extraction" clause (extraction now lives in `contracts.AgentResponse.from_openresponses`): e.g. `"""Shared AsyncOpenAI client construction for OpenResponses targets."""`.

- [ ] **Step 2: Repoint all importers**

Replace `from evaluatorq.simulation._client import build_simulation_client` with `from evaluatorq.openresponses.client import build_simulation_client` in every importer. Find them:

```bash
grep -rln "simulation._client\|simulation/_client" src/ tests/ | grep -v __pycache__
```

- [ ] **Step 3: Verify no stale references**

```bash
grep -rn "simulation._client\|simulation/_client" src/ tests/ | grep -v __pycache__
```
Expected: no output.

- [ ] **Step 4: Verify green** (same as Task 1 Step 4)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor(evaluatorq-py): move build_simulation_client to openresponses/client.py (RES-897)"
```

---

### Task 6: Move `target.py` → `openresponses/target.py` (kills the redteam→simulation edge)

`OrqResponsesTarget` has zero simulation-domain refs. Relocating it to `openresponses/` lets `redteam/backends/openresponses.py` import it from a shared leaf instead of from `simulation/` — eliminating the one in-scope cross-domain edge.

**Files:**
- Create: `src/evaluatorq/openresponses/target.py`
- Delete: `src/evaluatorq/simulation/target.py`
- Modify importers: `redteam/backends/openresponses.py` (**the edge**); `simulation/runner/simulation.py`, `simulation/api.py`, `simulation/adapters.py`, `simulation/__init__.py`, `simulation/tracing.py`
- Tests: `tests/redteam/test_orq_responses_target_as_agent_target.py`, `tests/integration/test_orq_responses_target_integration.py`, `tests/integration/test_sim_redteam_target_roundtrip.py`, `tests/simulation/test_orq_responses_target.py`, `tests/simulation/test_tracing.py`, `tests/openresponses/test_backend_registry.py`

- [ ] **Step 1: Move the file**

```bash
git mv src/evaluatorq/simulation/target.py src/evaluatorq/openresponses/target.py
```

- [ ] **Step 2: Fix internal imports in the moved file**

`openresponses/target.py` now imports siblings. Update:
- `from evaluatorq.simulation._client import build_simulation_client` → `from evaluatorq.openresponses.client import build_simulation_client` (already moved in T5; the line currently still says `_client` if T5 only repointed callers — verify and fix)
- It imports `record_openresponses_request/response`, `with_llm_span` from `simulation.tracing` — **leave these as `simulation.tracing` imports**. Per the ticket, tracing stays in `simulation/` for this PR (RES-899 relocates it). This is a temporary, documented openresponses→simulation edge on the *tracing* seam only, which RES-899 resolves. Do NOT try to move tracing here.
- `get_field` / `with_retry` already point to `common/` from Tasks 1-2.

- [ ] **Step 3: Repoint all importers**

Replace `from evaluatorq.simulation.target import OrqResponsesTarget` with `from evaluatorq.openresponses.target import OrqResponsesTarget` everywhere. Find them:

```bash
grep -rln "simulation.target\|simulation/target" src/ tests/ | grep -v __pycache__
```
Pay special attention to `redteam/backends/openresponses.py:15` — the in-scope edge.

- [ ] **Step 4: Verify the cross-domain edge is gone**

```bash
grep -rn "from evaluatorq.simulation" src/evaluatorq/redteam/ | grep -v __pycache__
```
Expected: no output (the sole `OrqResponsesTarget` import was the only redteam→simulation edge in scope).

```bash
grep -rn "simulation.target\|simulation/target" src/ tests/ | grep -v __pycache__
```
Expected: no output.

- [ ] **Step 5: Verify green** (same as Task 1 Step 4)

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor(evaluatorq-py): move OrqResponsesTarget to openresponses/target.py, remove redteam->simulation edge (RES-897)"
```

---

### Task 7: Name-collision clarity for `convert_models` vs `contracts` (NOT a dedupe)

`openresponses/convert_models.py` and `contracts.py` both define `FunctionCall`/`Message` (and `Usage`/`TokenUsage`), modeling **different layers**: `convert_models.*` = OpenResponses wire format (Responses API spec), `contracts.*` = internal canonical (chat-completions shape). They are legitimately separate — do NOT merge. Only remove the readability hazard.

**Files:**
- Audit candidates (import both): `src/evaluatorq/contracts.py`, `src/evaluatorq/redteam/contracts.py`, `src/evaluatorq/integrations/langchain_integration/convert.py`
- Modify: `src/evaluatorq/openresponses/convert_models.py` (add one-line docstrings)

- [ ] **Step 1: Audit for unaliased dual imports**

```bash
for f in src/evaluatorq/contracts.py src/evaluatorq/redteam/contracts.py src/evaluatorq/integrations/langchain_integration/convert.py; do
  echo "=== $f ==="; grep -n "FunctionCall\|Message\|Usage\|convert_models\|from evaluatorq.contracts" "$f"
done
```
Identify any single module importing a colliding name (`FunctionCall`/`Message`/`Usage`/`TokenUsage`) from BOTH `convert_models` and `contracts` without an alias.

- [ ] **Step 2: Alias where a collision exists**

Where found, alias the wire-format import: `from evaluatorq.openresponses.convert_models import Message as ORMessage, FunctionCall as ORFunctionCall` (and update local uses). If no module imports both unaliased (the likely case — `contracts.py` imports only specific convert_models symbols at line 67-70), record "no collision found" and skip aliasing. **No rename churn, no behavior change.**

- [ ] **Step 3: Add wire-format docstrings to `convert_models` models**

Add a one-line docstring to each model class in `openresponses/convert_models.py` stating it is the OpenResponses wire-format shape, e.g.:

```python
class Message(BaseModel):
    """OpenResponses wire-format message (Responses API spec) — distinct from contracts.Message (internal canonical)."""
```
Do the same for `FunctionCall`, `Usage`, and any other model that collides by name with `contracts`.

- [ ] **Step 4: Verify green** (same as Task 1 Step 4)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor(evaluatorq-py): clarify convert_models wire-format vs contracts canonical naming (RES-897)"
```

---

## Final Verification (after all tasks)

- [ ] Full green:
```bash
uv run basedpyright 2>&1 | tail -1          # 0 errors, 0 warnings
uv run pytest -m 'not integration' -q 2>&1 | tail -3   # all pass (>= 1414 + new test)
```
- [ ] No cross-domain edges remain:
```bash
grep -rn "from evaluatorq.simulation" src/evaluatorq/redteam/ | grep -v __pycache__   # empty
grep -rn "from evaluatorq.redteam" src/evaluatorq/simulation/ | grep -v __pycache__   # empty
```
- [ ] No leftover shims: search for any temporary re-export modules added mid-refactor and confirm none remain.
- [ ] Diff is moves + import rewrites + the one classmethod + name-collision aliasing/docstrings — no stray logic changes.
- [ ] Open PR titled: `refactor(evaluatorq-py): extract OpenResponses runtime + shared models out of simulation/`. Call out the single behavior delta explicitly in the description: **`response_id` is now populated on `AgentResponse` (previously dropped on every parse path) — this is a fix.** Note follow-up RES-899 (tracing-layer unification, blocked by this).

## Out of Scope (do NOT touch — RES-899)

- `simulation/tracing.py` in full: `record_openresponses_request/response`, `record_llm_response`/`record_token_usage`, `_truncate`/`_capture_message_content`, `with_simulation_span`, the `orq.simulation.*` span schema.
- judge, user_simulator, `prompt_builders`, runner, domain types (`Persona`, `Scenario`, `Judgment`, `EmotionalArc`, `Criterion`, `TurnMetrics`, `SimulationResult`, `Datapoint`).
- Do NOT add a free-floating `agent_response_to_openresponses` (dropped in merge `9b7eca8`, zero consumers). If reverse direction is ever needed, add `AgentResponse.to_openresponses()` symmetric with `from_openresponses`.
