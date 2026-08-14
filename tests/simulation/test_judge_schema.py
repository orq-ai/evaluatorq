"""Guardrails pinning the generated-from-pydantic judge tool schemas (RES-1308).

`_wire_schema` in `evaluatorq.simulation.agents.judge` generates the two OpenAI
function-tool schemas from `ContinueConversation` / `FinishConversation`
instead of hand-written JSON-schema dicts. These tests assert the properties
constraint 3 (see `.superpowers/sdd/plan.md`) depends on and that a future edit
to the generator could silently undo:

- `required` comes from the explicit `WIRE_REQUIRED`, not pydantic's own
  (empty, because every field has a parser-side default) `required` — that gap
  is RES-1308 itself.
- the same is true one nesting level down, for `CriterionVerdict`'s `evidence`
  field, via `_NESTED_WIRE_REQUIRED`.
- no `$ref`/`$defs` survive (the Orq router's narrower schema dialects do not
  all resolve refs).
- no `default` survives (it is a parser-side value; advertising it contradicts
  `WIRE_REQUIRED`).
- `criteria_verdicts` has no `null`/`anyOf` branch (an explicit `null` would be
  a cheap way for the model to produce "the judge told us nothing" on purpose).
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from evaluatorq.simulation.agents import judge as judge_module
from evaluatorq.simulation.agents.judge import (
    JUDGE_TOOLS,
    ContinueConversation,
    FinishConversation,
    _assert_wire_fields_exist,
    _JudgeToolArgs,
    _wire_schema,
)


def _tool(name: str) -> dict[str, Any]:
    (tool,) = [t for t in JUDGE_TOOLS if t['function']['name'] == name]
    return tool


def _find_keys(node: object, key: str) -> list[object]:
    """Every value found under `key` anywhere in a JSON-schema tree."""
    hits: list[object] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                hits.append(v)
            hits.extend(_find_keys(v, key))
    elif isinstance(node, list):
        for v in node:
            hits.extend(_find_keys(v, key))
    return hits


@pytest.mark.parametrize('name', ['continue_conversation', 'finish_conversation'])
class TestWireSchemaShape:
    def test_required_matches_wire_required_and_includes_criteria_verdicts(self, name: str) -> None:
        """The RES-1308 regression test: `required` must not silently go empty."""
        model = judge_module._TOOL_MODELS[name]
        params = _tool(name)['function']['parameters']
        assert params['required'] == sorted(model.WIRE_REQUIRED)
        assert 'criteria_verdicts' in params['required']

    def test_no_ref_or_defs_survive(self, name: str) -> None:
        params = _tool(name)['function']['parameters']
        assert _find_keys(params, '$ref') == []
        assert _find_keys(params, '$defs') == []
        assert '$defs' not in params

    def test_no_default_survives(self, name: str) -> None:
        params = _tool(name)['function']['parameters']
        assert _find_keys(params, 'default') == []

    def test_criteria_verdicts_has_no_null_branch(self, name: str) -> None:
        params = _tool(name)['function']['parameters']
        prop = params['properties']['criteria_verdicts']
        assert 'anyOf' not in prop
        assert prop.get('type') != 'null'
        types_found = _find_keys(prop, 'type')
        assert 'null' not in types_found

    def test_criteria_verdicts_items_inlined_with_expected_properties(self, name: str) -> None:
        params = _tool(name)['function']['parameters']
        prop = params['properties']['criteria_verdicts']
        items = prop['items']
        assert isinstance(items, dict)
        assert 'criterion_id' in items['properties']
        assert 'occurred' in items['properties']
        # Nested required set too — the same "quietly un-required" defect one
        # level down (`_NESTED_WIRE_REQUIRED`), not only at the top level.
        assert items['required'] == sorted(judge_module._NESTED_WIRE_REQUIRED[judge_module.CriterionVerdict])
        assert 'evidence' in items['required']


def test_object_node_description_stripped_but_field_description_kept() -> None:
    """`_strip_node` pops `description` from object nodes (docstrings), not field nodes."""
    params = _tool('continue_conversation')['function']['parameters']
    # The top-level object schema description comes from the class docstring —
    # maintainer rationale, not something the model should read as prompt text.
    assert 'description' not in params
    # A field's own description is the opposite: it IS written for the model.
    assert params['properties']['reason']['description']
    assert params['properties']['criteria_verdicts']['description']


def test_wire_schema_removed_object_description_would_leak_docstring() -> None:
    """Confirms the guardrail actually fires: reproduce the schema without the strip and see the docstring leak.

    `model_json_schema()` on its own includes the class docstring as
    `description` on the object node; `_wire_schema` strips it. If a future
    edit dropped that stripping, this assertion on the raw schema (not
    `_wire_schema`'s output) would start passing while the pinned test above
    would start failing — proving the guardrail is load-bearing.
    """
    raw = ContinueConversation.model_json_schema()
    assert 'description' in raw  # the un-stripped schema still carries it
    stripped = _wire_schema(ContinueConversation)
    assert 'description' not in stripped


def test_assert_wire_fields_exist_raises_for_unknown_top_level_field() -> None:
    class _BadModel(_JudgeToolArgs):
        WIRE_REQUIRED: ClassVar[frozenset[str]] = _JudgeToolArgs.WIRE_REQUIRED | {'not_a_real_field'}

    original = dict(judge_module._TOOL_MODELS)
    judge_module._TOOL_MODELS['throwaway'] = _BadModel
    try:
        with pytest.raises(RuntimeError, match=re.escape('_BadModel') + '.*not_a_real_field'):
            _assert_wire_fields_exist()
    finally:
        judge_module._TOOL_MODELS.clear()
        judge_module._TOOL_MODELS.update(original)


def test_assert_wire_fields_exist_raises_for_unknown_nested_field() -> None:
    class _ThrowawayNested(BaseModel):
        criterion_id: str = ''

    original_nested = dict(judge_module._NESTED_WIRE_REQUIRED)
    judge_module._NESTED_WIRE_REQUIRED[_ThrowawayNested] = frozenset({'not_a_real_field'})
    try:
        with pytest.raises(RuntimeError, match=re.escape('_ThrowawayNested') + '.*not_a_real_field'):
            _assert_wire_fields_exist()
    finally:
        judge_module._NESTED_WIRE_REQUIRED.clear()
        judge_module._NESTED_WIRE_REQUIRED.update(original_nested)


def test_assert_wire_fields_exist_passes_for_the_real_models() -> None:
    """The real models must never raise — proves the two failure tests above
    are actually exercising the unknown-field branch, not something else."""
    _assert_wire_fields_exist()


def test_finish_conversation_adds_goal_achieved_to_wire_required() -> None:
    assert FinishConversation.WIRE_REQUIRED == _JudgeToolArgs.WIRE_REQUIRED | {'goal_achieved'}
