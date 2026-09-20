"""A classify judge (Jev) can be seated on `llm_jury` / `llm_jury_pairwise` panels.

The jury layer never calls ``/classify`` itself: it builds the `ClassifyQuestion` and
hands it to `run_judge`, which decides the endpoint. These tests therefore capture the
``classify=`` keyword rather than any request body, and pin the one guarantee the change
had to keep — a panel with plain list labels and no ``levels`` produces exactly the
verdict schema and system prompt it produced before.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from evaluatorq.common import model_catalogue, structured_output
from evaluatorq.common.judge import ClassifyQuestion, EvaluatorResponsePayload, JudgeOutcome
from evaluatorq.contracts import AgentResponse, ToolCallOutputItem
from evaluatorq.llm_jury import (
    DEFAULT_TEMPLATE,
    PairwiseComparator,
    _build_verdict_model,  # pyright: ignore[reportPrivateUsage]
    _default_system_prompt,  # pyright: ignore[reportPrivateUsage]
    llm_jury,
    llm_jury_pairwise,
)
from evaluatorq.types import DataPoint, Evaluator, Output, ScorerParameter

# Patch via the module object, not the dotted string: the package re-exports the
# ``llm_jury`` function, which shadows the same-named submodule (see test_llm_jury_factory).
llm_jury_mod = importlib.import_module('evaluatorq.llm_jury')

JEV = 'typesafe/jev-latest'


@pytest.fixture(autouse=True)
def _catalogue_for_jury_tests(monkeypatch: pytest.MonkeyPatch):
    """Exercise jury routing as it runs through the Orq router, without network access."""
    model_catalogue.reset_catalogue_cache()
    jev = model_catalogue.ModelInfo(0.0, 0.0, 'typesafe', False, supports_classify=True)  # noqa: FBT003
    chat = model_catalogue.ModelInfo(0.0, 0.0, 'openai', True)

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {JEV: jev, 'jev-latest': jev, 'gpt-5-mini': chat}

    monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)
    monkeypatch.setattr(llm_jury_mod, 'client_routes_through_orq', lambda client: True)
    yield
    model_catalogue.reset_catalogue_cache()

# Captured from the pre-change `_build_verdict_model` / `_default_system_prompt`. A panel
# that names no label descriptions and no levels must keep producing these byte for byte.
BOOLEAN_SCHEMA = {
    'properties': {
        'explanation': {
            'default': '',
            'description': 'Explanation for the verdict',
            'title': 'Explanation',
            'type': 'string',
        },
        'value': {'title': 'Value', 'type': 'boolean'},
    },
    'required': ['value'],
    'title': 'VerdictModel',
    'type': 'object',
}
LIST_LABELS_SCHEMA = {
    'properties': {
        'explanation': {
            'default': '',
            'description': 'Explanation for the verdict',
            'title': 'Explanation',
            'type': 'string',
        },
        'value': {'enum': ['a', 'b'], 'title': 'Value', 'type': 'string'},
    },
    'required': ['value'],
    'title': 'VerdictModel',
    'type': 'object',
}
NUMERIC_SCHEMA = {
    'properties': {
        'explanation': {
            'default': '',
            'description': 'Explanation for the verdict',
            'title': 'Explanation',
            'type': 'string',
        },
        'value': {'title': 'Value', 'type': 'number'},
    },
    'required': ['value'],
    'title': 'VerdictModel',
    'type': 'object',
}
BOOLEAN_SYSTEM_PROMPT = (
    "You are a strict evaluator. Read the input, the model's output, any expected output, and judge "
    'against the stated criterion. Return a structured verdict with a 2-3 sentence explanation. '
    '`value` must be a boolean: true if the criterion is met, false otherwise.'
)
LIST_LABELS_SYSTEM_PROMPT = (
    "You are a strict evaluator. Read the input, the model's output, any expected output, and judge "
    'against the stated criterion. Return a structured verdict with a 2-3 sentence explanation. '
    '`value` must be exactly one of: a, b.'
)
NUMERIC_SYSTEM_PROMPT = (
    "You are a strict evaluator. Read the input, the model's output, any expected output, and judge "
    'against the stated criterion. Return a structured verdict with a 2-3 sentence explanation. '
    '`value` must be a number between 0.0 and 1.0 (higher is better).'
)


def _params(output: Output = 'the answer') -> ScorerParameter:
    data = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'q'}]}, expected_output='4')
    return ScorerParameter(data=data, output=output)


async def _run_and_capture(evaluator: Evaluator, sink: list[Any]) -> None:
    """Run the evaluator's scorer once with `run_judge` replaced by a capturing fake."""

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink.append(kwargs.get('classify'))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=True, explanation='ok'))

    with patch.object(llm_jury_mod, 'run_judge', side_effect=fake_run_judge):
        await evaluator['scorer'](_params())


async def _capture_verdict_model(evaluator: Evaluator) -> type[BaseModel]:
    """Run the evaluator once and return the ``response_model`` handed to `run_judge`."""
    sink: list[Any] = []

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink.append(kwargs.get('response_model'))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='a', explanation='ok'))

    with patch.object(llm_jury_mod, 'run_judge', side_effect=fake_run_judge):
        await evaluator['scorer'](_params())
    return cast('type[BaseModel]', sink[0])

# ---------------------------------------------------------------------------
# Verdict model and system prompt
# ---------------------------------------------------------------------------


def test_plain_panel_schema_and_prompt_are_unchanged() -> None:
    for kind, labels, schema, prompt in (
        ('categorical', None, BOOLEAN_SCHEMA, BOOLEAN_SYSTEM_PROMPT),
        ('categorical', ['a', 'b'], LIST_LABELS_SCHEMA, LIST_LABELS_SYSTEM_PROMPT),
        ('numeric', None, NUMERIC_SCHEMA, NUMERIC_SYSTEM_PROMPT),
    ):
        model = _build_verdict_model(kind, labels, (0.0, 1.0))
        assert model.model_json_schema() == schema
        assert _default_system_prompt(kind, labels, (0.0, 1.0)) == prompt


def test_label_descriptions_reach_schema_and_prompt() -> None:
    described = {'a': 'the good one', 'b': None}
    model = _build_verdict_model('categorical', ['a', 'b'], (0.0, 1.0), label_descriptions=described)
    value = model.model_json_schema()['properties']['value']
    assert value['enum'] == ['a', 'b']
    assert value['description'] == 'One of: a: the good one; b'
    prompt = _default_system_prompt('categorical', ['a', 'b'], (0.0, 1.0), label_descriptions=described)
    assert prompt.startswith(LIST_LABELS_SYSTEM_PROMPT)
    assert 'One of: a: the good one; b' in prompt


def test_levels_reach_schema_and_prompt() -> None:
    levels = ['useless', 'fine', 'excellent']
    model = _build_verdict_model('numeric', None, (0.0, 1.0), levels=levels)
    description = model.model_json_schema()['properties']['value']['description']
    assert description == 'Scale from 0.0 to 1.0. Levels: 0.0=useless; 0.5=fine; 1.0=excellent'
    prompt = _default_system_prompt('numeric', None, (0.0, 1.0), levels=levels)
    assert prompt.startswith(NUMERIC_SYSTEM_PROMPT)
    assert '0.0=useless; 0.5=fine; 1.0=excellent' in prompt


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_prompt_and_criteria_may_both_be_set_but_not_neither() -> None:
    with pytest.raises(ValueError, match='`criteria`, `prompt`, or both'):
        llm_jury(name='x')
    evaluator = llm_jury(name='x', criteria='c', prompt='judge {{criteria}}', client=MagicMock())
    assert evaluator['name'] == 'x'


def test_both_set_without_a_criteria_placeholder_warns_once() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury(name='x', criteria='c', prompt='judge {{output.response}}', client=MagicMock())
    messages = [call.args[0] for call in warn.call_args_list]
    assert sum('no judge reads it' in message for message in messages) == 1


def test_no_warning_when_the_prompt_reads_criteria() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury(name='x', criteria='c', prompt='judge {{criteria}} {{output.response}}', client=MagicMock())
    assert not any('no judge reads it' in call.args[0] for call in warn.call_args_list)


def test_no_unread_criteria_warning_for_a_classify_panel() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury(name='x', criteria='c', prompt='judge {{output.response}}', judges=[JEV], client=MagicMock())
    assert not any('no judge reads it' in call.args[0] for call in warn.call_args_list)


def test_pairwise_warns_when_criteria_is_set_but_never_rendered() -> None:
    """The pairwise factory drops an unread `criteria` exactly as the pointwise one does."""
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury_pairwise(criteria='c', prompt='compare {{response_a.output.response}}', client=MagicMock())
    messages = [call.args[0] for call in warn.call_args_list]
    assert sum('no judge reads it' in message for message in messages) == 1


def test_pairwise_is_quiet_when_the_prompt_reads_criteria() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury_pairwise(criteria='c', prompt='compare {{criteria}} {{response_a.output.response}}', client=MagicMock())
    assert not any('no judge reads it' in call.args[0] for call in warn.call_args_list)


def test_pairwise_classify_panel_never_warns_about_unread_criteria() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury_pairwise(
            criteria='c', prompt='compare {{response_a.output.response}}', judges=[JEV], client=MagicMock()
        )
    assert not any('no judge reads it' in call.args[0] for call in warn.call_args_list)


def test_levels_require_numeric_and_a_sane_length() -> None:
    with pytest.raises(ValueError, match='levels'):
        llm_jury(name='x', criteria='c', levels=['lo', 'hi'])
    with pytest.raises(ValueError, match='levels'):
        llm_jury(name='x', criteria='c', verdict_kind='numeric', levels=['only'])
    with pytest.raises(ValueError, match='levels'):
        llm_jury(name='x', criteria='c', verdict_kind='numeric', levels=[str(i) for i in range(11)])


def test_classify_panel_requires_criteria() -> None:
    with pytest.raises(ValueError, match='criteria'):
        llm_jury(name='x', prompt='judge {{output.response}}', judges=[JEV])


def test_classify_panel_numeric_requires_levels() -> None:
    with pytest.raises(ValueError, match='levels'):
        llm_jury(name='x', criteria='c', verdict_kind='numeric', judges=[JEV])


def test_classify_panel_requires_the_default_score_range() -> None:
    with pytest.raises(ValueError, match=r'score_range'):
        llm_jury(
            name='x',
            criteria='c',
            verdict_kind='numeric',
            levels=['lo', 'hi'],
            score_range=(1.0, 5.0),
            threshold=3.0,
            judges=[JEV],
        )


def test_classify_panel_allows_a_score_range_override_outside_numeric_mode() -> None:
    """`score_range` is meaningless on a categorical panel, so the classify check skips it."""
    evaluator = llm_jury(
        name='x', criteria='c', labels=['a', 'b'], score_range=(1.0, 5.0), judges=[JEV], client=MagicMock()
    )
    assert evaluator['name'] == 'x'


def test_classify_boolean_panel_requires_a_probability_threshold() -> None:
    with pytest.raises(ValueError, match='threshold'):
        llm_jury(name='x', criteria='c', judges=[JEV], threshold=5)
    evaluator = llm_jury(name='x', criteria='c', judges=[JEV], threshold=0.7, client=MagicMock())
    assert evaluator['name'] == 'x'


def test_a_replacement_classify_judge_seats_the_same_rules() -> None:
    with pytest.raises(ValueError, match='criteria'):
        llm_jury(name='x', prompt='p', judges=['gpt-5-mini'], replacement_judges=[JEV])


def test_classify_panel_warns_once_on_repetitions() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury(name='x', criteria='c', judges=[JEV], repetitions=3, client=MagicMock())
    assert sum('repetitions' in call.args[0] for call in warn.call_args_list) == 1


def test_pairwise_classify_panel_warns_once_on_repetitions() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury_pairwise(criteria='c', judges=[JEV], repetitions=3, client=MagicMock())
    messages = [call.args[0].format(*call.args[1:]) for call in warn.call_args_list]
    assert sum('repetitions' in message for message in messages) == 1
    assert 'each A/B ordering' in next(message for message in messages if 'repetitions' in message)


def test_direct_pairwise_classify_panel_warns_about_repetitions_and_ignored_settings() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        PairwiseComparator(
            panel=[JEV],
            criteria='c',
            system_prompt='s',
            swap=True,
            repetitions=3,
            replacement_judges=None,
            min_successful_judges=1,
            max_tokens=1234,
            timeout_ms=1000,
            temperature=None,
            structured_output=True,
            extra_kwargs=None,
            extra_body=None,
            client=MagicMock(),
        )
    messages = [call.args[0].format(*call.args[1:]) for call in warn.call_args_list]
    assert sum('repetitions' in message for message in messages) == 1
    ignored = [message for message in messages if 'do not reach' in message]
    assert len(ignored) == 1
    assert 'system_prompt, max_tokens' in ignored[0]


# ---------------------------------------------------------------------------
# Settings a classify judge cannot read
# ---------------------------------------------------------------------------


def test_a_classify_panel_that_sets_none_of_the_prompted_settings_is_quiet() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury(name='x', criteria='c', judges=[JEV, 'gpt-5-mini'], client=MagicMock())
    assert not any('do not reach' in call.args[0] for call in warn.call_args_list)


def test_a_classify_panel_names_the_settings_it_cannot_read_once() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury(
            name='x',
            criteria='c',
            judges=[JEV, 'gpt-5-mini'],
            system_prompt='x',
            temperature=0.2,
            client=MagicMock(),
        )
    named = [call for call in warn.call_args_list if 'do not reach' in call.args[0]]
    assert len(named) == 1
    assert named[0].args[2] == 'system_prompt, temperature'
    assert named[0].args[1] == JEV


def test_a_panel_of_only_classify_judges_does_not_claim_a_prompted_judge_reads_them() -> None:
    """There is no prompted judge on this panel, so the settings reach nobody at all."""
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury(name='x', criteria='c', judges=[JEV], temperature=0.3, client=MagicMock())
    named = [call for call in warn.call_args_list if 'do not reach' in call.args[0]]
    assert len(named) == 1
    assert named[0].args[3] == ''
    assert 'still apply' not in named[0].args[0].format(*named[0].args[1:])


def test_a_mixed_panel_still_says_any_prompted_judges_read_them() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury(name='x', criteria='c', judges=[JEV, 'gpt-5-mini'], temperature=0.3, client=MagicMock())
    named = [call for call in warn.call_args_list if 'do not reach' in call.args[0]]
    assert len(named) == 1
    assert named[0].args[3] == ' (they still apply to any prompted judges)'


def test_the_config_fields_a_classify_judge_never_receives_are_named_too() -> None:
    """`_run_single_judge` keeps these off the classify config, so nothing downstream reports them."""
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury(
            name='x',
            criteria='c',
            judges=[JEV, 'gpt-5-mini'],
            reasoning_effort='high',
            extra_kwargs={'top_p': 0.9},
            extra_body={'retry': {'count': 2}},
            client=MagicMock(),
        )
    named = [call for call in warn.call_args_list if 'do not reach' in call.args[0]]
    assert len(named) == 1
    assert named[0].args[2] == 'reasoning_effort, extra_kwargs, extra_body'


def test_a_non_default_max_tokens_is_named_as_unread_by_the_classify_judge() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury(
            name='x',
            criteria='c',
            judges=[JEV, 'gpt-5-mini'],
            max_tokens=1234,
            client=MagicMock(),
        )
    named = [call for call in warn.call_args_list if 'do not reach' in call.args[0]]
    assert len(named) == 1
    assert named[0].args[2] == 'max_tokens'
    assert named[0].args[3] == ' (they still apply to any prompted judges)'


def test_empty_extra_kwargs_and_extra_body_are_not_reported_as_set() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury(name='x', criteria='c', judges=[JEV], extra_kwargs={}, extra_body={}, client=MagicMock())
    assert not any('do not reach' in call.args[0] for call in warn.call_args_list)


def test_an_llm_only_panel_never_warns_about_unread_settings() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury(name='x', criteria='c', judges=['gpt-5-mini'], system_prompt='x', temperature=0.2, client=MagicMock())
    assert not any('do not reach' in call.args[0] for call in warn.call_args_list)


def test_a_pairwise_classify_panel_that_sets_none_of_the_prompted_settings_is_quiet() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury_pairwise(criteria='c', judges=[JEV, 'gpt-5-mini'], client=MagicMock())
    assert not any('do not reach' in call.args[0] for call in warn.call_args_list)


def test_a_pairwise_classify_panel_names_the_settings_it_cannot_read_once() -> None:
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        llm_jury_pairwise(
            criteria='c', judges=[JEV, 'gpt-5-mini'], system_prompt='x', temperature=0.2, client=MagicMock()
        )
    named = [call for call in warn.call_args_list if 'do not reach' in call.args[0]]
    assert len(named) == 1
    assert named[0].args[2] == 'system_prompt, temperature'


# ---------------------------------------------------------------------------
# Question building
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boolean_panel_builds_a_noul_question_from_the_default_template() -> None:
    sink: list[Any] = []
    evaluator = llm_jury(name='x', criteria='is it right?', judges=[JEV], threshold=0.8, client=MagicMock())
    await _run_and_capture(evaluator, sink)

    question = sink[0]
    assert isinstance(question, ClassifyQuestion)
    assert question.kind == 'noul'
    assert question.criteria is None
    assert question.instructions == 'is it right?'
    assert question.noul_threshold == 0.8
    assert set(cast(dict[str, Any], question.state)) == {
        'input.all_messages',
        'output.response',
        'input.expected_output',
    }
    assert 'criteria' not in cast(dict[str, Any], question.state)
    assert cast(dict[str, Any], question.state)['output.response'] == 'the answer'


@pytest.mark.asyncio
async def test_state_keys_track_the_default_template() -> None:
    sink: list[Any] = []
    await _run_and_capture(llm_jury(name='x', criteria='c', judges=[JEV], client=MagicMock()), sink)
    expected = [path for path in llm_jury_mod.extract_template_paths(DEFAULT_TEMPLATE) if path != 'criteria']
    assert list(cast(dict[str, Any], sink[0].state)) == expected


@pytest.mark.asyncio
async def test_state_fields_override_the_template_paths() -> None:
    sink: list[Any] = []
    evaluator = llm_jury(
        name='x', criteria='c', judges=[JEV], state_fields=['output.response', 'criteria'], client=MagicMock()
    )
    await _run_and_capture(evaluator, sink)
    assert list(cast(dict[str, Any], sink[0].state)) == ['output.response']


@pytest.mark.asyncio
async def test_a_missing_state_path_is_skipped() -> None:
    sink: list[Any] = []
    evaluator = llm_jury(
        name='x', criteria='c', judges=[JEV], state_fields=['output.response', 'input.nothing'], client=MagicMock()
    )
    await _run_and_capture(evaluator, sink)
    assert list(cast(dict[str, Any], sink[0].state)) == ['output.response']


@pytest.mark.asyncio
async def test_a_state_that_resolves_to_nothing_warns() -> None:
    sink: list[Any] = []
    evaluator = llm_jury(
        name='x',
        criteria='c',
        judges=[JEV],
        state_fields=['ouput.response'],  # typo on purpose: nothing resolves
        client=MagicMock(),
    )
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        await _run_and_capture(evaluator, sink)
    messages = [call.args[0] for call in warn.call_args_list]
    assert sum('classify state is empty' in message for message in messages) == 1
    assert cast(dict[str, Any], sink[0].state) == {}


@pytest.mark.asyncio
async def test_an_llm_only_panel_builds_no_question_and_no_state() -> None:
    """A panel of prompted judges must not pay for a question nobody answers — nor warn about it."""
    sink: list[Any] = []
    evaluator = llm_jury(
        name='x',
        criteria='c',
        judges=['gpt-5-mini'],
        state_fields=['ouput.response'],  # typo on purpose: nothing would resolve
        client=MagicMock(),
    )
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        await _run_and_capture(evaluator, sink)
    assert sink == [None]
    assert not any('classify state' in call.args[0] for call in warn.call_args_list)


@pytest.mark.asyncio
async def test_a_classify_panel_whose_prompt_renders_nothing_warns_about_the_empty_state() -> None:
    sink: list[Any] = []
    evaluator = llm_jury(
        name='x', criteria='c', prompt='judge against {{criteria}}', judges=[JEV], client=MagicMock()
    )
    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        await _run_and_capture(evaluator, sink)
    messages = [call.args[0] for call in warn.call_args_list]
    assert sum('classify state is empty' in message for message in messages) == 1
    assert cast(dict[str, Any], sink[0].state) == {}


@pytest.mark.asyncio
async def test_labeled_panel_builds_a_choice_question_carrying_the_descriptions() -> None:
    sink: list[Any] = []
    evaluator = llm_jury(
        name='x',
        criteria='grade it',
        labels={'good': 'fully correct', 'bad': None},
        passing_labels=['good'],
        judges=[JEV],
        client=MagicMock(),
    )
    await _run_and_capture(evaluator, sink)
    assert sink[0].kind == 'choice'
    assert sink[0].criteria == {'good': 'fully correct', 'bad': None}


@pytest.mark.asyncio
async def test_numeric_panel_builds_a_score_question_from_levels() -> None:
    sink: list[Any] = []
    evaluator = llm_jury(
        name='x',
        criteria='rate it',
        verdict_kind='numeric',
        levels=['useless', 'fine', 'excellent'],
        judges=[JEV],
        client=MagicMock(),
    )
    await _run_and_capture(evaluator, sink)
    assert sink[0].kind == 'score'
    assert sink[0].criteria == ['useless', 'fine', 'excellent']


@pytest.mark.asyncio
async def test_only_the_classify_judge_on_a_mixed_panel_gets_the_question() -> None:
    sink: dict[str, Any] = {}
    evaluator = llm_jury(name='x', criteria='c', judges=[JEV, 'gpt-5-mini'], client=MagicMock())

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink[cast(str, kwargs['model'])] = kwargs.get('classify')
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=True, explanation='ok'))

    with patch.object(llm_jury_mod, 'run_judge', side_effect=fake_run_judge):
        await evaluator['scorer'](_params())

    assert isinstance(sink[JEV], ClassifyQuestion)
    assert sink['gpt-5-mini'] is None


@pytest.mark.asyncio
async def test_labels_as_a_list_and_as_a_dict_build_the_same_verdict_literal() -> None:
    """Both label shapes must constrain ``value`` to the same enum, described or not."""
    from_list = await _capture_verdict_model(llm_jury(name='x', criteria='c', labels=['a', 'b'], client=MagicMock()))
    from_dict = await _capture_verdict_model(
        llm_jury(name='x', criteria='c', labels={'a': 'the good one', 'b': None}, client=MagicMock())
    )
    list_value = from_list.model_json_schema()['properties']['value']
    dict_value = from_dict.model_json_schema()['properties']['value']
    assert list_value['enum'] == dict_value['enum'] == ['a', 'b']
    assert 'description' not in list_value
    assert dict_value['description'] == 'One of: a: the good one; b'
    plain = _build_verdict_model('categorical', ['a', 'b'], (0.0, 1.0), label_descriptions={'a': None, 'b': None})
    assert plain.model_json_schema() == LIST_LABELS_SCHEMA


# ---------------------------------------------------------------------------
# The config a classify judge is handed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_vanilla_classify_panel_warns_about_no_unread_config_field(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A caller who set nothing must not be told their settings were ignored — on every datapoint."""
    jev = model_catalogue.ModelInfo(0.0, 0.0, 'typesafe', False, supports_classify=True)  # noqa: FBT003

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {JEV: jev, 'jev-latest': jev}

    monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)
    structured_output._WARNED_UNREAD.clear()  # pyright: ignore[reportPrivateUsage]
    client = MagicMock()
    client.base_url = 'https://my.orq.ai/v3/router'
    client.post = AsyncMock(
        return_value={
            'answers': {'verdict': {'type': 'noul', 'noul': 0.9}},
            'usage': {'input_tokens': 10, 'output_tokens': 1},
            'model': 'jev-latest',
        }
    )
    evaluator = llm_jury(name='x', criteria='is it right?', judges=[JEV], client=client)

    with caplog.at_level(logging.WARNING, logger='evaluatorq.common.structured_output'):
        result = await evaluator['scorer'](_params())

    assert cast('Any', result).value is True
    assert 'run_judge[classify]' not in caplog.text
    client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_catalogue_only_classify_model_is_seated_without_manual_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = 'acme/sorter'
    info = model_catalogue.ModelInfo(0.0, 0.0, 'acme', False, supports_classify=True)  # noqa: FBT003

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {model: info, 'sorter': info}

    monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)
    model_catalogue.reset_catalogue_cache()
    client = MagicMock()
    client.base_url = 'https://my.orq.ai/v3/router'
    client.post = AsyncMock(
        return_value={
            'answers': {'verdict': {'type': 'noul', 'noul': 0.9}},
            'usage': {'input_tokens': 10, 'output_tokens': 0},
            'model': 'sorter',
        }
    )
    evaluator = llm_jury(name='x', criteria='is it right?', judges=[model], max_tokens=1234, client=client)

    with patch.object(llm_jury_mod.logger, 'warning') as warn:
        result = await evaluator['scorer'](_params())

    assert cast('Any', result).value is True
    client.post.assert_awaited_once()
    messages = [call.args[0].format(*call.args[1:]) for call in warn.call_args_list]
    assert sum('max_tokens' in message and 'do not reach' in message for message in messages) == 1


@pytest.mark.asyncio
async def test_a_catalogue_only_classify_model_without_a_question_does_not_fall_back_to_prompting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = 'acme/sorter'
    info = model_catalogue.ModelInfo(0.0, 0.0, 'acme', False, supports_classify=True)  # noqa: FBT003

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {model: info, 'sorter': info}

    monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)
    model_catalogue.reset_catalogue_cache()
    evaluator = llm_jury(name='x', prompt='judge {{output.response}}', judges=[model], client=MagicMock())
    run_judge = AsyncMock(return_value=JudgeOutcome(error_message='should not run'))
    with (
        patch.object(llm_jury_mod, 'run_judge', run_judge),
        pytest.raises(ValueError, match='needs `criteria`'),
    ):
        await evaluator['scorer'](_params())

    run_judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_catalogue_only_numeric_classify_model_rejects_a_custom_score_range_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = 'acme/sorter'
    info = model_catalogue.ModelInfo(0.0, 0.0, 'acme', False, supports_classify=True)  # noqa: FBT003

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {model: info, 'sorter': info}

    monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)
    model_catalogue.reset_catalogue_cache()
    evaluator = llm_jury(
        name='x',
        criteria='score it',
        judges=[model],
        verdict_kind='numeric',
        levels=['bad', 'good'],
        score_range=(0.0, 10.0),
        client=MagicMock(),
    )
    run_judge = AsyncMock(return_value=JudgeOutcome(error_message='should not run'))
    with (
        patch.object(llm_jury_mod, 'run_judge', run_judge),
        pytest.raises(ValueError, match='cannot be honoured'),
    ):
        await evaluator['scorer'](_params())

    run_judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_catalogue_false_overrides_the_built_in_hint_and_keeps_the_prompted_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = model_catalogue.ModelInfo(0.0, 0.0, 'typesafe', True, supports_classify=False)  # noqa: FBT003

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {JEV: info, 'jev-latest': info}

    monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)
    model_catalogue.reset_catalogue_cache()
    sink: dict[str, Any] = {}

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink.update(kwargs)
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=True, explanation='ok'))

    evaluator = llm_jury(name='x', criteria='c', judges=[JEV], max_tokens=1234, client=MagicMock())
    with patch.object(llm_jury_mod, 'run_judge', side_effect=fake_run_judge):
        await evaluator['scorer'](_params())

    assert sink['classify'] is None
    assert sink['cfg'].api == 'responses'
    assert sink['cfg'].max_tokens == 1234


@pytest.mark.asyncio
async def test_a_prompted_judge_on_a_mixed_panel_keeps_the_full_config() -> None:
    """Only the classify seat loses the sampling settings; the prompted judge still reads them."""
    sink: dict[str, Any] = {}

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink[cast(str, kwargs['model'])] = kwargs['cfg']
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=True, explanation='ok'))

    evaluator = llm_jury(
        name='x', criteria='c', judges=[JEV, 'gpt-5-mini'], max_tokens=1234, reasoning_effort='high', client=MagicMock()
    )
    with patch.object(llm_jury_mod, 'run_judge', side_effect=fake_run_judge):
        await evaluator['scorer'](_params())

    assert sink['gpt-5-mini'].max_tokens == 1234
    assert sink['gpt-5-mini'].reasoning_effort == 'high'
    assert sink[JEV].model_fields_set == {'model', 'timeout_ms'}


# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pairwise_builds_an_a_b_tie_choice_question() -> None:
    sink: list[Any] = []
    comparator = llm_jury_pairwise(criteria='which is better?', judges=[JEV], swap=False, client=MagicMock())

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink.append(kwargs.get('classify'))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='A', explanation='ok'))

    with patch.object(llm_jury_mod, 'run_judge', side_effect=fake_run_judge):
        await comparator.compare(question='q?', response_a='alpha', response_b='beta')

    question = sink[0]
    assert isinstance(question, ClassifyQuestion)
    assert question.kind == 'choice'
    assert question.instructions == 'which is better?'
    assert question.criteria == {
        'A': 'Response A is better',
        'B': 'Response B is better',
        'tie': 'Neither is clearly better',
    }
    state = cast(dict[str, Any], question.state)
    assert 'criteria' not in state
    assert state['question'] == 'q?'
    assert state['response_a.output.response'] == 'alpha'
    assert state['response_b.output.response'] == 'beta'


@pytest.mark.asyncio
async def test_pairwise_state_fields_override() -> None:
    sink: list[Any] = []
    comparator = llm_jury_pairwise(
        criteria='c',
        judges=[JEV],
        swap=False,
        state_fields=['response_a.output.response', 'response_b.output.response'],
        client=MagicMock(),
    )

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink.append(kwargs.get('classify'))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='tie', explanation='ok'))

    with patch.object(llm_jury_mod, 'run_judge', side_effect=fake_run_judge):
        await comparator.compare(question='q?', response_a='alpha', response_b='beta')

    assert list(cast(dict[str, Any], sink[0].state)) == [
        'response_a.output.response',
        'response_b.output.response',
    ]


@pytest.mark.asyncio
async def test_pairwise_swap_rebuilds_classify_state_for_each_ordering() -> None:
    sink: list[ClassifyQuestion] = []
    comparator = llm_jury_pairwise(criteria='c', judges=[JEV], swap=True, client=MagicMock())

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink.append(cast(ClassifyQuestion, kwargs['classify']))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='A', explanation='ok'))

    with patch.object(llm_jury_mod, 'run_judge', side_effect=fake_run_judge):
        await comparator.compare(question='q?', response_a='alpha', response_b='beta')

    states = [cast(dict[str, Any], question.state) for question in sink]
    assert {(state['response_a.output.response'], state['response_b.output.response']) for state in states} == {
        ('alpha', 'beta'),
        ('beta', 'alpha'),
    }


@pytest.mark.asyncio
async def test_pairwise_llm_only_panel_builds_no_question() -> None:
    sink: list[Any] = []
    comparator = llm_jury_pairwise(criteria='c', judges=['gpt-5-mini'], swap=False, client=MagicMock())

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink.append(kwargs.get('classify'))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='A', explanation='ok'))

    with patch.object(llm_jury_mod, 'run_judge', side_effect=fake_run_judge):
        await comparator.compare(question='q?', response_a='alpha', response_b='beta')

    assert sink == [None]


def test_pairwise_comparator_accepts_state_fields_directly() -> None:
    comparator = PairwiseComparator(
        panel=[JEV],
        criteria='c',
        system_prompt='s',
        swap=False,
        repetitions=1,
        replacement_judges=None,
        min_successful_judges=1,
        max_tokens=100,
        timeout_ms=1000,
        temperature=None,
        structured_output=True,
        extra_kwargs=None,
        extra_body=None,
        client=MagicMock(),
        state_fields=['question'],
    )
    sink: list[Any] = []

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink.append(kwargs.get('classify'))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='tie', explanation='ok'))

    with patch.object(llm_jury_mod, 'run_judge', side_effect=fake_run_judge):
        asyncio.run(comparator.compare(question='q?', response_a='alpha', response_b='beta'))

    assert list(cast(dict[str, Any], sink[0].state)) == ['question']


# ---------------------------------------------------------------------------
# Classify state carries values, not their prompt rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_default_state_carries_the_transcript_as_messages_not_as_a_json_string() -> None:
    """The state is serialised as JSON; a pre-serialised value would arrive escaped inside it."""
    sink: list[Any] = []
    await _run_and_capture(llm_jury(name='x', criteria='c', judges=[JEV], client=MagicMock()), sink)

    state = cast(dict[str, Any], sink[0].state)
    assert state['input.all_messages'] == [{'role': 'user', 'content': 'q'}]
    # `output.response` has only one spelling and is unchanged by the nested-first read.
    assert state['output.response'] == 'the answer'


@pytest.mark.asyncio
async def test_a_tool_call_state_field_is_the_list_of_calls() -> None:
    sink: list[Any] = []
    evaluator = llm_jury(
        name='x', criteria='c', judges=[JEV], state_fields=['output.tools_called'], client=MagicMock()
    )
    output = AgentResponse(
        output=[ToolCallOutputItem(id='t1', name='search', arguments='{"q": "x"}', result='hit')]
    )

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink.append(kwargs.get('classify'))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=True, explanation='ok'))

    with patch.object(llm_jury_mod, 'run_judge', side_effect=fake_run_judge):
        await evaluator['scorer'](_params(output))

    calls = cast(dict[str, Any], sink[0].state)['output.tools_called']
    assert isinstance(calls, list)
    assert calls[0]['name'] == 'search'
    assert calls[0]['arguments'] == {'q': 'x'}


@pytest.mark.asyncio
async def test_a_pairwise_state_field_with_both_spellings_resolves_to_the_value() -> None:
    """`response_a.output.messages` exists nested (a list) and flat (a JSON string)."""
    sink: list[Any] = []
    comparator = llm_jury_pairwise(
        criteria='c',
        judges=[JEV],
        swap=False,
        state_fields=['response_a.output.messages', 'response_a.output.response'],
        client=MagicMock(),
    )

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink.append(kwargs.get('classify'))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='A', explanation='ok'))

    with patch.object(llm_jury_mod, 'run_judge', side_effect=fake_run_judge):
        await comparator.compare(question='q?', response_a='alpha', response_b='beta')

    state = cast(dict[str, Any], sink[0].state)
    assert state['response_a.output.messages'] == [{'role': 'assistant', 'content': 'alpha'}]
    assert state['response_a.output.response'] == 'alpha'


@pytest.mark.asyncio
async def test_the_empty_state_warning_fires_once_per_evaluator_not_once_per_datapoint() -> None:
    sink: list[Any] = []
    evaluator = llm_jury(
        name='x',
        criteria='c',
        judges=[JEV],
        state_fields=['ouput.response'],  # typo on purpose: nothing resolves
        client=MagicMock(),
    )

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink.append(kwargs.get('classify'))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=True, explanation='ok'))

    with patch.object(llm_jury_mod.logger, 'warning') as warn, patch.object(
        llm_jury_mod, 'run_judge', side_effect=fake_run_judge
    ):
        await evaluator['scorer'](_params())
        await evaluator['scorer'](_params('a second answer'))

    messages = [call.args[0] for call in warn.call_args_list]
    assert sum('classify state is empty' in message for message in messages) == 1
    assert len(sink) == 2
    assert all(cast(dict[str, Any], question.state) == {} for question in sink)


@pytest.mark.asyncio
async def test_the_pairwise_empty_state_warning_fires_once_per_comparator() -> None:
    sink: list[Any] = []
    comparator = llm_jury_pairwise(
        criteria='c',
        judges=[JEV],
        swap=False,
        state_fields=['respones_a.output.response'],  # typo on purpose: nothing resolves
        client=MagicMock(),
    )

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        sink.append(kwargs.get('classify'))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='A', explanation='ok'))

    with patch.object(llm_jury_mod.logger, 'warning') as warn, patch.object(
        llm_jury_mod, 'run_judge', side_effect=fake_run_judge
    ):
        await comparator.compare(question='q?', response_a='alpha', response_b='beta')
        await comparator.compare(question='q2?', response_a='gamma', response_b='delta')

    messages = [call.args[0] for call in warn.call_args_list]
    assert sum('classify state is empty' in message for message in messages) == 1
    assert len(sink) == 2
