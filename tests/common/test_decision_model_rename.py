"""Classify-model names keep working after the rename to decision models, and every decision model gets the decision state."""

from __future__ import annotations

import contextlib
import importlib
import warnings

import pytest
from typer.testing import CliRunner

import evaluatorq
from evaluatorq.common import judge, llm_call, model_catalogue, model_input
from evaluatorq.common.model_catalogue import ModelInfo
from evaluatorq.trace_finder import settings as finder_settings

RENAMED = [
    (evaluatorq, 'ClassifyOutcome', 'DecisionOutcome'),
    (evaluatorq, 'ClassifyQuestion', 'DecisionQuestion'),
    (evaluatorq, 'ClassifyRequest', 'DecisionRequest'),
    (evaluatorq, 'run_classify', 'run_decision'),
    (judge, 'ClassifyQuestion', 'DecisionQuestion'),
    (judge, 'ClassifyAnswer', 'DecisionAnswer'),
    (judge, 'ClassifyResponse', 'DecisionResponse'),
    (judge, 'ClassifyRequest', 'DecisionRequest'),
    (judge, 'ClassifyOutcome', 'DecisionOutcome'),
    (judge, 'run_classify', 'run_decision'),
    (judge, 'run_classify_judges', 'run_decision_judges'),
    (llm_call, 'classify_request_body', 'decision_request_body'),
    (llm_call, 'execute_classify', 'execute_decision'),
    (model_catalogue, 'KNOWN_CLASSIFY_MODELS', 'KNOWN_DECISION_MODELS'),
    (model_catalogue, 'is_known_classify_model', 'is_known_decision_model'),
    (model_catalogue, 'supports_classify', 'is_decision_model'),
]


@pytest.mark.parametrize(('module', 'old', 'new'), RENAMED)
def test_an_old_classify_name_resolves_to_its_decision_name_and_warns(module: object, old: str, new: str) -> None:
    with pytest.warns(DeprecationWarning, match=rf'^{old} is deprecated; use {new}\.$'):
        resolved = getattr(module, old)
    assert resolved is getattr(module, new)


def test_no_public_name_still_says_classify_in_the_renamed_modules() -> None:
    for module in (evaluatorq, judge, llm_call, model_catalogue):
        public = [name for name in getattr(module, '__all__', vars(module)) if not name.startswith('_')]
        assert [name for name in public if 'classify' in name.lower()] == [], module.__name__


def test_model_info_reads_the_old_field_name_with_a_warning() -> None:
    with pytest.warns(DeprecationWarning, match=r'supports_classify=\.\.\.\) is deprecated'):
        info = ModelInfo(0.0, 0.0, 'acme', False, supports_classify=True)  # noqa: FBT003
    assert info.supports_decisions is True
    with pytest.warns(DeprecationWarning, match=r'ModelInfo\.supports_classify is deprecated'):
        assert info.supports_classify is True


@pytest.mark.asyncio
async def test_models_by_provider_reads_the_old_kind_with_a_warning(fake_catalogue) -> None:
    fake_catalogue({'typesafe/jev-latest': ModelInfo(0.0, 0.0, 'typesafe', False, supports_decisions=True)})  # noqa: FBT003
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        current = await model_catalogue.models_by_provider(kind='decision')
    with pytest.warns(DeprecationWarning, match=r"kind='classify'\) is deprecated"):
        old = await model_catalogue.models_by_provider(kind='classify')
    assert old == current == {'typesafe': ['typesafe/jev-latest']}


def test_a_result_saved_with_the_old_endpoint_name_reads_as_decisions() -> None:
    assert judge.JudgeOutcome(endpoint='classify').endpoint == 'decisions'
    assert judge.JudgeOutcome(endpoint='decisions').endpoint == 'decisions'


@pytest.mark.asyncio
async def test_run_judge_reads_the_old_classify_keyword_with_a_warning() -> None:
    question = judge.DecisionQuestion(kind='noul', instructions='Is it fine?', state='reply')
    # The fake client fails the call itself; only the warning raised before it matters here.
    with pytest.warns(DeprecationWarning, match=r'^run_judge\(classify=\.\.\.\) is deprecated; use decision=\.$'), contextlib.suppress(Exception):
        await judge.run_judge(
            client=object(),  # ty: ignore[invalid-argument-type]
            model='typesafe/jev-latest',
            cfg=judge.LLMCallConfig(),
            prompt_template='',
            replacements={},
            classify=question,
        )


ENV = 'EVALUATORQ_DECISION_MODELS'


@pytest.mark.parametrize(
    ('value', 'model', 'expected'),
    [
        ('acme/sorter', 'acme/sorter', True),
        ('acme/sorter', 'sorter', True),
        ('sorter', 'acme/sorter', True),
        (' acme/sorter , other/x ', 'other/x', True),
        ('!typesafe/jev-latest', 'typesafe/jev-latest', False),
        ('acme/sorter', 'acme/other', None),
        ('', 'acme/sorter', None),
    ],
)
def test_the_override_names_decision_models_and_marks_others_off(
    monkeypatch: pytest.MonkeyPatch, value: str, model: str, expected: bool | None
) -> None:
    monkeypatch.setenv(ENV, value)
    assert model_catalogue.decision_model_override(model) is expected


@pytest.mark.asyncio
async def test_the_override_outranks_the_catalogue_everywhere(monkeypatch: pytest.MonkeyPatch, fake_catalogue) -> None:
    fake_catalogue({
        'typesafe/jev-latest': ModelInfo(0.0, 0.0, 'typesafe', False, supports_decisions=True),  # noqa: FBT003
        'acme/sorter': ModelInfo(0.0, 0.0, 'acme', True, supports_decisions=False),  # noqa: FBT003
    })
    monkeypatch.setenv(ENV, 'acme/sorter,!typesafe/jev-latest')
    assert await model_catalogue.is_decision_model('acme/sorter') is True
    assert await model_catalogue.is_decision_model('typesafe/jev-latest') is False
    assert model_catalogue.is_known_decision_model('acme/sorter') is True
    assert model_catalogue.is_known_decision_model('typesafe/jev-latest') is False
    assert model_input.uses_decision_state('acme/sorter') is True
    assert model_input.uses_decision_state('typesafe/jev-latest') is False


@pytest.mark.asyncio
async def test_a_decision_model_other_than_jev_gets_the_decision_state_once_looked_up(fake_catalogue) -> None:
    fake_catalogue({'acme/sorter': ModelInfo(0.0, 0.0, 'acme', True, supports_decisions=True)})  # noqa: FBT003
    assert model_input.uses_decision_state('acme/sorter') is False
    assert await model_catalogue.is_decision_model('acme/sorter') is True
    assert model_input.uses_decision_state('acme/sorter') is True
    # The judge's stricter check still prompts a model that serves chat as well.
    assert model_catalogue.is_known_decision_model('acme/sorter') is False


@pytest.mark.asyncio
async def test_a_plain_model_keeps_the_text_view(fake_catalogue) -> None:
    fake_catalogue({'openai/gpt-6-luna': ModelInfo(0.0, 0.0, 'openai', True)})  # noqa: FBT003
    assert await model_catalogue.is_decision_model('openai/gpt-6-luna') is False
    assert model_input.uses_decision_state('openai/gpt-6-luna') is False
    assert model_input.uses_decision_state('typesafe/jev-latest') is True


def test_the_cli_help_lists_only_the_current_filter_modes() -> None:
    import typer

    from evaluatorq import cli as cli_module

    app = typer.Typer()
    cli_module._register_subapps(app)
    for command in (['find', '--help'], ['insights', '--help']):
        result = CliRunner().invoke(app, command, env={'COLUMNS': '200', 'NO_COLOR': '1'})
        assert result.exit_code == 0, result.output
        assert '[llm|llm+decision]' in result.output
        assert 'llm+jev' not in result.output


def test_the_cli_type_reads_the_old_mode_with_the_same_warning() -> None:
    with pytest.warns(DeprecationWarning, match=r"^Filter selection mode 'llm\+jev' is deprecated; use 'llm\+decision'\.$"):
        assert finder_settings.FILTER_SELECTION_CLI_TYPE.convert('llm+jev', None, None) == 'llm+decision'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert finder_settings.FILTER_SELECTION_CLI_TYPE.convert('llm', None, None) == 'llm'


def test_a_saved_settings_file_reads_the_old_mode_with_the_same_warning() -> None:
    with pytest.warns(DeprecationWarning, match=r"^Filter selection mode 'llm\+jev' is deprecated"):
        loaded = finder_settings.DashboardSettings.model_validate({'filter_selection': 'llm+jev'})
    assert loaded.filter_selection == 'llm+decision'


def test_the_shared_module_imports_cleanly() -> None:
    importlib.reload(importlib.import_module('evaluatorq.common.deprecation'))
