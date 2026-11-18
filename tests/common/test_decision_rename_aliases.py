"""Names from before Jev was renamed to "decision" keep working, and say what replaces them."""

from __future__ import annotations

import warnings

import pytest

from evaluatorq.common import model_input
from evaluatorq.insights.models import InsightsPopulation
from evaluatorq.trace_finder.settings import normalize_filter_selection

RENAMED = [
    ('JEV_STATE_CHARS', 'DECISION_STATE_CHARS'),
    ('JEV_STATE_QUESTION_CHARS', 'DECISION_STATE_QUESTION_CHARS'),
    ('JEV_STATE_ALL_QUESTIONS_CHARS', 'DECISION_STATE_ALL_QUESTIONS_CHARS'),
    ('DEFAULT_JEV_OPTIONS', 'DEFAULT_DECISION_OPTIONS'),
    ('JevOptions', 'DecisionOptions'),
    ('is_jev_model', 'uses_decision_state'),
    ('jev_coverage', 'decision_coverage'),
    ('jev_state', 'decision_state'),
    ('jev_text_state', 'decision_text_state'),
    ('jev_state_char_limit', 'decision_state_char_limit'),
    ('fit_jev_state', 'fit_decision_state'),
]


@pytest.mark.parametrize(('old', 'new'), RENAMED)
def test_an_old_name_resolves_to_the_renamed_object_and_warns(old: str, new: str) -> None:
    with pytest.warns(DeprecationWarning, match=rf'^{old} is deprecated; use {new}\.$'):
        resolved = getattr(model_input, old)
    assert resolved is getattr(model_input, new)


def test_no_public_name_still_says_jev_and_every_alias_points_at_a_real_name() -> None:
    assert [name for name in vars(model_input) if 'jev' in name.lower() and not name.startswith('_')] == []
    assert sorted(model_input._RENAMED) == sorted(old for old, _ in RENAMED)
    for new in model_input._RENAMED.values():
        assert hasattr(model_input, new), new


def test_the_new_names_do_not_warn_and_an_unknown_name_still_fails() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert model_input.decision_state is not None
        assert model_input.DecisionOptions is not None
    with pytest.raises(AttributeError, match="has no attribute 'jev_nothing'"):
        _ = model_input.jev_nothing


def test_the_old_import_statement_still_works() -> None:
    with pytest.warns(DeprecationWarning):
        from evaluatorq.common.model_input import JevOptions, jev_state
    assert JevOptions is model_input.DecisionOptions
    assert jev_state is model_input.decision_state


@pytest.mark.parametrize(
    ('value', 'expected'),
    [('llm+jev', 'llm+decision'), ('llm+decision', 'llm+decision'), ('llm', 'llm'), ('merged', 'merged'), (None, None)],
)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_only_the_old_filter_mode_name_is_rewritten(value: object, expected: object) -> None:
    assert normalize_filter_selection(value) == expected


def test_a_population_accepts_the_old_filter_mode_name_with_a_warning() -> None:
    with pytest.warns(DeprecationWarning, match=r"^Filter selection mode 'llm\+jev' is deprecated; use 'llm\+decision'\.$") as caught:
        old = InsightsPopulation(query='refunds', filter_selection='llm+jev')
    assert old.filter_selection == 'llm+decision'
    # The warning names this file, not pydantic: Python's default filters hide a warning attributed to a library.
    assert [warning.filename for warning in caught] == [__file__]

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert InsightsPopulation(query='refunds', filter_selection='llm+decision').filter_selection == 'llm+decision'
        assert InsightsPopulation(query='refunds', filter_selection='llm').filter_selection == 'llm'
        assert InsightsPopulation(query='refunds').filter_selection is None


@pytest.mark.parametrize(('value', 'expected'), [('decision', True), ('jev', True), ('text', False), (None, False)])
def test_a_report_saved_before_the_rename_still_reads_as_decision_state(value: object, expected: bool) -> None:
    assert model_input.is_decision_state_format(value) is expected
