"""The deprecated ``parallelism=`` alias keeps working, loudly."""

import pytest

from evaluatorq.common.parallelism import resolve_datapoint_parallelism
from evaluatorq.types import EvaluatorParams


def _resolve(datapoint_parallelism, parallelism):
    return resolve_datapoint_parallelism(
        datapoint_parallelism, parallelism, default=10, caller='evaluatorq'
    )


def test_default_applies_when_neither_is_given():
    assert _resolve(None, None) == 10


def test_old_name_still_works_but_warns():
    with pytest.warns(DeprecationWarning, match='deprecated'):
        assert _resolve(None, 3) == 3


def test_new_name_wins_when_both_are_given():
    with pytest.warns(DeprecationWarning):
        assert _resolve(7, 3) == 7


def test_new_name_alone_does_not_warn(recwarn):
    assert _resolve(4, None) == 4
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_evaluator_params_accepts_the_old_field_name():
    params = EvaluatorParams.model_validate(
        {'data': [], 'jobs': [lambda d, i: None], 'parallelism': 3}
    )
    assert params.datapoint_parallelism == 3
