"""The deprecated ``parallelism=`` alias keeps working, loudly."""

import inspect

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


def test_every_entry_point_defaults_to_the_same_datapoint_count():
    """One number on every surface, so a caller can size it once.

    Simulation used to default to 5 while core and red teaming used 10; each
    default is its own literal, so nothing but this test notices them drifting.
    """
    from evaluatorq.simulation._config import SimulationConfig
    from evaluatorq.simulation.api import _generate_and_simulate_run, _simulate_run

    assert EvaluatorParams.model_fields['datapoint_parallelism'].default == 10
    assert SimulationConfig.model_fields['datapoint_parallelism'].default == 10
    for func in (_simulate_run, _generate_and_simulate_run):
        assert inspect.signature(func).parameters['datapoint_parallelism'].default == 10, func
    assert _resolve(None, None) == 10


def test_evaluator_params_accepts_the_old_field_name():
    params = EvaluatorParams.model_validate(
        {'data': [], 'jobs': [lambda d, i: None], 'parallelism': 3}
    )
    assert params.datapoint_parallelism == 3
