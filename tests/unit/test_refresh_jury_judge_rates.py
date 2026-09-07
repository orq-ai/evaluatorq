"""The decisions the monthly rate refresh has to surface for a human."""

import importlib.util
from pathlib import Path
from typing import Any

from evaluatorq.common.model_catalogue import ModelInfo

_spec = importlib.util.spec_from_file_location(
    'refresh_jury_judge_rates',
    Path(__file__).resolve().parents[2] / 'scripts' / 'refresh_jury_judge_rates.py',
)
assert _spec is not None and _spec.loader is not None
refresh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refresh)


def _row(input_rate: float = 1.0, output_rate: float = 4.0, effort: str | None = 'high') -> dict[str, Any]:
    return {'input_rate': input_rate, 'output_rate': output_rate, 'seated_effort': effort, 'priced_at_ceiling': True}


def _info(input_rate: float = 1.0, output_rate: float = 4.0, efforts: frozenset[str] | None = None) -> ModelInfo:
    return ModelInfo(
        input_cost_per_1k=input_rate / 1000,
        output_cost_per_1k=output_rate / 1000,
        provider='openai',
        supports_responses=True,
        reasoning_efforts=efforts,
    )


def test_rates_come_from_the_catalogue_and_the_derivation_is_carried_over() -> None:
    row = refresh.refreshed_row(_row(), _info(input_rate=2.5, output_rate=10.0))
    assert row['input_rate'] == 2.5
    assert row['output_rate'] == 10.0
    # Nothing here re-derives the frontier, so these must survive verbatim.
    assert row['seated_effort'] == 'high'
    assert row['priced_at_ceiling'] is True


def test_a_repricing_is_reported() -> None:
    previous = {'openai/gpt-5.6-luna': _row()}
    current = {'openai/gpt-5.6-luna': _row(input_rate=2.0)}
    (line,) = refresh.decisions(previous, current, {})
    assert line.startswith('REPRICED:')


def test_a_judge_missing_from_the_catalogue_is_reported() -> None:
    (line,) = refresh.decisions({'openai/gone': _row()}, {}, {})
    assert line.startswith('NOT SERVED:')


def test_a_seat_ranked_at_an_effort_the_model_does_not_accept_is_reported() -> None:
    """The check that catches a card-only effort label being priced as sendable."""
    row = _row(effort='reasoning')
    (line,) = refresh.decisions({'x/y': row}, {'x/y': row}, {'x/y': frozenset({'low', 'medium', 'high'})})
    assert line.startswith('EFFORT NOT ACCEPTED:')


def test_an_unchanged_table_produces_no_decisions() -> None:
    previous = {'openai/gpt-5.6-luna': _row()}
    assert refresh.decisions(previous, dict(previous), {'openai/gpt-5.6-luna': frozenset({'high'})}) == []
