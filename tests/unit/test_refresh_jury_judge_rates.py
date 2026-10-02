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
    return {'input_rate': input_rate, 'output_rate': output_rate, 'seated_effort': effort}


def _info(input_rate: float = 1.0, output_rate: float = 4.0, default_effort: str | None = None) -> ModelInfo:
    return ModelInfo(
        input_cost_per_1k=input_rate / 1000,
        output_cost_per_1k=output_rate / 1000,
        provider='openai',
        supports_responses=True,
        default_reasoning_effort=default_effort,
    )


def test_rates_and_default_effort_come_from_the_catalogue() -> None:
    row = refresh.refreshed_row(_info(input_rate=2.5, output_rate=10.0, default_effort='medium'))
    assert row == {'input_rate': 2.5, 'output_rate': 10.0, 'seated_effort': 'medium'}


def test_a_repricing_is_reported() -> None:
    previous = {'openai/gpt-5.6-luna': _row()}
    current = {'openai/gpt-5.6-luna': _row(input_rate=2.0)}
    (line,) = refresh.decisions(previous, current)
    assert line.startswith('REPRICED:')


def test_a_judge_missing_from_the_catalogue_is_reported() -> None:
    (line,) = refresh.decisions({'openai/gone': _row()}, {})
    assert line.startswith('NOT SERVED:')


def test_a_changed_default_effort_is_reported() -> None:
    (line,) = refresh.decisions({'x/y': _row(effort='high')}, {'x/y': _row(effort='medium')})
    assert line.startswith('DEFAULT EFFORT CHANGED:')


def test_an_unchanged_table_produces_no_decisions() -> None:
    previous = {'openai/gpt-5.6-luna': _row()}
    assert refresh.decisions(previous, dict(previous)) == []
