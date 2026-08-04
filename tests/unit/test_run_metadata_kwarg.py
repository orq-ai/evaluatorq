from __future__ import annotations

import evaluatorq.common.llm_call as llm_call
from evaluatorq.common.llm_call import apply_pipeline_metadata, run_metadata_kwarg
from evaluatorq.common.thread_context import evaluatorq_pipeline, evaluatorq_run_id


def test_empty_off_orq(monkeypatch) -> None:
    monkeypatch.setattr(llm_call, 'client_routes_through_orq', lambda _c: False)
    with evaluatorq_run_id('r1'):
        assert run_metadata_kwarg(object()) == {}


def test_empty_when_nothing_bound_on_orq(monkeypatch) -> None:
    monkeypatch.setattr(llm_call, 'client_routes_through_orq', lambda _c: True)
    assert run_metadata_kwarg(object()) == {}


def test_metadata_on_orq_when_bound(monkeypatch) -> None:
    monkeypatch.setattr(llm_call, 'client_routes_through_orq', lambda _c: True)
    with evaluatorq_pipeline('red_teaming'), evaluatorq_run_id('r1'):
        assert run_metadata_kwarg(object()) == {
            'metadata': {'evaluatorq_pipeline': 'red_teaming', 'evaluatorq_run_id': 'r1'}
        }


def test_caller_supplied_metadata_wins_on_conflict(monkeypatch) -> None:
    """A caller's own metadata key survives; the bound keys fill in around it."""
    monkeypatch.setattr(llm_call, 'client_routes_through_orq', lambda _c: True)
    params = {'metadata': {'evaluatorq_run_id': 'caller-id', 'custom': 'x'}}
    with evaluatorq_pipeline('red_teaming'), evaluatorq_run_id('bound-id'):
        apply_pipeline_metadata(object(), params)

    assert params['metadata'] == {
        'evaluatorq_run_id': 'caller-id',
        'evaluatorq_pipeline': 'red_teaming',
        'custom': 'x',
    }


def test_apply_is_a_no_op_off_orq(monkeypatch) -> None:
    """Off-Orq the params dict is left untouched — an unknown field would 400."""
    monkeypatch.setattr(llm_call, 'client_routes_through_orq', lambda _c: False)
    params: dict = {'model': 'gpt-4o-mini'}
    with evaluatorq_run_id('r1'):
        apply_pipeline_metadata(object(), params)

    assert params == {'model': 'gpt-4o-mini'}
