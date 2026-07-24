from __future__ import annotations

import evaluatorq.common.llm_call as llm_call
from evaluatorq.common.llm_call import run_metadata_kwarg
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
