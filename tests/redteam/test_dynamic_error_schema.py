"""Integration-style tests for dynamic error schema propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from types import SimpleNamespace

from evaluatorq.redteam.contracts import AgentContext
from evaluatorq.redteam.reports.converters import (
    _coerce_job_output_payload,  # pyright: ignore[reportPrivateUsage]
    dynamic_evaluatorq_results_to_report,
)


@dataclass
class _FakeDataPoint:
    inputs: dict[str, Any]


@dataclass
class _FakeJobResult:
    job_name: str
    output: object
    evaluator_scores: list[Any] = field(default_factory=list)
    error: str | None = None


@dataclass
class _FakeResult:
    data_point: _FakeDataPoint
    job_results: list[_FakeJobResult]
    error: str | None = None


def test_job_output_payload_coercion_keeps_structured_error_fields() -> None:
    payload = _coerce_job_output_payload({
        'final_response': 'done',
        'error': 'Target timeout',
        'error_type': 'target_error',
        'error_stage': 'target_call',
        'error_code': 'openai.http.504',
        'error_details': {'status_code': 504, 'raw_message': 'gateway timeout'},
    })

    assert payload.error_stage == 'target_call'
    assert payload.error_code == 'openai.http.504'
    assert payload.error_details == {'status_code': 504, 'raw_message': 'gateway timeout'}


def test_dynamic_results_to_report_maps_error_stage_code_and_details() -> None:
    score = SimpleNamespace(value=False, explanation='Evaluator marked this vulnerable')
    evaluator_score = SimpleNamespace(score=score)

    result = _FakeResult(
        data_point=_FakeDataPoint(
            inputs={
                'id': 'asi01-test-001',
                'category': 'ASI01',
                'strategy': {
                    'category': 'ASI01',
                    'name': 'tool_output_hijack',
                    'description': 'test strategy',
                    'attack_technique': 'direct-injection',
                    'delivery_methods': ['direct-request'],
                    'turn_type': 'single',
                    'severity': 'medium',
                    'objective_template': 'test objective',
                    'prompt_template': 'test prompt',
                    'is_generated': False,
                },
                'objective': 'induce harmful action',
            }
        ),
        job_results=[
            _FakeJobResult(
                job_name='dynamic:redteam:agent:custom-1',
                output={
                    'final_response': 'I can do that',
                    'turns': 1,
                    'error': 'Target failed',
                    'error_type': 'target_error',
                    'error_stage': 'target_call',
                    'error_code': 'orq.http.503',
                    'error_details': {'status_code': 503, 'provider_code': 'service_unavailable'},
                },
                evaluator_scores=[evaluator_score],
            )
        ],
    )

    report = dynamic_evaluatorq_results_to_report(
        agent_context=AgentContext(key='custom-1'),
        categories_tested=['ASI01'],
        results=[result],
        duration_seconds=0.1,
        description='dynamic error mapping test',
    )

    row = report.results[0]
    assert report.tested_agents == ['custom-1']
    assert row.error == 'Target failed'
    assert row.error_type == 'target_error'
    assert row.error_stage == 'target_call'
    assert row.error_code == 'orq.http.503'
    assert row.error_details == {'status_code': 503, 'provider_code': 'service_unavailable'}
