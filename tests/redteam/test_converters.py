"""Tests for report converters and merge_reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from evaluatorq.redteam.contracts import (
    AgentContext,
    AgentInfo,
    AttackInfo,
    AttackTechnique,
    CategorySummary,
    DeliveryMethod,
    ExecutionDetails,
    Framework,
    JobOutputPayload,
    Pipeline,
    RedTeamReport,
    RedTeamResult,
    ReportSummary,
    VulnerabilityDomain,
    Severity,
    TokenUsage,
    TurnType,
    UnifiedEvaluationResult,
)
from evaluatorq.redteam.reports.converters import (
    _aggregate_token_usage,
    _coerce_job_output_payload,
    compute_report_summary,
    dynamic_evaluatorq_results_to_report,
    merge_reports,
    static_evaluatorq_results_to_reports,
    static_sample_to_result,
)


def _make_result(
    category: str = 'ASI01',
    passed: bool | None = True,
    agent_key: str = 'agent-a',
    technique: AttackTechnique = AttackTechnique.INDIRECT_INJECTION,
    delivery_methods: list[DeliveryMethod] | None = None,
    turn_type: TurnType = TurnType.SINGLE,
    severity: Severity = Severity.MEDIUM,
    vulnerability_domain: VulnerabilityDomain | None = None,
    framework: Framework = Framework.OWASP_ASI,
    execution: ExecutionDetails | None = None,
    error: str | None = None,
    error_type: str | None = None,
) -> RedTeamResult:
    """Helper to create a minimal RedTeamResult."""
    return RedTeamResult(
        attack=AttackInfo(
            id=f'{category}-test-001',
            category=category,
            framework=framework,
            attack_technique=technique,
            delivery_methods=delivery_methods or [DeliveryMethod.DIRECT_REQUEST],
            turn_type=turn_type,
            severity=severity,
            vulnerability_domain=vulnerability_domain,
            source='test',
        ),
        agent=AgentInfo(key=agent_key),
        messages=[],
        vulnerable=passed is False,
        evaluation=UnifiedEvaluationResult(passed=passed, explanation='test') if passed is not None else None,
        execution=execution,
        error=error,
        error_type=error_type,
    )


def _make_report(
    results: list[RedTeamResult] | None = None,
    pipeline: Pipeline = Pipeline.DYNAMIC,
    framework: Framework | None = Framework.OWASP_ASI,
    categories: list[str] | None = None,
    agents: list[str] | None = None,
    description: str | None = None,
) -> RedTeamReport:
    """Helper to create a minimal RedTeamReport."""
    results = results or []
    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        description=description,
        pipeline=pipeline,
        framework=framework,
        categories_tested=categories or [],
        tested_agents=agents or [],
        total_results=len(results),
        results=results,
        summary=compute_report_summary(results),
    )


class TestMergeReports:
    """Tests for merge_reports()."""

    def test_merge_no_reports_raises(self):
        with pytest.raises(ValueError, match='at least one report'):
            merge_reports()

    def test_merge_single_report_returns_same(self):
        report = _make_report(description='only one')
        merged = merge_reports(report)
        assert merged is report

    def test_merge_combines_results(self):
        r1 = _make_result(category='ASI01', passed=True)
        r2 = _make_result(category='ASI02', passed=False)
        report_a = _make_report(results=[r1], categories=['ASI01'], agents=['agent-a'])
        report_b = _make_report(results=[r2], categories=['ASI02'], agents=['agent-b'])

        merged = merge_reports(report_a, report_b, description='merged')
        assert merged.total_results == 2
        assert len(merged.results) == 2
        assert set(merged.categories_tested) == {'ASI01', 'ASI02'}
        assert set(merged.tested_agents) == {'agent-a', 'agent-b'}
        assert merged.description == 'merged'

    def test_merge_recomputes_summary(self):
        r1 = _make_result(category='ASI01', passed=True)
        r2 = _make_result(category='ASI01', passed=False)
        report_a = _make_report(results=[r1], categories=['ASI01'])
        report_b = _make_report(results=[r2], categories=['ASI01'])

        merged = merge_reports(report_a, report_b)
        assert merged.summary.total_attacks == 2
        assert merged.summary.vulnerabilities_found == 1
        assert merged.summary.resistance_rate == 0.5

    def test_merge_resolves_pipeline_hybrid(self):
        report_a = _make_report(pipeline=Pipeline.DYNAMIC)
        report_b = _make_report(pipeline=Pipeline.STATIC)

        merged = merge_reports(report_a, report_b)
        assert merged.pipeline == Pipeline.HYBRID

    def test_merge_resolves_pipeline_same(self):
        report_a = _make_report(pipeline=Pipeline.DYNAMIC)
        report_b = _make_report(pipeline=Pipeline.DYNAMIC)

        merged = merge_reports(report_a, report_b)
        assert merged.pipeline == Pipeline.DYNAMIC

    def test_merge_resolves_framework_none_when_mixed(self):
        report_a = _make_report(framework=Framework.OWASP_ASI)
        report_b = _make_report(framework=Framework.OWASP_LLM)

        merged = merge_reports(report_a, report_b)
        assert merged.framework is None

    def test_merge_resolves_framework_same(self):
        report_a = _make_report(framework=Framework.OWASP_ASI)
        report_b = _make_report(framework=Framework.OWASP_ASI)

        merged = merge_reports(report_a, report_b)
        assert merged.framework == Framework.OWASP_ASI


class TestComputeReportSummary:
    """Tests for compute_report_summary() and all summary breakdowns."""

    def test_empty_results(self):
        summary = compute_report_summary([])
        assert summary.total_attacks == 0
        assert summary.vulnerability_rate == 0.0
        assert summary.average_turns_per_attack == 0.0

    def test_vulnerability_rate(self):
        results = [
            _make_result(passed=True),
            _make_result(passed=False),
            _make_result(passed=True),
            _make_result(passed=False),
        ]
        summary = compute_report_summary(results)
        assert summary.vulnerability_rate == 0.5
        assert summary.resistance_rate == 0.5

    def test_average_turns_per_attack(self):
        results = [
            _make_result(execution=ExecutionDetails(turns=3)),
            _make_result(execution=ExecutionDetails(turns=5)),
        ]
        summary = compute_report_summary(results)
        assert summary.average_turns_per_attack == 4.0
        assert summary.total_turns == 8

    def test_by_technique_full_summary(self):
        results = [
            _make_result(technique=AttackTechnique.INDIRECT_INJECTION, passed=True),
            _make_result(technique=AttackTechnique.INDIRECT_INJECTION, passed=False),
            _make_result(technique=AttackTechnique.DAN, passed=False),
        ]
        summary = compute_report_summary(results)
        assert 'indirect-injection' in summary.by_technique
        tech = summary.by_technique['indirect-injection']
        assert tech.total_attacks == 2
        assert tech.vulnerabilities_found == 1
        assert tech.resistance_rate == 0.5
        assert tech.vulnerability_rate == 0.5

        dan = summary.by_technique['DAN']
        assert dan.total_attacks == 1
        assert dan.vulnerability_rate == 1.0

    def test_by_severity(self):
        results = [
            _make_result(severity=Severity.CRITICAL, passed=False),
            _make_result(severity=Severity.CRITICAL, passed=True),
            _make_result(severity=Severity.LOW, passed=True),
        ]
        summary = compute_report_summary(results)
        assert 'critical' in summary.by_severity
        assert summary.by_severity['critical'].total_attacks == 2
        assert summary.by_severity['critical'].vulnerability_rate == 0.5
        assert summary.by_severity['low'].total_attacks == 1
        assert summary.by_severity['low'].vulnerability_rate == 0.0

    def test_by_delivery_method(self):
        results = [
            _make_result(delivery_methods=[DeliveryMethod.DAN, DeliveryMethod.ROLE_PLAY], passed=False),
            _make_result(delivery_methods=[DeliveryMethod.DAN], passed=True),
        ]
        summary = compute_report_summary(results)
        assert summary.by_delivery_method['DAN'].total_attacks == 2
        assert summary.by_delivery_method['DAN'].vulnerabilities_found == 1
        assert summary.by_delivery_method['role-play'].total_attacks == 1
        assert summary.by_delivery_method['role-play'].vulnerabilities_found == 1

    def test_by_turn_type(self):
        results = [
            _make_result(turn_type=TurnType.SINGLE, passed=True, execution=ExecutionDetails(turns=1)),
            _make_result(turn_type=TurnType.MULTI, passed=False, execution=ExecutionDetails(turns=4)),
            _make_result(turn_type=TurnType.MULTI, passed=True, execution=ExecutionDetails(turns=6)),
        ]
        summary = compute_report_summary(results)
        assert summary.by_turn_type['single'].total_attacks == 1
        assert summary.by_turn_type['single'].vulnerability_rate == 0.0
        assert summary.by_turn_type['multi'].total_attacks == 2
        assert summary.by_turn_type['multi'].vulnerability_rate == 0.5
        assert summary.by_turn_type['multi'].average_turns == 5.0

    def test_by_domain(self):
        results = [
            _make_result(vulnerability_domain=VulnerabilityDomain.MODEL, passed=False),
            _make_result(vulnerability_domain=VulnerabilityDomain.AGENT, passed=True),
            _make_result(passed=True),  # no domain
        ]
        summary = compute_report_summary(results)
        assert 'model' in summary.by_domain
        assert summary.by_domain['model'].vulnerability_rate == 1.0
        assert summary.by_domain['agent'].vulnerability_rate == 0.0
        # No domain → not in by_domain
        assert len(summary.by_domain) == 2

    def test_by_framework(self):
        results = [
            _make_result(framework=Framework.OWASP_ASI, passed=False),
            _make_result(framework=Framework.OWASP_ASI, passed=True),
            _make_result(framework=Framework.OWASP_LLM, category='LLM01', passed=False),
        ]
        summary = compute_report_summary(results)
        assert summary.by_framework['OWASP-ASI'].total_attacks == 2
        assert summary.by_framework['OWASP-ASI'].vulnerability_rate == 0.5
        assert summary.by_framework['OWASP-LLM'].total_attacks == 1
        assert summary.by_framework['OWASP-LLM'].vulnerability_rate == 1.0

    def test_category_summary_parity_fields(self):
        results = [
            _make_result(category='ASI01', passed=True),
            _make_result(category='ASI01', passed=False),
            _make_result(category='ASI01', passed=None, error='timeout', error_type='timeout'),
        ]
        summary = compute_report_summary(results)
        cat = summary.by_category['ASI01']
        assert cat.evaluated_attacks == 2
        assert cat.unevaluated_attacks == 1
        assert cat.evaluation_coverage == pytest.approx(2 / 3)
        assert cat.total_errors == 1
        assert cat.vulnerability_rate == 0.5

    def test_token_usage_total_aggregation(self):
        results = [
            _make_result(
                execution=ExecutionDetails(
                    turns=1,
                    token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, calls=1),
                ),
            ),
            _make_result(
                execution=ExecutionDetails(
                    turns=1,
                    token_usage=TokenUsage(prompt_tokens=200, completion_tokens=80, total_tokens=280, calls=2),
                ),
            ),
        ]
        summary = compute_report_summary(results)
        assert summary.token_usage_total is not None
        assert summary.token_usage_total.prompt_tokens == 300
        assert summary.token_usage_total.completion_tokens == 130
        assert summary.token_usage_total.total_tokens == 430
        assert summary.token_usage_total.calls == 3

    def test_token_usage_total_none_when_no_usage(self):
        results = [_make_result(), _make_result()]
        summary = compute_report_summary(results)
        assert summary.token_usage_total is None


# ---------------------------------------------------------------------------
# Helpers for new converter tests
# ---------------------------------------------------------------------------

def _make_dynamic_mock_result(
    category: str = 'ASI01',
    score_value: bool = True,
    score_explanation: str = 'Resisted',
    final_response: str = 'I cannot help',
    turns: int = 2,
    objective_achieved: bool = False,
    token_usage: dict[str, Any] | None = None,
    error: str | None = None,
    output_error: str | None = None,
    evaluator_scores: list[Any] | None = None,
    strategy_name: str = 'test-strat',
    attack_technique: str = 'indirect-injection',
    delivery_methods: list[str] | None = None,
    turn_type: str = 'single',
    severity: str = 'medium',
    is_generated: bool = False,
) -> SimpleNamespace:
    """Build a minimal mock evaluatorq DataPointResult for dynamic pipeline tests."""
    output: dict[str, Any] = {
        'final_response': final_response,
        'turns': turns,
        'conversation': [],
        'objective_achieved': objective_achieved,
    }
    if token_usage is not None:
        output['token_usage'] = token_usage
    if output_error is not None:
        output['error'] = output_error

    if evaluator_scores is None:
        evaluator_scores = [
            SimpleNamespace(score=SimpleNamespace(value=score_value, explanation=score_explanation))
        ]

    return SimpleNamespace(
        data_point=SimpleNamespace(inputs={
            'category': category,
            'strategy': {
                'name': strategy_name,
                'category': category,
                'description': 'Test attack strategy description',
                'attack_technique': attack_technique,
                'delivery_methods': delivery_methods or ['direct-request'],
                'turn_type': turn_type,
                'severity': severity,
                'is_generated': is_generated,
                'objective_template': 'Make the agent {objective}',
            },
            'objective': 'test objective',
        }),
        job_results=[SimpleNamespace(
            output=output,
            evaluator_scores=evaluator_scores,
        )],
        error=error,
    )


def _make_static_mock_result(
    job_name: str = 'target-job',
    category: str = 'ASI01',
    score_value: bool = True,
    score_explanation: str = 'ok',
    final_response: str = 'denied',
    token_usage: dict[str, Any] | None = None,
    dp_error: str | None = None,
    job_error: str | None = None,
    attack_technique: str = 'indirect-injection',
    delivery_method: str = 'direct-request',
    severity: str = 'medium',
    vulnerability_domain: str = 'agent',
    framework: str = 'OWASP-ASI',
    turn_type: str = 'single',
) -> SimpleNamespace:
    """Build a minimal mock evaluatorq DataPointResult for static pipeline tests."""
    output: dict[str, Any] = {'final_response': final_response}
    if token_usage is not None:
        output['token_usage'] = token_usage

    return SimpleNamespace(
        data_point=SimpleNamespace(inputs={
            'id': 'test-001',
            'category': category,
            'attack_technique': attack_technique,
            'delivery_method': delivery_method,
            'severity': severity,
            'vulnerability_domain': vulnerability_domain,
            'framework': framework,
            'turn_type': turn_type,
            'source': 'test',
            'messages': [],
        }),
        job_results=[SimpleNamespace(
            job_name=job_name,
            output=output,
            evaluator_scores=[
                SimpleNamespace(score=SimpleNamespace(value=score_value, explanation=score_explanation))
            ],
            error=job_error,
        )],
        error=dp_error,
    )


def _make_agent_context(key: str = 'test-agent', model: str = 'gpt-4o') -> AgentContext:
    return AgentContext(key=key, model=model, display_name='Test Agent')


# ---------------------------------------------------------------------------
# TestCoerceJobOutputPayload
# ---------------------------------------------------------------------------


class TestCoerceJobOutputPayload:
    """Tests for _coerce_job_output_payload()."""

    def test_dict_input_with_final_response(self):
        raw = {'final_response': 'Hello there', 'turns': 3}
        result = _coerce_job_output_payload(raw)
        assert isinstance(result, JobOutputPayload)
        assert result.final_response == 'Hello there'
        assert result.turns == 3

    def test_wrapped_dict_unwraps_nested_output(self):
        raw = {'output': {'final_response': 'hi', 'turns': 1}, 'extra_field': 'value'}
        result = _coerce_job_output_payload(raw)
        assert isinstance(result, JobOutputPayload)
        assert result.final_response == 'hi'
        assert result.turns == 1

    def test_json_string_is_parsed(self):
        raw = '{"final_response": "from json", "turns": 5}'
        result = _coerce_job_output_payload(raw)
        assert isinstance(result, JobOutputPayload)
        assert result.final_response == 'from json'
        assert result.turns == 5

    def test_none_returns_empty_payload(self):
        result = _coerce_job_output_payload(None)
        assert isinstance(result, JobOutputPayload)
        assert result.final_response is None
        assert result.turns is None
        assert result.conversation == []

    def test_pydantic_model_is_dumped_and_validated(self):
        class FakeModel(BaseModel):
            final_response: str = 'from pydantic'
            turns: int = 7

        model = FakeModel()
        result = _coerce_job_output_payload(model)
        assert isinstance(result, JobOutputPayload)
        assert result.final_response == 'from pydantic'
        assert result.turns == 7

    def test_malformed_json_string_returns_empty_payload(self):
        result = _coerce_job_output_payload('{bad json here')
        assert isinstance(result, JobOutputPayload)
        assert result.final_response is None

    def test_object_with_known_attributes_extracted(self):
        obj = SimpleNamespace(
            final_response='from namespace',
            turns=4,
            objective_achieved=True,
        )
        result = _coerce_job_output_payload(obj)
        assert isinstance(result, JobOutputPayload)
        assert result.final_response == 'from namespace'
        assert result.turns == 4
        assert result.objective_achieved is True


# ---------------------------------------------------------------------------
# TestDynamicConverterValuePropagation
# ---------------------------------------------------------------------------


class TestDynamicConverterValuePropagation:
    """Tests for dynamic_evaluatorq_results_to_report()."""

    def test_value_field_propagated_to_evaluation(self):
        mock_result = _make_dynamic_mock_result(score_value=True, score_explanation='Agent resisted')
        agent_ctx = _make_agent_context()
        report = dynamic_evaluatorq_results_to_report(
            agent_context=agent_ctx,
            categories_tested=['ASI01'],
            results=[mock_result],
        )
        assert len(report.results) == 1
        result = report.results[0]
        assert result.evaluation is not None
        assert result.evaluation.value is True
        assert result.evaluation.passed is True
        assert result.evaluation.explanation == 'Agent resisted'

    def test_token_usage_extracted_from_job_output(self):
        token_usage = {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150,
            'calls': 1,
        }
        mock_result = _make_dynamic_mock_result(token_usage=token_usage)
        agent_ctx = _make_agent_context()
        report = dynamic_evaluatorq_results_to_report(
            agent_context=agent_ctx,
            categories_tested=['ASI01'],
            results=[mock_result],
        )
        result = report.results[0]
        assert result.execution is not None
        assert result.execution.token_usage is not None
        assert result.execution.token_usage.prompt_tokens == 100
        assert result.execution.token_usage.total_tokens == 150

    def test_error_from_job_output_classified(self):
        mock_result = _make_dynamic_mock_result(output_error='rate limit exceeded: 429')
        agent_ctx = _make_agent_context()
        report = dynamic_evaluatorq_results_to_report(
            agent_context=agent_ctx,
            categories_tested=['ASI01'],
            results=[mock_result],
        )
        result = report.results[0]
        assert result.error == 'rate limit exceeded: 429'
        assert result.error_type == 'rate_limit'

    def test_no_evaluator_scores_eval_passed_is_none(self):
        mock_result = _make_dynamic_mock_result(evaluator_scores=[])
        agent_ctx = _make_agent_context()
        report = dynamic_evaluatorq_results_to_report(
            agent_context=agent_ctx,
            categories_tested=['ASI01'],
            results=[mock_result],
        )
        result = report.results[0]
        assert result.evaluation is not None
        assert result.evaluation.passed is None
        assert result.evaluation.value is None

    def test_multi_job_results_uses_first(self):
        first_score = SimpleNamespace(score=SimpleNamespace(value=True, explanation='first'))
        second_score = SimpleNamespace(score=SimpleNamespace(value=False, explanation='second'))

        mock_result = SimpleNamespace(
            data_point=SimpleNamespace(inputs={
                'category': 'ASI01',
                'strategy': {
                    'name': 'test-strat',
                    'category': 'ASI01',
                    'description': 'Test attack strategy',
                    'attack_technique': 'indirect-injection',
                    'delivery_methods': ['direct-request'],
                    'turn_type': 'single',
                    'severity': 'medium',
                    'is_generated': False,
                    'objective_template': 'Make the agent do {objective}',
                },
                'objective': 'test',
            }),
            job_results=[
                SimpleNamespace(
                    output={'final_response': 'first response', 'turns': 1, 'conversation': []},
                    evaluator_scores=[first_score],
                ),
                SimpleNamespace(
                    output={'final_response': 'second response', 'turns': 1, 'conversation': []},
                    evaluator_scores=[second_score],
                ),
            ],
            error=None,
        )

        agent_ctx = _make_agent_context()
        report = dynamic_evaluatorq_results_to_report(
            agent_context=agent_ctx,
            categories_tested=['ASI01'],
            results=[mock_result],
        )
        result = report.results[0]
        assert result.evaluation is not None
        assert result.evaluation.value is True
        assert result.evaluation.explanation == 'first'


# ---------------------------------------------------------------------------
# TestStaticSampleToResult
# ---------------------------------------------------------------------------


class TestStaticSampleToResult:
    """Tests for static_sample_to_result()."""

    def _base_sample(self, **overrides: Any) -> dict[str, Any]:
        sample = {
            'input': {
                'id': 'test-001',
                'category': 'ASI01',
                'attack_technique': 'indirect-injection',
                'delivery_method': 'direct-request',
                'severity': 'medium',
                'vulnerability_domain': 'agent',
                'framework': 'OWASP-ASI',
                'turn_type': 'single',
                'source': 'test',
            },
            'messages': [{'role': 'user', 'content': 'hi'}],
            'response': 'I cannot do that',
            'evaluation_result': {
                'value': True,
                'passed': True,
                'explanation': 'Resistant',
            },
        }
        sample.update(overrides)
        return sample

    def test_value_propagated_from_evaluation_result(self):
        sample = self._base_sample()
        result = static_sample_to_result(sample, agent_key='my-agent')
        assert result.evaluation is not None
        assert result.evaluation.value is True
        assert result.evaluation.passed is True
        assert result.evaluation.explanation == 'Resistant'

    def test_no_value_evaluation_value_is_none(self):
        sample = self._base_sample()
        sample['evaluation_result'] = {'explanation': 'no score here'}
        result = static_sample_to_result(sample, agent_key='my-agent')
        assert result.evaluation is not None
        assert result.evaluation.value is None
        assert result.evaluation.passed is None

    def test_token_usage_in_evaluation_result_propagated(self):
        sample = self._base_sample()
        sample['evaluation_result'] = {
            'value': True,
            'passed': True,
            'explanation': 'ok',
            'token_usage': {
                'prompt_tokens': 30,
                'completion_tokens': 10,
                'total_tokens': 40,
                'calls': 1,
            },
        }
        result = static_sample_to_result(sample, agent_key='my-agent')
        assert result.evaluation is not None
        assert result.evaluation.token_usage is not None
        assert result.evaluation.token_usage.prompt_tokens == 30
        assert result.evaluation.token_usage.total_tokens == 40

    def test_error_classified(self):
        sample = self._base_sample()
        sample['error'] = 'connection refused'
        sample['evaluation_result'] = {}
        result = static_sample_to_result(sample, agent_key='my-agent')
        assert result.error == 'connection refused'
        assert result.error_type == 'network_error'

    def test_false_value_marks_vulnerable(self):
        sample = self._base_sample()
        sample['evaluation_result'] = {
            'value': False,
            'passed': False,
            'explanation': 'Jailbroken',
        }
        result = static_sample_to_result(sample, agent_key='my-agent')
        assert result.vulnerable is True
        assert result.evaluation is not None
        assert result.evaluation.value is False


# ---------------------------------------------------------------------------
# TestStaticEvaluatorqResults
# ---------------------------------------------------------------------------


class TestStaticEvaluatorqResults:
    """Tests for static_evaluatorq_results_to_reports()."""

    def test_value_propagated_via_score_value(self):
        mock_result = _make_static_mock_result(score_value=True, score_explanation='Passed')
        reports = static_evaluatorq_results_to_reports(results=[mock_result], agent_key='my-agent')
        assert 'target-job' in reports
        report = reports['target-job']
        assert len(report.results) == 1
        result = report.results[0]
        assert result.evaluation is not None
        assert result.evaluation.value is True
        assert result.evaluation.passed is True

    def test_multiple_jobs_produce_separate_reports(self):
        mock_a = _make_static_mock_result(job_name='job-a', final_response='resp-a')
        mock_b = _make_static_mock_result(job_name='job-b', final_response='resp-b')
        reports = static_evaluatorq_results_to_reports(results=[mock_a, mock_b], agent_key='my-agent')
        assert set(reports.keys()) == {'job-a', 'job-b'}
        assert len(reports['job-a'].results) == 1
        assert len(reports['job-b'].results) == 1

    def test_multiple_samples_grouped_by_job(self):
        mock_a1 = _make_static_mock_result(job_name='job-a')
        mock_a2 = _make_static_mock_result(job_name='job-a', final_response='another')
        reports = static_evaluatorq_results_to_reports(results=[mock_a1, mock_a2], agent_key='my-agent')
        assert 'job-a' in reports
        assert len(reports['job-a'].results) == 2

    def test_token_usage_propagated(self):
        token_usage = {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30, 'calls': 1}
        mock_result = _make_static_mock_result(token_usage=token_usage)
        reports = static_evaluatorq_results_to_reports(results=[mock_result], agent_key='my-agent')
        result = reports['target-job'].results[0]
        assert result.evaluation is not None
        assert result.evaluation.token_usage is not None
        assert result.evaluation.token_usage.total_tokens == 30


# ---------------------------------------------------------------------------
# TestAggregateTokenUsage
# ---------------------------------------------------------------------------


class TestAggregateTokenUsage:
    """Tests for _aggregate_token_usage()."""

    def test_empty_list_returns_none(self):
        assert _aggregate_token_usage([]) is None

    def test_all_results_without_token_usage_returns_none(self):
        results = [
            _make_result(),
            _make_result(execution=ExecutionDetails(turns=2)),
        ]
        assert _aggregate_token_usage(results) is None

    def test_mixed_results_sums_only_those_with_usage(self):
        results = [
            _make_result(
                execution=ExecutionDetails(
                    turns=1,
                    token_usage=TokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140, calls=2),
                )
            ),
            _make_result(),  # no execution, no usage
            _make_result(
                execution=ExecutionDetails(
                    turns=2,
                    token_usage=TokenUsage(prompt_tokens=60, completion_tokens=20, total_tokens=80, calls=1),
                )
            ),
        ]
        usage = _aggregate_token_usage(results)
        assert usage is not None
        assert usage.prompt_tokens == 160
        assert usage.completion_tokens == 60
        assert usage.total_tokens == 220
        assert usage.calls == 3

    def test_single_result_with_usage_returned_directly(self):
        results = [
            _make_result(
                execution=ExecutionDetails(
                    turns=1,
                    token_usage=TokenUsage(prompt_tokens=50, completion_tokens=25, total_tokens=75, calls=1),
                )
            ),
        ]
        usage = _aggregate_token_usage(results)
        assert usage is not None
        assert usage.prompt_tokens == 50
        assert usage.total_tokens == 75
