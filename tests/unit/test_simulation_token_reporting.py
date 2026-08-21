"""Regression coverage for simulation token reporting and OpenResponses conversion."""

from __future__ import annotations

from unittest.mock import MagicMock

from evaluatorq.common.tracing import record_llm_response
from evaluatorq.contracts import Message, ReportSection, TokenUsage
from evaluatorq.simulation.convert import to_open_responses
from evaluatorq.simulation.reports import export_html, export_markdown
from evaluatorq.simulation.reports.sections import build_report_sections
from evaluatorq.simulation.types import SimulationResult, TerminatedBy, TurnMetrics
from evaluatorq.simulation.ui.token_display import token_metric_specs, token_overview_caption


def _result(token_usage: TokenUsage) -> SimulationResult:
    return SimulationResult(
        messages=[Message(role='user', content='hi'), Message(role='assistant', content='hello')],
        terminated_by=TerminatedBy.judge,
        reason='done',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=token_usage,
        turn_metrics=[TurnMetrics(turn_number=1, token_usage=TokenUsage(), judge_reason='ok')],
        metadata={'persona': 'Tester', 'scenario': 'Smoke'},
    )


def test_openresponses_conversion_preserves_cached_and_reasoning_tokens() -> None:
    response = to_open_responses(
        _result(
            TokenUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cached_tokens=3,
                reasoning_tokens=2,
            )
        )
    )

    assert response['usage']['input_tokens_details'] == {'cached_tokens': 3}
    assert response['usage']['output_tokens_details'] == {'reasoning_tokens': 2}


def test_openresponses_trace_preserves_output_reasoning_tokens() -> None:
    class _OutputDetails:
        reasoning_tokens = 2

    class _Usage:
        input_tokens = 10
        output_tokens = 5
        total_tokens = 15
        input_tokens_details = None
        output_tokens_details = _OutputDetails()

    class _Response:
        def __init__(self) -> None:
            self.usage = _Usage()
            self.choices: list[object] = []

    span = MagicMock()
    record_llm_response(span, _Response())
    attrs = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}

    assert attrs['gen_ai.usage.reasoning.output_tokens'] == 2


def test_token_usage_trace_accepts_canonical_detail_counts() -> None:
    from evaluatorq.common.tracing import record_token_usage

    span = MagicMock()
    record_token_usage(span, cached_tokens=3, reasoning_tokens=2)
    attrs = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}

    assert attrs['gen_ai.usage.cache_read.input_tokens'] == 3
    assert attrs['gen_ai.usage.reasoning.output_tokens'] == 2


def test_token_section_aggregates_canonical_and_optional_counts() -> None:
    sections = build_report_sections([
        _result(
            TokenUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cached_tokens=3,
                reasoning_tokens=2,
            )
        ),
        _result(
            TokenUsage(
                input_tokens=20,
                output_tokens=10,
                total_tokens=30,
                cached_tokens=4,
                reasoning_tokens=1,
            )
        ),
    ])
    tokens = next(section for section in sections if section.kind == 'token_usage')

    assert tokens.data['input_tokens'] == 30
    assert tokens.data['output_tokens'] == 15
    assert tokens.data['cached_tokens'] == 7
    assert tokens.data['reasoning_tokens'] == 3
    assert tokens.data['avg_input_per_conversation'] == 15
    assert tokens.data['avg_output_per_conversation'] == 7.5


def test_token_usage_rows_share_canonical_labels_and_legacy_fallbacks() -> None:
    from evaluatorq.simulation.reports.token_usage import build_token_usage_rows

    rows = build_token_usage_rows({
        'input_tokens': 10,
        'output_tokens': 5,
        'total_tokens': 15,
        'cached_tokens': 3,
        'reasoning_tokens': 2,
        'avg_total_per_conversation': 15,
        'avg_input_per_conversation': 10,
        'avg_output_per_conversation': 5,
    })

    assert rows == [
        ['Input Tokens (total)', '10'],
        ['Output Tokens (total)', '5'],
        ['Total Tokens', '15'],
        ['Avg Total / Conversation', '15'],
        ['Avg Input / Conversation', '10'],
        ['Avg Output / Conversation', '5'],
        ['Cached Tokens (retrieved)', '3 (30% of input)'],
        ['Reasoning Tokens', '2'],
    ]
    assert build_token_usage_rows({
        'prompt_tokens': 10,
        'completion_tokens': 5,
        'total_tokens': 15,
        'avg_total_per_conversation': 15,
        'avg_prompt_per_conversation': 10,
        'avg_completion_per_conversation': 5,
    })[0:2] == [
        ['Input Tokens (total)', '10'],
        ['Output Tokens (total)', '5'],
    ]


def test_token_usage_export_sections_render_the_same_rows() -> None:
    from evaluatorq.simulation.reports.export_html import _render_token_usage_html
    from evaluatorq.simulation.reports.export_md import _render_token_usage_section

    section = ReportSection(
        kind='token_usage',
        title='Token Usage',
        data={
            'input_tokens': 10,
            'output_tokens': 5,
            'total_tokens': 15,
            'cached_tokens': 3,
            'reasoning_tokens': 2,
            'avg_total_per_conversation': 15,
            'avg_input_per_conversation': 10,
            'avg_output_per_conversation': 5,
        },
    )
    markdown = _render_token_usage_section(section)
    html = _render_token_usage_html(section)

    for label, value in [
        ('Input Tokens (total)', '10'),
        ('Output Tokens (total)', '5'),
        ('Total Tokens', '15'),
        ('Avg Total / Conversation', '15'),
        ('Avg Input / Conversation', '10'),
        ('Avg Output / Conversation', '5'),
        ('Cached Tokens (retrieved)', '3 (30% of input)'),
        ('Reasoning Tokens', '2'),
    ]:
        assert f'| {label} | {value} |' in markdown
        assert f'<td data-label="Metric">{label}</td><td data-label="Value">{value}</td>' in html


def test_token_usage_exports_use_canonical_names_and_optional_details() -> None:
    result = _result(
        TokenUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cached_tokens=3,
            reasoning_tokens=2,
        )
    )

    for rendered in (export_markdown([result], target='t'), export_html([result], target='t')):
        assert 'Input Tokens (total)' in rendered
        assert 'Output Tokens (total)' in rendered
        assert 'Cached Tokens (retrieved)' in rendered
        assert 'Reasoning Tokens' in rendered
        assert 'Prompt Tokens (total)' not in rendered
        assert 'Completion Tokens (total)' not in rendered


def test_token_display_uses_canonical_names_with_optional_details() -> None:
    data = {
        'input_tokens': 10,
        'output_tokens': 5,
        'total_tokens': 15,
        'cached_tokens': 3,
        'reasoning_tokens': 2,
        'avg_total_per_conversation': 15,
    }

    assert token_metric_specs(data) == [
        ('Input', '10'),
        ('Output', '5'),
        ('Total', '15'),
        ('Cached (retrieved)', '3'),
        ('Reasoning', '2'),
    ]
    assert token_overview_caption(data) == 'Input 10 · Output 5 · Cached (retrieved) 3 · Reasoning 2 · Avg 15/conv'


def test_token_display_reads_legacy_saved_run_keys() -> None:
    assert token_metric_specs({'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}) == [
        ('Input', '10'),
        ('Output', '5'),
        ('Total', '15'),
    ]
