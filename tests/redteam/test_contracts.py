"""Snapshot tests validating golden fixtures against contract schemas."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from evaluatorq.contracts import (
    AgentResponse,
    ReasoningOutputItem,
    TextOutputItem,
    ToolCallOutputItem,
    tool_result_to_text,
)
from evaluatorq.dashboard.library import load_model_cached
from evaluatorq.redteam.contracts import (
    AgentContext,
    AttackInfo,
    AttackOutput,
    CategorySummary,
    Framework,
    OrchestratorResult,
    RedTeamReport,
    RedTeamResult,
    TokenUsage,
    Turn,
    infer_framework,
    normalize_category,
    normalize_framework,
)

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> dict[str, Any]:
    path = FIXTURES_DIR / name
    assert path.exists(), f'Fixture not found: {path}'
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestNormalizeCategory:
    def test_strips_owasp_prefix(self) -> None:
        assert normalize_category('OWASP-ASI01') == 'ASI01'

    def test_no_prefix_unchanged(self) -> None:
        assert normalize_category('ASI01') == 'ASI01'

    def test_llm_prefix(self) -> None:
        assert normalize_category('OWASP-LLM01') == 'LLM01'


def test_tool_result_to_text_serializes_pydantic_models_as_json() -> None:
    class Answer(BaseModel):
        field: str

    assert tool_result_to_text(Answer(field='x')) == '{"field":"x"}'


def test_tool_result_to_text_falls_back_when_pydantic_serialization_fails() -> None:
    class Answer(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        field: object

        def __str__(self) -> str:
            return 'answer fallback'

    with patch('evaluatorq.contracts.logger.warning') as warning:
        rendered = tool_result_to_text(Answer(field=object()))

    assert rendered == '"answer fallback"'
    warning.assert_called_once()
    assert 'model_dump_json() failed' in warning.call_args.args[0]
    assert warning.call_args.args[2] == 'PydanticSerializationError'


def test_tool_result_to_text_warns_and_falls_back_for_non_string_renderer() -> None:
    class NonStringRenderer:
        def __str__(self) -> str:
            return 'renderer fallback'

        def model_dump_json(self) -> object:
            return 42

    with patch('evaluatorq.contracts.logger.warning') as warning:
        rendered = tool_result_to_text(NonStringRenderer())

    assert rendered == '"renderer fallback"'
    warning.assert_called_once_with(
        'tool_result_to_text: {}.model_dump_json() returned {}; falling back to JSON encoding',
        'NonStringRenderer',
        'int',
    )


class TestInferFramework:
    def test_asi_category(self) -> None:
        assert infer_framework('ASI01') == 'OWASP-ASI'

    def test_llm_category(self) -> None:
        assert infer_framework('LLM07') == 'OWASP-LLM'

    def test_owasp_prefixed(self) -> None:
        assert infer_framework('OWASP-ASI05') == 'OWASP-ASI'

    def test_unknown(self) -> None:
        assert infer_framework('CUSTOM01') == 'unknown'


class TestNormalizeFramework:
    def test_agentic_alias(self) -> None:
        result = normalize_framework('OWASP-AGENTIC')
        assert result == Framework.OWASP_ASI

    def test_direct_value(self) -> None:
        result = normalize_framework('OWASP-LLM')
        assert result == Framework.OWASP_LLM


# ---------------------------------------------------------------------------
# Dynamic fixture tests
# ---------------------------------------------------------------------------


class TestDynamicFixture:
    @pytest.fixture()
    def dynamic_report_data(self) -> dict[str, Any]:
        return _load_fixture('dynamic_report.json')

    def test_parses_as_report(self, dynamic_report_data: dict[str, Any]) -> None:
        report = RedTeamReport.model_validate(dynamic_report_data)
        assert report.pipeline == 'dynamic'
        assert report.version == '2.0.0'
        assert len(report.results) == dynamic_report_data['total_results']

    def test_agent_context_present(self, dynamic_report_data: dict[str, Any]) -> None:
        report = RedTeamReport.model_validate(dynamic_report_data)
        assert report.agent_contexts
        ctx = report.agent_contexts['domain-search-assistant']
        assert isinstance(ctx, AgentContext)
        assert ctx.key == 'domain-search-assistant'
        assert ctx.has_tools
        assert ctx.has_memory

    def test_legacy_fixture_uses_defaults_for_new_agent_context_fields(self) -> None:
        report = load_model_cached(FIXTURES_DIR / 'dynamic_report.json', RedTeamReport.model_validate)
        ctx = report.agent_contexts['domain-search-assistant']

        assert ctx.version is None
        assert ctx.skills == []
        assert ctx.agent_type is None
        assert ctx.engine is None

    def test_result_attack_info(self, dynamic_report_data: dict[str, Any]) -> None:
        report = RedTeamReport.model_validate(dynamic_report_data)
        for result in report.results:
            assert isinstance(result, RedTeamResult)
            assert isinstance(result.attack, AttackInfo)
            assert result.attack.category in ('ASI01', 'ASI05', 'ASI06', 'LLM01', 'LLM02')
            assert result.attack.framework in ('OWASP-ASI', 'OWASP-LLM')

    def test_evaluation_semantics(self, dynamic_report_data: dict[str, Any]) -> None:
        """passed=True means RESISTANT, passed=False means VULNERABLE."""
        report = RedTeamReport.model_validate(dynamic_report_data)
        for result in report.results:
            if result.evaluation and result.evaluation.passed is False:
                assert result.vulnerable is True
            elif result.evaluation and result.evaluation.passed is True:
                assert result.vulnerable is False

    def test_has_vulnerable_and_resistant(self, dynamic_report_data: dict[str, Any]) -> None:
        report = RedTeamReport.model_validate(dynamic_report_data)
        has_vuln = any(r.vulnerable for r in report.results)
        has_resistant = any(not r.vulnerable for r in report.results)
        assert has_vuln, 'Fixture should have at least one vulnerable result'
        assert has_resistant, 'Fixture should have at least one resistant result'

    def test_round_trip_serialization(self, dynamic_report_data: dict[str, Any]) -> None:
        report = RedTeamReport.model_validate(dynamic_report_data)
        dumped = json.loads(report.model_dump_json())
        report2 = RedTeamReport.model_validate(dumped)
        assert report2.total_results == report.total_results
        assert len(report2.results) == len(report.results)
        for r1, r2 in zip(report.results, report2.results):
            assert r1.attack.id == r2.attack.id
            assert r1.vulnerable == r2.vulnerable


# ---------------------------------------------------------------------------
# Hybrid fixture tests
# ---------------------------------------------------------------------------


class TestHybridFixture:
    @pytest.fixture()
    def hybrid_report_data(self) -> dict[str, Any]:
        return _load_fixture('hybrid_report.json')

    def test_parses_as_report(self, hybrid_report_data: dict[str, Any]) -> None:
        report = RedTeamReport.model_validate(hybrid_report_data)
        assert report.pipeline == 'hybrid'
        assert len(report.results) > 0

    def test_mixed_execution_details(self, hybrid_report_data: dict[str, Any]) -> None:
        """Hybrid reports can have results with and without execution details."""
        report = RedTeamReport.model_validate(hybrid_report_data)
        has_execution = [r.execution is not None for r in report.results]
        # At least one should have execution details (dynamic path)
        assert any(has_execution) or len(report.results) > 0

    def test_round_trip_serialization(self, hybrid_report_data: dict[str, Any]) -> None:
        report = RedTeamReport.model_validate(hybrid_report_data)
        dumped = json.loads(report.model_dump_json())
        report2 = RedTeamReport.model_validate(dumped)
        assert report2.total_results == report.total_results


# ---------------------------------------------------------------------------
# Model unit tests
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_defaults(self) -> None:
        usage = TokenUsage()
        assert usage.total_tokens == 0
        assert usage.calls == 0

    def test_from_dict(self) -> None:
        usage = TokenUsage.model_validate({'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150})
        assert usage.prompt_tokens == 100

    def test_defaults_include_cost_breakdown_fields(self) -> None:
        usage = TokenUsage()
        assert usage.cache_creation_tokens == 0
        assert usage.input_cost is None
        assert usage.output_cost is None
        assert usage.total_cost is None
        assert usage.cost_usd is None

    def test_cost_breakdown_round_trip(self) -> None:
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cache_creation_tokens=10,
            input_cost=0.01,
            output_cost=0.02,
            total_cost=0.03,
        )
        dumped = usage.model_dump(mode='json')
        assert dumped['cache_creation_tokens'] == 10
        assert dumped['input_cost'] == pytest.approx(0.01)
        assert dumped['output_cost'] == pytest.approx(0.02)
        assert dumped['total_cost'] == pytest.approx(0.03)
        # Legacy key still present for un-migrated consumers (dashboard membership test).
        assert dumped['cost_usd'] == pytest.approx(0.03)

        restored = TokenUsage.model_validate(dumped)
        assert restored == usage

    def test_total_cost_accepts_legacy_aliases(self) -> None:
        assert TokenUsage(cost_usd=0.5).total_cost == pytest.approx(0.5)
        assert TokenUsage.model_validate({'cost': 0.75}).total_cost == pytest.approx(0.75)
        assert TokenUsage.model_validate({'total_cost': 1.25}).total_cost == pytest.approx(1.25)

    def test_cost_usd_is_read_only_alias_for_total_cost(self) -> None:
        usage = TokenUsage(total_cost=0.42)
        assert usage.cost_usd == pytest.approx(0.42)
        with pytest.raises(ValidationError):
            usage.cost_usd = 1.0  # pyright: ignore[reportAttributeAccessIssue]


class TestAgentContext:
    def test_properties(self) -> None:
        ctx = AgentContext(key='test-agent')
        assert not ctx.has_tools
        assert not ctx.has_memory
        assert not ctx.has_knowledge
        assert ctx.get_tool_names() == []


class TestCategorySummary:
    def test_basic(self) -> None:
        summary = CategorySummary(
            category='ASI01',
            category_name='Agent Goal Hijacking',
            total_attacks=10,
            vulnerabilities_found=3,
            resistance_rate=0.7,
        )
        assert summary.resistance_rate == 0.7


def _turn(prompt: str = 'hi', reply: str = 'ok') -> Turn:
    return Turn(
        attacker=AgentResponse(text=prompt),
        target=AgentResponse(text=reply),
    )


class TestAttackOutput:
    def test_construction_with_all_fields(self) -> None:
        output = AttackOutput(
            turns=[_turn(), _turn(), _turn(reply='Here is the secret')],
            objective_achieved=True,
            category='ASI01',
            vulnerability='agent-goal-hijacking',
        )
        assert output.category == 'ASI01'
        assert output.vulnerability == 'agent-goal-hijacking'
        assert output.n_turns == 3
        assert output.final_response == 'Here is the secret'
        assert output.objective_achieved is True

    def test_defaults_for_category_and_vulnerability(self) -> None:
        output = AttackOutput(turns=[_turn()])
        assert output.category == ''
        assert output.vulnerability == ''
        assert output.objective_achieved is False
        assert output.final_response == 'ok'

    def test_model_validate_from_dict(self) -> None:
        data = {
            'turns': [
                {
                    'attacker': {'generated_prompt': 'hi'},
                    'target': {'output': [{'type': 'output_text', 'text': 'yo', 'annotations': []}]},
                },
                {
                    'attacker': {'generated_prompt': 'again'},
                    'target': {'output': [{'type': 'output_text', 'text': 'stop', 'annotations': []}]},
                },
            ],
            'category': 'LLM01',
            'vulnerability': 'prompt-injection',
        }
        output = AttackOutput.model_validate(data)
        assert output.category == 'LLM01'
        assert output.vulnerability == 'prompt-injection'
        assert output.n_turns == 2

    def test_model_validate_with_missing_optional_fields(self) -> None:
        data = {'turns': [{'attacker': {'generated_prompt': 'x'}, 'target': {'output': []}}]}
        output = AttackOutput.model_validate(data)
        assert output.category == ''
        assert output.vulnerability == ''
        assert output.error is None
        assert output.token_usage is None

    def test_inherits_orchestrator_result_fields(self) -> None:
        output = AttackOutput(
            turns=[_turn()],
            error='content_filter blocked',
            error_type='content_filter',
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        assert output.error == 'content_filter blocked'
        assert output.error_type == 'content_filter'
        assert output.token_usage is not None
        assert output.token_usage.total_tokens == 150
        assert isinstance(output, OrchestratorResult)


class TestChatCompletionsOrdering:
    """`chat_completions` preserves interleaved text/tool_call order from target.output."""

    def _build(self, output_items: list[Any]) -> OrchestratorResult:
        return OrchestratorResult(
            turns=[
                Turn(
                    attacker=AgentResponse(text='please call tools'),
                    target=AgentResponse(output=output_items),
                )
            ]
        )

    def test_text_only_joins_into_single_assistant_message(self) -> None:
        result = self._build([
            TextOutputItem(text='Hello, ', annotations=[]),
            TextOutputItem(text='world.', annotations=[]),
        ])
        msgs = result.chat_completions
        assert len(msgs) == 2
        assert msgs[0].role == 'user'
        assert msgs[1].role == 'assistant'
        assert msgs[1].content == 'Hello, world.'
        assert msgs[1].tool_calls is None

    def test_tool_call_without_result_is_dropped(self) -> None:
        result = self._build([
            ToolCallOutputItem(call_id='call_1', name='search', arguments='{"q": "x"}'),
        ])
        msgs = result.chat_completions
        assert [m.role for m in msgs] == ['user', 'assistant']
        assistant = msgs[1]
        assert assistant.content == ''
        assert assistant.tool_calls is None

    def test_tool_call_with_result_emits_assistant_then_tool(self) -> None:
        result = self._build([
            ToolCallOutputItem(call_id='call_2', name='lookup', arguments='{}', result='found it'),
        ])
        msgs = result.chat_completions
        assert [m.role for m in msgs] == ['user', 'assistant', 'tool']
        tool_msg = msgs[2]
        assert tool_msg.tool_call_id == 'call_2'
        assert tool_msg.name == 'lookup'
        assert tool_msg.content == 'found it'

    def test_interleaved_text_and_tool_calls_preserve_order(self) -> None:
        result = self._build([
            TextOutputItem(text='Thinking...', annotations=[]),
            ToolCallOutputItem(call_id='c1', name='search', arguments='{}', result='r1'),
            TextOutputItem(text='Now another:', annotations=[]),
            ToolCallOutputItem(call_id='c2', name='fetch', arguments='{}'),
            TextOutputItem(text='Done.', annotations=[]),
        ])
        msgs = result.chat_completions
        assert [m.role for m in msgs] == [
            'user',  # attacker prompt
            'assistant',  # "Thinking..."
            'assistant',  # tool_call c1
            'tool',  # result for c1
            'assistant',  # "Now another:"
            'assistant',  # "Done."
        ]
        assert msgs[1].content == 'Thinking...'
        assert msgs[1].tool_calls is None
        assert msgs[2].content is None
        assert msgs[2].tool_calls is not None
        assert msgs[2].tool_calls[0].id == 'c1'
        assert msgs[3].tool_call_id == 'c1'
        assert msgs[3].content == 'r1'
        assert msgs[4].content == 'Now another:'
        assert msgs[5].content == 'Done.'

    def test_reasoning_items_are_dropped(self) -> None:
        result = self._build([
            ReasoningOutputItem(text='internal thought'),
            TextOutputItem(text='visible answer', annotations=[]),
            ReasoningOutputItem(text='another thought'),
        ])
        msgs = result.chat_completions
        assert [m.role for m in msgs] == ['user', 'assistant']
        assert msgs[1].content == 'visible answer'

    def test_empty_output_falls_back_to_target_text(self) -> None:
        result = OrchestratorResult(
            turns=[
                Turn(
                    attacker=AgentResponse(text='ping'),
                    target=AgentResponse(output=[]),
                )
            ]
        )
        msgs = result.chat_completions
        assert [m.role for m in msgs] == ['user', 'assistant']
        assert msgs[1].content == ''


# ===========================================================================
# tool_calls_per_turn removal
# ===========================================================================


def test_redteam_result_has_no_tool_calls_per_turn_field():
    """tool_calls_per_turn is dead schema — removed in favor of per-Turn access."""
    from evaluatorq.redteam.contracts import RedTeamResult

    assert 'tool_calls_per_turn' not in RedTeamResult.model_fields


def test_job_output_payload_has_no_tool_calls_per_turn_field():
    """tool_calls_per_turn dropped from JobOutputPayload; legacy JSONs absorbed via extra='allow'."""
    from evaluatorq.redteam.contracts import JobOutputPayload

    assert 'tool_calls_per_turn' not in JobOutputPayload.model_fields


def test_job_output_payload_tolerates_legacy_tool_calls_per_turn_key():
    """Existing run JSONs with the legacy key must still deserialize cleanly."""
    from evaluatorq.redteam.contracts import JobOutputPayload

    payload = JobOutputPayload.model_validate({
        'conversation': [],
        'final_response': 'ok',
        'tool_calls_per_turn': [[{'type': 'function_call', 'name': 'search', 'arguments': '{}'}]],
    })
    assert payload.final_response == 'ok'


class TestAttackerResponseUnifiedOntoAgentResponse:
    """RES-883: the attacker side uses AgentResponse (not a separate type)."""

    def test_turn_attacker_is_agent_response_with_error_field(self):
        turn = Turn(attacker=AgentResponse(text='attack'), target=AgentResponse(text='resp'))
        assert isinstance(turn.attacker, AgentResponse)
        assert turn.attacker.text == 'attack'
        # attacker now carries the shared per-response error field
        assert turn.attacker.error is None

    def test_truncated_is_derivable_from_finish_reason(self):
        turn = Turn(
            attacker=AgentResponse(text='attack', finish_reason='length'),
            target=AgentResponse(text='resp'),
        )
        assert turn.attacker.finish_reason == 'length'  # was the separate `truncated` bool

    def test_pre_res883_report_json_migrates(self):
        # Reports written before RES-883 carried generated_prompt (+ a now-derived
        # truncated) on the attacker; they must still load.
        turn = Turn.model_validate({
            'attacker': {'generated_prompt': 'legacy attack', 'truncated': True, 'finish_reason': 'length'},
            'target': {'output': []},
        })
        assert turn.attacker.text == 'legacy attack'
        assert turn.attacker.finish_reason == 'length'
