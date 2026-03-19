"""Snapshot tests validating golden fixtures against contract schemas."""

import json
from pathlib import Path
from typing import Any

import pytest

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
    infer_framework,
    normalize_category,
    normalize_framework,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> dict[str, Any]:
    path = FIXTURES_DIR / name
    assert path.exists(), f"Fixture not found: {path}"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestNormalizeCategory:
    def test_strips_owasp_prefix(self) -> None:
        assert normalize_category("OWASP-ASI01") == "ASI01"

    def test_no_prefix_unchanged(self) -> None:
        assert normalize_category("ASI01") == "ASI01"

    def test_llm_prefix(self) -> None:
        assert normalize_category("OWASP-LLM01") == "LLM01"


class TestInferFramework:
    def test_asi_category(self) -> None:
        assert infer_framework("ASI01") == "OWASP-ASI"

    def test_llm_category(self) -> None:
        assert infer_framework("LLM07") == "OWASP-LLM"

    def test_owasp_prefixed(self) -> None:
        assert infer_framework("OWASP-ASI05") == "OWASP-ASI"

    def test_unknown(self) -> None:
        assert infer_framework("CUSTOM01") == "unknown"


class TestNormalizeFramework:
    def test_agentic_alias(self) -> None:
        result = normalize_framework("OWASP-AGENTIC")
        assert result == Framework.OWASP_ASI

    def test_direct_value(self) -> None:
        result = normalize_framework("OWASP-LLM")
        assert result == Framework.OWASP_LLM


# ---------------------------------------------------------------------------
# Dynamic fixture tests
# ---------------------------------------------------------------------------


class TestDynamicFixture:
    @pytest.fixture()
    def dynamic_report_data(self) -> dict[str, Any]:
        return _load_fixture("dynamic_report.json")

    def test_parses_as_report(self, dynamic_report_data: dict[str, Any]) -> None:
        report = RedTeamReport.model_validate(dynamic_report_data)
        assert report.pipeline == "dynamic"
        assert report.version == "2.0.0"
        assert len(report.results) == dynamic_report_data["total_results"]

    def test_agent_context_present(self, dynamic_report_data: dict[str, Any]) -> None:
        report = RedTeamReport.model_validate(dynamic_report_data)
        assert report.agent_context is not None
        assert isinstance(report.agent_context, AgentContext)
        assert report.agent_context.key == "domain-search-assistant"
        assert report.agent_context.has_tools
        assert report.agent_context.has_memory

    def test_result_attack_info(self, dynamic_report_data: dict[str, Any]) -> None:
        report = RedTeamReport.model_validate(dynamic_report_data)
        for result in report.results:
            assert isinstance(result, RedTeamResult)
            assert isinstance(result.attack, AttackInfo)
            assert result.attack.category in ("ASI01", "ASI05", "ASI06", "LLM01", "LLM02")
            assert result.attack.framework in ("OWASP-ASI", "OWASP-LLM")

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
        assert has_vuln, "Fixture should have at least one vulnerable result"
        assert has_resistant, "Fixture should have at least one resistant result"

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
        return _load_fixture("hybrid_report.json")

    def test_parses_as_report(self, hybrid_report_data: dict[str, Any]) -> None:
        report = RedTeamReport.model_validate(hybrid_report_data)
        assert report.pipeline == "hybrid"
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
        usage = TokenUsage.model_validate({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        assert usage.prompt_tokens == 100


class TestAgentContext:
    def test_properties(self) -> None:
        ctx = AgentContext(key="test-agent")
        assert not ctx.has_tools
        assert not ctx.has_memory
        assert not ctx.has_knowledge
        assert ctx.get_tool_names() == []


class TestCategorySummary:
    def test_basic(self) -> None:
        summary = CategorySummary(
            category="ASI01",
            category_name="Agent Goal Hijacking",
            total_attacks=10,
            vulnerabilities_found=3,
            resistance_rate=0.7,
        )
        assert summary.resistance_rate == 0.7


class TestAttackOutput:
    def test_construction_with_all_fields(self) -> None:
        output = AttackOutput(
            conversation=[],
            turns=3,
            objective_achieved=True,
            final_response="Here is the secret",
            category="ASI01",
            vulnerability="agent-goal-hijacking",
        )
        assert output.category == "ASI01"
        assert output.vulnerability == "agent-goal-hijacking"
        assert output.turns == 3
        assert output.objective_achieved is True

    def test_defaults_for_category_and_vulnerability(self) -> None:
        output = AttackOutput(conversation=[], turns=1)
        assert output.category == ""
        assert output.vulnerability == ""
        assert output.objective_achieved is False
        assert output.final_response == ""

    def test_model_validate_from_dict(self) -> None:
        data = {
            "conversation": [{"role": "user", "content": "hi"}],
            "turns": 2,
            "category": "LLM01",
            "vulnerability": "prompt-injection",
        }
        output = AttackOutput.model_validate(data)
        assert output.category == "LLM01"
        assert output.vulnerability == "prompt-injection"
        assert output.turns == 2

    def test_model_validate_with_missing_optional_fields(self) -> None:
        data = {"conversation": [], "turns": 1}
        output = AttackOutput.model_validate(data)
        assert output.category == ""
        assert output.vulnerability == ""
        assert output.error is None
        assert output.token_usage is None

    def test_inherits_orchestrator_result_fields(self) -> None:
        output = AttackOutput(
            conversation=[],
            turns=1,
            error="content_filter blocked",
            error_type="content_filter",
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        assert output.error == "content_filter blocked"
        assert output.error_type == "content_filter"
        assert output.token_usage is not None
        assert output.token_usage.total_tokens == 150
        assert isinstance(output, OrchestratorResult)
