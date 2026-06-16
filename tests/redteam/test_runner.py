"""Tests for the unified red_team() entry point and category helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evaluatorq.contracts import AgentTarget, Message
from evaluatorq.redteam import get_category_info, list_categories, red_team
from evaluatorq.redteam.contracts import AgentResponse, Pipeline, RedTeamReport, ReportSummary
from evaluatorq.redteam.runner import _deduplicate_target_labels, _parse_target
from evaluatorq.redteam.vulnerability_registry import get_primary_category


def _make_report(target: str = "agent:test", **kwargs) -> RedTeamReport:
    """Create a minimal RedTeamReport for use in tests."""
    defaults = dict(
        created_at=datetime.now(tz=timezone.utc),
        description=f"Report for {target}",
        pipeline=Pipeline.DYNAMIC,
        framework=None,
        categories_tested=["ASI01"],
        tested_agents=[target],
        total_results=0,
        results=[],
        summary=ReportSummary(),
    )
    defaults.update(kwargs)
    return RedTeamReport(**defaults)  # pyright: ignore[reportArgumentType]


class TestParseTarget:
    """Tests for target string parsing."""

    def test_agent_target(self):
        kind, value = _parse_target('agent:my-agent-key')
        assert kind == 'agent'
        assert value == 'my-agent-key'

    def test_openai_target(self):
        with pytest.raises(ValueError, match='OpenAIModelTarget'):
            _parse_target('openai:gpt-4o')

    def test_bare_key_defaults_to_agent(self):
        kind, value = _parse_target('my-agent-key')
        assert kind == 'agent'
        assert value == 'my-agent-key'

    def test_case_insensitive_kind(self):
        kind, value = _parse_target('Agent:my-key')
        assert kind == 'agent'

    def test_empty_value_raises(self):
        with pytest.raises(ValueError, match='missing a value'):
            _parse_target('agent:')

    def test_multiple_colons(self):
        # openai:/llm: prefixes are removed; ensure migration error mentions OpenAIModelTarget.
        with pytest.raises(ValueError, match='OpenAIModelTarget'):
            _parse_target('openai:org/gpt-4o:latest')


class _FakeTarget:
    """Minimal agent target stub for label dedup tests."""
    def __init__(self, name: str | None = None):
        self._name = name

    @property
    def name(self) -> str | None:
        return self._name


class TestDeduplicateTargetLabels:
    """Tests for _deduplicate_target_labels()."""

    def test_no_targets(self):
        labels, label_map = _deduplicate_target_labels([], [])
        assert labels == []
        assert label_map == {}

    def test_string_targets_only(self):
        labels, label_map = _deduplicate_target_labels(["agent:a", "agent:b"], [])
        assert labels == ["agent:a", "agent:b"]
        assert label_map == {}

    def test_agent_targets_with_unique_names(self):
        t1 = _FakeTarget("bot-a")
        t2 = _FakeTarget("bot-b")
        labels, label_map = _deduplicate_target_labels([], [t1, t2])
        assert labels == ["bot-a", "bot-b"]
        assert label_map[id(t1)] == "bot-a"
        assert label_map[id(t2)] == "bot-b"

    def test_duplicate_agent_target_names_get_suffixed(self):
        t1 = _FakeTarget("my-bot")
        t2 = _FakeTarget("my-bot")
        t3 = _FakeTarget("my-bot")
        labels, label_map = _deduplicate_target_labels([], [t1, t2, t3])
        assert labels == ["my-bot", "my-bot-1", "my-bot-2"]

    def test_agent_target_collides_with_string_target(self):
        t1 = _FakeTarget("agent:foo")
        labels, label_map = _deduplicate_target_labels(["agent:foo"], [t1])
        assert labels == ["agent:foo", "agent:foo-1"]
        assert label_map[id(t1)] == "agent:foo-1"

    def test_fallback_to_class_name_when_no_name(self):
        t1 = _FakeTarget(None)
        labels, label_map = _deduplicate_target_labels([], [t1])
        assert labels == ["_FakeTarget"]

    def test_mixed_string_and_agent_targets(self):
        t1 = _FakeTarget("bot-a")
        t2 = _FakeTarget("bot-a")
        labels, _ = _deduplicate_target_labels(["agent:x", "bot-a"], [t1, t2])
        assert labels == ["agent:x", "bot-a", "bot-a-1", "bot-a-2"]


class TestListCategories:
    """Tests for list_categories()."""

    def test_returns_list(self):
        cats = list_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0

    def test_no_owasp_prefix(self):
        cats = list_categories()
        for cat in cats:
            assert not cat.startswith('OWASP-'), f'{cat} should not have OWASP- prefix'

    def test_contains_known_categories(self):
        cats = list_categories()
        assert 'ASI01' in cats


class TestGetCategoryInfo:
    """Tests for get_category_info()."""

    def test_returns_dict(self):
        info = get_category_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_info_structure(self):
        info = get_category_info()
        for cat, details in info.items():
            assert 'name' in details
            assert 'strategy_count' in details
            assert 'single_turn_count' in details
            assert 'multi_turn_count' in details
            assert isinstance(details['strategy_count'], int)
            assert details['strategy_count'] >= 0

    def test_asi01_has_strategies(self):
        info = get_category_info()
        assert 'ASI01' in info
        assert info['ASI01']['strategy_count'] > 0


class TestRedTeamValidation:
    """Tests for red_team() argument validation (no actual LLM calls)."""

    @pytest.mark.asyncio
    async def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match='is not a valid Pipeline'):
            await red_team('agent:test', mode='invalid')

    @pytest.mark.asyncio
    async def test_static_mode_dispatches(self):
        """Static mode dispatches to _run_static (no longer raises NotImplementedError)."""
        from unittest.mock import AsyncMock, patch

        mock_report = _make_report()
        with patch(
            'evaluatorq.redteam.runner._run_static',
            new_callable=AsyncMock,
            return_value=mock_report,
        ) as mock_static:
            result = await red_team('agent:test', mode='static')
            assert result is mock_report
            mock_static.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hybrid_mode_dispatches(self):
        """Hybrid mode dispatches to _run_dynamic_or_hybrid."""
        from unittest.mock import AsyncMock, patch

        mock_report = _make_report()
        with patch(
            'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
            new_callable=AsyncMock,
            return_value=mock_report,
        ) as mock_hybrid:
            result = await red_team('agent:test', mode='hybrid')
            assert result is mock_report
            mock_hybrid.assert_awaited_once()


class TestStaticCoverageGuard:
    """Tests for static mode evaluator coverage guard."""

    @pytest.mark.asyncio
    async def test_run_static_filters_uncovered_categories(self):
        """Static mode skips datapoints whose category has no evaluator."""
        from unittest.mock import MagicMock, patch

        from evaluatorq import DataPoint

        # Create datapoints with a mix of covered and uncovered categories
        datapoints = [
            DataPoint(inputs={'id': '1', 'category': 'ASI01', 'messages': []}),
            DataPoint(inputs={'id': '2', 'category': 'UNKNOWN_CAT', 'messages': []}),
            DataPoint(inputs={'id': '3', 'category': 'ASI02', 'messages': []}),
        ]

        def mock_get_evaluator(cat, model_id=None):
            if cat in ('ASI01', 'ASI02'):
                return MagicMock()
            return None

        with (
            patch(
                'evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.load_owasp_agentic_dataset',
                return_value=datapoints,
            ),
            patch(
                'evaluatorq.redteam.frameworks.owasp.evaluators.get_evaluator_for_category',
                side_effect=mock_get_evaluator,
            ),
            patch('evaluatorq.evaluatorq', return_value=[]) as mock_evaluatorq,
            patch('evaluatorq.redteam.runtime.jobs.create_model_job', return_value=MagicMock()),
            patch('evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.create_owasp_evaluator', return_value=MagicMock()),
            patch('evaluatorq.redteam.reports.converters.static_evaluatorq_results_to_reports', return_value={}),
        ):
            from evaluatorq.redteam.runner import _run_static

            await _run_static(
                targets=['agent:test'],
                categories=None,
                evaluator_model='azure/gpt-5-mini',
                parallelism=5,
                max_static_datapoints=None,
                dataset=None,
                description='test',
            )

            # Verify evaluatorq was called with only 2 datapoints (not 3)
            call_args = mock_evaluatorq.call_args
            submitted_data = call_args.kwargs.get('data')
            assert submitted_data is not None
            assert len(submitted_data) == 2

    @pytest.mark.asyncio
    async def test_run_static_raises_when_all_filtered(self):
        """Static mode raises ValueError when all datapoints are filtered."""
        from unittest.mock import patch

        from evaluatorq import DataPoint

        datapoints = [
            DataPoint(inputs={'id': '1', 'category': 'UNKNOWN', 'messages': []}),
        ]

        with (
            patch(
                'evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.load_owasp_agentic_dataset',
                return_value=datapoints,
            ),
            patch(
                'evaluatorq.redteam.frameworks.owasp.evaluators.get_evaluator_for_category',
                return_value=None,
            ),
        ):
            from evaluatorq.redteam.runner import _run_static

            with pytest.raises(ValueError, match='All datapoints were filtered out'):
                await _run_static(
                    targets=['agent:test'],
                    categories=None,
                    evaluator_model='azure/gpt-5-mini',
                    parallelism=5,
                    max_static_datapoints=None,
                    dataset=None,
                    description='test',
                )


class TestConfirmCallback:
    """Tests for hooks-based confirm support (replaces legacy confirm_callback)."""

    @pytest.mark.asyncio
    async def test_hooks_on_confirm_false_aborts(self):
        """Hooks returning False from on_confirm should abort execution."""
        from unittest.mock import patch

        from evaluatorq.redteam.hooks import DefaultHooks

        class FalseConfirmHooks(DefaultHooks):
            # Sync override of the now-async DefaultHooks.on_confirm — intentional:
            # exercises the sync-hook compatibility path (driven via await_maybe).
            def on_confirm(self, payload) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
                return False

        with (
            patch('evaluatorq.redteam.runner._run_dynamic_or_hybrid') as mock_dynamic,
        ):
            # Make _run_dynamic_or_hybrid call hooks.on_confirm and raise if False
            async def _fake_dynamic(**kwargs):
                hooks = kwargs.get('hooks')
                if hooks is not None and not hooks.on_confirm({}):
                    raise RuntimeError('Execution cancelled by confirmation callback')
                return _make_report()

            mock_dynamic.side_effect = _fake_dynamic

            with pytest.raises(RuntimeError, match='cancelled by confirmation callback'):
                await red_team(
                    'agent:test',
                    mode='dynamic',
                    hooks=FalseConfirmHooks(),
                )

    @pytest.mark.asyncio
    async def test_hooks_on_confirm_true_proceeds(self):
        """Hooks returning True from on_confirm should allow execution."""
        from unittest.mock import AsyncMock, patch

        from evaluatorq.redteam.hooks import DefaultHooks

        mock_report = _make_report()

        with patch(
            'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
            new_callable=AsyncMock,
            return_value=mock_report,
        ) as mock_dynamic:
            result = await red_team(
                'agent:test',
                mode='dynamic',
                hooks=DefaultHooks(),
            )
            assert result is mock_report
            # hooks is passed through to _run_dynamic_or_hybrid
            call_kwargs = mock_dynamic.call_args.kwargs
            assert call_kwargs['hooks'] is not None


class TestRedTeamMultiTarget:
    """Tests for red_team() with multiple targets."""

    @pytest.mark.asyncio
    async def test_empty_targets_raises(self):
        with pytest.raises(ValueError, match='at least one target'):
            await red_team([])

    @pytest.mark.asyncio
    async def test_multi_target_dispatches_to_unified_pipeline(self):
        """Multi-target passes all targets to _run_dynamic_or_hybrid."""
        from unittest.mock import AsyncMock, patch

        mock_report = _make_report()
        with patch(
            'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
            new_callable=AsyncMock,
            return_value=mock_report,
        ) as mock_pipeline:
            result = await red_team(['agent:a', 'agent:b'])
            assert result is mock_report
            call_kwargs = mock_pipeline.call_args.kwargs
            assert call_kwargs['targets'] == ['agent:a', 'agent:b']

    @pytest.mark.asyncio
    async def test_single_string_target_works(self):
        """A single string target dispatches directly."""
        from unittest.mock import AsyncMock, patch

        mock_report = _make_report()
        with patch(
            'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
            new_callable=AsyncMock,
            return_value=mock_report,
        ) as mock_pipeline:
            result = await red_team('agent:test')
            assert result is mock_report
            mock_pipeline.assert_awaited_once()


class TestRedTeamWithAgentTarget:
    """Tests for passing AgentTarget objects directly to red_team()."""

    @pytest.mark.asyncio
    async def test_agent_target_dispatches_to_dynamic(self):
        from unittest.mock import AsyncMock, patch

        class MockTarget(AgentTarget):
            async def respond(self, messages: list[Message]) -> AgentResponse:
                return AgentResponse(text='response')

            def new(self) -> MockTarget:
                return MockTarget()

        mock_report = _make_report()
        with patch(
            'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
            new_callable=AsyncMock,
            return_value=mock_report,
        ) as mock_dyn:
            result = await red_team(MockTarget())
            assert result is mock_report
            mock_dyn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_target_list_dispatches(self):
        from unittest.mock import AsyncMock, patch

        class MockTarget(AgentTarget):
            async def respond(self, messages: list[Message]) -> AgentResponse:
                return AgentResponse(text='response')

            def new(self) -> MockTarget:
                return MockTarget()

        mock_report = _make_report()
        with patch(
            'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            result = await red_team([MockTarget(), 'agent:test'])
            assert result is mock_report

    @pytest.mark.asyncio
    async def test_invalid_target_type_raises(self):
        """Passing an object that does not implement AgentTarget protocol raises TypeError."""
        with pytest.raises(TypeError, match='Invalid target type'):
            await red_team(42)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    @pytest.mark.asyncio
    async def test_invalid_item_in_list_raises(self):
        """Passing an invalid item inside a list raises TypeError."""

        class MockTarget(AgentTarget):
            async def respond(self, messages: list[Message]) -> AgentResponse:
                return AgentResponse(text='response')

            def new(self) -> MockTarget:
                return MockTarget()

        with pytest.raises(TypeError, match='Invalid target type'):
            await red_team([MockTarget(), 42])  # type: ignore[list-item, arg-type]  # pyright: ignore[reportArgumentType]


class TestEvaluabilityGate:
    """red_team() drops explicitly-requested vulnerabilities that have no automated evaluator.

    All ten built-in OWASP ASI categories now have prompt-based evaluators, and
    every built-in Vulnerability maps to one, so the gate never fires for the
    built-in set. It exists to guard the documented custom-vulnerability /
    custom-framework extension path: a user-registered vulnerability with no
    evaluator can be *attacked* but not *scored*, so generating attacks for it
    would burn attacker+target tokens on inconclusive results. These tests
    simulate that by patching the evaluator registry to omit a vulnerability.
    """

    @staticmethod
    def _registry_without(*vulns):
        """Return a copy of VULNERABILITY_EVALUATOR_REGISTRY with ``vulns`` removed."""
        from evaluatorq.redteam.contracts import Vulnerability
        from evaluatorq.redteam.frameworks.owasp.evaluators import (
            VULNERABILITY_EVALUATOR_REGISTRY,
        )

        drop = {Vulnerability(v) for v in vulns}
        return {k: v for k, v in VULNERABILITY_EVALUATOR_REGISTRY.items() if k not in drop}

    @pytest.mark.asyncio
    async def test_unevaluable_category_dropped_for_dynamic(self):
        from unittest.mock import AsyncMock, patch

        # Simulate ASI03 (identity_privilege_abuse) lacking an evaluator.
        patched = self._registry_without('identity_privilege_abuse')
        mock_report = _make_report()
        with (
            patch(
                'evaluatorq.redteam.frameworks.owasp.evaluators.VULNERABILITY_EVALUATOR_REGISTRY',
                patched,
            ),
            patch(
                'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
                new_callable=AsyncMock,
                return_value=mock_report,
            ) as mock_dyn,
        ):
            result = await red_team(
                'agent:test',
                mode='dynamic',
                categories=['ASI01', 'ASI03'],
            )

        kwargs = mock_dyn.call_args.kwargs
        assert 'ASI03' not in kwargs['categories']
        assert 'ASI01' in kwargs['categories']
        assert all(
            get_primary_category(v) != 'ASI03' for v in (kwargs['resolved_vulns'] or [])
        )
        assert any('ASI03' in w for w in result.pipeline_warnings)

    @pytest.mark.asyncio
    async def test_unevaluable_vulnerability_dropped_for_hybrid(self):
        from unittest.mock import AsyncMock, patch

        patched = self._registry_without('identity_privilege_abuse')
        mock_report = _make_report()
        with (
            patch(
                'evaluatorq.redteam.frameworks.owasp.evaluators.VULNERABILITY_EVALUATOR_REGISTRY',
                patched,
            ),
            patch(
                'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
                new_callable=AsyncMock,
                return_value=mock_report,
            ) as mock_dyn,
        ):
            await red_team(
                'agent:test',
                mode='hybrid',
                vulnerabilities=['goal_hijacking', 'identity_privilege_abuse'],
            )

        kwargs = mock_dyn.call_args.kwargs
        assert all(
            v.value != 'identity_privilege_abuse' for v in (kwargs['resolved_vulns'] or [])
        )

    @pytest.mark.asyncio
    async def test_all_unevaluable_raises(self):
        from unittest.mock import patch

        patched = self._registry_without('identity_privilege_abuse', 'inter_agent_comms')
        with (
            patch(
                'evaluatorq.redteam.frameworks.owasp.evaluators.VULNERABILITY_EVALUATOR_REGISTRY',
                patched,
            ),
            pytest.raises(ValueError, match='have an automated evaluator'),
        ):
            await red_team('agent:test', mode='dynamic', categories=['ASI03', 'ASI07'])

    @pytest.mark.asyncio
    async def test_evaluable_categories_unaffected(self):
        from unittest.mock import AsyncMock, patch

        mock_report = _make_report()
        with patch(
            'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
            new_callable=AsyncMock,
            return_value=mock_report,
        ) as mock_dyn:
            result = await red_team('agent:test', mode='dynamic', categories=['ASI01', 'ASI02'])

        kwargs = mock_dyn.call_args.kwargs
        assert set(kwargs['categories']) == {'ASI01', 'ASI02'}
        assert not any('no automated evaluator' in w for w in result.pipeline_warnings)

    @pytest.mark.asyncio
    async def test_real_llm03_is_evaluable_not_gated(self):
        """Unpatched: LLM03 resolves to supply_chain (scored by the ASI04 evaluator) and
        is NOT gated. Pins the real behavior Arian flagged — LLM03 Supply Chain is now
        scorable, so the gate must let it through rather than drop it as unevaluable.
        """
        from unittest.mock import AsyncMock, patch

        from evaluatorq.redteam.contracts import Vulnerability

        mock_report = _make_report()
        with patch(
            'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
            new_callable=AsyncMock,
            return_value=mock_report,
        ) as mock_dyn:
            result = await red_team('agent:test', mode='dynamic', categories=['LLM03'])

        kwargs = mock_dyn.call_args.kwargs
        assert Vulnerability.SUPPLY_CHAIN in (kwargs['resolved_vulns'] or [])
        assert 'LLM03' in kwargs['categories']
        assert not any('no automated evaluator' in w for w in result.pipeline_warnings)

    @pytest.mark.asyncio
    async def test_real_unknown_category_raises(self):
        """Unpatched: an unrecognized category (e.g. LLM10, which is not an OWASP
        category) raises instead of silently swallowing the error and proceeding with no
        vulnerabilities resolved — which would run a meaningless, unscored attack.
        """
        with pytest.raises(ValueError, match='Unrecognized categor'):
            await red_team('agent:test', mode='dynamic', categories=['LLM10'])


class TestHuggingFacePreflight:
    """Static/hybrid runs that pull datapoints from HuggingFace fail fast when
    huggingface-hub is missing, instead of dying deep in the static leg (which in
    hybrid mode runs only after the entire dynamic leg)."""

    def test_is_huggingface_source(self):
        from pathlib import Path

        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import is_huggingface_source

        assert is_huggingface_source(None) is True  # default HF dataset
        assert is_huggingface_source('hf:org/repo') is True
        assert is_huggingface_source('hf:org/repo/file.json') is True
        assert is_huggingface_source('/tmp/local.json') is False
        assert is_huggingface_source(Path('/tmp/local.json')) is False
        assert is_huggingface_source('orq:dataset-id') is False

    def test_ensure_huggingface_available_raises_when_missing(self, monkeypatch):
        import sys

        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import ensure_huggingface_available

        # Simulate an uninstalled package: import huggingface_hub -> ImportError
        monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
        with pytest.raises(ImportError, match=r"evaluatorq\[redteam\]"):
            ensure_huggingface_available()

    @pytest.mark.asyncio
    async def test_static_hf_run_preflights_hf(self, monkeypatch):
        import sys
        from unittest.mock import AsyncMock, patch

        monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
        with patch(
            'evaluatorq.redteam.runner._run_static',
            new_callable=AsyncMock,
            return_value=_make_report(),
        ) as mock_static:
            with pytest.raises(ImportError, match=r"evaluatorq\[redteam\]"):
                await red_team('agent:test', mode='static', dataset=None)
        mock_static.assert_not_awaited()  # failed before dispatching the leg

    @pytest.mark.asyncio
    async def test_static_local_dataset_skips_hf_preflight(self, monkeypatch):
        import sys
        from unittest.mock import AsyncMock, patch

        # Even with huggingface_hub "missing", a local dataset must not trip the check.
        monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
        with patch(
            'evaluatorq.redteam.runner._run_static',
            new_callable=AsyncMock,
            return_value=_make_report(),
        ) as mock_static:
            await red_team('agent:test', mode='static', dataset='/tmp/local.json')
        mock_static.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dynamic_run_skips_hf_preflight(self, monkeypatch):
        import sys
        from unittest.mock import AsyncMock, patch

        # Dynamic mode never loads static datapoints -> no HF dependency.
        monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
        with patch(
            'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
            new_callable=AsyncMock,
            return_value=_make_report(),
        ) as mock_dyn:
            await red_team('agent:test', mode='dynamic', categories=['ASI01'])
        mock_dyn.assert_awaited_once()
