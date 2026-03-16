"""Tests for the unified red_team() entry point and category helpers."""

from __future__ import annotations

import pytest

from evaluatorq.redteam import get_category_info, list_categories, red_team
from evaluatorq.redteam._runner import _parse_target  # pyright: ignore[reportPrivateUsage]


class TestParseTarget:
    """Tests for target string parsing."""

    def test_agent_target(self):
        kind, value = _parse_target('agent:my-agent-key')
        assert kind == 'agent'
        assert value == 'my-agent-key'

    def test_openai_target(self):
        kind, value = _parse_target('openai:gpt-4o')
        assert kind == 'openai'
        assert value == 'gpt-4o'

    def test_bare_key_defaults_to_agent(self):
        kind, value = _parse_target('my-agent-key')
        assert kind == 'agent'
        assert value == 'my-agent-key'

    def test_case_insensitive_kind(self):
        kind, _value = _parse_target('Agent:my-key')
        assert kind == 'agent'

    def test_empty_value_raises(self):
        with pytest.raises(ValueError, match='missing a value'):
            _parse_target('agent:')

    def test_multiple_colons(self):
        kind, value = _parse_target('openai:org/gpt-4o:latest')
        assert kind == 'openai'
        assert value == 'org/gpt-4o:latest'


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
        for _cat, details in info.items():
            assert 'name' in details
            assert 'strategy_count' in details
            assert 'single_turn_count' in details
            assert 'multi_turn_count' in details
            assert isinstance(details['strategy_count'], int)
            assert details['strategy_count'] > 0

    def test_asi01_has_strategies(self):
        info = get_category_info()
        assert 'ASI01' in info
        assert info['ASI01']['strategy_count'] > 0


class TestRedTeamValidation:
    """Tests for red_team() argument validation (no actual LLM calls)."""

    @pytest.mark.asyncio
    async def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match='Invalid mode'):
            await red_team('agent:test', mode='invalid')

    @pytest.mark.asyncio
    async def test_static_mode_dispatches(self):
        """Static mode dispatches to _run_static (no longer raises NotImplementedError)."""
        from unittest.mock import AsyncMock, patch

        sentinel = object()
        with patch(
            'evaluatorq.redteam._runner._run_static',
            new_callable=AsyncMock,
            return_value=sentinel,
        ) as mock_static:
            result = await red_team('agent:test', mode='static')
            assert result is sentinel
            mock_static.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hybrid_mode_not_implemented(self):
        with pytest.raises(NotImplementedError, match='not yet available'):
            await red_team('agent:test', mode='hybrid')
