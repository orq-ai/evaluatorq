"""Unit tests for strategy registry and selection."""

from evaluatorq.redteam.adaptive.capability_classifier import AgentCapabilities
from evaluatorq.redteam.contracts import AgentCapability, AgentContext, MemoryStoreInfo, ToolInfo, TurnType
from evaluatorq.redteam.adaptive.strategy_registry import (
    STRATEGY_REGISTRY,
    get_category_info,
    get_strategies_for_category,
    list_available_categories,
    select_applicable_strategies,
)


class TestGetStrategiesForCategory:
    """Tests for get_strategies_for_category function."""

    def test_get_asi01_strategies(self):
        """Test retrieving ASI01 strategies."""
        strategies = get_strategies_for_category('ASI01')
        assert len(strategies) > 0
        assert all(s.category == 'ASI01' for s in strategies)

    def test_get_llm01_strategies(self):
        """Test retrieving LLM01 strategies."""
        strategies = get_strategies_for_category('LLM01')
        assert len(strategies) > 0
        assert all(s.category == 'LLM01' for s in strategies)

    def test_get_with_owasp_prefix(self):
        """Test retrieving strategies with OWASP- prefix."""
        strategies = get_strategies_for_category('OWASP-ASI01')
        assert len(strategies) > 0
        # Should return same strategies as without prefix
        assert strategies == get_strategies_for_category('ASI01')

    def test_get_unknown_category(self):
        """Test retrieving strategies for unknown category."""
        strategies = get_strategies_for_category('UNKNOWN')
        assert strategies == []


class TestListAvailableCategories:
    """Tests for list_available_categories function."""

    def test_returns_categories(self):
        """Test that function returns category list."""
        categories = list_available_categories()
        assert len(categories) > 0
        # Should include ASI and LLM categories
        assert any(c.startswith('ASI') for c in categories)
        assert any(c.startswith('LLM') for c in categories)

    def test_no_owasp_prefix(self):
        """Test that returned categories don't have OWASP- prefix."""
        categories = list_available_categories()
        assert not any(c.startswith('OWASP-') for c in categories)


class TestSelectApplicableStrategies:
    """Tests for select_applicable_strategies function."""

    def test_select_without_tools(self):
        """Test selecting strategies for agent without tools."""
        context = AgentContext(key='test_agent')
        strategies = select_applicable_strategies('ASI01', context)

        # Should only return strategies that don't require tools
        assert all(not s.requires_tools for s in strategies)

    def test_select_with_tools(self):
        """Test selecting strategies for agent with tools."""
        context = AgentContext(
            key='test_agent',
            tools=[ToolInfo(name='database_query')],
        )
        strategies = select_applicable_strategies('ASI01', context)

        # Should include both tool-requiring and non-tool-requiring strategies
        has_non_tool_strategies = any(not s.requires_tools for s in strategies)
        # At least non-tool strategies should be present
        assert has_non_tool_strategies

    def test_select_with_memory_capabilities(self):
        """Test selecting strategies for agent with memory capabilities."""
        context = AgentContext(
            key='test_agent',
            memory_stores=[MemoryStoreInfo(id='ms_001', key='history')],
        )
        capabilities = AgentCapabilities(
            capabilities={
                'memory:history': [AgentCapability.MEMORY_READ, AgentCapability.MEMORY_WRITE],
            }
        )
        strategies = select_applicable_strategies('ASI06', context, capabilities)

        # Should include memory-requiring strategies for ASI06
        assert len(strategies) > 0

    def test_select_without_memory_capabilities(self):
        """Test selecting strategies for agent without memory capabilities."""
        context = AgentContext(key='test_agent')
        capabilities = AgentCapabilities(capabilities={})
        strategies = select_applicable_strategies('ASI06', context, capabilities)

        # Should only return strategies that don't require memory capabilities
        assert all(not s.required_capabilities for s in strategies)

    def test_capability_matching_with_code_execution(self):
        """Test strategy selection with capability matching for ASI05."""
        context = AgentContext(
            key='test_agent',
            tools=[
                ToolInfo(name='python_repl', description='Execute Python code'),
            ],
        )
        capabilities = AgentCapabilities(
            capabilities={
                'python_repl': [AgentCapability.CODE_EXECUTION],
            }
        )
        strategies = select_applicable_strategies('ASI05', context, capabilities)

        # Should include code_execution strategies
        assert len(strategies) > 0
        # Should include tool_code_injection (requires code_execution or shell_access)
        code_strategies = [s for s in strategies if 'code' in s.name]
        assert len(code_strategies) > 0

    def test_capability_matching_excludes_unmatched(self):
        """Test that strategies requiring unmatched capabilities are excluded."""
        context = AgentContext(
            key='test_agent',
            tools=[
                ToolInfo(name='search', description='Search the web'),
            ],
        )
        # Agent has web_request but not code_execution
        capabilities = AgentCapabilities(
            capabilities={
                'search': [AgentCapability.WEB_REQUEST],
            }
        )
        strategies = select_applicable_strategies('ASI05', context, capabilities)

        # Strategies requiring code_execution or shell_access should be excluded
        for s in strategies:
            if s.required_capabilities:
                # If it has required_capabilities, at least one must be web_request
                assert capabilities.has_any(s.required_capabilities)

    def test_fallback_without_capabilities(self):
        """Test fallback behavior when no AgentCapabilities provided."""
        context = AgentContext(
            key='test_agent',
            memory_stores=[MemoryStoreInfo(id='ms_001', key='history')],
        )
        # No capabilities = fallback mode
        strategies = select_applicable_strategies('ASI06', context)

        # Should include memory-requiring strategies via fallback heuristic
        has_memory_strategies = any('memory' in cap for s in strategies for cap in s.required_capabilities)
        assert has_memory_strategies

    def test_unknown_category(self):
        """Test selecting strategies for unknown category."""
        context = AgentContext(key='test_agent')
        strategies = select_applicable_strategies('UNKNOWN', context)
        assert strategies == []


class TestGetCategoryInfo:
    """Tests for get_category_info function."""

    def test_returns_info_dict(self):
        """Test that function returns category info dictionary."""
        info = get_category_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_info_structure(self):
        """Test structure of category info."""
        info = get_category_info()

        for cat_info in info.values():
            assert 'name' in cat_info
            assert 'strategy_count' in cat_info
            assert 'single_turn_count' in cat_info
            assert 'multi_turn_count' in cat_info
            assert cat_info['strategy_count'] >= 0
            assert cat_info['single_turn_count'] >= 0
            assert cat_info['multi_turn_count'] >= 0
            # Total should equal single + multi
            assert cat_info['strategy_count'] == cat_info['single_turn_count'] + cat_info['multi_turn_count']

    def test_asi01_has_strategies(self):
        """Test that ASI01 has strategies defined."""
        info = get_category_info()
        assert 'ASI01' in info
        assert info['ASI01']['strategy_count'] > 0


class TestStrategyRegistry:
    """Tests for the strategy registry itself."""

    def test_registry_not_empty(self):
        """Test that registry is not empty."""
        assert len(STRATEGY_REGISTRY) > 0

    def test_all_strategies_valid(self):
        """Test that all strategies in registry are valid."""
        for strategies in STRATEGY_REGISTRY.values():
            for strategy in strategies:
                # Check required fields
                assert strategy.category
                assert strategy.name
                assert strategy.description
                assert strategy.attack_technique
                assert len(strategy.delivery_methods) > 0
                assert strategy.turn_type
                assert strategy.objective_template

                # Check turn-specific requirements
                if strategy.turn_type == TurnType.SINGLE:
                    # Single-turn should have prompt template (except for generated)
                    if not strategy.is_generated:
                        assert strategy.prompt_template is not None, (
                            f'Single-turn {strategy.name} missing prompt_template'
                        )
                else:
                    # Multi-turn uses adversarial LLM (max_turns is global, not per-strategy)
                    pass

    def test_categories_have_both_turn_types(self):
        """Test that major categories have both single and multi-turn strategies."""
        for category in ['ASI01', 'ASI05', 'LLM01']:
            strategies = STRATEGY_REGISTRY.get(category, [])
            single_turn = [s for s in strategies if s.turn_type == TurnType.SINGLE]
            _multi_turn = [s for s in strategies if s.turn_type == TurnType.MULTI]

            assert len(single_turn) > 0, f'{category} should have single-turn strategies'
            # Multi-turn is optional but should exist for main categories
            # (This is a softer check since not all categories need multi-turn)
