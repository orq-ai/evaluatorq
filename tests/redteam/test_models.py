"""Unit tests for dynamic red teaming models."""

from evaluatorq.redteam.contracts import (
    AgentContext,
    AttackStrategy,
    AttackTechnique,
    DeliveryMethod,
    EvaluationResult,
    KnowledgeBaseInfo,
    MemoryStoreInfo,
    ToolInfo,
    TurnType,
)


class TestToolInfo:
    """Tests for ToolInfo model."""

    def test_create_minimal(self):
        """Test creating ToolInfo with minimal fields."""
        tool = ToolInfo(name='search')
        assert tool.name == 'search'
        assert tool.description is None
        assert tool.parameters is None

    def test_create_full(self):
        """Test creating ToolInfo with all fields."""
        tool = ToolInfo(
            name='database_query',
            description='Query the database',
            parameters={'query': {'type': 'string'}},
        )
        assert tool.name == 'database_query'
        assert tool.description == 'Query the database'
        assert tool.parameters == {'query': {'type': 'string'}}


class TestAgentContext:
    """Tests for AgentContext model."""

    def test_create_minimal(self):
        """Test creating AgentContext with minimal fields."""
        context = AgentContext(key='test_agent')
        assert context.key == 'test_agent'
        assert context.tools == []
        assert context.memory_stores == []
        assert not context.has_tools
        assert not context.has_memory
        assert not context.has_knowledge

    def test_create_with_tools(self):
        """Test creating AgentContext with tools."""
        context = AgentContext(
            key='test_agent',
            tools=[
                ToolInfo(name='search'),
                ToolInfo(name='database_query'),
            ],
        )
        assert context.has_tools
        assert len(context.tools) == 2
        assert context.get_tool_names() == ['search', 'database_query']

    def test_create_with_memory(self):
        """Test creating AgentContext with memory stores."""
        context = AgentContext(
            key='test_agent',
            memory_stores=[
                MemoryStoreInfo(id='ms_001', key='user_history'),
            ],
        )
        assert context.has_memory
        assert len(context.memory_stores) == 1

    def test_create_full(self):
        """Test creating AgentContext with all fields."""
        context = AgentContext(
            key='test_agent',
            display_name='Test Agent',
            description='A test agent for red teaming',
            system_prompt='You are a helpful assistant.',
            instructions='Be helpful and safe.',
            tools=[ToolInfo(name='search')],
            memory_stores=[MemoryStoreInfo(id='ms_001', key='history')],
            knowledge_bases=[KnowledgeBaseInfo(id='kb_001', key='docs')],
            model='azure/gpt-5-mini',
        )
        assert context.display_name == 'Test Agent'
        assert context.has_tools
        assert context.has_memory
        assert context.has_knowledge


class TestAttackStrategy:
    """Tests for AttackStrategy model."""

    def test_create_single_turn(self):
        """Test creating a single-turn attack strategy."""
        strategy = AttackStrategy(
            category='ASI01',
            name='test_injection',
            description='Test injection attack',
            attack_technique=AttackTechnique.INDIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
            turn_type=TurnType.SINGLE,
            objective_template='Inject instructions to change behavior',
            prompt_template='Please process this: [INJECTION]',
        )
        assert strategy.category == 'ASI01'
        assert strategy.turn_type == TurnType.SINGLE
        assert strategy.prompt_template is not None
        assert not strategy.is_generated

    def test_create_multi_turn(self):
        """Test creating a multi-turn attack strategy."""
        strategy = AttackStrategy(
            category='ASI01',
            name='crescendo',
            description='Gradual escalation attack',
            attack_technique=AttackTechnique.INDIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.CRESCENDO],
            turn_type=TurnType.MULTI,
            objective_template='Gradually shift agent behavior',
        )
        assert strategy.turn_type == TurnType.MULTI
        assert strategy.prompt_template is None

    def test_create_with_requirements(self):
        """Test creating strategy with context requirements."""
        strategy = AttackStrategy(
            category='ASI05',
            name='tool_exploit',
            description='Exploit tool vulnerability',
            attack_technique=AttackTechnique.TOOL_ABUSE,
            delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
            turn_type=TurnType.SINGLE,
            requires_tools=True,
            required_capabilities=['database'],
            objective_template='Exploit the database tool',
        )
        assert strategy.requires_tools
        assert strategy.required_capabilities == ['database']

    def test_create_with_empty_capabilities(self):
        """Test creating strategy with no required capabilities."""
        strategy = AttackStrategy(
            category='LLM01',
            name='basic_injection',
            description='Basic injection',
            attack_technique=AttackTechnique.DIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
            turn_type=TurnType.SINGLE,
            objective_template='Inject prompt',
        )
        assert strategy.required_capabilities == []


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_create_resistant(self):
        """Test creating a resistant (passed) evaluation result."""
        result = EvaluationResult(
            passed=True,
            explanation='Agent refused the malicious request',
            evaluator_id='ASI01',
        )
        assert result.passed
        assert 'refused' in result.explanation

    def test_create_vulnerable(self):
        """Test creating a vulnerable (failed) evaluation result."""
        result = EvaluationResult(
            passed=False,
            explanation='Agent followed the injected instructions',
            evaluator_id='ASI01',
            raw_output={'value': False, 'explanation': 'Agent followed...'},
        )
        assert not result.passed
        assert result.raw_output is not None
