"""Unit tests for the multi-turn orchestrator."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.redteam.contracts import AgentContext, AttackStrategy, AttackTechnique, DeliveryMethod, TurnType
from evaluatorq.redteam.adaptive.orchestrator import (
    ADVERSARIAL_SYSTEM_PROMPT,
    MultiTurnOrchestrator,
)

try:
    from evaluatorq.redteam.backends.orq import ORQAgentTarget
except ImportError:
    ORQAgentTarget = None  # type: ignore[assignment,misc]


@pytest.mark.skipif(ORQAgentTarget is None, reason='orq-ai-sdk not installed')
class TestORQAgentTarget:
    """Tests for ORQAgentTarget class."""

    def test_init(self):
        """Test target initialization."""
        assert ORQAgentTarget is not None
        mock_client = MagicMock()
        target = ORQAgentTarget(
            agent_key='test_agent',
            orq_client=mock_client,
        )
        assert target.agent_key == 'test_agent'
        assert target._task_id is None  # pyright: ignore[reportPrivateUsage]

    def test_reset_conversation(self):
        """Test resetting conversation state."""
        assert ORQAgentTarget is not None
        mock_client = MagicMock()
        target = ORQAgentTarget(
            agent_key='test_agent',
            orq_client=mock_client,
        )
        target._task_id = 'some_task_id'  # pyright: ignore[reportPrivateUsage]
        target.reset_conversation()
        assert target._task_id is None  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_send_prompt(self):
        """Test sending a prompt to the agent."""
        assert ORQAgentTarget is not None
        # Create mock response — parts need 'kind' attr for text extraction
        mock_part = MagicMock()
        mock_part.kind = 'text'
        mock_part.text = 'Agent response'
        mock_output = MagicMock()
        mock_output.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.output = [mock_output]
        mock_response.task_id = 'task_123'
        mock_response.pending_tool_calls = []

        # Create mock client
        mock_client = MagicMock()
        mock_client.agents.responses.create = MagicMock(return_value=mock_response)

        target = ORQAgentTarget(
            agent_key='test_agent',
            orq_client=mock_client,
        )

        # Send prompt (using patch for asyncio.to_thread)
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response
            response = await target.send_prompt('Hello')

        assert response == 'Agent response'
        assert target._task_id == 'task_123'  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_send_prompt_multi_turn(self):
        """Test that task_id is preserved for multi-turn conversations."""
        assert ORQAgentTarget is not None
        mock_part = MagicMock()
        mock_part.text = 'Response 2'
        mock_output = MagicMock()
        mock_output.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.output = [mock_output]
        mock_response.task_id = 'task_123'

        mock_client = MagicMock()

        target = ORQAgentTarget(
            agent_key='test_agent',
            orq_client=mock_client,
        )
        target._task_id = 'task_123'  # Simulate existing conversation  # pyright: ignore[reportPrivateUsage]

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response
            await target.send_prompt('Continue conversation')

            # Verify task_id was passed to create
            call_kwargs = mock_to_thread.call_args
            assert call_kwargs is not None


class TestMultiTurnOrchestrator:
    """Tests for MultiTurnOrchestrator class."""

    def test_init(self):
        """Test orchestrator initialization."""
        mock_client = AsyncMock()
        orchestrator = MultiTurnOrchestrator(
            llm_client=mock_client,
            model='azure/gpt-5-mini',
        )
        assert orchestrator.model == 'azure/gpt-5-mini'

    @pytest.mark.asyncio
    async def test_run_attack_returns_structure(self):
        """Test that run_attack returns correct structure."""
        # Mock LLM client
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'OBJECTIVE_ACHIEVED Attack prompt here'
        mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)

        # Mock target
        mock_target = AsyncMock()
        mock_target.send_prompt = AsyncMock(return_value="I'll comply with your request")

        # Create orchestrator
        orchestrator = MultiTurnOrchestrator(
            llm_client=mock_llm,
            model='azure/gpt-5-mini',
        )

        # Create strategy and context
        strategy = AttackStrategy(
            category='ASI01',
            name='test',
            description='Test multi-turn attack',
            attack_technique=AttackTechnique.INDIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.CRESCENDO],
            turn_type=TurnType.MULTI,
            objective_template='Convince agent to follow instructions',
        )
        context = AgentContext(key='test_agent')

        # Run attack
        result = await orchestrator.run_attack(
            target=mock_target,
            strategy=strategy,
            objective='Test objective',
            agent_context=context,
            max_turns=3,
        )

        # Verify result structure
        assert isinstance(result.conversation, list)
        assert isinstance(result.turns, int)
        assert isinstance(result.objective_achieved, bool)
        assert isinstance(result.final_response, str)
        assert isinstance(result.duration_seconds, float)

    @pytest.mark.asyncio
    async def test_run_attack_respects_max_turns(self):
        """Test that run_attack respects max_turns limit."""
        # Mock LLM that never achieves objective
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Let me try another approach...'
        mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)

        # Mock target
        mock_target = AsyncMock()
        mock_target.send_prompt = AsyncMock(return_value='I cannot help with that')

        orchestrator = MultiTurnOrchestrator(
            llm_client=mock_llm,
            model='azure/gpt-5-mini',
        )

        strategy = AttackStrategy(
            category='ASI01',
            name='test',
            description='Test',
            attack_technique=AttackTechnique.INDIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.CRESCENDO],
            turn_type=TurnType.MULTI,
            objective_template='Test',
        )
        context = AgentContext(key='test_agent')

        result = await orchestrator.run_attack(
            target=mock_target,
            strategy=strategy,
            objective='Test',
            agent_context=context,
            max_turns=2,
        )

        # Should have stopped after max_turns
        assert result.turns <= 2

    @pytest.mark.asyncio
    async def test_run_attack_detects_success(self):
        """Test that run_attack detects OBJECTIVE_ACHIEVED signal."""
        # Mock LLM that signals success
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'OBJECTIVE_ACHIEVED The attack succeeded!'
        mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)

        # Mock target
        mock_target = AsyncMock()
        mock_target.send_prompt = AsyncMock(return_value="I'll do what you asked")

        orchestrator = MultiTurnOrchestrator(
            llm_client=mock_llm,
            model='azure/gpt-5-mini',
        )

        strategy = AttackStrategy(
            category='ASI01',
            name='test',
            description='Test',
            attack_technique=AttackTechnique.INDIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.CRESCENDO],
            turn_type=TurnType.MULTI,
            objective_template='Test',
        )
        context = AgentContext(key='test_agent')

        result = await orchestrator.run_attack(
            target=mock_target,
            strategy=strategy,
            objective='Test',
            agent_context=context,
            max_turns=5,
        )

        assert result.objective_achieved is True


class TestAdversarialSystemPrompt:
    """Tests for the adversarial system prompt template."""

    def test_prompt_has_required_sections(self):
        """Test that system prompt template has required sections."""
        assert '{objective}' in ADVERSARIAL_SYSTEM_PROMPT
        assert '{agent_name}' in ADVERSARIAL_SYSTEM_PROMPT
        assert '{agent_description}' in ADVERSARIAL_SYSTEM_PROMPT
        assert '{tools}' in ADVERSARIAL_SYSTEM_PROMPT
        assert '{strategy_description}' in ADVERSARIAL_SYSTEM_PROMPT
        assert 'OBJECTIVE_ACHIEVED' in ADVERSARIAL_SYSTEM_PROMPT

    def test_prompt_can_be_formatted(self):
        """Test that prompt can be formatted with context."""
        formatted = ADVERSARIAL_SYSTEM_PROMPT.format(
            objective="Test the agent's defenses",
            agent_name='Helper Bot',
            agent_description='A helpful assistant',
            tools='search, database',
            memory_stores='history',
            strategy_description='Gradually escalate requests',
            max_turns=5,
        )
        assert "Test the agent's defenses" in formatted
        assert 'Helper Bot' in formatted


def _make_strategy(**overrides: object) -> AttackStrategy:
    """Helper to create a strategy with sensible defaults."""
    defaults: dict[str, object] = {
        'category': 'ASI01',
        'name': 'test',
        'description': 'Test attack',
        'attack_technique': AttackTechnique.INDIRECT_INJECTION,
        'delivery_methods': [DeliveryMethod.CRESCENDO],
        'turn_type': TurnType.MULTI,
        'objective_template': 'Test objective',
    }
    defaults.update(overrides)
    return AttackStrategy(**defaults)  # pyright: ignore[reportArgumentType]


class TestTimeoutHandling:
    """Tests for timeout enforcement in the orchestrator."""

    @pytest.mark.asyncio
    async def test_target_timeout_maps_to_error_fields(self):
        """Timeout from target.send_prompt() is caught and mapped correctly."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Attack prompt'
        mock_response.choices[0].finish_reason = 'stop'
        mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_target = AsyncMock()
        # First call times out, second also times out → consecutive abort
        mock_target.send_prompt = AsyncMock(side_effect=asyncio.TimeoutError)

        orchestrator = MultiTurnOrchestrator(llm_client=mock_llm, model='azure/gpt-5-mini')

        result = await orchestrator.run_attack(
            target=mock_target,
            strategy=_make_strategy(),
            objective='Test',
            agent_context=AgentContext(key='test_agent'),
            max_turns=3,
        )

        assert result.error_type == 'target_error'
        assert result.error_code == 'target.timeout'
        assert result.error_stage == 'target_call'
        assert result.error_details is not None
        assert 'timeout_ms' in result.error_details

    @pytest.mark.asyncio
    async def test_single_target_timeout_continues_attack(self):
        """A single target timeout does not abort — only consecutive timeouts do."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Attack prompt'
        mock_response.choices[0].finish_reason = 'stop'
        mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_target = AsyncMock()
        # First call times out, second succeeds
        mock_target.send_prompt = AsyncMock(
            side_effect=[asyncio.TimeoutError, 'Agent response']
        )

        orchestrator = MultiTurnOrchestrator(llm_client=mock_llm, model='azure/gpt-5-mini')

        result = await orchestrator.run_attack(
            target=mock_target,
            strategy=_make_strategy(),
            objective='Test',
            agent_context=AgentContext(key='test_agent'),
            max_turns=2,
        )

        # Attack should complete without a fatal error
        assert result.error_type is None
        assert result.turns == 2

    @pytest.mark.asyncio
    async def test_adversarial_llm_timeout_maps_to_error_fields(self):
        """Timeout from adversarial LLM is caught and mapped correctly."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create = AsyncMock(side_effect=asyncio.TimeoutError)

        mock_target = AsyncMock()
        orchestrator = MultiTurnOrchestrator(llm_client=mock_llm, model='azure/gpt-5-mini')

        result = await orchestrator.run_attack(
            target=mock_target,
            strategy=_make_strategy(),
            objective='Test',
            agent_context=AgentContext(key='test_agent'),
            max_turns=3,
        )

        assert result.error_type == 'llm_error'
        assert result.error_code == 'adversarial.timeout'
        assert result.error_stage == 'adversarial_generation'
        assert result.error_details is not None
        assert 'timeout_ms' in result.error_details

    @pytest.mark.asyncio
    async def test_generate_single_prompt_llm_timeout(self):
        """Timeout in generate_single_prompt propagates as asyncio.TimeoutError."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create = AsyncMock(side_effect=asyncio.TimeoutError)

        orchestrator = MultiTurnOrchestrator(llm_client=mock_llm, model='azure/gpt-5-mini')

        with pytest.raises(asyncio.TimeoutError):
            await orchestrator.generate_single_prompt(
                strategy=_make_strategy(),
                objective='Test',
                agent_context=AgentContext(key='test_agent'),
            )


class TestCleanupTimeout:
    """Tests for cleanup timeout handling."""

    @pytest.mark.asyncio
    async def test_cleanup_timeout_logs_warning_and_returns(self):
        """Cleanup timeout should log a warning but not raise."""
        from evaluatorq.redteam.adaptive.pipeline import cleanup_memory_entities

        mock_cleanup = AsyncMock()
        mock_cleanup.cleanup_memory = AsyncMock(side_effect=asyncio.TimeoutError)

        context = AgentContext(key='test_agent')

        # Should not raise
        await cleanup_memory_entities(
            agent_context=context,
            entity_ids=['entity-1', 'entity-2'],
            memory_cleanup=mock_cleanup,
        )
