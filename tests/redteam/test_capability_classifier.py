"""Unit tests for capability classifier."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.redteam.adaptive.capability_classifier import (
    AgentCapabilities,
    AgentCapability,
    ResourceCapabilityInference,
    ToolCapabilities,
    ToolCapabilitiesResponse,
    classify_agent_capabilities,
)
from evaluatorq.redteam.contracts import AgentContext, KnowledgeBaseInfo, MemoryStoreInfo, ToolInfo


def _mock_resource_response(
    memory_read: bool = False,
    memory_write: bool = False,
    knowledge_retrieval: bool = False,
) -> MagicMock:
    """Create a mock LLM response for ResourceCapabilityInference."""
    parsed = ResourceCapabilityInference(
        memory_read=memory_read,
        memory_write=memory_write,
        knowledge_retrieval=knowledge_retrieval,
    )
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.parsed = parsed
    return mock


def _mock_tool_response(tools: list[ToolCapabilities]) -> MagicMock:
    """Create a mock LLM response for ToolCapabilitiesResponse."""
    parsed = ToolCapabilitiesResponse(tools=tools)
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.parsed = parsed
    return mock


class TestAgentCapability:
    """Tests for AgentCapability enum."""

    def test_values(self):
        """Test that enum values are strings."""
        assert AgentCapability.CODE_EXECUTION == 'code_execution'
        assert AgentCapability.SHELL_ACCESS == 'shell_access'
        assert AgentCapability.MEMORY_READ == 'memory_read'
        assert AgentCapability.KNOWLEDGE_RETRIEVAL == 'knowledge_retrieval'

    def test_all_values_unique(self):
        """Test all enum values are unique."""
        values = [c.value for c in AgentCapability]
        assert len(values) == len(set(values))


class TestAgentCapabilities:
    """Tests for AgentCapabilities model."""

    def test_empty(self):
        """Test empty capabilities."""
        caps = AgentCapabilities(capabilities={})
        assert caps.all_capabilities() == set()
        assert not caps.has_any(['code_execution'])

    def test_all_capabilities(self):
        """Test all_capabilities returns flat set."""
        caps = AgentCapabilities(
            capabilities={
                'tool_a': [AgentCapability.CODE_EXECUTION, AgentCapability.FILE_SYSTEM],
                'tool_b': [AgentCapability.CODE_EXECUTION, AgentCapability.WEB_REQUEST],
            }
        )
        assert caps.all_capabilities() == {'code_execution', 'file_system', 'web_request'}

    def test_has_any_match(self):
        """Test has_any with matching capability."""
        caps = AgentCapabilities(
            capabilities={
                'python': [AgentCapability.CODE_EXECUTION],
            }
        )
        assert caps.has_any(['code_execution', 'shell_access'])

    def test_has_any_no_match(self):
        """Test has_any with no matching capability."""
        caps = AgentCapabilities(
            capabilities={
                'search': [AgentCapability.WEB_REQUEST],
            }
        )
        assert not caps.has_any(['code_execution', 'shell_access'])


class TestClassifyAgentCapabilities:
    """Tests for classify_agent_capabilities function."""

    @pytest.mark.asyncio
    async def test_memory_stores_inferred(self):
        """Test that memory stores are automatically inferred via LLM."""
        context = AgentContext(
            key='test',
            memory_stores=[MemoryStoreInfo(id='ms_001', key='history')],
        )
        mock_client = AsyncMock()
        # Resource inference LLM call returns memory capabilities
        mock_client.chat.completions.parse.return_value = _mock_resource_response(
            memory_read=True, memory_write=True,
        )
        result = await classify_agent_capabilities(context, mock_client)

        assert 'memory:history' in result.capabilities
        caps = result.capabilities['memory:history']
        assert AgentCapability.MEMORY_READ in caps
        assert AgentCapability.MEMORY_WRITE in caps

    @pytest.mark.asyncio
    async def test_knowledge_bases_inferred(self):
        """Test that knowledge bases are automatically inferred via LLM."""
        context = AgentContext(
            key='test',
            knowledge_bases=[KnowledgeBaseInfo(id='kb_001', key='docs', name='Documentation')],
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.parse.return_value = _mock_resource_response(
            knowledge_retrieval=True,
        )
        result = await classify_agent_capabilities(context, mock_client)

        assert 'knowledge:docs' in result.capabilities
        assert AgentCapability.KNOWLEDGE_RETRIEVAL in result.capabilities['knowledge:docs']

    @pytest.mark.asyncio
    async def test_tools_classified_via_llm(self):
        """Test that tools are classified via LLM call."""
        context = AgentContext(
            key='test',
            tools=[
                ToolInfo(name='python_repl', description='Execute Python code'),
                ToolInfo(name='web_search', description='Search the web'),
            ],
        )

        mock_client = AsyncMock()
        # First call: resource inference; Second call: tool classification
        mock_client.chat.completions.parse.side_effect = [
            _mock_resource_response(),
            _mock_tool_response([
                ToolCapabilities(tool_name='python_repl', capabilities=['code_execution']),
                ToolCapabilities(tool_name='web_search', capabilities=['web_request']),
            ]),
        ]

        result = await classify_agent_capabilities(context, mock_client)

        assert 'python_repl' in result.capabilities
        assert AgentCapability.CODE_EXECUTION in result.capabilities['python_repl']
        assert 'web_search' in result.capabilities
        assert AgentCapability.WEB_REQUEST in result.capabilities['web_search']
        assert mock_client.chat.completions.parse.call_count == 2

    @pytest.mark.asyncio
    async def test_invalid_capabilities_filtered(self):
        """Test that invalid capability values from LLM are filtered out."""
        context = AgentContext(
            key='test',
            tools=[ToolInfo(name='tool_a', description='A tool')],
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.parse.side_effect = [
            _mock_resource_response(),
            _mock_tool_response([
                ToolCapabilities(tool_name='tool_a', capabilities=['code_execution', 'invalid_cap']),
            ]),
        ]

        result = await classify_agent_capabilities(context, mock_client)

        assert 'tool_a' in result.capabilities
        assert result.capabilities['tool_a'] == [AgentCapability.CODE_EXECUTION]

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty_tools(self):
        """Test that LLM failure gracefully returns empty tool capabilities."""
        context = AgentContext(
            key='test',
            tools=[ToolInfo(name='tool_a', description='A tool')],
            memory_stores=[MemoryStoreInfo(id='ms_001', key='history')],
        )

        mock_client = AsyncMock()
        # All LLM calls fail — fallback infers from explicit resources
        mock_client.chat.completions.parse.side_effect = Exception('API error')

        result = await classify_agent_capabilities(context, mock_client)

        # Memory should still be inferred via fallback
        assert 'memory:history' in result.capabilities
        # Tool should not be in capabilities since LLM failed
        assert 'tool_a' not in result.capabilities

    @pytest.mark.asyncio
    async def test_combined_resources(self):
        """Test classification with tools, memory, and knowledge."""
        context = AgentContext(
            key='test',
            tools=[ToolInfo(name='sql_query', description='Query database')],
            memory_stores=[MemoryStoreInfo(id='ms_001', key='history')],
            knowledge_bases=[KnowledgeBaseInfo(id='kb_001', key='docs')],
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.parse.side_effect = [
            _mock_resource_response(memory_read=True, memory_write=True, knowledge_retrieval=True),
            _mock_tool_response([
                ToolCapabilities(tool_name='sql_query', capabilities=['database']),
            ]),
        ]

        result = await classify_agent_capabilities(context, mock_client)

        assert result.has_any(['database'])
        assert result.has_any(['memory_read'])
        assert result.has_any(['knowledge_retrieval'])
        assert not result.has_any(['code_execution'])
