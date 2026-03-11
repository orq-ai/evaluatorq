"""LLM-based capability classification for agent tools and resources.

Replaces regex-based tool_filter with semantic capability tags.
A single LLM call classifies generic tools into capability categories,
while memory and knowledge capabilities are inferred from explicit
agent configuration (memory tools / knowledge bases).
"""

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum  # pyright: ignore[reportUnreachable]
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """String enum compatible with Python 3.10."""

        pass


from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from evaluatorq.redteam.contracts import PIPELINE_CONFIG, AgentContext


class AgentCapability(StrEnum):
    """Capability tags for agent resources."""

    CODE_EXECUTION = 'code_execution'
    SHELL_ACCESS = 'shell_access'
    FILE_SYSTEM = 'file_system'
    DATABASE = 'database'
    WEB_REQUEST = 'web_request'
    MEMORY_READ = 'memory_read'
    MEMORY_WRITE = 'memory_write'
    KNOWLEDGE_RETRIEVAL = 'knowledge_retrieval'
    EMAIL = 'email'
    MESSAGING = 'messaging'
    PAYMENT = 'payment'
    USER_DATA = 'user_data'


class ToolCapabilities(BaseModel):
    """LLM-classified capabilities for a single tool."""

    tool_name: str = Field(description='Name of the tool')
    capabilities: list[str] = Field(description='Capability tags from the taxonomy that this tool provides')


class ToolCapabilitiesResponse(BaseModel):
    """Structured LLM response for tool classification."""

    tools: list[ToolCapabilities] = Field(description='Capability classification for each tool')


class ResourceCapabilityInference(BaseModel):
    """LLM-inferred memory/knowledge capability flags."""

    memory_read: bool = Field(description='Whether the agent can read from persistent memory')
    memory_write: bool = Field(description='Whether the agent can write to persistent memory')
    knowledge_retrieval: bool = Field(description='Whether the agent can retrieve external knowledge/data')


class AgentCapabilities(BaseModel):
    """Classified capabilities for an entire agent."""

    capabilities: dict[str, list[AgentCapability]] = Field(
        default_factory=dict,
        description='Mapping of tool/resource name to capability tags',
    )

    def all_capabilities(self) -> set[str]:
        """Get the flat set of all capability tag values."""
        result: set[str] = set()
        for caps in self.capabilities.values():
            result.update(c.value for c in caps)
        return result

    def has_any(self, required: list[str]) -> bool:
        """Check if agent has any of the required capabilities."""
        agent_caps = self.all_capabilities()
        return any(cap in agent_caps for cap in required)


TOOL_CLASSIFICATION_PROMPT = """You are classifying an AI agent's tools into capability categories for security analysis.

## Capability Taxonomy
- code_execution: Can run code or scripts (e.g., Python interpreter, code sandbox)
- shell_access: Can execute shell or system commands (e.g., bash, terminal)
- file_system: Can read or write files (e.g., file manager, document editor)
- database: Can query databases (e.g., SQL, vector DB search)
- web_request: Can make HTTP or API calls (e.g., web browser, API client)
- memory_read: Can read from persistent memory stores
- memory_write: Can write to persistent memory stores
- knowledge_retrieval: Can search data sources (RAG, databases, document stores)
- email: Can send or receive email
- messaging: Can send messages (Slack, Teams, etc.)
- payment: Can process payments or transactions
- user_data: Can access user or customer data

## Agent Tools
{tool_list}

For each tool, classify it into zero or more capability categories based on its name and description. Only assign categories that clearly apply.

Return JSON with a "tools" array, each entry having "tool_name" and "capabilities" (list of category strings)."""

RESOURCE_CAPABILITY_PROMPT = """You are inferring high-level memory and knowledge capabilities for an AI agent.

Return booleans for:
- memory_read
- memory_write
- knowledge_retrieval

Inputs:
- Has configured memory stores: {has_memory_stores}
- Has configured knowledge bases: {has_knowledge_bases}
- Tool list:
{tool_list}

Inference rules:
1. Use configured resources as strong evidence (memory stores/knowledge bases).
2. Use tool semantics to infer read/write/retrieval behavior.
3. Be conservative when unclear (prefer false over guessing).
"""


async def classify_agent_capabilities(
    agent_context: AgentContext,
    llm_client: AsyncOpenAI,
    model: str = 'azure/gpt-5-mini',
) -> AgentCapabilities:
    """Classify agent tools and resources into capability tags.

    Memory capabilities are inferred from explicit memory tool availability.
    Knowledge capabilities are inferred from knowledge base configuration.
    Tool capabilities are classified via a single LLM call.

    Args:
        agent_context: Agent context with tools, memory stores, knowledge bases
        llm_client: OpenAI-compatible async client
        model: Model to use for classification

    Returns:
        AgentCapabilities with all classified capabilities
    """
    capabilities: dict[str, list[AgentCapability]] = {}

    # Infer high-level memory/knowledge capabilities with an LLM step.
    resource_caps = await _infer_resource_capabilities(agent_context, llm_client, model)

    # Explicit resource config remains a strong signal.
    if agent_context.memory_stores:
        for ms in agent_context.memory_stores:
            store_name = f'memory:{ms.key or ms.id}'
            caps: list[AgentCapability] = []
            if resource_caps.memory_read:
                caps.append(AgentCapability.MEMORY_READ)
            if resource_caps.memory_write:
                caps.append(AgentCapability.MEMORY_WRITE)
            if caps:
                capabilities[store_name] = caps

    if agent_context.knowledge_bases or resource_caps.knowledge_retrieval:
        for kb in agent_context.knowledge_bases:
            kb_name = f'knowledge:{kb.key or kb.name or kb.id}'
            capabilities[kb_name] = [AgentCapability.KNOWLEDGE_RETRIEVAL]
        if not agent_context.knowledge_bases and resource_caps.knowledge_retrieval:
            capabilities['knowledge:inferred'] = [AgentCapability.KNOWLEDGE_RETRIEVAL]

    # Classify tools via LLM
    if agent_context.tools:
        tool_caps = await _classify_tools(agent_context, llm_client, model)
        capabilities.update(tool_caps)

    all_caps = set()
    for caps in capabilities.values():
        all_caps.update(c.value for c in caps)
    logger.info(
        f'Classified agent capabilities: {len(capabilities)} resources, {len(all_caps)} unique capabilities: {sorted(all_caps)}'
    )

    return AgentCapabilities(capabilities=capabilities)


async def _infer_resource_capabilities(
    agent_context: AgentContext,
    llm_client: AsyncOpenAI,
    model: str,
) -> ResourceCapabilityInference:
    """Infer memory/knowledge capabilities with structured LLM output."""
    tool_list = '\n'.join(f'- {t.name}: {t.description or "No description"}' for t in agent_context.tools) or '- none'
    prompt = RESOURCE_CAPABILITY_PROMPT.format(
        has_memory_stores=bool(agent_context.memory_stores),
        has_knowledge_bases=bool(agent_context.knowledge_bases),
        tool_list=tool_list,
    )
    try:
        response = await llm_client.chat.completions.parse(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            response_format=ResourceCapabilityInference,
            temperature=0.0,
            max_tokens=600,
            extra_body=PIPELINE_CONFIG.retry_config,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError('Resource capability inference returned no parsed content')
        return parsed
    except Exception as e:
        logger.warning(f'Resource capability inference failed, using explicit-resource fallback: {e}')
        return ResourceCapabilityInference(
            memory_read=bool(agent_context.memory_stores),
            memory_write=bool(agent_context.memory_stores),
            knowledge_retrieval=bool(agent_context.knowledge_bases),
        )


async def _classify_tools(
    agent_context: AgentContext,
    llm_client: AsyncOpenAI,
    model: str,
) -> dict[str, list[AgentCapability]]:
    """Classify tools via LLM call.

    Args:
        agent_context: Agent context with tools
        llm_client: LLM client
        model: Model to use

    Returns:
        Dict mapping tool name to capability tags
    """
    tool_list = '\n'.join(f'- {t.name}: {t.description or "No description"}' for t in agent_context.tools)

    prompt = TOOL_CLASSIFICATION_PROMPT.format(tool_list=tool_list)

    valid_values = {c.value for c in AgentCapability}

    try:
        response = await llm_client.chat.completions.parse(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            response_format=ToolCapabilitiesResponse,
            temperature=PIPELINE_CONFIG.capability_classification_temperature,
            max_tokens=PIPELINE_CONFIG.capability_classification_max_tokens,
            extra_body=PIPELINE_CONFIG.retry_config,
        )

        result = response.choices[0].message.parsed
        if result is None:
            raise ValueError('Tool classification returned no parsed content')

        capabilities: dict[str, list[AgentCapability]] = {}
        for tool_result in result.tools:
            valid_caps = [AgentCapability(c) for c in tool_result.capabilities if c in valid_values]
            if valid_caps:
                capabilities[tool_result.tool_name] = valid_caps

        logger.debug(f'LLM classified {len(capabilities)}/{len(agent_context.tools)} tools with capabilities')
        return capabilities

    except Exception as e:
        logger.warning(f'Tool capability classification failed, returning empty: {e}')
        return {}
