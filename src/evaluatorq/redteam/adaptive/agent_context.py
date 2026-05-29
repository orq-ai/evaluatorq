"""Agent context retrieval from ORQ API."""

import asyncio
from typing import Any

from evaluatorq.redteam.contracts import AgentContext


async def retrieve_agent_context(orq_client: Any, agent_key: str) -> AgentContext:
    """Retrieve agent context from ORQ API.

    Delegates to :class:`ORQAgentTarget.get_agent_context`.

    Args:
        orq_client: ORQ SDK client instance
        agent_key: Unique agent key/identifier

    Returns:
        AgentContext with parsed and enriched agent configuration
    """
    try:
        from evaluatorq.redteam.backends.orq import ORQAgentTarget
    except ImportError:
        raise ImportError("orq_ai_sdk is required for ORQ context retrieval. Install with: pip install orq-ai-sdk")
    probe = ORQAgentTarget(agent_key=agent_key, orq_client=orq_client)
    return await probe.get_agent_context()


def retrieve_agent_context_sync(orq_client: Any, agent_key: str) -> AgentContext:
    """Synchronous version of retrieve_agent_context.

    Args:
        orq_client: ORQ SDK client instance
        agent_key: Unique agent key/identifier

    Returns:
        AgentContext with parsed agent configuration
    """
    return asyncio.run(retrieve_agent_context(orq_client, agent_key))
