"""Agent context retrieval from ORQ API."""

import asyncio
from typing import Any

from evaluatorq.redteam.contracts import AgentContext

try:
    from evaluatorq.redteam.backends.orq import ORQContextProvider as _ORQContextProvider
    _orq_context_provider_cls: Any = _ORQContextProvider
except ImportError:
    _orq_context_provider_cls = None


async def retrieve_agent_context(orq_client: Any, agent_key: str) -> AgentContext:
    """Retrieve agent context from ORQ API.

    Delegates to :class:`ORQContextProvider`.

    Args:
        orq_client: ORQ SDK client instance
        agent_key: Unique agent key/identifier

    Returns:
        AgentContext with parsed and enriched agent configuration
    """
    if _orq_context_provider_cls is None:
        raise ImportError("orq_ai_sdk is required for ORQ context retrieval. Install with: pip install orq-ai-sdk")
    return await _orq_context_provider_cls(orq_client).get_agent_context(agent_key)


def retrieve_agent_context_sync(orq_client: Any, agent_key: str) -> AgentContext:
    """Synchronous version of retrieve_agent_context.

    Args:
        orq_client: ORQ SDK client instance
        agent_key: Unique agent key/identifier

    Returns:
        AgentContext with parsed agent configuration
    """
    return asyncio.run(retrieve_agent_context(orq_client, agent_key))
