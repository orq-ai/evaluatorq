"""Agent context retrieval from ORQ API."""

import asyncio
from typing import TYPE_CHECKING

try:
    from evaluatorq.redteam.backends.orq import ORQContextProvider
except ImportError:
    ORQContextProvider = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from orq_ai_sdk import Orq as OrqType
else:
    OrqType = None  # type: ignore[assignment,misc]

from evaluatorq.redteam.contracts import AgentContext


async def retrieve_agent_context(orq_client: 'OrqType', agent_key: str) -> AgentContext:
    """Retrieve agent context from ORQ API.

    Delegates to :class:`ORQContextProvider`.

    Args:
        orq_client: ORQ SDK client instance
        agent_key: Unique agent key/identifier

    Returns:
        AgentContext with parsed and enriched agent configuration
    """
    if ORQContextProvider is None:
        msg = 'ORQ dependencies are not installed. Install the ORQ extras to use retrieve_agent_context.'
        raise RuntimeError(msg)
    return await ORQContextProvider(orq_client).get_agent_context(agent_key)


def retrieve_agent_context_sync(orq_client: 'OrqType', agent_key: str) -> AgentContext:
    """Synchronous version of retrieve_agent_context.

    Args:
        orq_client: ORQ SDK client instance
        agent_key: Unique agent key/identifier

    Returns:
        AgentContext with parsed agent configuration

    Raises:
        RuntimeError: If called from within an already-running event loop.
            Use ``await retrieve_agent_context(...)`` instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop — safe to use asyncio.run()
        return asyncio.run(retrieve_agent_context(orq_client, agent_key))
    else:
        raise RuntimeError(
            'retrieve_agent_context_sync cannot be called from within an async context. Use `await retrieve_agent_context(...)` instead.'
        )
