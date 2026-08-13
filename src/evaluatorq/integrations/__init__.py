"""Integration modules for evaluatorq.

Available integrations:

- langchain_integration: LangChain agent wrapper for OpenResponses format
- langgraph_integration: LangGraph agent target
- openai_agents_integration: OpenAI Agents SDK agent target
- pydantic_ai_integration: Pydantic AI agent target
- crewai_integration: CrewAI crew target
- vercel_ai_sdk_integration: Vercel AI SDK agent target (HTTP)
- callable_integration: Custom callable agent target

Integrations with optional dependencies (langgraph, openai-agents, pydantic-ai,
crewai) use lazy imports so that importing this package does not fail when those
libraries are not installed.
"""

import importlib

from . import langchain_integration

__all__ = [
    'callable_integration',  # noqa: F822
    'crewai_integration',  # noqa: F822
    'langchain_integration',
    'langgraph_integration',  # noqa: F822
    'openai_agents_integration',  # noqa: F822
    'pydantic_ai_integration',  # noqa: F822
    'vercel_ai_sdk_integration',  # noqa: F822
]


def __getattr__(name: str):
    # `from . import <name>` here would recurse: its `_handle_fromlist` calls
    # getattr on this package, which re-enters __getattr__ forever whenever the
    # sub-module is not already imported. `import_module` goes through the
    # import machinery instead, and binds the sub-module on this package itself,
    # so later lookups never reach __getattr__ again.
    if name in __all__:
        return importlib.import_module(f'.{name}', __name__)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
