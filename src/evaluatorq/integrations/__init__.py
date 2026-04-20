# ruff: noqa: F401
"""Integration modules for evaluatorq.

Available integrations:
- langchain_integration: LangChain agent wrapper for OpenResponses format
- langgraph_integration: LangGraph red teaming target
- openai_agents_integration: OpenAI Agents SDK red teaming target
- vercel_ai_sdk_integration: Vercel AI SDK red teaming target (HTTP)
- callable_integration: Custom callable red teaming target

Integrations with optional dependencies (langgraph, openai-agents) use lazy
imports so that importing this package does not fail when those libraries are
not installed.
"""

from . import langchain_integration

__all__ = [
    "callable_integration",
    "langchain_integration",
    "langgraph_integration",
    "openai_agents_integration",
    "vercel_ai_sdk_integration",
]


def __getattr__(name: str):  # noqa: ANN202
    if name == "langgraph_integration":
        from . import langgraph_integration

        return langgraph_integration
    if name == "openai_agents_integration":
        from . import openai_agents_integration

        return openai_agents_integration
    if name == "vercel_ai_sdk_integration":
        from . import vercel_ai_sdk_integration

        return vercel_ai_sdk_integration
    if name == "callable_integration":
        from . import callable_integration

        return callable_integration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
