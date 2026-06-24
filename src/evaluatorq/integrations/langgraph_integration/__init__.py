"""LangGraph integration for evaluatorq simulation and red teaming.

Provides a wrapper to use any LangGraph compiled graph as an AgentTarget.
"""

from .target import LangGraphTarget

__all__ = ["LangGraphTarget"]
