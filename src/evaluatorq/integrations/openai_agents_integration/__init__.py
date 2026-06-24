"""OpenAI Agents SDK integration for evaluatorq simulation and red teaming.

Provides a wrapper to use any OpenAI Agents SDK agent as an AgentTarget.
"""

from .target import OpenAIAgentTarget

__all__ = ["OpenAIAgentTarget"]
