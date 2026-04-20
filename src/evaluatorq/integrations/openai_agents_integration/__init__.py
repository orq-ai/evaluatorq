"""OpenAI Agents SDK integration for evaluatorq red teaming.

Provides a wrapper to use any OpenAI Agents SDK agent as a red teaming target.
"""

from .target import OpenAIAgentTarget

__all__ = ["OpenAIAgentTarget"]
