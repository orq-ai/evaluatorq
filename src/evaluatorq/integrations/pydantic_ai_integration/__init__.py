"""Pydantic AI integration for evaluatorq simulation and red teaming.

Provides a wrapper to use any Pydantic AI ``Agent`` as a unified ``AgentTarget``.
"""

from .target import PydanticAITarget

__all__ = ["PydanticAITarget"]
