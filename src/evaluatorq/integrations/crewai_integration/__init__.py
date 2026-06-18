"""CrewAI integration for evaluatorq simulation and red teaming.

Provides a wrapper to use a CrewAI ``Crew`` as a unified ``AgentTarget``.
"""

from .target import CrewAITarget

__all__ = ["CrewAITarget"]
