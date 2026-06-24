"""Custom callable integration for evaluatorq simulation and red teaming.

Provides a wrapper to use any sync or async function as an AgentTarget.
"""

from .target import CallableTarget

__all__ = ["CallableTarget"]
