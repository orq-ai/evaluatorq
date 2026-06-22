"""Vercel AI SDK integration for evaluatorq simulation and red teaming.

Provides a wrapper to use any Vercel AI SDK agent (served over HTTP)
as an AgentTarget (usable in simulation and red teaming).
"""

from .target import VercelAISdkTarget

__all__ = ["VercelAISdkTarget"]
