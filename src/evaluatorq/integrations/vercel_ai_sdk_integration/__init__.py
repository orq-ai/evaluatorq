"""Vercel AI SDK integration for evaluatorq red teaming.

Provides a wrapper to use any Vercel AI SDK agent (served over HTTP)
as a red teaming target.
"""

from .target import VercelAISdkTarget

__all__ = ["VercelAISdkTarget"]
