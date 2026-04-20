"""Custom callable integration for evaluatorq red teaming.

Provides a wrapper to use any sync or async function as a red teaming target.
"""

from .target import CallableTarget

__all__ = ["CallableTarget"]
