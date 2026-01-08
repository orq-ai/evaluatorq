"""
Tracing context utilities.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class TracingContext:
    """Context for tracing an evaluation run."""

    run_id: str
    """Unique identifier for the evaluation run"""

    run_name: str
    """Human-readable name for the evaluation run"""

    enabled: bool
    """Whether tracing is enabled"""

    parent_context: Any | None = None
    """Parent OTEL context, if any"""


def generate_run_id() -> str:
    """Generate a unique run ID for an evaluation run."""
    return str(uuid.uuid4())


async def capture_parent_context() -> Any | None:
    """
    Capture the current OTEL context as a parent context.
    Returns None if OTEL is not available.
    """
    try:
        from opentelemetry import context

        return context.get_current()
    except ImportError:
        return None
