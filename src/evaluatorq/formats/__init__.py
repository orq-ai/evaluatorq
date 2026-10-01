"""Convert an agent run between chat, Responses, OTel and ATIF."""

from __future__ import annotations

from evaluatorq.formats.atif import AtifTrajectory
from evaluatorq.formats.chat import ChatConversation
from evaluatorq.formats.otel import OtelSpan, OtelTrace
from evaluatorq.formats.responses import ResponsesConversation

__all__ = [
    'AtifTrajectory',
    'ChatConversation',
    'OtelSpan',
    'OtelTrace',
    'ResponsesConversation',
]
