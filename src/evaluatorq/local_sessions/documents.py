"""ATIF documents for parsed local sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.trace_document import TraceDocument, TraceMetadata
from evaluatorq.formats import ResponsesConversation

if TYPE_CHECKING:
    from evaluatorq.local_sessions.models import ParsedSession


def session_document(parsed: ParsedSession) -> TraceDocument:
    """Convert a parsed session to ATIF through the shared Responses converter."""
    summary = parsed.summary
    trajectory = ResponsesConversation(items=parsed.items).to_atif(
        agent_name=summary.source,
        agent_version=summary.agent_version or 'unknown',
        session_id=summary.session_id,
    )
    tool_names = tuple(sorted({item['name'] for item in parsed.items if item.get('type') == 'function_call'}))
    duration = summary.updated_at - summary.started_at
    return TraceDocument(
        metadata=TraceMetadata(
            trace_id=f'{summary.source}:{summary.session_id}',
            span_id=summary.session_id,
            timestamp=summary.started_at,
            project=summary.project_dir,
            model=parsed.model,
            product='local-session',
            trace_type='agent-session',
            agent_name=summary.source,
            tool_names=tool_names,
            total_tokens=parsed.total_tokens,
            duration_ms=max(0, int(duration.total_seconds() * 1000)),
            capture_metadata={
                'source': f'local:{summary.source}',
                'start': summary.started_at.isoformat(),
                'end': summary.updated_at.isoformat(),
            },
        ),
        trajectory=trajectory,
    )
