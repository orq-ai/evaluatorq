"""Build local-only Jev compression previews and load one Orq trace safely."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from evaluatorq.common.model_input import jev_state, serialized_chars
from evaluatorq.common.orq_client import close_orq_client
from evaluatorq.common.trace_document import TraceDocument, ensure_trace_document, prompt_messages
from evaluatorq.dashboard.auth import build_orq_client
from evaluatorq.dashboard.trace_finder.routes import selected_dashboard_auth
from evaluatorq.insights.transcript import full_conversation_view_with_message_spans
from evaluatorq.trace_finder import FacetSelection, NumericFilters
from evaluatorq.trace_finder.orq_source import OrqTraceSource, require_complete_targeted_load

_SAFE_ORQ_ERROR = 'The selected Orq trace could not be loaded.'
_OMISSION_MARKER = re.compile(r'\[\.\.\. (\d+) messages left out \.\.\.\]')


@dataclass(frozen=True)
class CompressionPreview:
    """The uncapped readable conversation and its exact structured Jev state preview."""

    original: str
    compressed: dict[str, Any]
    original_chars: int
    compressed_chars: int
    message_count: int
    omitted_messages: int
    global_cap: int


class InspectorSourceError(ValueError):
    """A source failure whose text is fixed and cannot expose selected input or provider data."""

    def __init__(self, *_args: object) -> None:
        super().__init__(_SAFE_ORQ_ERROR)


def preview_document(document: TraceDocument, *, global_char_cap: int) -> CompressionPreview:
    """Render a local-only preview without mutating its canonical document or calling a model.

    The original is the uncapped readable transcript, not raw source bytes. Its shared renderer redacts known
    secrets and omits harness reminders. Jev state is independently scrubbed and may represent media with markers;
    neither view downloads media URLs or changes the document.
    """
    messages = prompt_messages(document)
    compressed = jev_state(messages, global_char_cap=global_char_cap)
    original, _, message_count = full_conversation_view_with_message_spans(document)
    marker = compressed.get('omission')
    omission = _OMISSION_MARKER.fullmatch(marker) if isinstance(marker, str) else None
    omitted_messages = int(omission.group(1)) if omission else 0
    return CompressionPreview(
        original=original,
        compressed=compressed,
        original_chars=len(original),
        compressed_chars=serialized_chars(compressed),
        message_count=message_count,
        omitted_messages=omitted_messages,
        global_cap=global_char_cap,
    )


def _window(start: datetime | None, end: datetime | None) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    if start is not None and (start.tzinfo is None or start.utcoffset() is None):
        raise InspectorSourceError(_SAFE_ORQ_ERROR)
    if end is not None and (end.tzinfo is None or end.utcoffset() is None):
        raise InspectorSourceError(_SAFE_ORQ_ERROR)
    resolved_end = end or now
    resolved_start = start or resolved_end - timedelta(days=7)
    if resolved_start > resolved_end:
        raise InspectorSourceError(_SAFE_ORQ_ERROR)
    return resolved_start, resolved_end


async def fetch_orq_document(
    app: Any,
    trace_id: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> TraceDocument | None:
    """Load one Orq document in the selected dashboard account and requested (or recent) time window.

    The default window is the last seven days; callers can supply explicit timezone-aware bounds for older traces.
    Authentication, provider, and targeted-load errors are replaced with one fixed message so trace identifiers,
    credential data, paths, and provider payloads never reach the inspector.
    """
    try:
        resolved_start, resolved_end = _window(start, end)
        auth = selected_dashboard_auth(app)
        settings = app.state.finder_settings
        client = build_orq_client(auth, workspace=settings.orq_workspace, project=settings.orq_project_id)
    except Exception:  # noqa: BLE001 — auth and SDK errors must never reach the inspector.
        raise InspectorSourceError(_SAFE_ORQ_ERROR) from None

    source: OrqTraceSource | None = None
    try:
        source = OrqTraceSource(client)
        snapshot = await source.load_async(
            resolved_start,
            resolved_end,
            limit=1,
            facets=FacetSelection(),
            numeric=NumericFilters(),
            target_trace_ids=frozenset({trace_id}),
        )
        require_complete_targeted_load(
            snapshot,
            error_type=InspectorSourceError,
            operation='Loading selected Orq trace',
        )
        record = next((item for item in snapshot.traces if item.trace_id == trace_id), None)
        return ensure_trace_document(record) if record is not None else None
    except Exception:  # noqa: BLE001 — source SDK errors may contain provider payloads or identifiers.
        raise InspectorSourceError(_SAFE_ORQ_ERROR) from None
    finally:
        if source is not None:
            try:
                source.close()
            except Exception as exc:  # noqa: BLE001 — log only type to avoid leaking source data.
                logger.warning('Compression inspector source cleanup failed ({})', type(exc).__name__)
        try:
            await close_orq_client(client)
        except Exception as exc:  # noqa: BLE001 — log only type to avoid leaking credentials or payloads.
            logger.warning('Compression inspector client cleanup failed ({})', type(exc).__name__)
