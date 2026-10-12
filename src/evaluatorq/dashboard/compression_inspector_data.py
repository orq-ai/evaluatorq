"""Build local-only Jev compression previews and load one Orq trace safely."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

from loguru import logger

from evaluatorq.common.model_input import (
    DEFAULT_JEV_OPTIONS,
    EdgeLimit,
    JevOptions,
    cap_text,
    jev_state,
    omission_count,
    serialized_chars,
)
from evaluatorq.common.orq_client import close_orq_client
from evaluatorq.common.redact import scrub_known_secrets
from evaluatorq.common.trace_document import TraceDocument, ensure_trace_document, prompt_messages
from evaluatorq.dashboard.auth import build_orq_client
from evaluatorq.dashboard.trace_finder.routes import selected_dashboard_auth
from evaluatorq.insights.transcript import (
    DEFAULT_MESSAGE_KINDS,
    MESSAGE_KINDS,
    MessageKind,
    full_conversation_view,
    full_conversation_view_with_message_spans,
)
from evaluatorq.trace_finder import FacetSelection, NumericFilters
from evaluatorq.trace_finder.orq_source import OrqTraceSource, require_complete_targeted_load

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_SAFE_ORQ_ERROR = 'The selected Orq trace could not be loaded.'

MAX_KNOB_CHARS = 5_000_000
ToolsMode = Literal['collapsed', 'per_call']
TOOLS_MODES: tuple[ToolsMode, ...] = get_args(ToolsMode)
OriginalView = Literal['atif', 'source', 'readable']
ORIGINAL_VIEWS: tuple[OriginalView, ...] = get_args(OriginalView)
ORIGINAL_VIEW_LABELS: dict[OriginalView, str] = {
    'atif': 'ATIF trajectory',
    'source': 'raw source file',
    'readable': 'readable text',
}
# The original pane is cut to this many characters; a raw session file is read up to this many bytes.
ORIGINAL_CHARS = 2_000_000
SOURCE_BYTES = 5_000_000
# Posted (total, edge) field names of each Jev edge limit.
JEV_EDGE_FIELDS: dict[str, tuple[str, str]] = {
    'assistant': ('jev_assistant_total', 'jev_assistant_edge'),
    'user': ('jev_user_total', 'jev_user_edge'),
    'system': ('jev_system_total', 'jev_system_edge'),
    'tool_input': ('jev_tool_input_total', 'jev_tool_input_edge'),
    'tool_result': ('jev_tool_result_total', 'jev_tool_result_edge'),
}
JEV_FLAGS = ('system', 'developer', 'user', 'assistant', 'tool_calls', 'tool_results')
# Readable-view caps: `-1` keeps the part whole, and tool results may also be 0 (left out).
VIEW_CAP_FIELDS: dict[str, str] = {
    'view_system_chars': 'system_chars',
    'view_user_chars': 'user_chars',
    'view_assistant_edge': 'assistant_edge',
    'view_reasoning_chars': 'reasoning_chars',
    'view_tool_input_chars': 'tool_input_chars',
    'view_tool_max_chars': 'tool_max_chars',
}


_Choice = TypeVar('_Choice', bound=str)


class KnobError(ValueError):
    """An invalid knob value; the text names the field and its allowed range and never echoes the input."""


@dataclass(frozen=True)
class CompressionKnobs:
    """Every compression setting the inspector exposes: the Jev state limits and the readable-view options."""

    global_char_cap: int
    jev: JevOptions = DEFAULT_JEV_OPTIONS
    budget: int = 0
    include: frozenset[MessageKind] = DEFAULT_MESSAGE_KINDS
    system_chars: int = -1
    user_chars: int = -1
    assistant_edge: int = -1
    reasoning_chars: int = -1
    tools: ToolsMode = 'per_call'
    tool_input_chars: int = -1
    tool_max_chars: int = -1
    original_view: OriginalView = 'atif'
    wrap: bool = False
    coverage: bool = False

    @classmethod
    def defaults(cls, global_char_cap: int) -> CompressionKnobs:
        """Reproduce the original pane (every part whole) beside the classifier's Jev state."""
        return cls(global_char_cap=global_char_cap, budget=global_char_cap, coverage=True)

    def as_form(self) -> dict[str, int | str | bool]:
        """Flatten to the form fields the inspector posts; ``from_form`` reads them back."""
        values: dict[str, int | str | bool] = {
            'global_char_cap': self.global_char_cap,
            'jev_state_chars': self.jev.state_chars,
            'view_budget': self.budget,
            'view_tools': self.tools,
            'original_view': self.original_view,
            'wrap': self.wrap,
            'coverage': self.coverage,
        }
        for attribute, (total_name, edge_name) in JEV_EDGE_FIELDS.items():
            limit: EdgeLimit = getattr(self.jev, attribute)
            values[total_name], values[edge_name] = limit.total, limit.edge
        values.update({f'jev_include_{flag}': getattr(self.jev, f'include_{flag}') for flag in JEV_FLAGS})
        values.update({f'view_include_{kind}': kind in self.include for kind in sorted(MESSAGE_KINDS)})
        values.update({name: getattr(self, attribute) for name, attribute in VIEW_CAP_FIELDS.items()})
        return values

    @classmethod
    def from_form(cls, form: Mapping[str, Any]) -> CompressionKnobs:
        """Parse and validate the posted fields; every number must be present and in range."""
        jev = JevOptions(
            state_chars=_whole(form, 'jev_state_chars', 'Jev state limit'),
            assistant=_edge_limit(form, 'assistant'),
            user=_edge_limit(form, 'user'),
            system=_edge_limit(form, 'system'),
            tool_input=_edge_limit(form, 'tool_input'),
            tool_result=_edge_limit(form, 'tool_result'),
            include_system=_checked(form, 'jev_include_system'),
            include_developer=_checked(form, 'jev_include_developer'),
            include_user=_checked(form, 'jev_include_user'),
            include_assistant=_checked(form, 'jev_include_assistant'),
            include_tool_calls=_checked(form, 'jev_include_tool_calls'),
            include_tool_results=_checked(form, 'jev_include_tool_results'),
        )
        caps = {
            attribute: _whole(
                form,
                name,
                f'Readable view {attribute.replace("_", " ")}',
                minimum=0 if attribute == 'tool_max_chars' else 1,
                keep_whole=True,
            )
            for name, attribute in VIEW_CAP_FIELDS.items()
        }
        return cls(
            global_char_cap=_whole(form, 'global_char_cap', 'Global character cap'),
            jev=jev,
            budget=_whole(form, 'view_budget', 'Readable view budget'),
            include=frozenset(kind for kind in sorted(MESSAGE_KINDS) if _checked(form, f'view_include_{kind}')),
            tools=_choice(form, 'view_tools', 'Readable view tool display', TOOLS_MODES),
            original_view=_choice(form, 'original_view', 'Original shows', ORIGINAL_VIEWS),
            wrap=_checked(form, 'wrap'),
            coverage=_checked(form, 'coverage'),
            system_chars=caps['system_chars'],
            user_chars=caps['user_chars'],
            assistant_edge=caps['assistant_edge'],
            reasoning_chars=caps['reasoning_chars'],
            tool_input_chars=caps['tool_input_chars'],
            tool_max_chars=caps['tool_max_chars'],
        )


def _whole(form: Mapping[str, Any], name: str, label: str, *, minimum: int = 1, keep_whole: bool = False) -> int:
    """Read one whole number in range; `-1` (keep whole) is allowed only when `keep_whole`."""
    text = str(form.get(name) or '').strip()
    if re.fullmatch(r'-?\d{1,9}', text):
        value = int(text)
        if minimum <= value <= MAX_KNOB_CHARS or (keep_whole and value == -1):
            return value
    allowed = f'{minimum:,} to {MAX_KNOB_CHARS:,}'
    if keep_whole:
        allowed = f'-1 (whole) or {allowed}'
    raise KnobError(f'{label}: enter a whole number, {allowed}.')


def _checked(form: Mapping[str, Any], name: str) -> bool:
    return form.get(name) is not None


def _choice(form: Mapping[str, Any], name: str, label: str, options: tuple[_Choice, ...]) -> _Choice:
    text = str(form.get(name) or '')
    for option in options:
        if option == text:
            return option
    raise KnobError(f'{label}: choose one of the listed options.')


def _edge_limit(form: Mapping[str, Any], attribute: str) -> EdgeLimit:
    total_name, edge_name = JEV_EDGE_FIELDS[attribute]
    label = f'Jev {attribute.replace("_", " ")} limit'
    total = _whole(form, total_name, f'{label} total')
    edge = _whole(form, edge_name, f'{label} edge')
    if edge * 2 > total:
        raise KnobError(f'{label}: the edge must be at most half the total.')
    return EdgeLimit(total, edge)


@dataclass(frozen=True)
class CompressionPreview:
    """The readable view and the structured Jev state of one document under a set of knobs."""

    readable: str
    original: str
    original_view: OriginalView
    original_note: str
    compressed: dict[str, Any]
    original_chars: int
    transcript_chars: int
    readable_chars: int
    compressed_chars: int
    message_count: int
    omitted_messages: int
    global_cap: int


class InspectorSourceError(ValueError):
    """A source failure whose text is fixed and cannot expose selected input or provider data."""

    def __init__(self, *_args: object) -> None:
        super().__init__(_SAFE_ORQ_ERROR)


def preview_document(
    document: TraceDocument, *, knobs: CompressionKnobs, source_path: Path | None = None
) -> CompressionPreview:
    """Render a local-only preview without mutating its canonical document or calling a model.

    The original is the canonical ATIF JSON, the raw session file (`source_path`, local sessions only) or the
    readable text, as `knobs.original_view` says; every original is scrubbed of known secrets and cut to
    ``ORIGINAL_CHARS``. The readable text uses the Insights ``full_conversation_view`` options and the Jev state the
    classifier's Jev limits, and neither downloads media URLs or changes the document. The transcript character
    count is that of the uncapped readable transcript. `knobs.coverage` adds the opt-in coverage description to both
    compressed formats, a leading `[context: ...]` line in the readable text and a leading `context` object in the
    Jev state, and never to the original pane. Limits too small for either renderer, and a source that is
    unavailable, raise ``ValueError`` with a fixed message.
    """
    messages = prompt_messages(document)
    compressed = jev_state(
        messages, global_char_cap=knobs.global_char_cap, options=replace(knobs.jev, coverage=knobs.coverage)
    )
    full, _, message_count = full_conversation_view_with_message_spans(document)
    readable = full_conversation_view(
        document,
        knobs.budget,
        include=knobs.include,
        system_chars=knobs.system_chars,
        user_chars=knobs.user_chars,
        assistant_edge=knobs.assistant_edge,
        reasoning_chars=knobs.reasoning_chars,
        tools=knobs.tools,
        tool_input_chars=knobs.tool_input_chars,
        tool_max_chars=knobs.tool_max_chars,
        coverage_header=knobs.coverage,
    )
    original, note = _original(document, knobs.original_view, readable, source_path)
    return CompressionPreview(
        readable=readable,
        original=original,
        original_view=knobs.original_view,
        original_note=note,
        compressed=compressed,
        original_chars=len(original),
        transcript_chars=len(full),
        readable_chars=len(readable),
        compressed_chars=serialized_chars(compressed),
        message_count=message_count,
        omitted_messages=omission_count(compressed.get('omission')),
        global_cap=knobs.global_char_cap,
    )


def _original(document: TraceDocument, view: OriginalView, readable: str, source_path: Path | None) -> tuple[str, str]:
    """The scrubbed, capped text of the chosen original and a note on anything cut from it."""
    notes: list[str] = []
    if view == 'atif':
        text = document.model_dump_json(indent=2, exclude_none=True)
    elif view == 'readable':
        text = readable
    elif source_path is None:
        raise KnobError('Original shows: the raw source is only available for local sessions; choose another view.')
    else:
        try:
            with source_path.open('rb') as handle:
                data = handle.read(SOURCE_BYTES)
            total = source_path.stat().st_size
        except OSError:
            raise KnobError('Original shows: the source file could not be read.') from None
        if total > len(data):
            notes.append(f'Showing the first {len(data):,} of {total:,} bytes of the source file.')
        text = data.decode('utf-8', errors='replace')
    text = scrub_known_secrets(text)
    shown, omitted = cap_text(text, ORIGINAL_CHARS)
    if omitted:
        notes.append(f'Cut to {ORIGINAL_CHARS:,} characters; {omitted:,} characters are left out.')
    return shown, ' '.join(notes)


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
