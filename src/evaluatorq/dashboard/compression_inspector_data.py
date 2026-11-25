"""Build local-only decision-state compression previews and load one Orq trace safely."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

from loguru import logger

from evaluatorq.common.messages import coerce_content_text, reasoning_text
from evaluatorq.common.model_input import (
    DEFAULT_DECISION_OPTIONS,
    DecisionOptions,
    EdgeLimit,
    cap_text,
    decision_state,
    serialized_chars,
    state_omitted_messages,
)
from evaluatorq.common.orq_client import close_orq_client
from evaluatorq.common.redact import scrub_known_secrets
from evaluatorq.common.trace_document import TraceDocument, ensure_trace_document, to_chat_messages
from evaluatorq.dashboard.auth import build_orq_client
from evaluatorq.dashboard.trace_finder.routes import selected_dashboard_auth
from evaluatorq.insights.readable_style import ReadableStyle
from evaluatorq.insights.transcript import (
    MESSAGE_KINDS,
    WHOLE,
    MessageKind,
    full_conversation_view,
    full_conversation_view_entries,
    full_conversation_view_with_message_spans,
)
from evaluatorq.trace_finder import FacetSelection, NumericFilters
from evaluatorq.trace_finder.orq_source import OrqTraceSource, require_complete_targeted_load

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

_SAFE_ORQ_ERROR = 'The selected Orq trace could not be loaded.'

MAX_KNOB_CHARS = 5_000_000
ToolsMode = Literal['collapsed', 'per_call']
TOOLS_MODES: tuple[ToolsMode, ...] = get_args(ToolsMode)
OriginalView = Literal['atif', 'source', 'readable']
ORIGINAL_VIEWS: tuple[OriginalView, ...] = get_args(OriginalView)
BarView = Literal['types', 'messages']
BAR_VIEWS: tuple[BarView, ...] = get_args(BarView)
ORIGINAL_VIEW_LABELS: dict[OriginalView, str] = {
    'atif': 'ATIF trajectory',
    'source': 'raw source file',
    'readable': 'readable text',
}
# The original pane is cut to this many characters; a raw session file is read up to this many bytes.
ORIGINAL_CHARS = 2_000_000
SOURCE_BYTES = 5_000_000
# Posted (total, edge) field names of each decision-state edge limit.
DECISION_EDGE_FIELDS: dict[str, tuple[str, str]] = {
    'assistant': ('decision_assistant_total', 'decision_assistant_edge'),
    'user': ('decision_user_total', 'decision_user_edge'),
    'system': ('decision_system_total', 'decision_system_edge'),
    'tool_input': ('decision_tool_input_total', 'decision_tool_input_edge'),
    'tool_result': ('decision_tool_result_total', 'decision_tool_result_edge'),
}
DECISION_FLAGS = ('system', 'developer', 'user', 'assistant', 'tool_calls', 'tool_results')
DECISION_FORMAT_FLAGS = ('located_omission', 'field_order', 'drop_empty')
READABLE_STYLE_FLAGS = (
    'blocks',
    'pair_results',
    'plain_tool_input',
    'labelled_cuts',
    'strip_terminal_noise',
    'shorten_cwd',
    'collapse_repeats',
    'collapse_blobs',
)
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
    """Every compression setting the inspector exposes: the decision state limits and the readable-view options."""

    global_char_cap: int
    decision: DecisionOptions = DEFAULT_DECISION_OPTIONS
    budget: int = 0
    include: frozenset[MessageKind] = WHOLE.include
    system_chars: int = -1
    user_chars: int = -1
    assistant_edge: int = -1
    reasoning_chars: int = -1
    tools: ToolsMode = 'per_call'
    tool_input_chars: int = -1
    tool_max_chars: int = -1
    original_view: OriginalView = 'atif'
    wrap: bool = False
    bar_view: BarView = 'types'
    coverage: bool = False
    style: ReadableStyle = field(default_factory=ReadableStyle)

    @classmethod
    def defaults(cls, global_char_cap: int) -> CompressionKnobs:
        """Enable inspector-only formatting beside the classifier's existing part limits."""
        return cls(
            global_char_cap=global_char_cap,
            budget=global_char_cap,
            coverage=True,
            decision=replace(DEFAULT_DECISION_OPTIONS, **dict.fromkeys(DECISION_FORMAT_FLAGS, True)),
            style=ReadableStyle(**dict.fromkeys(READABLE_STYLE_FLAGS, True)),
        )

    def as_form(self) -> dict[str, int | str | bool]:
        """Flatten to the form fields the inspector posts; ``from_form`` reads them back."""
        values: dict[str, int | str | bool] = {
            'global_char_cap': self.global_char_cap,
            'decision_state_chars': self.decision.state_chars,
            'view_budget': self.budget,
            'view_tools': self.tools,
            'original_view': self.original_view,
            'wrap': self.wrap,
            'bar_view': self.bar_view,
            'coverage': self.coverage,
        }
        for attribute, (total_name, edge_name) in DECISION_EDGE_FIELDS.items():
            limit: EdgeLimit = getattr(self.decision, attribute)
            values[total_name], values[edge_name] = limit.total, limit.edge
        values.update({
            f'decision_include_{flag}': getattr(self.decision, f'include_{flag}') for flag in DECISION_FLAGS
        })
        values.update({f'decision_{flag}': getattr(self.decision, flag) for flag in DECISION_FORMAT_FLAGS})
        values.update({f'view_{flag}': getattr(self.style, flag) for flag in READABLE_STYLE_FLAGS})
        values.update({f'view_include_{kind}': kind in self.include for kind in sorted(MESSAGE_KINDS)})
        values.update({name: getattr(self, attribute) for name, attribute in VIEW_CAP_FIELDS.items()})
        return values

    @classmethod
    def from_form(cls, form: Mapping[str, Any]) -> CompressionKnobs:
        """Parse and validate the posted fields; every number must be present and in range."""
        decision = DecisionOptions(
            state_chars=_whole(form, 'decision_state_chars', 'Decision state limit'),
            assistant=_edge_limit(form, 'assistant'),
            user=_edge_limit(form, 'user'),
            system=_edge_limit(form, 'system'),
            tool_input=_edge_limit(form, 'tool_input'),
            tool_result=_edge_limit(form, 'tool_result'),
            include_system=_checked(form, 'decision_include_system'),
            include_developer=_checked(form, 'decision_include_developer'),
            include_user=_checked(form, 'decision_include_user'),
            include_assistant=_checked(form, 'decision_include_assistant'),
            include_tool_calls=_checked(form, 'decision_include_tool_calls'),
            include_tool_results=_checked(form, 'decision_include_tool_results'),
            **{flag: _checked(form, f'decision_{flag}') for flag in DECISION_FORMAT_FLAGS},
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
            decision=decision,
            budget=_whole(form, 'view_budget', 'Readable view budget'),
            include=frozenset(kind for kind in sorted(MESSAGE_KINDS) if _checked(form, f'view_include_{kind}')),
            tools=_choice(form, 'view_tools', 'Readable view tool display', TOOLS_MODES),
            original_view=_choice(form, 'original_view', 'Original shows', ORIGINAL_VIEWS),
            wrap=_checked(form, 'wrap'),
            # The bar switch sits beside the bars, which an error notice replaces, so a missing value means the default.
            bar_view=_choice(form, 'bar_view', 'Size bars show', BAR_VIEWS) if form.get('bar_view') else 'types',
            coverage=_checked(form, 'coverage'),
            style=ReadableStyle(**{flag: _checked(form, f'view_{flag}') for flag in READABLE_STYLE_FLAGS}),
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
    total_name, edge_name = DECISION_EDGE_FIELDS[attribute]
    label = f'Decision state {attribute.replace("_", " ")} limit'
    total = _whole(form, total_name, f'{label} total')
    edge = _whole(form, edge_name, f'{label} edge')
    if edge * 2 > total:
        raise KnobError(f'{label}: the edge must be at most half the total.')
    return EdgeLimit(total, edge)


@dataclass(frozen=True)
class CompressionPreview:
    """The readable view and the structured decision state of one document under a set of knobs."""

    readable: str
    original: str
    original_view: OriginalView
    original_note: str
    compressed: dict[str, Any]
    original_chars: int
    full_json_chars: int
    transcript_chars: int
    readable_chars: int
    compressed_chars: int
    message_count: int
    omitted_messages: int
    global_cap: int
    readable_body_present: bool = True
    layers: tuple[CompressionLayer, ...] = ()

    @property
    def readable_reduction_percent(self) -> float:
        """Character reduction against the complete compact trace JSON; negative means expansion."""
        return 100 * (1 - self.readable_chars / self.full_json_chars)

    @property
    def decision_reduction_percent(self) -> float:
        """Character reduction against the complete compact trace JSON; negative means expansion."""
        return 100 * (1 - self.compressed_chars / self.full_json_chars)


# Segment order in the size bars; `other` holds what belongs to no message: structure, metadata, headers, markers.
LayerKind = Literal['system', 'user', 'assistant', 'reasoning', 'tool_input', 'tool_result', 'other']
LAYER_KINDS: tuple[LayerKind, ...] = get_args(LayerKind)
LAYER_KIND_LABELS: dict[LayerKind, str] = {
    'system': 'System',
    'user': 'User',
    'assistant': 'Assistant',
    'reasoning': 'Reasoning',
    'tool_input': 'Tool call',
    'tool_result': 'Tool result',
    'other': 'Structure and metadata',
}
_ROLE_LAYER_KINDS: dict[str, LayerKind] = {
    'user': 'user',
    'assistant': 'assistant',
    'tool': 'tool_result',
    'function': 'tool_result',
}
# The per-message bar merges neighbouring messages of one type; past this many runs it merges neighbours of any type.
MAX_LAYER_RUNS = 240
PREVIEW_CHARS = 120


@dataclass(frozen=True)
class LayerEntry:
    """One measured piece of a layer: the source message it came from, its message type and its characters."""

    index: int
    kind: LayerKind
    chars: int


@dataclass(frozen=True)
class LayerRun:
    """Neighbouring entries of a layer drawn as one segment of the per-message bar.

    `kind` is the type with the most characters in the run and `kinds` the characters of every type in it, so a run
    merged past ``MAX_LAYER_RUNS`` still says what it holds. `preview` is the scrubbed start of its first entry's
    source text.
    """

    kind: LayerKind
    first: int
    last: int
    chars: int
    kinds: tuple[tuple[LayerKind, int], ...]
    preview: str = ''


@dataclass(frozen=True)
class CompressionLayer:
    """One representation of the trace: its size, its characters by message type, and its messages in order."""

    label: str
    total: int
    segments: tuple[tuple[LayerKind, int], ...]
    runs: tuple[LayerRun, ...] = ()
    estimated: bool = False

    def messages_of(self, kind: LayerKind) -> int:
        """How many source messages contribute characters of `kind` to this layer."""
        return len({
            index for run in self.runs for index in range(run.first, run.last + 1) if dict(run.kinds).get(kind)
        })


def _layer(
    label: str, total: int, entries: Iterable[LayerEntry], previews: Mapping[tuple[int, LayerKind], str]
) -> CompressionLayer:
    """Measure a layer from its entries and give what they do not cover to `other`, so the segments sum to `total`."""
    measured = [entry for entry in entries if entry.chars > 0]
    covered = sum(entry.chars for entry in measured)
    estimated = covered > total
    if estimated:
        # The view was cut after rendering: share the kept characters out in the measured proportions.
        measured = [replace(entry, chars=entry.chars * total // covered) for entry in measured]
        measured = [entry for entry in measured if entry.chars > 0]
    sizes: dict[LayerKind, int] = dict.fromkeys(LAYER_KINDS, 0)
    for entry in measured:
        sizes[entry.kind] += entry.chars
    sizes['other'] = total - sum(sizes.values())
    return CompressionLayer(
        label=label,
        total=total,
        segments=tuple((kind, sizes[kind]) for kind in LAYER_KINDS if sizes[kind]),
        runs=_runs(measured, previews),
        estimated=estimated,
    )


def _runs(entries: list[LayerEntry], previews: Mapping[tuple[int, LayerKind], str]) -> tuple[LayerRun, ...]:
    """Merge neighbouring entries of one type, then neighbours of any type until at most ``MAX_LAYER_RUNS`` are left."""
    groups: list[list[LayerEntry]] = []
    for entry in entries:
        if groups and groups[-1][-1].kind == entry.kind:
            groups[-1].append(entry)
        else:
            groups.append([entry])
    if len(groups) > MAX_LAYER_RUNS:
        size = -(-len(groups) // MAX_LAYER_RUNS)
        groups = [
            [entry for group in groups[start : start + size] for entry in group]
            for start in range(0, len(groups), size)
        ]
    runs: list[LayerRun] = []
    for group in groups:
        kinds: dict[LayerKind, int] = {}
        for entry in group:
            kinds[entry.kind] = kinds.get(entry.kind, 0) + entry.chars
        first = group[0]
        runs.append(
            LayerRun(
                kind=max(kinds, key=lambda kind: kinds[kind]),
                first=min(entry.index for entry in group),
                last=max(entry.index for entry in group),
                chars=sum(kinds.values()),
                kinds=tuple((kind, kinds[kind]) for kind in LAYER_KINDS if kind in kinds),
                preview=previews.get((first.index, first.kind), ''),
            )
        )
    return tuple(runs)


def _preview_text(text: str) -> str:
    """The scrubbed start of a text on one line, for a hover card."""
    flat = ' '.join(scrub_known_secrets(text[: PREVIEW_CHARS * 4]).split())
    return flat if len(flat) <= PREVIEW_CHARS else f'{flat[: PREVIEW_CHARS - 1]}…'


def _role_kind(role: object) -> LayerKind:
    return _ROLE_LAYER_KINDS.get(str(role or ''), 'system')


def _arguments_text(call: dict[str, Any]) -> str:
    function = call.get('function')
    raw = function.get('arguments') if isinstance(function, dict) else call.get('arguments')
    return raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)


def message_entries(messages: list[dict[str, Any]]) -> tuple[list[LayerEntry], dict[tuple[int, LayerKind], str]]:
    """Measure text, reasoning, tool-call arguments and tool results in chat messages, with a preview of each."""
    entries: list[LayerEntry] = []
    previews: dict[tuple[int, LayerKind], str] = {}

    def add(index: int, kind: LayerKind, text: str) -> None:
        if text:
            entries.append(LayerEntry(index=index, kind=kind, chars=len(text)))
            previews.setdefault((index, kind), _preview_text(text))

    for index, message in enumerate(messages):
        add(index, 'reasoning', reasoning_text(message) or '')
        add(index, _role_kind(message.get('role')), coerce_content_text(message.get('content')))
        for call in message.get('tool_calls') or []:
            if isinstance(call, dict):
                add(index, 'tool_input', _arguments_text(call))
    return entries, previews


def decision_entries(state: Mapping[str, Any]) -> list[LayerEntry]:
    """Measure message text, tool inputs and tool results in a decision state, in its order."""
    entries: list[LayerEntry] = []
    for entry in state.get('messages') or []:
        index, kind = int(entry.get('index', 0)), entry.get('type')
        if kind == 'message':
            entries.append(
                LayerEntry(index=index, kind=_role_kind(entry.get('role')), chars=len(str(entry.get('text') or '')))
            )
        elif kind == 'tool_call':
            entries.append(LayerEntry(index=index, kind='tool_input', chars=len(str(entry.get('input') or ''))))
            entries.extend(
                LayerEntry(
                    index=int(result.get('index', index)), kind='tool_result', chars=len(str(result.get('text') or ''))
                )
                for result in entry.get('results') or []
            )
        elif kind == 'orphan_result':
            entries.append(LayerEntry(index=index, kind='tool_result', chars=len(str(entry.get('text') or ''))))
    return entries


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
    ``ORIGINAL_CHARS``. The readable text uses the Insights ``full_conversation_view`` options and the decision state the
    classifier's decision-state limits, and neither downloads media URLs or changes the document. The transcript character
    count is that of the uncapped readable transcript. `knobs.coverage` adds the opt-in coverage description to both
    compressed formats, a leading `[context: ...]` line in the readable text and a leading `context` object in the
    decision state, and never to the original pane. Limits too small for either renderer, and a source that is
    unavailable, raise ``ValueError`` with a fixed message.
    """
    messages = to_chat_messages(document)
    compressed = decision_state(
        messages, global_char_cap=knobs.global_char_cap, options=replace(knobs.decision, coverage=knobs.coverage)
    )
    full, _, message_count = full_conversation_view_with_message_spans(document)

    view_options: dict[str, Any] = {
        'options': WHOLE,
        'include': knobs.include,
        'system_chars': knobs.system_chars,
        'user_chars': knobs.user_chars,
        'assistant_edge': knobs.assistant_edge,
        'reasoning_chars': knobs.reasoning_chars,
        'tools': knobs.tools,
        'tool_input_chars': knobs.tool_input_chars,
        'tool_max_chars': knobs.tool_max_chars,
        'style': knobs.style,
    }

    def render_readable(*, coverage: bool) -> str:
        return full_conversation_view(document, budget=knobs.budget, coverage_header=coverage, **view_options)

    body = render_readable(coverage=False)
    readable = render_readable(coverage=True) if knobs.coverage else body
    original, note = _original(document, knobs.original_view, readable, source_path)
    full_json_chars = len(document.model_dump_json())
    compressed_chars = serialized_chars(compressed)
    source_entries, previews = message_entries(messages)
    view_entries = [
        LayerEntry(index=entry.index, kind=entry.kind, chars=entry.chars)
        for entry in full_conversation_view_entries(document, **view_options)
    ]
    layers = (
        _layer('Full trace JSON', full_json_chars, source_entries, previews),
        _layer('Readable view', len(readable), view_entries, previews),
        _layer('Decision state', compressed_chars, decision_entries(compressed), previews),
    )
    return CompressionPreview(
        readable=readable,
        original=original,
        original_view=knobs.original_view,
        original_note=note,
        compressed=compressed,
        original_chars=len(original),
        full_json_chars=full_json_chars,
        transcript_chars=len(full),
        readable_chars=len(readable),
        compressed_chars=compressed_chars,
        message_count=message_count,
        omitted_messages=state_omitted_messages(compressed),
        global_cap=knobs.global_char_cap,
        readable_body_present=bool(body),
        layers=layers,
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
