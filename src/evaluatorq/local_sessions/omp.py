"""omp session reader (`$PI_CODING_AGENT_DIR/sessions/<project>/<timestamp>_<id>.jsonl`)."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.local_sessions import _jsonl, items
from evaluatorq.local_sessions.models import ParsedSession, SessionFamily, SessionLoadError, SessionSummary

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_HEAD_BYTES = 262_144
_TAIL_BYTES = 65_536
_TEXT_CHARS = 200
_USER_INVOKED = ('skill-prompt', 'collab-prompt')
# Record types that carry no conversation content. They are dropped without a warning.
_DROPPED = frozenset({
    'title',
    'session',
    'custom',
    'model_usage',
    'mode_change',
    'ttsr_injection',
    'model_change',
    'thinking_level_change',
    'title_change',
    'credential_pin',
})


def _message(record: dict[str, Any]) -> dict[str, Any]:
    message = record.get('message')
    return message if isinstance(message, dict) else {}


class _OmpReader:
    family: SessionFamily = 'omp'

    def iter_files(self, root: Path) -> Iterator[Path]:
        yield from root.glob('*/*.jsonl')

    def is_session_path(self, path: Path, root: Path) -> bool:
        return path.suffix == '.jsonl' and path.parent.parent == root

    def summarize(self, path: Path) -> SessionSummary | None:
        try:
            head = _jsonl.read_head(path, max_bytes=_HEAD_BYTES)
            session = next((record for record in head[:2] if record.get('type') == 'session'), None)
            if session is None or session.get('parentSession'):
                return None
            first_prompt = next(
                (
                    text
                    for record in head
                    if record.get('type') == 'message'
                    and (message := _message(record)).get('role') == 'user'
                    and (text := items.blocks_text(message.get('content')))
                ),
                None,
            )
            if first_prompt is None:
                return None
            tail = _jsonl.read_tail(path, max_bytes=_TAIL_BYTES)
            stat = path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            title_record = head[0] if head[0].get('type') == 'title' else {}
            title = next(
                (
                    value
                    for value in (title_record.get('title'), session.get('title'))
                    if isinstance(value, str) and value
                ),
                '',
            )
            session_id = session.get('id')
            if not isinstance(session_id, str) or not session_id:
                session_id = path.stem
            updated = next(
                (record['timestamp'] for record in reversed(tail) if isinstance(record.get('timestamp'), str)), ''
            )
            return SessionSummary(
                source='omp',
                path=path.absolute(),
                session_id=session_id,
                title=title[:_TEXT_CHARS],
                project_dir=session['cwd'] if isinstance(session.get('cwd'), str) else '',
                started_at=_jsonl.timestamp_or(session.get('timestamp'), mtime),
                updated_at=_jsonl.timestamp_or(updated, mtime),
                size_bytes=stat.st_size,
                first_prompt=first_prompt.strip()[:_TEXT_CHARS],
                agent_version=session['version'] if isinstance(session.get('version'), str) else '',
            )
        except SessionLoadError:
            raise
        except (OSError, ValueError) as exc:
            raise SessionLoadError(f'{path}: unreadable session ({type(exc).__name__})') from exc

    def parse(self, path: Path) -> ParsedSession:
        summary = self.summarize(path)
        if summary is None:
            raise SessionLoadError(f'{path}: not a main session')
        try:
            records, skipped = _jsonl.read_records(path)
        except OSError as exc:
            raise SessionLoadError(f'{path}: unreadable session ({type(exc).__name__})') from exc
        _jsonl.warn_skipped(path, skipped)
        converted, model, total_tokens = _map_records(_chain(records), path)
        if not converted:
            raise SessionLoadError(f'{path}: no conversation items')
        return ParsedSession(
            summary=summary, items=converted, model=model, total_tokens=total_tokens, skipped_lines=skipped
        )


READER = _OmpReader()


def _chain(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records from the root to the last entry with an `id`: omp branches only when the user rewinds."""
    by_id: dict[str, dict[str, Any]] = {}
    leaf: str | None = None
    for record in records:
        record_id = record.get('id')
        if isinstance(record_id, str):
            by_id[record_id] = record
            leaf = record_id
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor = leaf
    while cursor is not None and cursor in by_id and cursor not in seen:
        seen.add(cursor)
        chain.append(by_id[cursor])
        parent = by_id[cursor].get('parentId')
        cursor = parent if isinstance(parent, str) else None
    chain.reverse()
    return chain


def _map_records(records: list[dict[str, Any]], path: Path) -> tuple[list[dict[str, Any]], str, int | None]:
    out: list[dict[str, Any]] = []
    model = ''
    total_tokens: int | None = None
    skipped_types: Counter[str] = Counter()
    for record in records:
        kind = record.get('type')
        if kind == 'message':
            message = _message(record)
            _map_message(message, out, skipped_types)
            if message.get('role') == 'assistant':
                if isinstance(message.get('model'), str) and message['model']:
                    provider = message.get('provider')
                    model = (
                        f'{provider}/{message["model"]}' if isinstance(provider, str) and provider else message['model']
                    )
                usage = message.get('usage')
                total = usage.get('totalTokens') if isinstance(usage, dict) else None
                if isinstance(total, int):
                    total_tokens = (total_tokens or 0) + total
        elif kind == 'compaction':
            summary = record.get('summary')
            out.append(items.compaction_summary(summary if isinstance(summary, str) else ''))
        elif kind == 'reset_boundary':
            out.append(items.system_text('Conversation context was reset.'))
        elif kind == 'custom_message':
            _map_custom_message(record, out)
        elif kind not in _DROPPED:
            skipped_types[kind if isinstance(kind, str) else 'unknown'] += 1
    if skipped_types:
        logger.warning('Skipped omp record types {} in {}', dict(skipped_types), path)
    return out, model, total_tokens


def _map_custom_message(record: dict[str, Any], out: list[dict[str, Any]]) -> None:
    custom_type = record.get('customType')
    text = items.blocks_text(record.get('content'))
    if not text.strip():
        return
    if custom_type in _USER_INVOKED:
        out.append(items.user_text(text))
    elif record.get('display') is True:
        out.append(items.system_text(items.trim(f'[{custom_type}]\n{text}')))


def _map_message(message: dict[str, Any], out: list[dict[str, Any]], skipped_types: Counter[str]) -> None:
    role = message.get('role')
    content = message.get('content')
    if role == 'user':
        text = items.blocks_text(content)
        if text:
            out.append(items.user_text(text))
    elif role == 'assistant':
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict):
                _map_assistant_block(block, out)
    elif role == 'toolResult':
        call_id = message.get('toolCallId')
        if isinstance(call_id, str):
            text = items.blocks_text(content)
            out.append(
                items.function_call_output(
                    call_id=call_id, output=f'[error] {text}' if message.get('isError') is True else text
                )
            )
    elif role == 'bashExecution':
        if message.get('excludeFromContext') is not True:
            command, output = message.get('command'), message.get('output')
            out.append(
                items.user_text(
                    items.trim(
                        f'$ {command if isinstance(command, str) else ""}\n{output if isinstance(output, str) else ""}'
                    )
                )
            )
    elif role != 'developer':
        skipped_types['message'] += 1


def _map_assistant_block(block: dict[str, Any], out: list[dict[str, Any]]) -> None:
    kind = block.get('type')
    if kind == 'thinking':
        thinking = block.get('thinking')
        if isinstance(thinking, str) and thinking:
            out.append(items.reasoning(thinking))
    elif kind == 'text':
        text = block.get('text')
        if isinstance(text, str) and text:
            out.append(items.assistant_text(text))
    elif kind == 'toolCall' and isinstance(block.get('id'), str) and isinstance(block.get('name'), str):
        out.append(items.function_call(call_id=block['id'], name=block['name'], arguments=block.get('arguments')))
