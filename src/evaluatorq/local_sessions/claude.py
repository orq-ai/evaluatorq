"""Claude Code and Claude desktop session reader (`~/.claude/projects` and the Cowork store)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from evaluatorq.contracts import tool_result_to_text
from evaluatorq.formats._shared import parse_iso
from evaluatorq.local_sessions import _jsonl, items
from evaluatorq.local_sessions.models import (
    ParsedSession,
    SessionFamily,
    SessionLoadError,
    SessionSource,
    SessionSummary,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_COWORK_ROOT = 'local-agent-mode-sessions'
_HEAD_TAIL_BYTES = 262_144
_TEXT_CHARS = 200
_DESKTOP_ENTRYPOINTS = frozenset({'claude-desktop', 'local-agent'})
_CONVERSATION = frozenset({'user', 'assistant'})


def _content_blocks(record: dict[str, Any]) -> object:
    message = record.get('message')
    return message.get('content') if isinstance(message, dict) else None


def _prompt_text(record: dict[str, Any]) -> str | None:
    """Text of a human-typed user line; `None` for meta, compaction and tool-result carriers."""
    if record.get('type') != 'user' or record.get('isMeta') or record.get('isCompactSummary'):
        return None
    content = _content_blocks(record)
    if isinstance(content, str):
        return content or None
    if isinstance(content, list) and any(
        isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str)
        for block in content
    ):
        return items.blocks_text([
            block for block in content if isinstance(block, dict) and block.get('type') != 'tool_result'
        ])
    return None


def _first(records: list[dict[str, Any]], key: str) -> str:
    for record in records:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return ''


def _last_value(records: list[dict[str, Any]], kind: str, key: str) -> str:
    for record in reversed(records):
        if record.get('type') == kind and isinstance(record.get(key), str) and record[key]:
            return record[key]
    return ''


class _ClaudeReader:
    family: SessionFamily = 'claude'

    def iter_files(self, root: Path) -> Iterator[Path]:
        pattern = '**/.claude/projects/*/*.jsonl' if root.name == _COWORK_ROOT else '*/*.jsonl'
        for path in root.glob(pattern):
            if self.is_session_path(path, root):
                yield path

    def is_session_path(self, path: Path, root: Path) -> bool:
        if path.suffix != '.jsonl' or 'subagents' in path.parts:
            return False
        projects = path.parent.parent
        if root.name == _COWORK_ROOT:
            return projects.name == 'projects' and '.claude' in path.parts
        return projects == root

    def summarize(self, path: Path) -> SessionSummary | None:
        try:
            head = _jsonl.read_head(path, max_bytes=_HEAD_TAIL_BYTES)
            tail = _jsonl.read_tail(path, max_bytes=_HEAD_TAIL_BYTES)
            stat = path.stat()
            first_line = next((record for record in head if record.get('type') in _CONVERSATION), None)
            if first_line is None or first_line.get('isSidechain') is True:
                return None
            first_prompt = next((text for record in head if (text := _prompt_text(record))), None)
            if first_prompt is None:
                return None
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            started = _first(head, 'timestamp')
            updated = next(
                (record['timestamp'] for record in reversed(tail) if isinstance(record.get('timestamp'), str)), ''
            )
            source: SessionSource = (
                'claude-desktop'
                if any(record.get('entrypoint') in _DESKTOP_ENTRYPOINTS for record in (*head, *tail))
                else 'claude-code'
            )
            return SessionSummary(
                source=source,
                path=path.absolute(),
                session_id=_first(head, 'sessionId') or path.stem,
                title=(_last_value(tail, 'custom-title', 'customTitle') or _last_value(tail, 'ai-title', 'aiTitle'))[
                    :_TEXT_CHARS
                ],
                project_dir=_first(head, 'cwd'),
                started_at=parse_iso(started) if started else mtime,
                updated_at=parse_iso(updated) if updated else mtime,
                size_bytes=stat.st_size,
                first_prompt=first_prompt.strip()[:_TEXT_CHARS],
                agent_version=_first(head, 'version'),
            )
        except (OSError, ValueError) as exc:
            if isinstance(exc, SessionLoadError):
                raise
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
        kept = _active_branch(records)
        converted, model, total_tokens = _map_records(kept)
        if not converted:
            raise SessionLoadError(f'{path}: no conversation items')
        return ParsedSession(
            summary=summary, items=converted, model=model, total_tokens=total_tokens, skipped_lines=skipped
        )


READER = _ClaudeReader()


def _parent_of(record: dict[str, Any]) -> str | None:
    parent = record.get('parentUuid')
    if isinstance(parent, str):
        return parent
    logical = record.get('logicalParentUuid')
    if record.get('type') == 'system' and parent is None and isinstance(logical, str):
        return logical
    return None


def _active_branch(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records on the live branch: rewound forks are dropped, parallel tool-call forks are kept whole."""
    index: dict[str, int] = {}
    for position, record in enumerate(records):
        uuid = record.get('uuid')
        if isinstance(uuid, str):
            index[uuid] = position
    if not index:
        return []
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for uuid in index:
        parent = _parent_of(records[index[uuid]])
        if parent is not None and parent in index:
            children.setdefault(parent, []).append(uuid)
        else:
            roots.append(uuid)

    leaf = next(
        (
            record['leafUuid']
            for record in reversed(records)
            if record.get('type') == 'last-prompt' and record.get('leafUuid') in index
        ),
        None,
    )
    head = leaf if leaf is not None else max(index, key=index.__getitem__)
    ancestors: set[str] = set()
    cursor: str | None = head
    while cursor is not None and cursor in index and cursor not in ancestors:
        ancestors.add(cursor)
        cursor = _parent_of(records[index[cursor]])

    has_prompt: dict[str, bool] = {}
    last_position: dict[str, int] = {}
    for root in roots:
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            uuid, done = stack.pop()
            if not done:
                stack.append((uuid, True))
                stack.extend((child, False) for child in children.get(uuid, ()))
                continue
            kids = children.get(uuid, ())
            has_prompt[uuid] = _prompt_text(records[index[uuid]]) is not None or any(has_prompt[k] for k in kids)
            last_position[uuid] = max([index[uuid], *(last_position[k] for k in kids)])

    kept: set[str] = set()
    walk = list(roots)
    while walk:
        uuid = walk.pop()
        kept.add(uuid)
        kids = children.get(uuid, [])
        branches = [kid for kid in kids if has_prompt[kid]]
        if len(branches) >= 2:
            on_path = [kid for kid in branches if kid in ancestors]
            chosen = on_path[0] if on_path else max(branches, key=last_position.__getitem__)
            kids = [kid for kid in kids if kid == chosen or not has_prompt[kid]]
        walk.extend(kids)
    return [record for record in records if record.get('uuid') in kept]


def _map_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, int | None]:
    out: list[dict[str, Any]] = []
    model = ''
    usage_by_message: dict[str, int] = {}
    for record in records:
        if record.get('isSidechain') is True:
            continue
        kind = record.get('type')
        content = _content_blocks(record)
        if kind == 'user':
            _map_user(record, content, out)
        elif kind == 'assistant':
            message = record.get('message')
            if not isinstance(message, dict):
                continue
            if isinstance(message.get('model'), str) and message['model']:
                model = message['model']
            usage = message.get('usage')
            if isinstance(usage, dict):
                key = message.get('id') if isinstance(message.get('id'), str) else str(record.get('uuid'))
                usage_by_message[key] = sum(
                    value
                    for name in (
                        'input_tokens',
                        'output_tokens',
                        'cache_creation_input_tokens',
                        'cache_read_input_tokens',
                    )
                    if isinstance(value := usage.get(name), int)
                )
            _map_assistant(content, out)
    return out, model, (sum(usage_by_message.values()) if usage_by_message else None)


def _map_user(record: dict[str, Any], content: object, out: list[dict[str, Any]]) -> None:
    if isinstance(content, list):
        for block in content:
            if (
                isinstance(block, dict)
                and block.get('type') == 'tool_result'
                and isinstance(block.get('tool_use_id'), str)
            ):
                raw = block.get('content')
                text = (
                    items.blocks_text(raw)
                    if isinstance(raw, (str, list))
                    else ('' if raw is None else tool_result_to_text(raw))
                )
                out.append(items.function_call_output(call_id=block['tool_use_id'], output=text))
    if record.get('isMeta'):
        return
    text = (
        content
        if isinstance(content, str)
        else items.blocks_text([b for b in content if isinstance(b, dict) and b.get('type') != 'tool_result'])
        if isinstance(content, list)
        else ''
    )
    if not text:
        return
    out.append(items.compaction_summary(text) if record.get('isCompactSummary') else items.user_text(text))


def _map_assistant(content: object, out: list[dict[str, Any]]) -> None:
    if isinstance(content, str):
        if content:
            out.append(items.assistant_text(content))
        return
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get('type')
        if kind == 'thinking' and isinstance(block.get('thinking'), str) and block['thinking']:
            out.append(items.reasoning(block['thinking']))
        elif kind == 'text' and isinstance(block.get('text'), str) and block['text']:
            out.append(items.assistant_text(block['text']))
        elif kind == 'tool_use' and isinstance(block.get('id'), str) and isinstance(block.get('name'), str):
            out.append(items.function_call(call_id=block['id'], name=block['name'], arguments=block.get('input')))
