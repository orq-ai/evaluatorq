"""Codex session reader (`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` and `archived_sessions`)."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.contracts import tool_result_to_text
from evaluatorq.local_sessions import _jsonl, items
from evaluatorq.local_sessions.roots import codex_home

if TYPE_CHECKING:
    from collections.abc import Iterator

    from evaluatorq.local_sessions.models import ParsedSession, SessionFamily, SessionSummary

_DATED_DEPTH = 3
_INJECTED_PREFIXES = (
    '<environment_context>',
    '# AGENTS.md instructions',
    '<recommended_plugins>',
    '<system_instruction>',
    '<subagent_notification>',
    '<turn_aborted>',
    '# Context from my IDE',
    '<user_instructions>',
)


def _index_path() -> Path:
    return codex_home() / 'session_index.jsonl'


@lru_cache(maxsize=4)
def _titles(path_str: str, mtime_ns: int) -> dict[str, str]:  # mtime_ns is the cache key
    """`id -> thread_name` from the session index; the last line for an id wins."""
    records, _ = _jsonl.read_records(Path(path_str))
    titles: dict[str, str] = {}
    for record in records:
        session_id, name = record.get('id'), record.get('thread_name')
        if isinstance(session_id, str) and isinstance(name, str):
            titles[session_id] = name
    return titles


def _title(session_id: str) -> str:
    path = _index_path()
    try:
        return _titles(str(path), path.stat().st_mtime_ns).get(session_id, '')
    except OSError:
        return ''


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get('payload')
    return payload if isinstance(payload, dict) else {}


def _is_injected(text: str) -> bool:
    return text.lstrip().startswith(_INJECTED_PREFIXES)


def _message_text(payload: dict[str, Any]) -> str:
    return items.blocks_text(payload.get('content'))


def _output_text(output: object) -> str:
    if isinstance(output, str):
        return output
    return '' if output is None else tool_result_to_text(output)


_SCRIPT_FAILED = 'Script failed'
_EXIT_CODE_LINE = re.compile(r'^Exit code: (\d+)\r?$', re.MULTILINE)
_EXIT_CODE_WINDOW = 600


def _failed(text: str) -> bool:
    """Codex marks a failed call in the output text: a first line `Script failed`, or a non-zero `Exit code: N` near the top."""
    if text.split('\n', 1)[0].strip() == _SCRIPT_FAILED:
        return True
    return any(int(code) != 0 for code in _EXIT_CODE_LINE.findall(text[:_EXIT_CODE_WINDOW]))


class _CodexReader:
    family: SessionFamily = 'codex'

    def iter_files(self, root: Path) -> Iterator[Path]:
        for path in root.rglob('rollout-*.jsonl'):
            if self.is_session_path(path, root):
                yield path

    def is_session_path(self, path: Path, root: Path) -> bool:
        if not (path.name.startswith('rollout-') and path.name.endswith('.jsonl')):
            return False
        if path.parent == root:
            return True
        try:
            return len(path.parent.relative_to(root).parts) == _DATED_DEPTH
        except ValueError:
            return False

    def summarize(self, path: Path) -> SessionSummary | None:
        return _jsonl.summarize_session(path, _extract)

    def parse(self, path: Path, summary: SessionSummary) -> ParsedSession:
        return _jsonl.parse_session(path, summary, lambda records: _map_records(records, path))


READER = _CodexReader()


def _extract(head: _jsonl.Records, _tail: _jsonl.Records) -> _jsonl.SummaryParts | None:
    if not head or head[0].get('type') != 'session_meta':
        return None
    meta = _payload(head[0])
    source = meta.get('source')
    if (isinstance(source, dict) and 'subagent' in source) or meta.get('parent_thread_id'):
        return None
    first_prompt = next(
        (
            text
            for record in head
            if record.get('type') == 'response_item'
            and (payload := _payload(record)).get('type') == 'message'
            and payload.get('role') == 'user'
            and not _is_injected(text := _message_text(payload))
        ),
        '',
    )
    session_id = meta.get('id')
    if not isinstance(session_id, str):
        session_id = ''
    return _jsonl.SummaryParts(
        source='codex',
        session_id=session_id,
        title=_title(session_id) if session_id else '',
        project_dir=meta['cwd'] if isinstance(meta.get('cwd'), str) else '',
        started=meta.get('timestamp'),
        first_prompt=first_prompt,
        agent_version=meta['cli_version'] if isinstance(meta.get('cli_version'), str) else '',
    )


def _map_records(records: list[dict[str, Any]], path: Path) -> tuple[list[dict[str, Any]], str, int | None]:
    out: list[dict[str, Any]] = []
    model = ''
    total_tokens: int | None = None
    skipped_types: Counter[str] = Counter()
    for record in records:
        kind = record.get('type')
        payload = _payload(record)
        if kind == 'response_item':
            _map_item(payload, out, skipped_types)
        elif kind == 'compacted':
            message = payload.get('message')
            out.append(items.compaction_summary(message if isinstance(message, str) else ''))
        elif kind == 'turn_context':
            if isinstance(payload.get('model'), str) and payload['model']:
                model = payload['model']
        elif kind == 'event_msg' and payload.get('type') == 'token_count':
            info = payload.get('info')
            usage = info.get('total_token_usage') if isinstance(info, dict) else None
            total = usage.get('total_tokens') if isinstance(usage, dict) else None
            if isinstance(total, int):
                total_tokens = total
    if skipped_types:
        logger.warning('Skipped Codex item types {} in {}', dict(skipped_types), path)
    return out, model, total_tokens


def _map_item(payload: dict[str, Any], out: list[dict[str, Any]], skipped_types: Counter[str]) -> None:
    kind = payload.get('type')
    call_id = payload.get('call_id')
    name = payload.get('name')
    if kind == 'message':
        role = payload.get('role')
        text = _message_text(payload)
        if role == 'user' and text and not _is_injected(text):
            out.append(items.user_text(text))
        elif role == 'assistant' and text:
            out.append(items.assistant_text(text))
    elif kind == 'reasoning':
        summary = payload.get('summary')
        text = (
            '\n'.join(part['text'] for part in summary if isinstance(part, dict) and isinstance(part.get('text'), str))
            if isinstance(summary, list)
            else ''
        )
        if text:
            out.append(items.reasoning(text))
    elif kind == 'agent_message':
        content = payload.get('content')
        text = items.blocks_text(
            [b for b in content if isinstance(b, dict) and b.get('type') == 'input_text']
            if isinstance(content, list)
            else []
        )
        if text:
            author = payload.get('author')
            label = (
                f'Message from subagent {author}:' if isinstance(author, str) and author else 'Message from subagent:'
            )
            out.append(items.system_text(items.trim(f'{label}\n{text}')))
    elif kind == 'function_call' and isinstance(call_id, str) and isinstance(name, str):
        out.append(items.function_call(call_id=call_id, name=name, arguments=payload.get('arguments')))
    elif kind == 'custom_tool_call' and isinstance(call_id, str) and isinstance(name, str):
        out.append(items.function_call(call_id=call_id, name=name, arguments={'input': payload.get('input')}))
    elif kind in ('function_call_output', 'custom_tool_call_output') and isinstance(call_id, str):
        text = _output_text(payload.get('output'))
        out.append(items.function_call_output(call_id=call_id, output=text, is_error=_failed(text)))
    else:
        skipped_types[kind if isinstance(kind, str) else 'unknown'] += 1
