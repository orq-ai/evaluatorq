"""Mechanical guardrails for the reuse table in CLAUDE.md.

Prose conventions get skipped; a failing test does not. Each rule below names the
canonical helper in its failure message. When one fires, route the call through
the helper — do not append your file to the allowlist. The allowlists are a
freeze on the sites that predate the rule and are expected to shrink.

Detection is AST-based, not regex: a docstring or comment mentioning a call is
not a call, and `x = client.chat.completions` followed by `x.create(...)` should
not be a way around the rule by accident.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src' / 'evaluatorq'

# Attribute suffixes that mean "an LLM API call happened here".
LLM_CALL_SUFFIXES = (
    'chat.completions.create',
    'chat.completions.parse',
    'responses.create',
    'responses.parse',
)
# Callables that construct an SDK client, by SDK.
OPENAI_CLIENT_NAMES = frozenset({'OpenAI', 'AsyncOpenAI', 'AzureOpenAI', 'AsyncAzureOpenAI'})
ORQ_CLIENT_NAMES = frozenset({'Orq', 'AsyncOrq'})
SPAN_NAMES = frozenset({'get_tracer', 'start_span', 'start_as_current_span'})


def _dotted(node: ast.expr) -> str:
    """Best-effort dotted name for a call target (``a.b.c`` -> ``'a.b.c'``)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return '.'.join(reversed(parts))


def _calls(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, dotted_name)`` for every call in ``source``."""
    return [
        (node.lineno, _dotted(node.func))
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
    ]


@cache
def _src_calls() -> tuple[tuple[str, int, str], ...]:
    """Every ``(relative_path, lineno, dotted_name)`` call in the package, parsed once."""
    return tuple(
        (path.relative_to(SRC).as_posix(), lineno, name)
        for path in sorted(SRC.rglob('*.py'))
        for lineno, name in _calls(path.read_text(encoding='utf-8'))
    )


def _hits(predicate: object, *, allow: frozenset[str] | set[str] | None = None) -> list[str]:
    """Return ``path:line`` for every call matching ``predicate`` outside ``allow``."""
    assert callable(predicate)
    allowed = allow or ()
    return [f'{rel}:{lineno}' for rel, lineno, name in _src_calls() if rel not in allowed and predicate(name)]


def _is_llm_call(name: str) -> bool:
    return any(name == suffix or name.endswith('.' + suffix) for suffix in LLM_CALL_SUFFIXES)


def _tail(name: str) -> str:
    return name.rsplit('.', 1)[-1]


# Call sites that predate common.llm_call. Shrink, never grow — the length
# assertion below exists so growing it is a deliberate, visible edit.
LLM_CALL_ALLOW = frozenset({
    'common/llm_call.py',  # canonical
    'common/structured_output.py',  # canonical for schema output
    'openresponses/target.py',  # Responses API transport
    'redteam/backends/openai.py',  # backend transport
    'redteam/backends/orq.py',  # backend transport
    'redteam/adaptive/blackbox_classifier.py',
    'redteam/adaptive/tool_chaining.py',
    'simulation/agents/base.py',
    'simulation/generators/first_message_generator.py',
    'common/reports/executive_summary.py',
})


def test_llm_call_allowlist_does_not_grow() -> None:
    assert len(LLM_CALL_ALLOW) == 10, (
        'LLM_CALL_ALLOW changed size. Removing an entry (good) means lowering this '
        'number; adding one means a new direct call site slipped in — route it '
        'through evaluatorq.common.llm_call instead.'
    )


def test_llm_calls_go_through_the_shared_helper() -> None:
    extra = _hits(_is_llm_call, allow=LLM_CALL_ALLOW)
    assert not extra, (
        'Direct LLM API call outside the shared path: '
        + ', '.join(extra)
        + '. Use evaluatorq.common.llm_call.execute_chat_completion / execute_chat_parse '
        '(or common.structured_output for schema output) — they own span recording, '
        'trace-header injection, token usage and the reserved-key guard.'
    )


def test_openai_clients_come_from_llm_client() -> None:
    extra = _hits(lambda name: _tail(name) in OPENAI_CLIENT_NAMES, allow={'common/llm_client.py'})
    assert not extra, (
        'OpenAI client constructed outside common/llm_client.py: '
        + ', '.join(extra)
        + '. Use resolve_llm_client() — it owns env-var precedence, base-URL '
        'resolution and max_retries.'
    )


def test_orq_clients_come_from_orq_client() -> None:
    extra = _hits(lambda name: _tail(name) in ORQ_CLIENT_NAMES, allow={'common/orq_client.py'})
    assert not extra, (
        'Orq SDK client constructed outside common/orq_client.py: '
        + ', '.join(extra)
        + '. Use resolve_orq_client() — it owns the lazy import, the install hint, '
        'the ORQ_API_KEY check and the ORQ_BASE_URL default.'
    )


def test_spans_are_opened_only_in_tracing_modules() -> None:
    extra = [
        hit
        for hit in _hits(lambda name: _tail(name) in SPAN_NAMES)
        if not (hit.startswith('tracing/') or Path(hit.split(':')[0]).name == 'tracing.py')
    ]
    assert not extra, (
        'Span created outside a tracing module: '
        + ', '.join(extra)
        + '. Use evaluatorq.common.tracing.with_llm_span, or add a thin wrapper in your '
        "surface's tracing.py that delegates to it — span naming stays one vocabulary."
    )


@pytest.mark.parametrize(
    ('source', 'predicate', 'expected'),
    [
        ('await client.chat.completions.create(**params)', _is_llm_call, True),
        ('client.beta.chat.completions.parse(**params)', _is_llm_call, True),
        ('await self._client.responses.create(**kwargs)', _is_llm_call, True),
        ('"""Prose about client.chat.completions.create."""', _is_llm_call, False),
        ('# client.chat.completions.create(...)', _is_llm_call, False),
        ('client = AsyncOpenAI(api_key=key)', lambda n: _tail(n) in OPENAI_CLIENT_NAMES, True),
        ('client = openai.AzureOpenAI(api_key=key)', lambda n: _tail(n) in OPENAI_CLIENT_NAMES, True),
        ('client = Orq(api_key=key)', lambda n: _tail(n) in ORQ_CLIENT_NAMES, True),
        ('with tracer.start_as_current_span("x"):\n    pass', lambda n: _tail(n) in SPAN_NAMES, True),
    ],
)
def test_detection_actually_fires(source: str, predicate: object, expected: bool) -> None:
    """A guardrail nobody proved can fail is a guardrail that silently no-ops."""
    assert callable(predicate)
    assert any(predicate(name) for _, name in _calls(source)) is expected
