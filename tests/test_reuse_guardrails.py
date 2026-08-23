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
    'openresponses/target.py',  # Responses API transport
    'redteam/backends/openai.py',  # backend transport
    'redteam/backends/orq.py',  # backend transport
    'redteam/adaptive/blackbox_classifier.py',
    'redteam/adaptive/tool_chaining.py',
    'simulation/generators/first_message_generator.py',
})


def test_llm_call_allowlist_does_not_grow() -> None:
    assert len(LLM_CALL_ALLOW) == 7, (
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


def _hand_written_properties_dict_lines(source: str) -> list[int]:
    """Line numbers of dict literals mapping a literal ``'properties'`` key to a
    literal dict of field definitions.

    AST-based so a docstring or comment mentioning ``properties`` is not a hit,
    and a value that happens to equal the string ``'properties'`` (not a key)
    is not one either. An *empty* ``'properties': {}`` is not flagged — that is
    a "this tool takes no parameters" fallback, not a hand-rolled schema with
    fields a pydantic model should own instead.
    """
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == 'properties' and isinstance(value, ast.Dict) and value.keys
    ]


# Scope: the whole package. The judge's tool schemas (simulation/) are where
# RES-1308's schema/parser drift happened, but the remedy — generate the schema
# from a pydantic model — applies just as well to a future hand-rolled schema in
# redteam/, openresponses/ or integrations/, and the measured hit count repo-wide
# is 0, so the wider scope costs nothing and catches more. A schema genuinely
# handed to a provider verbatim is the escape hatch named in the failure message.
_HAND_WRITTEN_PROPERTIES_ROOT = SRC


@cache
def _hand_written_properties_hits() -> list[str]:
    return [
        f'{path.relative_to(SRC).as_posix()}:{lineno}'
        for path in sorted(_HAND_WRITTEN_PROPERTIES_ROOT.rglob('*.py'))
        for lineno in _hand_written_properties_dict_lines(path.read_text(encoding='utf-8'))
    ]


def test_no_hand_written_json_schema_properties_dicts() -> None:
    """A JSON-schema `'properties': {...}` dict literal is a schema written by hand.

    RES-1308 happened because a hand-written tool schema and the parser reading
    it drifted apart. `judge.py`'s two tool schemas are generated from pydantic
    models via `_wire_schema` / `model_json_schema()` for exactly this reason;
    this measured 3 hits in `simulation/` before that refactor and 0 after — and
    0 across the whole package, which is why the scope is package-wide. A hit
    here means a schema is being hand-rolled again — define a `pydantic.BaseModel`
    and generate its schema instead (see `judge.py`'s `_wire_schema`).

    The scope is all of `src/evaluatorq/`; see `_HAND_WRITTEN_PROPERTIES_ROOT`.
    """
    hits = _hand_written_properties_hits()
    assert not hits, (
        "Hand-written JSON-schema 'properties' dict literal: "
        + ', '.join(hits)
        + '. Define a pydantic BaseModel and generate the schema via model_json_schema() '
        '/ _wire_schema (see simulation/agents/judge.py) instead of hand-writing it. '
        'Escape hatch: a schema this package hands to a provider verbatim and never '
        'parses back (an HTTP/json_schema evaluator definition) — add that one path to '
        'an explicit exemption here, with the reason, rather than narrowing the scope.'
    )


def test_hand_written_properties_detector_actually_fires() -> None:
    """A guardrail nobody proved can fail is a guardrail that silently no-ops."""
    assert _hand_written_properties_dict_lines('x = {"properties": {"a": {"type": "string"}}}') == [1]
    assert _hand_written_properties_dict_lines('x = {"foo": "properties"}') == []
    assert _hand_written_properties_dict_lines('# {"properties": {}}') == []
    assert _hand_written_properties_dict_lines('"""Docstring mentions properties."""') == []
    # An empty properties dict is "no parameters", not a hand-rolled schema —
    # this is the real shape at simulation/agents/base.py's Responses fallback.
    assert _hand_written_properties_dict_lines('x = {"type": "object", "properties": {}}') == []


# --- usage harvested off a raised exception -------------------------------
# `generate_structured` bills its rungs before it raises, so a failed structured
# call still cost money. Five modules each read the attribute directly, each with
# its own copy of the rationale; `usage_from_exception` is now the one place that
# rule is written down.
USAGE_HARVEST_ALLOW = frozenset({
    'common/structured_output.py',  # canonical
})


def _getattr_usage_sites() -> list[str]:
    """``path:line`` for every ``getattr(<caught exception>, 'usage', ...)`` in the package.

    Scoped to names bound by an ``except ... as`` in the same file. Reading
    ``usage`` off a *response* is a different thing and stays allowed.
    """
    sites: list[str] = []
    for path in sorted(SRC.rglob('*.py')):
        rel = path.relative_to(SRC).as_posix()
        if rel in USAGE_HARVEST_ALLOW:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        caught = {n.name for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.name}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'getattr'
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in caught
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == 'usage'
            ):
                sites.append(f'{rel}:{node.lineno}')
    return sites


def test_exception_usage_goes_through_the_shared_helper() -> None:
    sites = _getattr_usage_sites()
    assert not sites, (
        "getattr(<caught exception>, 'usage') outside the shared helper: "
        + ', '.join(sites)
        + '. Use evaluatorq.common.structured_output.usage_from_exception — it owns '
        'the harvest rule (why getattr rather than an except clause, and when the '
        'exception carries no total so the result must not be double-counted).'
    )
