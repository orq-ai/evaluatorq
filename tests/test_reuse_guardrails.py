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
import re
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


def _cache_control_dict_lines(source: str) -> list[int]:
    """Line numbers of dict literals carrying a literal ``'cache_control'`` key.

    AST-based, so the comments and docstrings that explain the convention are not
    hits — only a marker actually written into a request body is.
    """
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and key.value == 'cache_control'
    ]


# The one module allowed to write the marker. Every other call site imports
# `apply_cache_breakpoints` / `mark_responses_input` from it.
_PROMPT_CACHE_OWNER = 'common/prompt_cache.py'


def test_cache_control_is_written_only_by_prompt_cache() -> None:
    """A hand-placed `cache_control` is the expensive kind of mistake.

    A misplaced breakpoint costs a 1.25x write that nothing reads back and
    nothing errors — it surfaces as a bill, not a bug. `common/prompt_cache.py`
    owns placement so the `volatile_tail` contract, the router+model gate and the
    minimum-size guard cannot be bypassed by writing the key directly.
    """
    hits = [
        f'{path.relative_to(SRC).as_posix()}:{lineno}'
        for path in sorted(SRC.rglob('*.py'))
        if path.relative_to(SRC).as_posix() != _PROMPT_CACHE_OWNER
        for lineno in _cache_control_dict_lines(path.read_text(encoding='utf-8'))
    ]
    assert not hits, (
        'Hand-placed cache_control marker: '
        + ', '.join(hits)
        + '. Use common.prompt_cache.apply_cache_breakpoints (chat) or '
        'mark_responses_input (Responses), gated on caching_applies — writing the '
        'key directly skips the volatile_tail contract and the size guard.'
    )


def test_cache_control_detector_actually_fires() -> None:
    """A guardrail nobody proved can fail is a guardrail that silently no-ops."""
    assert _cache_control_dict_lines("x = {'cache_control': {'type': 'ephemeral'}}") == [1]
    assert _cache_control_dict_lines("x = {'foo': 'cache_control'}") == []
    assert _cache_control_dict_lines('# {"cache_control": {}}') == []
    assert _cache_control_dict_lines('"""Docstring mentions cache_control."""') == []
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


# Sphinx cross-reference roles (`:class:`Foo``) render as literal text on the
# MkDocs site — the role prefix and the `~` shorthand both print. RES-1278.
# Covers the autodoc object-reference roles plus the three doc-level roles
# (`ref`, `paramref`, `term`) that break the same way. Domain-specific roles
# nobody here writes (`:option:`, `:envvar:`, …) are out of scope until one shows up.
SPHINX_ROLE = re.compile(r':(?:class|func|meth|attr|mod|data|exc|obj|ref|paramref|term):`')


def _sphinx_role_lines(source: str) -> list[int]:
    """Line numbers of every docstring in ``source`` carrying a Sphinx role.

    A module docstring reports line 1: ``ast.Module`` has no ``lineno``.
    Roles in comments or in ordinary strings are not docstrings and not hits.
    """
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc and SPHINX_ROLE.search(doc):
            lines.append(getattr(node, 'lineno', 1))
    return sorted(lines)


def _sphinx_role_sites() -> list[str]:
    """``path:line`` for every docstring carrying a Sphinx cross-reference role."""
    return [
        f'{path.relative_to(SRC).as_posix()}:{line}'
        for path in sorted(SRC.rglob('*.py'))
        for line in _sphinx_role_lines(path.read_text(encoding='utf-8'))
    ]


def test_sphinx_role_detector_actually_fires() -> None:
    """A guardrail nobody proved can fail is a guardrail that silently no-ops."""
    assert _sphinx_role_lines('"""See :class:`Foo`."""') == [1]
    assert _sphinx_role_lines('def f():\n    """Calls :func:`bar`."""') == [1]
    assert _sphinx_role_lines('class C:\n    """Wraps :meth:`C.go`."""') == [1]
    # The `~` shorthand prints too, and a fully qualified target is still a hit.
    assert _sphinx_role_lines('"""A :class:`~evaluatorq.contracts.Message`."""') == [1]
    # Every role in the pattern fires, including the three doc-level ones.
    for role in ('attr', 'mod', 'data', 'exc', 'obj', 'ref', 'paramref', 'term'):
        assert _sphinx_role_lines(f'"""See :{role}:`x`."""') == [1], role
    # Two docstrings, two sites — the failure message names both.
    assert _sphinx_role_lines('def f():\n    """:func:`a`."""\n\n\ndef g():\n    """:func:`b`."""') == [1, 5]
    # Not docstrings: a comment, a bare expression string, an assigned string.
    assert _sphinx_role_lines('# :class:`Foo`') == []
    assert _sphinx_role_lines('def f():\n    pass\n\n\n":class:`Foo`"') == []
    assert _sphinx_role_lines('x = ":class:`Foo`"') == []
    # A plain code span is the fix, and must not be flagged.
    assert _sphinx_role_lines('"""See `Foo`, or [Message][evaluatorq.contracts.Message]."""') == []


def test_docstrings_carry_no_sphinx_roles() -> None:
    sites = _sphinx_role_sites()
    assert not sites, (
        'Sphinx cross-reference role in a docstring: '
        + ', '.join(sites)
        + '. mkdocstrings has no idea what a role is, so `:class:` and the `~` '
        'shorthand render as literal text on the API reference pages. Use a plain '
        'code span, or an mkdocstrings autoref: [Message][evaluatorq.contracts.Message].'
    )


def _new_returns_hardcoded_class(source: str, path: str) -> list[str]:
    """Return ``path:lineno`` for every ``new()``/``clone()`` that constructs a class from this module.

    A ``new()`` returning ``MyTarget(...)`` instead of ``type(self)(...)`` silently
    hands back a base instance for any subclass, on every parallel job. ``clone()``
    is checked too because ``OpenAIAgentTarget.new()`` is ``return self.clone()``,
    which puts the construction one call away from the method the contract names.

    Constructing a class *defined in the same module* is the signal, rather than a
    name convention: this stays AST-only, so no optional extra has to be installed
    to run it.
    """
    tree = ast.parse(source)
    local_classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) or node.name not in ('new', 'clone'):
            continue
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            if _dotted(call.func) in local_classes:
                hits.append(f'{path}:{call.lineno}')
    return hits


def test_new_constructs_via_type_self() -> None:
    hits = [
        hit
        for path in sorted(SRC.rglob('*.py'))
        for hit in _new_returns_hardcoded_class(path.read_text(encoding='utf-8'), str(path.relative_to(SRC)))
    ]
    assert not hits, (
        'AgentTarget.new()/clone() constructs a hardcoded class: '
        + ', '.join(hits)
        + '. Use type(self)(...) so a subclass does not silently degrade to its base class.'
    )


def test_new_hardcoded_class_detector_actually_fires() -> None:
    source = 'class MyTarget:\n    def new(self):\n        return MyTarget(self._x)\n'
    assert _new_returns_hardcoded_class(source, 'x.py') == ['x.py:3']
    assert _new_returns_hardcoded_class(source.replace('return MyTarget', 'return type(self)'), 'x.py') == []
    # The shape this repo actually ships: new() delegates, clone() constructs.
    delegating = (
        'class MyTarget:\n'
        '    def clone(self):\n'
        '        return MyTarget(self._x)\n'
        '\n'
        '    def new(self):\n'
        '        return self.clone()\n'
    )
    assert _new_returns_hardcoded_class(delegating, 'x.py') == ['x.py:3']
    # An async new() is a different node type; ast.walk must still see it.
    assert _new_returns_hardcoded_class(
        'class MyTarget:\n    async def new(self):\n        return MyTarget()\n', 'x.py'
    ) == ['x.py:3']
    # No naming convention required: the old detector missed anything not ending in "Target".
    assert _new_returns_hardcoded_class('class Backend:\n    def new(self):\n        return Backend()\n', 'x.py') == [
        'x.py:3'
    ]


# --- an llm_config consumed in part, silently --------------------------------
# `FirstMessageGenerator` dropped a caller's `llm_config.max_tokens` for as long
# as it existed: it sizes its own budget and calls the Responses API itself, so
# it never went through `generate_structured`, which is the thing that warns
# about the fields it does not consume. That was found by hand. This is the
# mechanical version.
_SIM_GENERATORS = SRC / 'simulation' / 'generators'

# Either one accounts for the whole config: the first warns on the caller's own
# behalf, the second warns on its callers'.
_CONFIG_ACCOUNTING_CALLS = frozenset({'warn_unread_config_fields', 'generate_structured'})

# Accessors that turn a config into call parameters. Fields come from the model
# itself so a new one is covered the day it is added.
_CONFIG_ACCESSORS = frozenset({'set_values', 'timeout_s', 'request_params'})


@cache
def _call_setting_names() -> frozenset[str]:
    """Every way of reading a *call setting* off an `LLMCallConfig`.

    `model` and `client` are excluded: reading those is routing, not sampling,
    and a generator that only routes (`DatapointGenerator`) hands the config on
    to the generators that do the calling, which each account for it there.
    """
    from evaluatorq.contracts import LLMCallConfig

    return (frozenset(LLMCallConfig.model_fields) | _CONFIG_ACCESSORS) - {'model', 'client'}


def _scope_nodes(root: ast.AST) -> list[ast.AST]:
    """Every node under ``root``, minus the bodies of the classes nested in it.

    A nested class is its own scope, so excluding it here keeps the enclosing
    walk from seeing — or clearing on — code it does not own.
    """
    nodes: list[ast.AST] = []
    stack: list[ast.AST] = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef) and child is not root:
                continue
            stack.append(child)
    return nodes


def _resolved_config_names(nodes: list[ast.AST]) -> set[str]:
    """Names bound to a ``resolve_sim_llm_config(...)`` result in ``nodes``.

    Catches a config held under a name that does not contain ``config``.
    """
    names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, ast.Call) or _tail(_dotted(value.func)) != 'resolve_sim_llm_config':
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                names.add(f'{target.value.id}.{target.attr}')
    return names


def _scope_accounting_gap(nodes: list[ast.AST], settings: frozenset[str]) -> int | None:
    """First unaccounted-for config read in one scope, or ``None``."""
    called = {_tail(_dotted(node.func)) for node in nodes if isinstance(node, ast.Call)}
    if 'resolve_sim_llm_config' not in called or called & _CONFIG_ACCOUNTING_CALLS:
        return None
    receivers = _resolved_config_names(nodes)
    reads = sorted(
        node.lineno
        for node in nodes
        if isinstance(node, ast.Attribute)
        and node.attr in settings
        and ('config' in (name := _dotted(node.value)).lower() or name in receivers)
    )
    return reads[0] if reads else None


def _config_accounting_gap(source: str) -> int | None:
    """First line where ``source`` reads a call setting off a resolved config
    without accounting for the settings it ignores, or ``None``.

    Each ``class`` is its own scope, and module-level code outside every class
    is one more. A scope qualifies when it resolves a config
    (`resolve_sim_llm_config`) and then reads sampling settings off it; it is
    clear when it calls `warn_unread_config_fields` or `generate_structured`.
    Scoping matters: module-wide clearing let one `generate_structured` call
    anywhere in a file excuse every other class in that same file.
    """
    tree = ast.parse(source)
    settings = _call_setting_names()
    scopes: list[ast.AST] = [tree, *(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))]
    hits = [line for scope in scopes if (line := _scope_accounting_gap(_scope_nodes(scope), settings)) is not None]
    return min(hits) if hits else None


def test_simulation_generators_account_for_the_config_they_are_given() -> None:
    """A generator that reads part of an `llm_config` must say what it drops.

    Scope is `simulation/generators/` on purpose. A general "any function taking
    `config: LLMCallConfig`" walk sweeps in `BaseAgent`, `contracts` and
    `structured_output` itself and would need an allowlist to stay green, which
    is the failure mode this file exists to avoid.
    """
    hits = [
        f'{path.relative_to(SRC).as_posix()}:{line}'
        for path in sorted(_SIM_GENERATORS.rglob('*.py'))
        if (line := _config_accounting_gap(path.read_text(encoding='utf-8'))) is not None
    ]
    assert not hits, (
        'Generator reads part of a resolved llm_config and drops the rest in silence: '
        + ', '.join(hits)
        + '. Call evaluatorq.common.structured_output.warn_unread_config_fields(config, '
        '<fields you read>, caller=...) next to the resolve_sim_llm_config call — see '
        '_READ_CONFIG_FIELDS in simulation/generators/first_message_generator.py — or '
        'route the call through generate_structured, which warns on your behalf.'
    )


def test_config_accounting_detector_actually_fires() -> None:
    """A guardrail nobody proved can fail is a guardrail that silently no-ops."""
    violating = (
        'class G:\n'
        '    def __init__(self, config=None):\n'
        '        self._config = resolve_sim_llm_config(sim_model=m, llm_config=config, caller="G")\n'
        '    async def go(self):\n'
        '        await self._client.responses.create(temperature=self._config.temperature)\n'
    )
    assert _config_accounting_gap(violating) == 5
    # Cleared by warning itself, in the same class that read the config...
    warned = violating.replace(
        'self._config = resolve',
        'warn_unread_config_fields(config, {"temperature"}, caller="G")\n        self._config = resolve',
    )
    assert _config_accounting_gap(warned) is None
    # ...but a warn call outside that class accounts for nothing.
    assert _config_accounting_gap(violating + '\nwarn_unread_config_fields(c, f, caller="other")\n') == 5
    # ...or by letting generate_structured warn on its behalf.
    assert _config_accounting_gap(violating.replace('self._client.responses.create', 'generate_structured')) is None
    # A module that never resolves a config is out of scope.
    assert _config_accounting_gap('x = self._config.temperature') is None
    # Routing-only reads are not consumption: DatapointGenerator's shape.
    assert (
        _config_accounting_gap(
            'self._config = resolve_sim_llm_config(sim_model=m, llm_config=config, caller="D")\n'
            'self._model = self._config.model\n'
            'sub = PersonaGenerator(client=self._config.client, config=self._config)\n'
        )
        is None
    )
    # Prose is not a read.
    assert _config_accounting_gap('resolve_sim_llm_config()\n"""Mentions config.temperature."""') is None
    # A cleared class does not clear its neighbours: the module-wide loophole.
    cleared = 'class Fine:\n    def go(self):\n        return generate_structured()\n\n\n'
    assert _config_accounting_gap(cleared + violating) == 10
    # A config read through a name that never says "config" still counts.
    assert _config_accounting_gap('cfg = resolve_sim_llm_config(sim_model=m, llm_config=c, caller="G")\nx = cfg.temperature\n') == 2
    assert (
        _config_accounting_gap(
            'class G:\n'
            '    def __init__(self):\n'
            '        self._cfg: LLMCallConfig = resolve_sim_llm_config(sim_model=m, llm_config=c, caller="G")\n'
            '    def go(self):\n'
            '        return self._client.responses.create(**self._cfg.set_values("temperature"))\n'
        )
        == 5
    )
