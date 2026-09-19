"""Pure ``{{...}}`` template substitution engine.

Faithful port of Orq's canonical evaluator template engine. Source of truth:
orquesta-web ``apps/evals/python-runner/evals_python_runner/utils/evaluator_manager/
llm/evaluator.py`` (``replace_curly_entries`` / ``is_valid_template_path`` /
``VALID_PATH_PATTERN``), mirrored in Go at ``libs/go/graders/template_engine.go``.
Ported from orquesta-web commit 95d9a2fef32229c179c24e44a49ccdffcc2d9361.

This is a FORK: upstream evolves in a repo evaluatorq-py does not depend on, with no
CI link. The parity suite (tests/common/test_template_engine.py) pins behaviour at
the port SHA; drift is a manual re-sync.

Security: every ``{{path}}`` is validated against a whitelist before resolution, and
substitution is single-pass (``re.sub`` with a callback), so a resolved value that
itself contains a ``{{...}}`` string is emitted verbatim and never re-expanded.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

# Whitelist: bare identifier, dot-separated identifiers / pure-numeric segments,
# bracketed (possibly negative) numeric indices, or any mix. Rejects function
# calls, string literals, assignments, ``;``, ``{}``, etc. Byte-identical to
# upstream VALID_PATH_PATTERN as of the port SHA (manual re-sync; not CI-enforced).
VALID_PATH_PATTERN = r'^[a-zA-Z_][a-zA-Z0-9_]*(?:\.(?:[a-zA-Z_][a-zA-Z0-9_]*|\d+)|\[-?\d+\])*$'

_CURLY = re.compile(r'{{(.*?)}}')
_BRACKET_INDEX = re.compile(r'\[(-?\d+)\]')
_NOT_FOUND = object()

# Bare reserved keys resolve to intact literal: the namespace dicts must remain
# present for dotted/indexed subpaths, but a bare {{input}} must not dump them.
# This is an intentional divergence from the upstream Orq engine (this file is a
# fork; see module docstring). Pinned by the parity suite below.
_RESERVED_BARE_KEYS = frozenset({'input', 'output', 'log', 'response_a', 'response_b'})


def is_valid_template_path(path: str) -> bool:
    """Return True if ``path`` is safe to resolve (whitelist match)."""
    return bool(re.match(VALID_PATH_PATTERN, path))


def _resolve_nested(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for segment in path.split('.'):
        bracket_at = segment.find('[')
        if bracket_at == -1:
            if not isinstance(current, dict) or segment not in current:
                return _NOT_FOUND
            current = current[segment]
            continue
        key = segment[:bracket_at]
        if key:
            if not isinstance(current, dict) or key not in current:
                return _NOT_FOUND
            current = current[key]
        for match in _BRACKET_INDEX.finditer(segment):
            if not isinstance(current, list):
                return _NOT_FOUND
            idx = int(match.group(1))
            if idx < 0:
                idx += len(current)
            if idx < 0 or idx >= len(current):
                return _NOT_FOUND
            current = current[idx]
    return current


def _placeholder_path(body: str) -> str | None:
    """The resolvable path inside one ``{{...}}`` body, or ``None`` to leave it intact.

    Holds the strip / internal-whitespace / whitelist / reserved-bare-key rules in
    one place, so `render_template` and `extract_template_paths` cannot drift on
    which placeholders count.
    """
    path = body.strip()
    # \n unreachable (no re.DOTALL) but kept for parity with upstream
    if ' ' in path or '\t' in path or '\n' in path or '\r' in path:
        return None
    if not is_valid_template_path(path):
        logger.warning('Rejected template path: {!r}', path)
        return None
    if path in _RESERVED_BARE_KEYS:
        logger.warning(
            'Bare reserved template key {!r} left unresolved — use a dotted path instead (e.g. {!r}).',
            path,
            f'{path}.<field>',
        )
        return None
    return path


def extract_template_paths(template: str) -> list[str]:
    """Every path `render_template` would try to resolve in ``template``, in order.

    Unique, in order of first appearance. Placeholders `render_template` leaves
    intact — internal whitespace, a non-whitelisted path, a bare reserved key — are
    excluded, so the list is what a caller must be able to supply, not every
    ``{{...}}`` in the string.
    """
    paths: list[str] = []
    for match in _CURLY.finditer(template):
        path = _placeholder_path(match.group(1))
        if path is not None and path not in paths:
            paths.append(path)
    return paths


def resolve_template_path(replacements: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Look ``path`` up in ``replacements``: ``(found, value)``.

    Flat exact match first (``{'a.b': ...}`` beats ``{'a': {'b': ...}}``), then
    nested traversal with ``[0]`` / ``[-1]`` indices. ``found`` is separate from
    the value so a stored ``None`` reads as found, not as missing.
    """
    if path in replacements:
        return True, replacements[path]
    value = _resolve_nested(replacements, path)
    return (False, None) if value is _NOT_FOUND else (True, value)


def render_template(template: str, replacements: dict[str, Any]) -> str:
    """Substitute every ``{{key}}`` / ``{{key.nested[0].path}}`` in ``template``.

    Resolution order: strip whitespace (tolerate ``{{ key }}``); reject internal
    whitespace and non-whitelisted paths (placeholder left intact); flat exact-match
    against ``replacements`` first; then nested traversal; unresolved → intact.

    Substitution runs through one ``re.sub`` pass rather than looping over
    `extract_template_paths`, because replacing resolved text placeholder by
    placeholder would re-expand a ``{{...}}`` string that a value itself contains.
    """

    def _format(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        if isinstance(value, str):
            return value
        # Parity with upstream: bool/None/number use Python str (True/False/None),
        # NOT JSON (true/false/null). Pinned by the parity suite — do not "fix".
        return str(value)

    def _replacer(match: re.Match[str]) -> str:
        path = _placeholder_path(match.group(1))
        if path is None:
            return match.group(0)
        found, value = resolve_template_path(replacements, path)
        return _format(value) if found else match.group(0)

    return _CURLY.sub(_replacer, template)
