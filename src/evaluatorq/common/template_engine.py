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


def is_valid_template_path(path: str) -> bool:
    """Return True if ``path`` is safe to resolve (whitelist match)."""
    return bool(re.match(VALID_PATH_PATTERN, path))


def render_template(template: str, replacements: dict[str, Any]) -> str:
    """Substitute every ``{{key}}`` / ``{{key.nested[0].path}}`` in ``template``.

    Resolution order: strip whitespace (tolerate ``{{ key }}``); reject internal
    whitespace and non-whitelisted paths (placeholder left intact); flat exact-match
    against ``replacements`` first; then nested traversal; unresolved → intact.
    """

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

    def _format(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        if isinstance(value, str):
            return value
        # Parity with upstream: bool/None/number use Python str (True/False/None),
        # NOT JSON (true/false/null). Pinned by the parity suite — do not "fix".
        return str(value)

    def _replacer(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        # \n unreachable (no re.DOTALL) but kept for parity with upstream
        if ' ' in key or '\t' in key or '\n' in key or '\r' in key:
            return match.group(0)
        if not is_valid_template_path(key):
            logger.warning('Rejected template path: {!r}', key)
            return match.group(0)
        if key in replacements:
            return _format(replacements[key])
        value = _resolve_nested(replacements, key)
        if value is _NOT_FOUND:
            return match.group(0)
        return _format(value)

    return _CURLY.sub(_replacer, template)
