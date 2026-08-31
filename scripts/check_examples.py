#!/usr/bin/env python
"""Check every example still refers to symbols evaluatorq actually exports.

Covers `examples/` (.py files and notebook code cells) and the fenced `python`
blocks in `docs/`. A docs sample is code a reader copies, so it rots the same way
an example does, and until this covered it the only thing standing between a
renamed symbol and a published page was someone running the docs-autofill receipt
runner by hand.

Static only — nothing is executed. That is deliberate: several examples build an
LLM client and call it at module level, so importing them means real network
traffic (and, with a placeholder key, minutes of retry backoff). The drift worth
catching here is a renamed or deleted `evaluatorq` symbol, and an AST walk catches
that for free.

Every source is parsed, so this also serves as the syntax check that the old
`scripts/smoke_examples.py` (py_compile only) used to do — it is a superset, and
that script was removed rather than run alongside this one.

What this does NOT catch: runtime breakage that only appears when an example is
actually run. The weekly `examples-weekly.yml` workflow covers that with real
credentials on a curated subset.

Run: uv run python scripts/check_examples.py
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / 'examples'
DOCS = ROOT / 'docs'
PACKAGE = 'evaluatorq'

# Plans and specs are working notes, not published prose: their snippets are
# deliberately partial and are not code anyone copies.
DOCS_SKIP_DIRS = {'superpowers'}

# A page marks a block that is not a Python module — an API signature display, a
# fragment showing one dict entry — by putting this on the line before the fence.
# Explicit and greppable on purpose: a checker that silently skips whatever fails
# to parse cannot tell a signature display from a sample someone broke.
SKIP_MARKER = 'check-examples: skip'


def _fenced_python_blocks(text: str) -> list[tuple[int, str]]:
    """(block number, source) for each fenced ``python`` block not marked skip.

    Closes a fence only on a backtick run at least as long as the one that opened
    it, per CommonMark. Docstring samples in this repo are fenced, so a 3-backtick
    fence inside a 4-backtick block is content — closing on the inner one would
    hand the parser half a program and blame the wrong line.
    """
    out: list[tuple[int, str]] = []
    lang: str | None = None
    buf: list[str] = []
    indent = ticks = 0
    number = 0
    skip_next = False
    in_fence = False
    for line in text.splitlines():
        stripped = line.lstrip()
        run = len(stripped) - len(stripped.lstrip('`'))
        if not in_fence:
            if run >= 3:
                in_fence, ticks, indent = True, run, len(line) - len(stripped)
                lang, buf = stripped[run:].strip().lower(), []
            elif stripped:
                skip_next = SKIP_MARKER in stripped
            continue
        if run >= ticks and not stripped[run:].strip():
            in_fence = False
            if lang in {'python', 'py'}:
                number += 1
                if not skip_next:
                    out.append((number, '\n'.join(buf)))
            skip_next = False
            continue
        buf.append(line[indent:] if line[:indent].strip() == '' else line)
    return out


def _iter_sources() -> list[tuple[Path, str, str]]:
    """(path, label, source) for every example .py file, notebook cell and docs block."""
    out: list[tuple[Path, str, str]] = [
        (path, str(path.relative_to(ROOT)), path.read_text()) for path in sorted(EXAMPLES.rglob('*.py'))
    ]
    for path in sorted(DOCS.rglob('*.md')):
        rel = path.relative_to(ROOT)
        if DOCS_SKIP_DIRS.intersection(rel.parts):
            continue
        for number, source in _fenced_python_blocks(path.read_text()):
            out.append((path, f'{rel} python block {number}', source))
    for path in sorted(EXAMPLES.rglob('*.ipynb')):
        rel = path.relative_to(ROOT)
        try:
            cells = json.loads(path.read_text())['cells']
        except Exception as exc:
            print(f'FAIL {rel}: unreadable notebook: {exc}')
            continue
        for i, cell in enumerate(cells):
            if cell.get('cell_type') != 'code':
                continue
            source = ''.join(cell.get('source', ''))
            # IPython magics and shell escapes are not Python; skip those cells.
            if any(line.lstrip().startswith(('%', '!')) for line in source.splitlines()):
                continue
            out.append((path, f'{rel} cell {i}', source))
    return out


def _parse(source: str) -> ast.AST:
    """Parse a source block, tolerating a bare top-level ``await``.

    A docs sample often shows the awaited call on its own rather than wrapping it
    in a `main()` nobody would copy. That is not a module, so retry it inside an
    async function before calling it broken.

    Raises:
        SyntaxError: if it parses as neither.
    """
    try:
        return ast.parse(source)
    except SyntaxError:
        indented = '\n'.join(f'    {line}' for line in source.splitlines())
        return ast.parse(f'async def _top_level():\n{indented}')


def _imported_symbols(tree: ast.AST) -> list[tuple[str, str, int]]:
    """(module, name, lineno) for every `from evaluatorq... import name`."""
    found: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if module == PACKAGE or module.startswith(f'{PACKAGE}.'):
                found += [(module, alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.')[0]
                if root == PACKAGE:
                    found.append((alias.name, '', node.lineno))
    return found


def main() -> int:
    failures: list[str] = []
    checked = 0
    module_cache: dict[str, object | None] = {}

    for _path, label, source in _iter_sources():
        try:
            tree = _parse(source)
        except SyntaxError as exc:
            failures.append(f'{label}: syntax error: {exc}')
            continue

        for module, name, lineno in _imported_symbols(tree):
            checked += 1
            if module not in module_cache:
                try:
                    module_cache[module] = importlib.import_module(module)
                except Exception as exc:
                    module_cache[module] = None
                    failures.append(f'{label}:{lineno}: cannot import {module}: {exc}')
            mod = module_cache[module]
            if mod is None or not name:
                continue
            if not hasattr(mod, name):
                failures.append(f'{label}:{lineno}: {module} has no attribute {name!r}')

    for item in failures:
        print(f'FAIL {item}')
    print(f'\n{checked} evaluatorq imports checked, {len(failures)} failed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
