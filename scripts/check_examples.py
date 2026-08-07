#!/usr/bin/env python
"""Check every example still refers to symbols evaluatorq actually exports.

Static only — examples are parsed, never executed. That is deliberate: several
build an LLM client and call it at module level, so importing them means real
network traffic (and, with a placeholder key, minutes of retry backoff). The
drift worth catching here is a renamed or deleted `evaluatorq` symbol, and an AST
walk catches that for free.

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
PACKAGE = 'evaluatorq'


def _iter_sources() -> list[tuple[Path, str, str]]:
    """(path, label, source) for every example .py file and notebook code cell."""
    out: list[tuple[Path, str, str]] = [
        (path, str(path.relative_to(ROOT)), path.read_text()) for path in sorted(EXAMPLES.rglob('*.py'))
    ]
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
            tree = ast.parse(source)
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
