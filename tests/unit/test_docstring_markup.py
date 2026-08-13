"""Sphinx markup in docstrings renders as literal text on the docs site.

Sphinx has never run on this repo — mkdocstrings passes `:class:`X`` and
`.. note::` straight through, so they appear verbatim on the API reference.
227 roles and 2 directives accumulated before anyone noticed. This fails the
moment one comes back, which is cheaper than another site-wide sweep.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

# `:class:`Thing``, `:func:`~pkg.mod.thing``, etc. Use autorefs (`[Text][path]`)
# or plain inline code instead. Matched generically rather than as a fixed keyword
# list: `:command:`, `:type:` and `:py:class:` render just as literally, and an
# allowlist only fails once someone reaches for a role that isn't on it.
ROLE = re.compile(r":[a-z][a-z:+-]*:`")
# `.. note::`, `.. deprecated::`, `.. versionadded::`, ... Write plain prose.
DIRECTIVE = re.compile(r"^\s*\.\. [a-z]+::", re.MULTILINE)


@pytest.mark.parametrize("pattern", [ROLE, DIRECTIVE], ids=["role", "directive"])
def test_no_sphinx_markup_in_src(pattern: re.Pattern[str]) -> None:
    offenders = [
        f"{path.relative_to(SRC)}:{text.count(chr(10), 0, m.start()) + 1}"
        for path in sorted(SRC.rglob("*.py"))
        for text in [path.read_text()]
        for m in pattern.finditer(text)
    ]
    assert not offenders, "Sphinx markup renders literally on the docs site: " + ", ".join(offenders)


# `Usage:` too — the integration targets spell their samples that way, and an
# indented body under it renders exactly as badly as one under `Example:`.
EXAMPLE_HEADING = re.compile(r"^(\s*)(?:Examples?|Usage):\s*$")


def _unfenced_example_sections(doc: str) -> list[str]:
    """`Example:` bodies that are indented code instead of a ```python fence.

    An indented body reaches the renderer as prose: multi-line code collapses into
    one paragraph and a leading `# comment` becomes an <h3> with its own ToC entry.
    docs/hooks.py cannot see this — it only inspects <pre> elements, and this defect
    never produces one.
    """
    lines = doc.splitlines()
    offenders = []
    for i, line in enumerate(lines):
        heading = EXAMPLE_HEADING.match(line)
        if not heading:
            continue
        indent = len(heading.group(1))
        body = []
        for nxt in lines[i + 1 :]:
            if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
                break
            body.append(nxt)
        text = textwrap.dedent("\n".join(body)).strip()
        if text and not text.startswith("```"):
            offenders.append(text.splitlines()[0])
    return offenders


def test_example_sections_are_fenced() -> None:
    offenders = [
        f"{path.relative_to(SRC)}: {first!r}"
        for path in sorted(SRC.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        for doc in [ast.get_docstring(node, clean=False)]
        if doc
        for first in _unfenced_example_sections(doc)
    ]
    assert not offenders, "Example: sections must fence their code as ```python: " + ", ".join(offenders)
