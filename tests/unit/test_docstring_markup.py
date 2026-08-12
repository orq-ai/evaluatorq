"""Sphinx markup in docstrings renders as literal text on the docs site.

Sphinx has never run on this repo — mkdocstrings passes `:class:`X`` and
`.. note::` straight through, so they appear verbatim on the API reference.
227 roles and 2 directives accumulated before anyone noticed. This fails the
moment one comes back, which is cheaper than another site-wide sweep.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

# `:class:`Thing``, `:func:`~pkg.mod.thing``, etc. Use autorefs (`[Text][path]`)
# or plain inline code instead.
ROLE = re.compile(r":(?:class|func|meth|attr|mod|obj|exc|data|ref):`")
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
