"""Guards against public symbols silently vanishing from the API reference.

``evaluatorq/__init__.py`` re-exports several functions from a sub-module of the
same name (``from .evaluatorq import evaluatorq``). At runtime the function wins;
griffe resolves statically, where the sub-module does — so mkdocstrings emitted
nothing at all for ``evaluatorq()``, ``deployment()`` and ``llm_jury()`` even
though every one of them is pinned in the generated ``members:`` list. Nothing
warned: ``mkdocs build --strict`` stayed green with the symbols missing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs"))

gen_pages = pytest.importorskip(
    "gen_pages", reason="docs dependency group not installed"
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("evaluatorq", "evaluatorq.evaluatorq.evaluatorq"),
        ("deployment", "evaluatorq.deployment.deployment"),
        ("llm_jury", "evaluatorq.llm_jury.llm_jury"),
    ],
)
def test_submodule_shadowed_symbols_are_detected(name: str, expected: str) -> None:
    mod = importlib.import_module("evaluatorq")
    assert name in mod.__all__, f"{name} is no longer exported; drop this case"
    assert gen_pages._submodule_shadowed(mod, "evaluatorq", name) == expected


@pytest.mark.parametrize("name", ["job", "DataPoint", "llm_jury_pairwise"])
def test_unshadowed_symbols_are_left_on_the_package_page(name: str) -> None:
    """Only same-named-module clashes get the escape hatch; everything else
    stays in the package's own ``members:`` list."""
    mod = importlib.import_module("evaluatorq")
    assert gen_pages._submodule_shadowed(mod, "evaluatorq", name) is None


def test_every_shadowed_symbol_resolves_to_a_real_object() -> None:
    """The generated ``::: pkg.mod.symbol`` path must actually exist, or
    mkdocstrings emits a heading for nothing."""
    mod = importlib.import_module("evaluatorq")
    for name in mod.__all__:
        path = gen_pages._submodule_shadowed(mod, "evaluatorq", name)
        if path is None:
            continue
        module_path, _, symbol = path.rpartition(".")
        assert hasattr(importlib.import_module(module_path), symbol), path
