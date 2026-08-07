"""The core-dependency check must name the interpreter it ran under.

A missing ``openai``/``typer`` almost always means pip installed into a
different environment than the one executing, so the message has to point at
``sys.executable`` rather than suggesting an extra.
"""

from __future__ import annotations

import sys

import pytest

import evaluatorq


def test_passes_when_core_deps_present() -> None:
    evaluatorq._require_core_deps()


def test_names_missing_dep_and_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluatorq.importlib.util,
        'find_spec',
        lambda name: None if name == 'openai' else object(),
    )

    with pytest.raises(ImportError) as excinfo:
        evaluatorq._require_core_deps()

    message = str(excinfo.value)
    assert 'openai' in message
    assert 'typer' not in message
    assert sys.executable in message
    assert 'evaluatorq[redteam]' not in message
