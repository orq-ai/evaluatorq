"""Every name in ``evaluatorq.redteam.__all__`` must actually be importable.

``ORQAgentTarget`` was documented and referenced in docstrings while missing from
this package's ``__init__``, so ``from evaluatorq.redteam import ORQAgentTarget``
raised ImportError. Ruff's F822 only catches the reverse (a name in ``__all__``
with nothing behind it) when the module has no ``__getattr__`` — this one has.
"""

from __future__ import annotations

import evaluatorq.redteam as rt
import pytest


@pytest.mark.parametrize('name', sorted(rt.__all__))
def test_public_name_is_importable(name: str) -> None:
    assert getattr(rt, name, None) is not None


def test_orq_agent_target_is_exported() -> None:
    from evaluatorq.redteam import ORQAgentTarget
    from evaluatorq.redteam.backends.orq import ORQAgentTarget as _Canonical

    assert ORQAgentTarget is _Canonical
