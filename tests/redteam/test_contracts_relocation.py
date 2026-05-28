"""Guards for the RES-808 PR2 relocation invariants.

Two things must stay true after `AgentTarget` and the agent-context models
moved into `evaluatorq.contracts`:

1. The context models re-exported from `evaluatorq.redteam.contracts` are the
   *same class objects* as the canonical ones in `evaluatorq.contracts`. If a
   future edit redefines a model locally in the redteam module instead of
   re-importing, the two become distinct classes and `isinstance` silently
   breaks across the path boundary — while the rest of the suite stays green
   (each module imports one path consistently). These tests pin the identity.

2. `AgentTarget` is a deliberate clean break: it is NOT re-exported from
   `evaluatorq.redteam.backends.base`. Re-adding that export would reintroduce
   the circular-import direction PR2 removed.
"""

from __future__ import annotations

import pytest

import evaluatorq.contracts as canonical
import evaluatorq.redteam.contracts as redteam_contracts


@pytest.mark.parametrize(
    "name",
    ["AgentContext", "ToolInfo", "MemoryStoreInfo", "KnowledgeBaseInfo"],
)
def test_context_models_are_identical_across_paths(name: str) -> None:
    """The redteam.contracts re-export must be the same object as the canonical class."""
    assert getattr(redteam_contracts, name) is getattr(canonical, name), (
        f"{name} diverged between evaluatorq.contracts and evaluatorq.redteam.contracts; "
        "the redteam module must re-import, not redefine, the relocated model"
    )


def test_agent_target_identical_via_redteam_package() -> None:
    """`evaluatorq.redteam.AgentTarget` must be the canonical contracts class."""
    import evaluatorq.redteam as redteam_pkg

    assert redteam_pkg.AgentTarget is canonical.AgentTarget


def test_agent_target_not_re_exported_from_base() -> None:
    """Clean break: AgentTarget must not be importable from redteam.backends.base."""
    import evaluatorq.redteam.backends.base as base_mod

    assert not hasattr(base_mod, "AgentTarget"), (
        "AgentTarget must NOT be re-exported from redteam.backends.base "
        "(RES-808 PR2 clean break); import it from evaluatorq.contracts"
    )

    with pytest.raises(ImportError):
        from evaluatorq.redteam.backends.base import AgentTarget  # noqa: F401
