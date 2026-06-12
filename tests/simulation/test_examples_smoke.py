"""Import-only smoke tests for examples/agent_simulation/*.

Catches drift between evaluatorq.simulation internals and the public examples
(API renames, removed imports, etc.) without running any live API calls.

Examples 03/04 depend on the `agent-simulation` research package which is not
installed by default — those are skipped when the package is missing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "agent_simulation"

RESEARCH_PACKAGE_REQUIRED = {"03_tool_simulation", "04_hardening_loop"}


def _load(name: str) -> None:
    path = EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_example_{name}", path)
    assert spec is not None  # noqa: S101
    assert spec.loader is not None  # noqa: S101
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "example",
    [
        "01_basic_simulation",
        "02_orq_deployment_simulation",
        "03_tool_simulation",
        "04_hardening_loop",
        "05_wrap_and_experiment",
    ],
)
def test_example_imports(example: str) -> None:
    if example in RESEARCH_PACKAGE_REQUIRED:
        if importlib.util.find_spec("agent_simulation") is None:
            pytest.skip(f"{example} requires the agent-simulation research package")
    _load(example)
