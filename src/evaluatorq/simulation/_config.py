"""Internal ``SimulationConfig`` — carries shared simulation params through the
internal call layers (``_simulate_run`` / ``_generate_and_simulate_run`` ->
``_simulate_core`` -> ``_simulate_via_evaluatorq``).

INTERNAL ONLY. Not exported from ``evaluatorq.simulation``'s public surface —
the public ``simulate()`` / ``generate_and_simulate()`` / ``generate()``
signatures are unaffected by this type; it exists purely to reduce the
parameter-threading boilerplate between the internal layers.

Field names here are the canonical INTERNAL names, which intentionally diverge
from the public keyword names at the one seam where they're built
(``_simulate_run`` / ``_generate_and_simulate_run``):

* ``model`` (was the public ``sim_model``)
* ``run_output`` (was the public ``report``)

Never call ``model_dump`` / serialize this model — several fields hold
callables, an ``AsyncOpenAI`` client, an ``AgentTarget`` instance, and hook
objects that aren't JSON-serializable.
"""

from __future__ import annotations

from collections.abc import (  # noqa: TC003 — used in real (non-TYPE_CHECKING) field annotations pydantic resolves at runtime
    Awaitable,
    Callable,
)
from pathlib import Path  # noqa: TC003 — used in a real (non-TYPE_CHECKING) field annotation, see note below
from typing import Any

from pydantic import BaseModel, ConfigDict

# NOTE: these two are used directly in field annotations below (not just for
# static typing) — pydantic resolves annotations at class-creation time even
# with `from __future__ import annotations`, so they must be real, importable
# names in this module's namespace and can't live behind `TYPE_CHECKING`.
from evaluatorq.contracts import AgentTarget  # noqa: TC001
from evaluatorq.simulation.hooks import SimulationHooks  # noqa: TC001
from evaluatorq.simulation.types import DEFAULT_MODEL, Message, Persona, Scenario, SimulationDatapoint


class SimulationConfig(BaseModel):
    """Shared simulation parameters, threaded through the internal call layers.

    Built once at the CLI/SDK seam (``_simulate_run`` /
    ``_generate_and_simulate_run``) and read from thereafter by
    ``_simulate_core`` and ``_simulate_via_evaluatorq``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    # --- Run identity / target resolution inputs -------------------------
    evaluation_name: str = ''
    target: str | Callable[[list[Message]], str | Awaitable[str]] | AgentTarget | None = None
    personas: list[Persona] | None = None
    scenarios: list[Scenario] | None = None
    datapoints: list[SimulationDatapoint] | None = None
    dataset_id: str | None = None
    experiment_id: str | None = None
    experiment_run_id: str | None = None
    memory_entity_id: str | None = None
    """Memory ``entity_id`` sent with every ``agent:<key>`` (or bare ``<key>``)
    target call — the Responses router requires a memory scope when the target
    agent has a memory store. ``None`` means a fresh per-target id is minted."""
    previous_run: str | None = None

    # --- Run behaviour -----------------------------------------------------
    max_turns: int | None = None
    """None means "unset": resolved in ``_simulate_core`` to a replayed run's
    cap when replaying, else to ``DEFAULT_MAX_TURNS``."""
    model: str = DEFAULT_MODEL
    evaluator_names: list[str] | None = None
    parallelism: int = 5
    user_simulator: Any = None
    """``BaseAgent | None`` — ``Any`` at runtime, see module note above."""
    judge: Any = None
    """``BaseAgent | None`` — ``Any`` at runtime, see module note above."""
    hooks: SimulationHooks | None = None
    generation_client: Any = None
    """``AsyncOpenAI | None`` — ``Any`` at runtime, see module note above."""
    upload_results: bool = True
    evaluation_description: str | None = None
    orq_results_path: str | None = None
    exit_on_failure: bool = True
    save: bool = False
    run_output: str | Path | None = None
    executive_summary: bool = False
    """Generate the LLM narrative summary in-core (before save). Off by default;
    the public ``simulate``/``generate_and_simulate`` flip it on. The CLI keeps
    it off here and generates its own after the run (avoids double generation)."""
