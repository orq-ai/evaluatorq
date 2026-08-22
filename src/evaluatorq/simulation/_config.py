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
from evaluatorq.simulation.evaluators.scorers import SimulationScoringConfig  # noqa: TC001
from evaluatorq.simulation.hooks import SimulationHooks  # noqa: TC001
from evaluatorq.simulation.reports.recommendations import SimulationRecommendationConfig  # noqa: TC001
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
    scoring: SimulationScoringConfig | None = None
    """Policy knobs for the ``turn_efficiency`` / ``conversation_quality`` scorers
    (cliffs, decay, floor, composite weights). ``None`` uses
    ``DEFAULT_SCORING_CONFIG`` — the shipped defaults."""
    datapoint_parallelism: int = 10
    target_agent_timeout_ms: int = 240_000
    """Per-call timeout for the target under test, threaded into
    ``SimulationRunner``. Mirrors red team's equivalent knob — a slow
    self-hosted target needs this raised; ``EVALUATORQ_LLM_TIMEOUT_S`` only
    covers the simulator's own (user-simulator / judge) LLM calls."""
    max_target_retries: int = 2
    """Retries for a failed target call, threaded into ``SimulationRunner``.
    Mirrors red team's equivalent knob."""
    target_reasoning_effort: str | None = None
    """Reasoning effort pinned on the agent *under test* (``agent:<key>`` /
    bare ``<key>`` targets only) — the simulation counterpart of red team's
    ``LLMConfig.target_reasoning_effort``. ``None`` leaves the provider
    default. Distinct from the simulator's own ``EVALUATORQ_REASONING_EFFORT``
    fallback (see ``simulation.agents.base``), which drives the user-simulator
    and judge, not the target."""
    max_tool_result_chars: int = 500
    """Cap on each tool result rendered into the text the user-simulator sees
    for a tool-only turn (``runner.simulation._tool_traffic_text``). 500 by
    default; raise it for a tool-heavy agent whose results are being cut
    before the simulator — and the judge, which scores the same transcript —
    can react to them."""
    per_simulation_timeout_s: float | None = None
    """Overall wall-clock bound for one datapoint's simulation (all turns,
    target + user-simulator + judge calls included), applied via
    ``SimulationRunner._run_with_timeout`` — the same guard ``run_batch`` uses.
    ``None`` (the default) leaves no bound beyond the per-call timeouts
    (``target_agent_timeout_ms`` / the simulator's own LLM timeout): a stalled
    conversation was previously unbounded on the ``simulate()`` path, since it
    calls ``runner.run()`` per row instead of ``run_batch``."""
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
    recommendations: SimulationRecommendationConfig | None = None
    """Generate remediation suggestions in-core (before save), mirroring red teaming's
    ``recommendations=`` on ``red_team``. ``None`` means off. The public
    ``simulate``/``generate_and_simulate`` leave it off (they return bare results, which
    have nowhere to carry suggestions); the CLI resolves its ``--recommendations`` flag
    into a config here."""

    executive_summary: bool = False
    """Generate the LLM narrative summary in-core (before save). Off by default;
    the public ``simulate``/``generate_and_simulate`` flip it on. The CLI keeps
    it off here and generates its own after the run (avoids double generation)."""
