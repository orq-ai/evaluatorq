"""Apply agent-simulation remediation suggestions back onto the agent.

Thin wrapper over the shared engine in :mod:`evaluatorq.common.apply` (the
red-team and simulation apply flows were ~90% identical and are consolidated
there); this module supplies the simulation prompt framing. The result shape
is the shared :class:`ApplyRecommendationsResult` - the surfaces returning
differently-named fields was review feedback - with ``ApplySuggestionsResult``
kept as an alias.

Preview by default: nothing is written to the platform unless ``apply=True``.
To avoid re-applying the same fix, the caller passes the suggestions already
applied to the agent (tracked on ``SimulationRun.applied_suggestions``); those
are skipped, and the newly applied ones come back on
``result.recommendations`` to append.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluatorq.common.apply import ApplyRecommendationsResult
from evaluatorq.common.apply import apply_recommendations as _apply_common

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai import AsyncOpenAI

    from evaluatorq.simulation.types import SimulationRecommendation

__all__ = ['ApplyRecommendationsResult', 'ApplySuggestionsResult', 'apply_suggestions']

# Harmonized result model (review): one shape for both surfaces.
ApplySuggestionsResult = ApplyRecommendationsResult

_MAX_SUGGESTIONS = 20

_CONTEXT = "remediation suggestions from an evaluation of the agent's behavior"


async def apply_suggestions(
    recommendations: list[SimulationRecommendation],
    agent_key: str,
    orq_client: Any,
    llm_client: AsyncOpenAI,
    model: str,
    *,
    apply: bool = False,
    # None = model default; reasoning models (e.g. gpt-5.6-luna) reject any
    # explicit temperature, so only pass one deliberately.
    temperature: float | None = None,
    max_suggestions: int = _MAX_SUGGESTIONS,
    already_applied: Sequence[str] = (),
) -> ApplyRecommendationsResult:
    """Fold simulation suggestions into an agent's instructions.

    See :func:`evaluatorq.common.apply.apply_recommendations` for the full
    contract; this wrapper takes the run's ``recommendations`` directly (their
    ``suggestions`` bullets are exposed via the harmonized ``recommendations``
    property) and keeps simulation's explicit ``temperature`` knob.
    """
    return await _apply_common(
        recommendations,
        agent_key,
        orq_client,
        llm_client,
        model,
        apply=apply,
        max_recommendations=max_suggestions,
        already_applied=already_applied,
        temperature=temperature,
        context=_CONTEXT,
    )
