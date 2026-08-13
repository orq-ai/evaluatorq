"""Apply red-team remediation recommendations back onto the agent.

Thin wrapper over the shared engine in `evaluatorq.common.apply` (the
red-team and simulation apply flows were ~90% identical and are consolidated
there); this module supplies the red-team prompt framing and pipeline config.

Preview by default: nothing is written to the platform unless ``apply=True``.
To avoid re-applying the same fix, the caller passes the recommendations
already applied to the agent (tracked on
``RedTeamReport.applied_recommendations``); those are skipped, and the newly
applied ones come back on the result to append.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluatorq.common.apply import ApplyRecommendationsResult
from evaluatorq.common.apply import apply_recommendations as _apply_common
from evaluatorq.redteam.contracts import PIPELINE_CONFIG

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai import AsyncOpenAI

    from evaluatorq.redteam.contracts import FocusAreaRecommendation, LLMConfig

__all__ = ['ApplyRecommendationsResult', 'apply_recommendations']

_MAX_RECOMMENDATIONS = 20

_INTRO = 'You are an AI security expert. '
_CONTEXT = 'security-remediation recommendations from a red-team assessment of the agent'


async def apply_recommendations(
    focus_area_recommendations: list[FocusAreaRecommendation],
    agent_key: str,
    orq_client: Any,
    llm_client: AsyncOpenAI,
    model: str,
    *,
    apply: bool = False,
    max_recommendations: int = _MAX_RECOMMENDATIONS,
    already_applied: Sequence[str] = (),
    cfg: LLMConfig | None = None,
) -> ApplyRecommendationsResult:
    """Fold red-team recommendations into an agent's instructions.

    See `evaluatorq.common.apply.apply_recommendations` for the full
    contract; this wrapper takes ``RedTeamReport.focus_area_recommendations``
    directly and defaults ``cfg`` to the red-team pipeline config so reasoning
    models keep their required call parameters.
    """
    return await _apply_common(
        focus_area_recommendations,
        agent_key,
        orq_client,
        llm_client,
        model,
        apply=apply,
        max_recommendations=max_recommendations,
        already_applied=already_applied,
        cfg=cfg or PIPELINE_CONFIG,
        intro=_INTRO,
        context=_CONTEXT,
    )
