"""Opt-in Jev classification for tool names missing from a signals config."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from loguru import logger

from evaluatorq.common.judge import ClassifyQuestion, run_judge
from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.signals.config import SignalsConfig, ToolRole
from evaluatorq.signals.walk import calls, walk

if TYPE_CHECKING:
    from collections.abc import Iterable

    from openai import AsyncOpenAI

    from evaluatorq.formats.atif import AtifTrajectory

_ROLE_CRITERIA: dict[str, str | None] = {
    'bash': 'Runs shell commands or otherwise executes operating-system commands.',
    'webview': 'Browses, fetches, or views web pages and other online content.',
    'skill': 'Loads or invokes a named skill or reusable capability.',
    'subagent': 'Delegates work to another agent or starts a child agent.',
    'other': 'Does not fit any of the other tool roles.',
}


async def classify_tool_roles(
    trajectories: Iterable[AtifTrajectory],
    config: SignalsConfig | None = None,
    *,
    client: AsyncOpenAI | None = None,
) -> SignalsConfig:
    """Classify unknown tool names with Jev and return a config with merged roles.

    Existing ``tool_roles`` entries always win. Each unknown name is sent once,
    with the first call's arguments as context. Failed, abstained, or invalid
    classifications are recorded as ``'other'`` and logged. The resolved client
    is closed when this function creates it; an injected client remains caller-owned.
    """
    resolved_config = config or SignalsConfig()
    samples: dict[str, dict[str, object]] = {}
    for trajectory in trajectories:
        for record in calls(walk(trajectory)):
            name = record.call.function_name
            if name not in resolved_config.tool_roles:
                samples.setdefault(name, record.call.arguments)
    if not samples:
        return resolved_config

    resolved_client = None
    active_client = client
    if active_client is None:
        try:
            resolved_client = resolve_llm_client(max_retries=0)
            active_client = resolved_client.client
        except Exception as exc:  # noqa: BLE001 - unavailable credentials degrades each requested classification
            for name in samples:
                logger.warning('Tool role classification failed for tool {}: {}', name, exc)
            return resolved_config.model_copy(
                update={'tool_roles': {**resolved_config.tool_roles, **dict.fromkeys(samples, 'other')}}
            )

    roles: dict[str, ToolRole] = {}
    try:
        for name, arguments in samples.items():
            question = ClassifyQuestion(
                kind='choice',
                instructions='Choose the role that best describes this tool.',
                criteria=_ROLE_CRITERIA,
                state={'tool_name': name, 'sample_arguments': arguments},
            )
            try:
                outcome = await run_judge(
                    client=active_client,
                    model=resolved_config.classifier.model,
                    cfg=LLMCallConfig(model=resolved_config.classifier.model, timeout_ms=90_000),
                    prompt_template='',
                    replacements={},
                    span_attributes={'orq.llm.purpose': 'judge'},
                    classify=question,
                )
                value = outcome.payload.value if outcome.payload is not None else None
                if outcome.error_kind is not None or outcome.payload is None or outcome.payload.abstain:
                    raise ValueError(outcome.error_message or 'classifier returned no decisive verdict')
                if not isinstance(value, str) or value not in _ROLE_CRITERIA:
                    raise ValueError(f'classifier returned invalid role {value!r}')
                roles[name] = cast('ToolRole', value)
            except Exception as exc:  # noqa: BLE001 - one tool's failure must not lose other classifications
                logger.warning('Tool role classification failed for tool {}: {}', name, exc)
                roles[name] = 'other'
    finally:
        if resolved_client is not None and resolved_client.owned:
            await resolved_client.client.close()

    # Reapply caller entries last in case a caller supplied a subclass or unusual mapping.
    return resolved_config.model_copy(update={'tool_roles': {**roles, **resolved_config.tool_roles}})
