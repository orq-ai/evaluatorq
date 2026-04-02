"""Generic evaluatorq model-under-test job factories."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from evaluatorq import DataPoint, Job, job
from loguru import logger

from evaluatorq.redteam.adaptive.orchestrator import _get_active_progress
from evaluatorq.redteam.backends.registry import create_async_llm_client
from evaluatorq.redteam.contracts import Message, TokenUsage
from evaluatorq.redteam.exceptions import CredentialError
from evaluatorq.redteam.runtime.orq_agent_job import _sanitize_job_name, create_orq_platform_agent_job

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def create_model_job(
    model: str | None = None,
    deployment_key: str | None = None,
    agent_key: str | None = None,
    llm_client: AsyncOpenAI | None = None,
    system_prompt: str | None = None,
    max_tokens: int = 5000,
) -> Job:
    """Create an evaluatorq job for router model, deployment, or ORQ agent.

    ``max_tokens`` is applied to direct model calls (router jobs).
    Agent and deployment targets manage their own token limits via platform
    configuration, so ``max_tokens`` is ignored for those paths.

    Args:
        model: Model name for direct LLM calls via the ORQ router or OpenAI.
        deployment_key: ORQ deployment key for deployment-based inference.
        agent_key: ORQ platform agent key (``max_tokens`` is not applied).
        max_tokens: Maximum tokens for direct model responses (default 5000).

    Returns:
        An evaluatorq Job.

    Raises:
        ValueError: If no target parameter is provided.
    """
    if agent_key:
        return create_orq_platform_agent_job(agent_key)

    if deployment_key:
        safe_key = _sanitize_job_name(deployment_key)

        try:
            from orq_ai_sdk import Orq
        except ImportError as e:
            msg = (
                'Deployment jobs require the orq-ai-sdk package. '
                'Install it with: pip install evaluatorq[orq]'
            )
            raise ImportError(msg) from e

        api_key = os.environ.get('ORQ_API_KEY')
        if not api_key:
            raise CredentialError('ORQ_API_KEY environment variable is not set')
        deployment_client = Orq(api_key=api_key)

        @job(f'redteam:static:{safe_key}')
        async def deployment_job(data: DataPoint, _row: int) -> dict[str, Any]:
            """Invoke the ORQ deployment and return the response with token usage."""
            messages = _build_messages(data)
            completion = await deployment_client.deployments.invoke_async(
                key=deployment_key,
                messages=messages,  # pyright: ignore[reportArgumentType]
            )

            # Advance the global progress bar for static attacks.
            _active_progress = _get_active_progress()
            if _active_progress is not None:
                await _active_progress.finish_attack(None)

            return {
                'response': _extract_deployment_content(completion),
                'token_usage': TokenUsage.from_completion(completion),
            }

        return deployment_job

    if model is None:
        msg = "Provide one of: 'model', 'deployment_key', or 'agent_key'"
        raise ValueError(msg)

    safe_model = _sanitize_job_name(model)

    @job(f'redteam:static:{safe_model}')
    async def router_job(data: DataPoint, _row: int) -> dict[str, Any]:
        """Call the router model and return the response with token usage and finish reason."""
        messages = _build_messages(data)
        if system_prompt:
            messages = [{'role': 'system', 'content': system_prompt}] + messages
        client = llm_client or create_async_llm_client()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,  # pyright: ignore[reportArgumentType]
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ''
        if not content:
            sample_id = data.inputs.get('id', 'unknown')
            finish_reason = response.choices[0].finish_reason
            logger.warning(
                f'Empty router response for {sample_id}: '
                f'content={response.choices[0].message.content}, finish_reason={finish_reason}'
            )
        # Advance the global progress bar for static attacks.
        _active_progress = _get_active_progress()
        if _active_progress is not None:
            await _active_progress.finish_attack(None)

        return {
            'response': content,
            'token_usage': TokenUsage.from_completion(response),
            'finish_reason': response.choices[0].finish_reason,
        }

    return router_job


def _build_messages(data: DataPoint) -> list[dict[str, Any]]:
    """Extract messages from a DataPoint and normalize known fields."""
    messages: list[dict[str, Any]] = []
    for raw in list(data.inputs.get('messages', [])):
        if isinstance(raw, Message):
            messages.append(raw.model_dump(mode='json', exclude_none=True))
            continue
        if isinstance(raw, dict):
            try:
                parsed = Message.model_validate(raw)
                messages.append(parsed.model_dump(mode='json', exclude_none=True))
            except Exception as e:
                logger.debug(f'Message validation failed, using raw dict: {e}')
                messages.append(dict(raw))
            continue
        logger.warning(f'Unexpected message type {type(raw).__name__} in DataPoint, coercing to string: {str(raw)[:100]}')
        messages.append({'role': 'user', 'content': str(raw)})
    return messages


def _extract_deployment_content(completion: object) -> str:
    """Extract text content from an ORQ deployment response."""
    choices = getattr(completion, 'choices', None)
    if not choices:
        logger.warning(f'Deployment returned no choices: {type(completion).__name__}')
        return ''

    message = getattr(choices[0], 'message', None)
    if not message:
        logger.warning('Deployment choice has no message')
        return ''

    msg_content = getattr(message, 'content', None)
    if isinstance(msg_content, str):
        return msg_content
    if isinstance(msg_content, list):
        return '\n'.join(
            str(getattr(part, 'text', '')) for part in msg_content if getattr(part, 'type', None) == 'text'
        )
    logger.warning(f'Unexpected content type in deployment response: {type(msg_content).__name__}')
    return ''


def _safe_int(value: Any) -> int:
    """Convert to int, returning 0 for non-numeric values."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_usage(raw_usage: Any) -> TokenUsage | None:
    """Normalize usage payloads to TokenUsage."""
    if isinstance(raw_usage, TokenUsage):
        return raw_usage
    if not isinstance(raw_usage, dict):
        return None

    prompt = _safe_int(raw_usage.get('prompt_tokens', raw_usage.get('prompt', 0)))
    completion = _safe_int(raw_usage.get('completion_tokens', raw_usage.get('completion', 0)))
    total = _safe_int(raw_usage.get('total_tokens', raw_usage.get('total', prompt + completion)))
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


