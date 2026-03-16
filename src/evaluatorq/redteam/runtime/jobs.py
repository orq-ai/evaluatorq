"""Generic evaluatorq model-under-test job factories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluatorq import DataPoint, Job, job
from loguru import logger

from evaluatorq.redteam.backends.registry import create_async_llm_client
from pydantic import ValidationError

from evaluatorq.redteam.contracts import Message, TokenUsage
from evaluatorq.redteam.runtime.orq_agent_job import create_orq_platform_agent_job

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def create_model_job(
    model: str | None = None,
    deployment_key: str | None = None,
    agent_key: str | None = None,
    llm_client: AsyncOpenAI | None = None,
) -> Job:
    """Create an evaluatorq job for router model, deployment, or ORQ agent.

    Args:
        model: Model name for direct LLM calls via the ORQ router or OpenAI.
        deployment_key: ORQ deployment key for deployment-based inference.
        agent_key: ORQ platform agent key.

    Returns:
        An evaluatorq Job.

    Raises:
        ValueError: If no target parameter is provided.
    """
    if agent_key:
        return create_orq_platform_agent_job(agent_key)

    if deployment_key:
        try:
            from orq_ai_sdk import Orq
        except ImportError as e:
            msg = (
                'Deployment jobs require the orq-ai-sdk package. '
                'Install it with: pip install evaluatorq[orq]'
            )
            raise ImportError(msg) from e

        import os

        deployment_client = Orq(api_key=os.environ.get('ORQ_API_KEY', ''))

        @job('model-under-test')
        async def deployment_job(data: DataPoint, _row: int) -> dict[str, Any]:
            messages = _build_messages(data)
            completion = await deployment_client.deployments.invoke_async(
                key=deployment_key,
                messages=messages,  # pyright: ignore[reportArgumentType]
            )

            return {
                'response': _extract_deployment_content(completion),
                'token_usage': _extract_usage_from_deployment_completion(completion),
            }

        return deployment_job

    if model is None:
        msg = "Provide one of: 'model', 'deployment_key', or 'agent_key'"
        raise ValueError(msg)

    resolved_client = llm_client or create_async_llm_client()

    @job('model-under-test')
    async def router_job(data: DataPoint, _row: int) -> dict[str, Any]:
        messages = _build_messages(data)
        response = await resolved_client.chat.completions.create(
            model=model,
            messages=messages,  # pyright: ignore[reportArgumentType]
        )
        content = response.choices[0].message.content or ''
        if not content:
            sample_id = data.inputs.get('id', 'unknown')
            finish_reason = response.choices[0].finish_reason
            logger.warning(
                f'Empty router response for {sample_id}: '
                + f'content={response.choices[0].message.content}, finish_reason={finish_reason}'
            )
        return {
            'response': content,
            'token_usage': _extract_usage_from_chat_completion(response),
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
            except (ValidationError, TypeError):
                messages.append(dict(raw))
            continue
        messages.append({'role': 'user', 'content': str(raw)})
    return messages


def _extract_deployment_content(completion: object) -> str:
    """Extract text content from an ORQ deployment response."""
    choices = getattr(completion, 'choices', None)
    if not choices:
        return ''

    message = getattr(choices[0], 'message', None)
    if not message:
        return ''

    msg_content = getattr(message, 'content', None)
    if isinstance(msg_content, str):
        return msg_content
    if isinstance(msg_content, list):
        return '\n'.join(
            str(getattr(part, 'text', '')) for part in msg_content if getattr(part, 'type', None) == 'text'
        )
    return ''


def _safe_int(value: Any) -> int:
    """Convert to int, returning 0 for non-numeric values."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _extract_usage_from_chat_completion(response: Any) -> TokenUsage | None:
    """Extract token usage from OpenAI-compatible completion responses."""
    usage = getattr(response, 'usage', None)
    if usage is None:
        return None

    prompt = _safe_int(getattr(usage, 'prompt_tokens', 0))
    completion = _safe_int(getattr(usage, 'completion_tokens', 0))
    total = _safe_int(getattr(usage, 'total_tokens', prompt + completion))
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _extract_usage_from_deployment_completion(completion: Any) -> TokenUsage | None:
    """Best-effort extraction of usage from ORQ deployment completion objects."""
    usage = getattr(completion, 'usage', None)
    if usage is None:
        return None

    prompt = _safe_int(getattr(usage, 'prompt_tokens', 0))
    completion_tokens = _safe_int(getattr(usage, 'completion_tokens', 0))
    total = _safe_int(getattr(usage, 'total_tokens', prompt + completion_tokens))
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion_tokens, total_tokens=total)
