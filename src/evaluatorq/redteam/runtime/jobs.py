"""Generic evaluatorq model-under-test job factories."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq import DataPoint, Job, job
from evaluatorq.common.llm_call import apply_pipeline_metadata, execute_chat_completion
from evaluatorq.common.llm_client import client_routes_through_orq
from evaluatorq.common.messages import coerce_content_text
from evaluatorq.common.thread_context import build_static_thread_id, conversation_thread, thread_body_param
from evaluatorq.common.tracing import record_llm_response, set_span_attrs, truncate_for_span
from evaluatorq.redteam.adaptive.orchestrator import _get_active_progress
from evaluatorq.redteam.backends.registry import create_async_llm_client
from evaluatorq.redteam.contracts import Message, TokenUsage
from evaluatorq.redteam.exceptions import CredentialError
from evaluatorq.redteam.tracing import with_llm_span, with_redteam_span

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def _sanitize_job_name(value: str) -> str:
    """Sanitize a value for use in job names (alphanumeric, dash, underscore)."""
    return ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '-' for ch in value).strip('-') or 'unknown'


def _static_attack_attrs(data: DataPoint) -> dict[str, str]:
    """Return the common trace attributes for a static red-team datapoint."""
    return {
        'orq.redteam.category': data.inputs.get('category', ''),
        'orq.redteam.vulnerability': data.inputs.get('vulnerability', data.inputs.get('category', '')),
        'orq.redteam.strategy_name': data.inputs.get('strategy_name', ''),
    }


def _static_target_input(messages: list[dict[str, Any]]) -> str:
    """Flatten the exact static request messages for bounded trace capture."""
    return truncate_for_span('\n\n'.join(coerce_content_text(message.get('content')) for message in messages))


def create_model_job(
    model: str | None = None,
    deployment_key: str | None = None,
    llm_client: AsyncOpenAI | None = None,
    system_prompt: str | None = None,
    max_tokens: int = 5000,
    run_id: str | None = None,
) -> Job:
    """Create an evaluatorq job for a router model or ORQ deployment.

    ``max_tokens`` is applied to direct model calls (router jobs).
    Deployment targets manage their own token limits via platform configuration,
    so ``max_tokens`` is ignored for that path.

    Args:
        model: Model name for direct LLM calls via the ORQ router or OpenAI.
        deployment_key: ORQ deployment key for deployment-based inference.
        max_tokens: Maximum tokens for direct model responses (default 5000).
        run_id: Red-team run id used to build the static-trace thread id so
            job spans correlate with the red-team pipeline; a per-target
            fallback is used when omitted.

    Returns:
        An evaluatorq Job.

    Raises:
        ValueError: If no target parameter is provided.
    """
    if deployment_key:
        safe_key = _sanitize_job_name(deployment_key)

        try:
            from orq_ai_sdk import Orq
        except ImportError as e:
            msg = (
                'Deployment jobs require the orq-ai-sdk package. '
                'Install it with: uv add "evaluatorq[orq]" (or: python -m pip install "evaluatorq[orq]")'
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
            attack_attrs = _static_attack_attrs(data)
            target_input = _static_target_input(messages)
            thread_id = build_static_thread_id(run_id, safe_key, _row)
            async with (
                with_redteam_span('orq.redteam.attack', attack_attrs),
                with_redteam_span(
                    'orq.redteam.target_call',
                    {
                        **attack_attrs,
                        'input': target_input,
                        'orq.redteam.input': target_input,
                    },
                ) as target_span,
            ):
                with conversation_thread(thread_id):
                    async with with_llm_span(
                        model=f'deployment:{deployment_key}',
                        operation='invoke',
                        provider='orq',
                        input_messages=messages,
                        attributes={'orq.redteam.llm_purpose': 'target'},
                    ) as llm_span:
                        invoke_kwargs: dict[str, Any] = {
                            'key': deployment_key,
                            'messages': messages,
                        }
                        # Deployment SDK calls do not go through the shared
                        # chat-completion helper, so apply the active run
                        # metadata explicitly. The helper preserves any
                        # caller-supplied metadata if this path gains one.
                        apply_pipeline_metadata(invoke_kwargs)
                        invoke_kwargs.update(thread_body_param())
                        completion = await deployment_client.deployments.invoke_async(**invoke_kwargs)
                        content = _extract_deployment_content(completion)
                        record_llm_response(llm_span, completion, output_content=content)
                        output = truncate_for_span(content)
                        set_span_attrs(target_span, {'output': output, 'orq.redteam.output': output})

            # Advance the global progress bar for static attacks.
            active_progress = _get_active_progress()
            if active_progress is not None:
                await active_progress.finish_attack(None)

            return {
                'response': content,
                'token_usage': TokenUsage.from_completion(completion),
                'thread_id': thread_id,
            }

        return deployment_job

    if model is None:
        msg = "Provide one of: 'model' or 'deployment_key'"
        raise ValueError(msg)

    safe_model = _sanitize_job_name(model)

    @job(f'redteam:static:{safe_model}')
    async def router_job(data: DataPoint, _row: int) -> dict[str, Any]:
        """Call the router model and return the response with token usage and finish reason."""
        messages = _build_messages(data)
        if system_prompt:
            messages = [{'role': 'system', 'content': system_prompt}, *messages]
        client = llm_client or create_async_llm_client()
        attack_attrs = _static_attack_attrs(data)
        target_input = _static_target_input(messages)
        thread_id = build_static_thread_id(run_id, safe_model, _row)
        async with (
            with_redteam_span('orq.redteam.attack', attack_attrs),
            with_redteam_span(
                'orq.redteam.target_call',
                {
                    **attack_attrs,
                    'input': target_input,
                    'orq.redteam.input': target_input,
                },
            ) as target_span,
        ):
            with conversation_thread(thread_id):
                async with with_llm_span(
                    model=model,
                    max_tokens=max_tokens,
                    input_messages=messages,
                    attributes={'orq.redteam.llm_purpose': 'target'},
                ) as llm_span:
                    extra_kwargs: dict[str, Any] = {}
                    if client_routes_through_orq(client):
                        # Run metadata is applied natively by execute_chat_completion
                        # (llm_call.apply_pipeline_metadata); only thread grouping here.
                        extra_kwargs['extra_body'] = thread_body_param()
                    # ponytail: fixed 300s ceiling (was unbounded); thread a cfg
                    # target timeout through create_model_job if per-run tuning is needed.
                    response, _ = await execute_chat_completion(
                        client=client,
                        model=model,
                        messages=messages,
                        span=llm_span,
                        timeout_s=300.0,
                        max_tokens=max_tokens,
                        extra_kwargs=extra_kwargs or None,
                    )
                    content = response.choices[0].message.content or ''
                    output = truncate_for_span(content)
                    set_span_attrs(target_span, {'output': output, 'orq.redteam.output': output})
        if not content:
            sample_id = data.inputs.get('id', 'unknown')
            finish_reason = response.choices[0].finish_reason
            logger.warning(
                f'Empty router response for {sample_id}: '
                f'content={response.choices[0].message.content}, finish_reason={finish_reason}'
            )
        # Advance the global progress bar for static attacks.
        active_progress = _get_active_progress()
        if active_progress is not None:
            await active_progress.finish_attack(None)

        return {
            'response': content,
            'token_usage': TokenUsage.from_completion(response),
            'finish_reason': response.choices[0].finish_reason,
            'thread_id': thread_id,
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
        logger.warning(
            f'Unexpected message type {type(raw).__name__} in DataPoint, coercing to string: {str(raw)[:100]}'
        )
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


def _normalize_usage(raw_usage: Any) -> TokenUsage | None:
    """Normalize usage payloads to TokenUsage via the shared extractor.

    Accepts both the new (``input/output_tokens``) and legacy
    (``prompt/completion_tokens``) shapes, carries cached/reasoning/cost, and
    honours a ``calls`` count already present in the payload (no longer hardcoded).
    """
    if isinstance(raw_usage, TokenUsage):
        return raw_usage
    if not isinstance(raw_usage, dict):
        return None
    return TokenUsage.extract(raw_usage)
