"""ORQ platform-agent evaluatorq job utilities."""

from __future__ import annotations

import uuid
from typing import Any

from evaluatorq import DataPoint, Job, job
from loguru import logger

from pydantic import ValidationError

from evaluatorq.redteam.contracts import Message


def create_orq_platform_agent_job(agent_key: str) -> Job:
    """Create an evaluatorq job that calls an ORQ platform agent.

    Supports single- and multi-turn datapoints by replaying all user turns while
    preserving conversation state via ``task_id``.

    Requires the ``orq-ai-sdk`` optional dependency.
    """

    try:
        from orq_ai_sdk import Orq
        from orq_ai_sdk.models import A2AMessage, CreateAgentResponseRequestMemory, TextPart
    except ImportError as e:
        msg = (
            'ORQ platform agent jobs require the orq-ai-sdk package. '
            'Install it with: pip install evaluatorq[orq]'
        )
        raise ImportError(msg) from e

    import os

    agent_client = Orq(api_key=os.environ.get('ORQ_API_KEY', ''))

    @job('agent')
    async def platform_agent_job(data: DataPoint, _row: int) -> str:
        messages = list(data.inputs.get('messages', []))
        sample_id = data.inputs.get('id') or str(uuid.uuid4())
        task_id: str | None = None
        result_text = ''
        user_turns = 0

        for raw_msg in messages:
            msg_obj = _normalize_message(raw_msg)
            role = msg_obj.role if isinstance(msg_obj, Message) else str(msg_obj.get('role', ''))
            if role != 'user':
                continue

            content = _message_content_to_text(
                msg_obj.content if isinstance(msg_obj, Message) else msg_obj.get('content')
            )
            if not content.strip():
                continue

            user_turns += 1
            kwargs: dict[str, Any] = {
                'agent_key': agent_key,
                'message': A2AMessage(
                    role='user',
                    parts=[TextPart(kind='text', text=content)],
                ),
                'memory': CreateAgentResponseRequestMemory(entity_id=sample_id),
                'background': False,
            }
            if task_id is not None:
                kwargs['task_id'] = task_id

            response = await agent_client.agents.responses.create_async(**kwargs)
            response_task_id = getattr(response, 'task_id', None)
            if response_task_id:
                task_id = response_task_id
            result_text = _extract_agent_response_text(response)

        if user_turns == 0:
            logger.warning(f'No user turns found for platform agent sample {sample_id}')
            return ''

        logger.debug(
            f'Platform agent response for {sample_id}: turns={user_turns}, '
            + f'len={len(result_text)}, repr={result_text[:200]!r}'
        )
        if not result_text:
            logger.warning(f'Empty platform agent response for {sample_id}')
        return result_text

    return platform_agent_job


def _message_content_to_text(content: Any) -> str:
    """Normalize message content payloads into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(str(part.get('text', '')) for part in content if part.get('type') == 'text')
    return str(content or '')


def _normalize_message(message: Any) -> Message | dict[str, Any]:
    """Normalize to a typed message when possible, otherwise keep dict fallback."""
    if isinstance(message, Message):
        return message
    if isinstance(message, dict):
        try:
            return Message.model_validate(message)
        except (ValidationError, TypeError):
            return message
    return {'role': 'user', 'content': str(message)}


def _extract_agent_response_text(response: Any) -> str:
    """Extract text content from an ORQ platform agent response."""
    parts_text: list[str] = []
    for msg_item in getattr(response, 'output', []):
        if getattr(msg_item, 'role', None) != 'agent':
            continue
        for part in getattr(msg_item, 'parts', []):
            if getattr(part, 'kind', None) == 'text':
                text = getattr(part, 'text', '')
                if text:
                    parts_text.append(text)
    return '\n'.join(parts_text)
