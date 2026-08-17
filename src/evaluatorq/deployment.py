"""
Deployment helper functions for invoking Orq deployments.

This module provides convenient functions for calling Orq deployments
from within evaluation jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from loguru import logger

from evaluatorq.common.orq_client import resolve_orq_client
from evaluatorq.contracts import TokenUsage

__all__ = [
    'DeploymentResponse',
    'MessageDict',
    'ThreadConfig',
    'deployment',
    'invoke',
]

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from orq_ai_sdk import Orq

# Cached client instance
_cached_client: Orq | None = None


class ThreadConfig(TypedDict, total=False):
    """Thread configuration for conversation tracking."""

    id: str
    tags: list[str] | None


class MessageDict(TypedDict, total=False):
    """Chat message structure compatible with Orq SDK."""

    role: Literal['system', 'user', 'assistant', 'developer', 'tool']
    content: str
    name: str | None


@dataclass
class DeploymentResponse:
    """Response from a deployment invocation."""

    content: str
    """The text content of the response"""

    raw: object
    """The raw response from the API"""

    usage: TokenUsage | None = None
    """Token usage extracted from the response, when available"""


def _get_or_create_client() -> Orq:
    """
    Get or create an Orq client instance.
    Reuses the same client for the entire process.

    Returns:
        Orq client instance

    Raises:
        ValueError: If ORQ_API_KEY is not set
        ImportError: If orq_ai_sdk is not installed
    """
    global _cached_client

    if _cached_client is None:
        _cached_client = resolve_orq_client()
    return _cached_client


async def deployment(
    key: str,
    inputs: dict[str, object] | None = None,
    context: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    thread: ThreadConfig | None = None,
    messages: list[MessageDict] | None = None,
) -> DeploymentResponse:
    """
    Invoke an Orq deployment and return the response content.

    Args:
        key: The deployment key (name)
        inputs: Input variables for the deployment template
        context: Context attributes for routing
        metadata: Metadata to attach to the request
        thread: Thread configuration for conversation tracking. Must include 'id' key.
        messages: Chat messages for conversational deployments

    Returns:
        DeploymentResponse with content and raw response

    Example:
        ```python
        # Simple invocation
        response = await deployment("my-deployment")
        print(response.content)

        # With inputs
        response = await deployment("summarizer", inputs={"text": "Long text..."})

        # With messages for chat-style deployments
        response = await deployment("chatbot", messages=[
            {"role": "user", "content": "Hello!"}
        ])

        # With thread tracking
        response = await deployment("assistant",
            inputs={"query": "What is AI?"},
            thread={"id": "conversation-123"}
        )
        ```
    """
    client = _get_or_create_client()

    # Convert our types to SDK-compatible types
    from orq_ai_sdk.models import Thread

    sdk_thread: Thread | None = None
    if thread is not None:
        sdk_thread = Thread(
            id=thread.get('id', ''),
            tags=thread.get('tags'),
        )

    # The SDK accepts list of message dicts directly
    # Cast to Any because the SDK's type hints are stricter than the actual runtime behavior
    completion = await client.deployments.invoke_async(
        key=key,
        inputs=inputs,
        context=context,
        metadata=metadata,
        thread=sdk_thread,
        messages=cast('Any', messages),
    )

    # Extract content from the response
    content = _extract_content_from_response(completion)
    usage = TokenUsage.extract(getattr(completion, 'usage', None), calls=1)

    return DeploymentResponse(content=content, raw=completion, usage=usage)


def _join_text_parts(parts: list[object]) -> str:
    """Join the ``text`` parts of a multimodal content array, dict or object form."""
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and part.get('type') == 'text':
            text_parts.append(str(part.get('text', '')))
        elif getattr(part, 'type', None) == 'text':
            text_parts.append(str(getattr(part, 'text', '')))
    return '\n'.join(text_parts)


def _extract_content_from_response(completion: object) -> str:
    """
    Extract text content from an Orq deployment response.

    An empty reply always logs: a recognised union arm that carried no text
    reaches the ``msg_type in ('content', 'tool_calls')`` warning rather than
    returning ``''`` silently, so the empty-content case is distinguishable in
    the logs from an unrecognised shape — and from a run that simply worked.

    Uses getattr-based introspection because the Orq SDK returns dynamically
    typed response objects. This approach handles various response structures
    (text, multimodal) without requiring tight coupling to specific SDK versions.
    """
    content = ''
    choices: list[object] | None = getattr(completion, 'choices', None)

    if not choices:
        return content

    first_choice: object = choices[0]
    message: object | None = getattr(first_choice, 'message', None)

    if not message:
        return content

    msg_type: str | None = getattr(message, 'type', None)
    msg_content: str | list[object] | None = getattr(message, 'content', None)

    if msg_type == 'content' and isinstance(msg_content, str) and msg_content:
        content = msg_content
    elif msg_type == 'content' and isinstance(msg_content, list) and (text := _join_text_parts(msg_content)):
        # Handle array content (e.g., multimodal responses)
        content = text
    elif isinstance(msg_content, str) and msg_content:
        # Other union arms (e.g. 'tool_calls') still carry a nullable str content
        # field per the SDK schema; surface it instead of silently dropping it.
        content = msg_content
    elif msg_type in ('content', 'tool_calls'):
        # A recognised union arm with empty/None content is routine (e.g. a pure
        # tool-call turn) — not a shape we failed to handle. Note it without
        # implying the type itself is unrecognised.
        logger.warning(
            'Deployment response message type {!r} carried no text content; returning empty reply',
            msg_type,
        )
    else:
        logger.warning(
            'Unrecognised deployment response message type {!r}; returning empty content',
            msg_type,
        )

    return content


async def invoke(
    key: str,
    inputs: dict[str, object] | None = None,
    context: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    thread: ThreadConfig | None = None,
    messages: list[MessageDict] | None = None,
) -> str:
    """
    Invoke an Orq deployment and return just the text content.
    This is a convenience wrapper around `deployment()` for simple use cases.

    Args:
        key: The deployment key (name)
        inputs: Input variables for the deployment template
        context: Context attributes for routing
        metadata: Metadata to attach to the request
        thread: Thread configuration for conversation tracking. Must include 'id' key.
        messages: Chat messages for conversational deployments

    Returns:
        The text content of the response

    Example:
        ```python
        # In a job
        @job("my-job")
        async def my_job(data, row):
            return await invoke("summarizer", inputs=data.inputs)
        ```
    """
    response = await deployment(
        key,
        inputs=inputs,
        context=context,
        metadata=metadata,
        thread=thread,
        messages=messages,
    )
    return response.content
