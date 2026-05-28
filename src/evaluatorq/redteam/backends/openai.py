"""OpenAI SDK backend implementation for dynamic red teaming."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from evaluatorq.contracts import AgentTarget
from evaluatorq.redteam.backends._errors import extract_provider_error_code, extract_status_code
from evaluatorq.redteam.backends.base import Backend
from evaluatorq.redteam.contracts import (
    DEFAULT_TARGET_MAX_TOKENS,
    DEFAULT_TARGET_TIMEOUT_MS,
    AgentContext,
    AgentResponse,
    OutputMessage,
    TargetKind,
    TextOutputItem,
    TokenUsage,
    ToolCallOutputItem,
)
from evaluatorq.redteam.tracing import record_llm_response, with_llm_span

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam


def create_async_llm_client(role_config=None) -> AsyncOpenAI:
    """Lazy proxy to :func:`~evaluatorq.redteam.backends.registry.create_async_llm_client`.

    Defined here so that tests can patch
    ``evaluatorq.redteam.backends.openai.create_async_llm_client`` and to avoid
    a circular import between this module and ``registry.py`` (which imports our
    classes at the top level).
    """
    from evaluatorq.redteam.backends.registry import create_async_llm_client as _create

    return _create(role_config)


def _openai_map_error(exc: Exception) -> tuple[str, str]:
    """Map an OpenAI exception to a normalized (error_code, error_message) tuple."""
    name = type(exc).__name__.lower()
    status_code = extract_status_code(exc)
    provider_code = extract_provider_error_code(exc)
    if status_code is not None:
        return f'openai.http.{status_code}', f'{type(exc).__name__}: {exc}'
    if provider_code:
        return f'openai.code.{provider_code}', f'{type(exc).__name__}: {exc}'
    if 'ratelimit' in name:
        return 'openai.rate_limit', f'{type(exc).__name__}: {exc}'
    if 'authentication' in name:
        return 'openai.auth', f'{type(exc).__name__}: {exc}'
    if 'timeout' in name:
        return 'openai.timeout', f'{type(exc).__name__}: {exc}'
    return 'openai.unknown', f'{type(exc).__name__}: {exc}'


class OpenAIModelTarget(AgentTarget):
    """Target adapter that treats ``agent_key`` as an OpenAI model identifier."""

    def __init__(
        self,
        model: str,
        system_prompt: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
        max_tokens: int | None = None,
        timeout_ms: int | None = None,
    ):
        """Initialize the target with a model name, optional async client, and optional system prompt.

        If ``client`` is not provided, one is created automatically via
        :func:`~evaluatorq.redteam.backends.registry.create_async_llm_client`.
        OpenAI models are stateless — no server-side memory to isolate.
        """
        super().__init__(memory_entity_id=None)
        self.model = model
        self.client = client or create_async_llm_client()
        self.system_prompt = system_prompt or 'You are a helpful assistant.'
        self.max_tokens = max_tokens or DEFAULT_TARGET_MAX_TOKENS
        self.timeout_ms = timeout_ms or DEFAULT_TARGET_TIMEOUT_MS
        self._history: list[ChatCompletionMessageParam] = []

    async def send_prompt(self, prompt: str) -> AgentResponse:
        """Send a prompt to the OpenAI model and return its response with usage + any tool calls."""
        user_msg: ChatCompletionMessageParam = {'role': 'user', 'content': prompt}
        messages: list[ChatCompletionMessageParam] = [
            {'role': 'system', 'content': self.system_prompt},
            *self._history,
            user_msg,
        ]
        async with with_llm_span(
            model=self.model,
            input_messages=messages,
            attributes={"orq.redteam.llm_purpose": "target"},
        ) as span:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                ),
                timeout=self.timeout_ms / 1000.0,
            )
            msg = response.choices[0].message
            content = msg.content or ''
            record_llm_response(span, response, output_content=content)

            # Capture tool calls made by the model.
            # Only ChatCompletionMessageToolCall has a .function attribute; skip custom tool calls.
            tool_call_items: list[ToolCallOutputItem] = []
            for tc in (getattr(msg, 'tool_calls', None) or []):
                func = getattr(tc, 'function', None)
                if func is None:
                    continue
                tc_id = getattr(tc, 'id', None)
                kwargs: dict[str, Any] = {
                    'name': func.name,
                    'arguments': func.arguments or '{}',
                }
                if isinstance(tc_id, str) and tc_id:
                    kwargs['id'] = tc_id
                tool_call_items.append(ToolCallOutputItem(**kwargs))

            usage = TokenUsage.from_completion(response)
            response_id = getattr(response, 'id', None)
            finish_reason = None
            choices = getattr(response, 'choices', None) or []
            if choices:
                finish_reason = getattr(choices[0], 'finish_reason', None)

        # Accumulate history for multi-turn context.
        assistant_msg: ChatCompletionMessageParam
        raw_tool_calls = getattr(msg, 'tool_calls', None) or []
        if raw_tool_calls:
            assistant_msg = {
                'role': 'assistant',
                'content': content or None,
                'tool_calls': [
                    {
                        'id': tc.id,
                        'type': 'function',
                        'function': {'name': tc.function.name, 'arguments': tc.function.arguments},
                    }
                    for tc in raw_tool_calls
                    if getattr(tc, 'function', None) is not None
                ],
            }
        else:
            assistant_msg = {'role': 'assistant', 'content': content}
        self._history.append(user_msg)
        self._history.append(assistant_msg)

        output: list[OutputMessage] = cast('list[OutputMessage]', list(tool_call_items))
        output.append(TextOutputItem(text=content, annotations=[]))
        return AgentResponse(
            output=output,
            usage=usage,
            model=getattr(response, 'model', None),
            response_id=response_id,
            finish_reason=finish_reason,
        )

    def new(self) -> OpenAIModelTarget:
        """Return a fresh target instance for parallel job safety (satisfies the ``AgentTarget`` ABC)."""
        return OpenAIModelTarget(model=self.model, system_prompt=self.system_prompt, client=self.client, max_tokens=self.max_tokens, timeout_ms=self.timeout_ms)

    target_kind: TargetKind = TargetKind.OPENAI
    """Used by the runner to populate report metadata correctly."""

    @property
    def name(self) -> str:
        """Return the model name as the target name."""
        return self.model

    async def get_agent_context(self) -> AgentContext:
        """Return a minimal agent context for this model target."""
        return AgentContext(
            key=self.model,
            display_name=self.model,
            description='OpenAI model target',
            model=self.model,
        )

    def map_error(self, exc: Exception) -> tuple[str, str]:
        """Map an OpenAI exception to a normalized error code and message tuple."""
        return _openai_map_error(exc)


class OpenAIBackend(Backend):
    """Backend for direct OpenAI model targets.

    Targets are stateless. ``cleanup_memory`` is a no-op (OpenAI models do not
    own server-side memory).
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        super().__init__(name="openai")
        self._client = client or create_async_llm_client()
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._timeout_ms = timeout_ms

    def create_target(self, agent_key: str) -> OpenAIModelTarget:
        return OpenAIModelTarget(
            model=agent_key,
            system_prompt=self._system_prompt,
            client=self._client,
            max_tokens=self._max_tokens,
            timeout_ms=self._timeout_ms,
        )

    async def cleanup_memory(self, ctx: AgentContext, entity_ids: list[str]) -> None:
        logger.debug('OpenAI backend has no memory store; cleanup is a no-op')

    def map_error(self, exc: Exception) -> tuple[str, str]:
        return _openai_map_error(exc)
