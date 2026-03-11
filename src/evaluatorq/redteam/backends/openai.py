"""OpenAI SDK backend implementation for dynamic red teaming."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from loguru import logger

from evaluatorq.redteam.backends.base import extract_provider_error_code, extract_status_code
from evaluatorq.redteam.contracts import AgentContext

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam


class OpenAIModelTarget:
    """Target adapter that treats ``agent_key`` as an OpenAI model identifier."""

    def __init__(
        self,
        model_id: str,
        client: AsyncOpenAI,
        system_prompt: str | None = None,
    ):
        self.model_id: str = model_id
        self.client: AsyncOpenAI = client
        self.system_prompt: str = system_prompt or 'You are a helpful assistant.'
        self._messages: list[dict[str, str]] = [
            {'role': 'system', 'content': self.system_prompt},
        ]

    async def send_prompt(self, prompt: str) -> str:
        self._messages.append({'role': 'user', 'content': prompt})
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=cast('list[ChatCompletionMessageParam]', self._messages),
        )
        content = response.choices[0].message.content or ''
        self._messages.append({'role': 'assistant', 'content': content})
        return content

    def reset_conversation(self) -> None:
        """Reset conversation history, keeping only the system prompt."""
        self._messages = [
            {'role': 'system', 'content': self.system_prompt},
        ]

    def clone(self) -> OpenAIModelTarget:
        """Create a fresh target instance for parallel job safety."""
        return OpenAIModelTarget(
            model_id=self.model_id,
            client=self.client,
            system_prompt=self.system_prompt,
        )


class OpenAIContextProvider:
    """Context provider for plain model targets.

    There is no tool/memory/KB metadata in raw model mode.
    """

    async def get_agent_context(self, agent_key: str) -> AgentContext:
        logger.info(f'Using OpenAI model target context for model={agent_key}')
        return AgentContext(
            key=agent_key,
            display_name=agent_key,
            description='OpenAI model target',
            tools=[],
            memory_stores=[],
            knowledge_bases=[],
            model=agent_key,
        )


class OpenAITargetFactory:
    """Factory creating OpenAI model targets."""

    def __init__(self, client: AsyncOpenAI):
        self._client: AsyncOpenAI = client

    def create_target(self, agent_key: str, memory_entity_id: str | None = None) -> OpenAIModelTarget:
        _ = memory_entity_id  # OpenAI model target does not support memory entity routing.
        return OpenAIModelTarget(model_id=agent_key, client=self._client)


class NoopMemoryCleanup:
    """No-op cleanup for backends without managed memory stores."""

    async def cleanup_memory(self, agent_context: AgentContext, entity_ids: list[str]) -> None:
        _ = (agent_context, entity_ids)
        logger.debug('Skipping memory cleanup: backend has no managed memory API')


class OpenAIErrorMapper:
    """Normalize OpenAI exceptions into runtime error taxonomy."""

    def map_error(self, exc: Exception) -> tuple[str, str]:
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
