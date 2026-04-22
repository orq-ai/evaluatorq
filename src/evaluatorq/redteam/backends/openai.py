"""OpenAI SDK backend implementation for dynamic red teaming."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from evaluatorq.redteam.backends.base import extract_provider_error_code, extract_status_code
from evaluatorq.redteam.contracts import AgentContext, TargetKind, TokenUsage
from evaluatorq.redteam.tracing import record_llm_response, with_llm_span

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenAIModelTarget:
    """Target adapter that treats ``agent_key`` as an OpenAI model identifier."""

    memory_entity_id: str | None = None
    """OpenAI models are stateless — no server-side memory to isolate."""

    def __init__(
        self,
        model_id: str,
        client: AsyncOpenAI,
        system_prompt: str | None = None,
    ):
        """Initialize the target with a model ID, async client, and optional system prompt."""
        self.model_id = model_id
        self.client = client
        self.system_prompt = system_prompt or 'You are a helpful assistant.'
        self._last_token_usage: TokenUsage | None = None

    async def send_prompt(self, prompt: str) -> str:
        """Send a prompt to the OpenAI model and return its response."""
        messages: list[ChatCompletionMessageParam] = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt},
        ]
        async with with_llm_span(
            model=self.model_id,
            input_messages=messages,
            attributes={"orq.redteam.llm_purpose": "target"},
        ) as span:
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
            )
            content = response.choices[0].message.content or ''
            record_llm_response(span, response, output_content=content)

            # Store usage for consume_last_token_usage()
            usage = getattr(response, 'usage', None)
            if usage is not None:
                _prompt = int(getattr(usage, 'prompt_tokens', 0) or 0)
                _completion = int(getattr(usage, 'completion_tokens', 0) or 0)
                _raw_total = getattr(usage, 'total_tokens', None)
                _total = int(_raw_total) if _raw_total else (_prompt + _completion)
                self._last_token_usage = TokenUsage(
                    prompt_tokens=_prompt,
                    completion_tokens=_completion,
                    total_tokens=_total,
                    calls=1,
                )
            else:
                self._last_token_usage = None

        return content

    def consume_last_token_usage(self) -> TokenUsage | None:
        """Return and clear usage from last call."""
        usage = self._last_token_usage
        self._last_token_usage = None
        return usage

    def reset_conversation(self) -> None:
        """Stateless adapter; nothing to reset."""
        return

    def clone(self) -> OpenAIModelTarget:
        """Create a fresh target instance for parallel job safety."""
        return OpenAIModelTarget(
            model_id=self.model_id,
            client=self.client,
            system_prompt=self.system_prompt,
        )

    target_kind: TargetKind = TargetKind.OPENAI
    """Used by the runner to populate report metadata correctly."""

    @property
    def name(self) -> str:
        """Return the model ID as the target name."""
        return self.model_id

    async def get_agent_context(self) -> AgentContext:
        """Return a minimal agent context for this model target."""
        return AgentContext(
            key=self.model_id,
            display_name=self.model_id,
            description='OpenAI model target',
            model=self.model_id,
        )

    def create_target(self, agent_key: str) -> OpenAIModelTarget:
        """Create a new OpenAI model target for the given model ID."""
        return OpenAIModelTarget(
            model_id=agent_key,
            client=self.client,
            system_prompt=self.system_prompt,
        )

    def map_error(self, exc: Exception) -> tuple[str, str]:
        """Map an OpenAI exception to a normalized error code and message tuple."""
        return OpenAIErrorMapper().map_error(exc)


class OpenAIContextProvider:
    """Context provider for plain model targets.

    There is no tool/memory/KB metadata in raw model mode.
    """

    def __init__(self, system_prompt: str | None = None):
        """Initialize the context provider with an optional system prompt override."""
        self._system_prompt = system_prompt

    async def get_agent_context(self, agent_key: str) -> AgentContext:
        """Return agent context for the specified OpenAI model."""
        logger.debug(f'Using OpenAI model target context for model={agent_key}')
        return AgentContext(
            key=agent_key,
            display_name=agent_key,
            description='OpenAI model target',
            system_prompt=self._system_prompt,
            tools=[],
            memory_stores=[],
            knowledge_bases=[],
            model=agent_key,
        )


class OpenAITargetFactory:
    """Factory creating OpenAI model targets."""

    def __init__(self, client: AsyncOpenAI, system_prompt: str | None = None):
        """Initialize the factory with a shared async OpenAI client and optional system prompt."""
        self._client = client
        self._system_prompt = system_prompt

    def create_target(self, agent_key: str) -> OpenAIModelTarget:
        """Create a new OpenAI model target instance."""
        return OpenAIModelTarget(model_id=agent_key, client=self._client, system_prompt=self._system_prompt)


class OpenAIErrorMapper:
    """Normalize OpenAI exceptions into runtime error taxonomy."""

    def map_error(self, exc: Exception) -> tuple[str, str]:
        """Map an OpenAI exception to a normalized error code and message tuple."""
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


def create_openai_target(model_id: str, client: Any = None) -> OpenAIModelTarget:
    """Create an OpenAIModelTarget from environment config."""
    from evaluatorq.redteam.backends.registry import create_async_llm_client
    resolved = client or create_async_llm_client()
    return OpenAIModelTarget(model_id=model_id, client=resolved)
