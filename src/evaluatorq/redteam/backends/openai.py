"""OpenAI SDK backend implementation for dynamic red teaming."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from evaluatorq.redteam.backends.base import extract_provider_error_code, extract_status_code
from evaluatorq.redteam.contracts import AgentContext, TargetKind, TokenUsage
from evaluatorq.redteam.tracing import record_llm_response, with_llm_span

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def create_async_llm_client() -> AsyncOpenAI:
    """Lazy proxy to :func:`~evaluatorq.redteam.backends.registry.create_async_llm_client`.

    Defined here so that tests can patch
    ``evaluatorq.redteam.backends.openai.create_async_llm_client`` and to avoid
    a circular import between this module and ``registry.py`` (which imports our
    classes at the top level).
    """
    from evaluatorq.redteam.backends.registry import create_async_llm_client as _create

    return _create()


class OpenAIModelTarget:
    """Target adapter that treats ``agent_key`` as an OpenAI model identifier."""

    memory_entity_id: str | None = None
    """OpenAI models are stateless — no server-side memory to isolate."""

    def __init__(
        self,
        model: str,
        system_prompt: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
    ):
        """Initialize the target with a model name, optional async client, and optional system prompt.

        If ``client`` is not provided, one is created automatically via
        :func:`~evaluatorq.redteam.backends.registry.create_async_llm_client`.
        """
        self.model = model
        self.client = client or create_async_llm_client()
        self.system_prompt = system_prompt or 'You are a helpful assistant.'
        self._last_token_usage: TokenUsage | None = None

    async def send_prompt(self, prompt: str) -> str:
        """Send a prompt to the OpenAI model and return its response."""
        messages: list[ChatCompletionMessageParam] = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt},
        ]
        async with with_llm_span(
            model=self.model,
            input_messages=messages,
            attributes={"orq.redteam.llm_purpose": "target"},
        ) as span:
            response = await self.client.chat.completions.create(
                model=self.model,
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

    def clone(self) -> OpenAIModelTarget:
        """Return a fresh target instance for parallel job safety."""
        return OpenAIModelTarget(model=self.model, system_prompt=self.system_prompt, client=self.client)

    def new(self) -> OpenAIModelTarget:
        """Return a fresh target instance (alias for :meth:`clone`, satisfies ``AgentTarget`` protocol)."""
        return self.clone()

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

    def create_target(self, agent_key: str) -> OpenAIModelTarget:
        """Create a new OpenAI model target for the given model name."""
        return OpenAIModelTarget(model=agent_key, system_prompt=self.system_prompt, client=self.client)

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
        return OpenAIModelTarget(model=agent_key, system_prompt=self._system_prompt, client=self._client)


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


