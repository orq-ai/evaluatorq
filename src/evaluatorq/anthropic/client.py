"""An OpenAI-shaped facade over an Anthropic SDK client."""

from __future__ import annotations

import inspect
from typing import Any

from evaluatorq.anthropic.convert import build_messages_request, to_chat_completion


class _ChatCompletions:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def create(self, **params: Any) -> Any:
        """Call ``messages.create`` and return a ``ChatCompletion``."""
        message = await self._client.messages.create(**build_messages_request(params))
        return to_chat_completion(message)


class _Chat:
    def __init__(self, client: Any) -> None:
        self.completions = _ChatCompletions(client)


class _Embeddings:
    @property
    def create(self) -> Any:
        raise NotImplementedError(
            'Anthropic has no embeddings API. Pass an OpenAI-compatible client as embedding_client= '
            '(Insights) to embed text.'
        )


class AnthropicLLMClient:
    """Wraps ``AsyncAnthropic``, ``AsyncAnthropicBedrock`` or ``AsyncAnthropicVertex``.

    Exposes the attributes evaluatorq reads from an ``AsyncOpenAI`` client: ``chat.completions.create``, ``base_url``, ``api_key``, ``max_retries``, ``with_options`` and ``close``.
    """

    evaluatorq_provider = 'anthropic'
    api_key = None

    def __init__(self, client: Any) -> None:
        if not hasattr(getattr(client, 'messages', None), 'create'):
            raise TypeError(
                f'AnthropicLLMClient needs an Anthropic SDK client with messages.create, got {type(client).__name__}.'
            )
        self._client = client
        self.chat = _Chat(client)
        self.embeddings = _Embeddings()

    @property
    def base_url(self) -> Any:
        return getattr(self._client, 'base_url', None)

    @property
    def max_retries(self) -> Any:
        return getattr(self._client, 'max_retries', 0)

    def with_options(self, **kwargs: Any) -> AnthropicLLMClient:
        """Wrap ``client.with_options(**kwargs)``; the wrapped client is never mutated."""
        return AnthropicLLMClient(self._client.with_options(**kwargs))

    async def close(self) -> None:
        closer = getattr(self._client, 'close', None)
        if closer is None:
            return
        result = closer()
        if inspect.isawaitable(result):
            await result


def wrap_anthropic_client(client: Any) -> AnthropicLLMClient:
    """Return ``client`` as an `AnthropicLLMClient`; an already-wrapped client is returned unchanged."""
    if isinstance(client, AnthropicLLMClient):
        return client
    return AnthropicLLMClient(client)
