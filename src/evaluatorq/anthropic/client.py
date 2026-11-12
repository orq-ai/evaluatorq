"""An OpenAI-shaped facade over an Anthropic SDK client."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from loguru import logger

from evaluatorq.anthropic.convert import build_messages_request, to_chat_completion, to_parsed_chat_completion
from evaluatorq.anthropic.responses import build_responses_request, to_parsed_response, to_response
from evaluatorq.common.llm_client import LLMConfigurationError

FALLBACK_BETA = 'server-side-fallback-2026-07-01'
_PLATFORMS_WITHOUT_FALLBACKS = ('Bedrock', 'Vertex')


@dataclass
class _RefusalLog:
    """Whether the refusal warning was logged. One instance is shared by every clone `with_options` makes."""

    logged: bool = False


class _Sender:
    """Sends one built request to Anthropic and logs a refusal once; shared by the Chat Completions and Responses faces."""

    def __init__(self, client: Any, refusal_log: _RefusalLog) -> None:
        self._client = client
        self._refusal_log = refusal_log

    async def send(self, request: dict[str, Any]) -> Any:
        """Send ``request`` through ``messages.create`` (or the beta endpoint for fallbacks) and return the raw Anthropic message."""
        if 'fallbacks' in request.get('extra_body', {}):
            platform = type(self._client).__name__
            hierarchy = [cls.__name__ for cls in type(self._client).__mro__]
            if any(name in cls for cls in hierarchy for name in _PLATFORMS_WITHOUT_FALLBACKS):
                raise LLMConfigurationError(
                    f'Server-side fallbacks are available on the Claude API only, not on {platform} '
                    '(Amazon Bedrock and Google Vertex AI). Remove extra_kwargs={"fallbacks": ...} from the LLMCallConfig.'
                )
            message = await self._client.beta.messages.create(**request, betas=[FALLBACK_BETA])
        else:
            message = await self._client.messages.create(**request)
        if message.stop_reason == 'refusal' and not self._refusal_log.logged:
            self._refusal_log.logged = True
            logger.warning(
                'The Anthropic model refused a request (stop_reason "refusal"). '
                'Server-side fallbacks can retry it on another model: pass extra_kwargs={"fallbacks": "default"} '
                "on the role's LLMCallConfig (Claude API only)."
            )
        return message


class _ChatCompletions:
    def __init__(self, sender: _Sender) -> None:
        self._sender = sender

    async def _send(self, params: dict[str, Any]) -> Any:
        return await self._sender.send(build_messages_request(params))

    async def create(self, **params: Any) -> Any:
        """Call ``messages.create`` and return a ``ChatCompletion``."""
        return to_chat_completion(await self._send(params))

    async def parse(self, **params: Any) -> Any:
        """Call ``messages.create`` with ``output_config.format`` and return a ``ParsedChatCompletion``."""
        completion = to_chat_completion(await self._send(params))
        return to_parsed_chat_completion(
            completion, response_format=params.get('response_format'), tools=params.get('tools')
        )


class _Chat:
    def __init__(self, sender: _Sender) -> None:
        self.completions = _ChatCompletions(sender)


class _Responses:
    def __init__(self, sender: _Sender) -> None:
        self._sender = sender

    async def create(self, **params: Any) -> Any:
        """Call ``messages.create`` and return an ``openai.types.responses.Response``."""
        return to_response(await self._sender.send(build_responses_request(params)))

    async def parse(self, *, text_format: Any, **params: Any) -> Any:
        """Like ``create`` with ``text.format`` built from ``text_format``; returns a ``ParsedResponse`` with ``output_parsed``."""
        request = build_responses_request(params, text_format=text_format)
        return to_parsed_response(await self._sender.send(request), text_format=text_format, tools=params.get('tools'))


class _Embeddings:
    @property
    def create(self) -> Any:
        raise NotImplementedError(
            'Anthropic has no embeddings API. Pass an OpenAI-compatible client as embedding_client= '
            '(Insights) to embed text.'
        )


def _creates_asynchronously(create: Any) -> bool:
    """True when ``messages.create`` is a coroutine function; the SDK wraps it in a plain function, so look through ``__wrapped__``."""
    try:
        return inspect.iscoroutinefunction(inspect.unwrap(create))
    except ValueError:
        return False


class AnthropicLLMClient:
    """Wraps ``AsyncAnthropic``, ``AsyncAnthropicBedrock`` or ``AsyncAnthropicVertex``.

    Exposes the attributes evaluatorq reads from an ``AsyncOpenAI`` client: ``chat.completions.create`` and ``parse``, ``responses.create`` and ``parse``, ``base_url``, ``api_key``, ``max_retries``, ``with_options`` and ``close``.
    """

    evaluatorq_provider = 'anthropic'
    api_key = None

    def __init__(self, client: Any, *, _refusal_log: _RefusalLog | None = None) -> None:
        if not hasattr(getattr(client, 'messages', None), 'create'):
            raise TypeError(
                f'AnthropicLLMClient needs an Anthropic SDK client with messages.create, got {type(client).__name__}.'
            )
        if not _creates_asynchronously(client.messages.create):
            raise TypeError(
                f'AnthropicLLMClient needs an async Anthropic client, got {type(client).__name__}. '
                'Use AsyncAnthropic, AsyncAnthropicBedrock or AsyncAnthropicVertex.'
            )
        self._client = client
        self._refusal_log = _refusal_log if _refusal_log is not None else _RefusalLog()
        sender = _Sender(client, self._refusal_log)
        self.chat = _Chat(sender)
        self.responses = _Responses(sender)
        self.embeddings = _Embeddings()

    @property
    def base_url(self) -> Any:
        return getattr(self._client, 'base_url', None)

    @property
    def max_retries(self) -> Any:
        return getattr(self._client, 'max_retries', 0)

    def with_options(self, **kwargs: Any) -> AnthropicLLMClient:
        """Wrap ``client.with_options(**kwargs)``; the wrapped client is never mutated."""
        return AnthropicLLMClient(self._client.with_options(**kwargs), _refusal_log=self._refusal_log)

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
