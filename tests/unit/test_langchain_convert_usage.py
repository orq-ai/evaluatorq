"""Usage extraction from LangChain messages into OpenResponses format.

Fixtures are built from langchain-core's own ``UsageMetadata`` /
``InputTokenDetails`` / ``OutputTokenDetails`` types rather than hand-written
dicts, so a schema move in langchain-core fails these tests instead of
confirming a guess (CLAUDE.md: provider usage shapes are not interchangeable).
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip('langchain_core')

from langchain_core.messages import AIMessage, HumanMessage, UsageMetadata  # noqa: E402
from langchain_core.messages.ai import InputTokenDetails, OutputTokenDetails  # noqa: E402

from evaluatorq.contracts import Usage  # noqa: E402
from evaluatorq.integrations.langchain_integration.convert import convert_to_open_responses  # noqa: E402
from evaluatorq.openresponses import ResponseResourceDict  # noqa: E402


def _ai_message_with_usage(usage: UsageMetadata) -> AIMessage:
    return AIMessage(content='answer', id='chatcmpl-1', usage_metadata=usage)


def _usage_of(response: ResponseResourceDict) -> dict[str, Any]:
    payload = response['usage']  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert payload is not None
    return payload


def test_cache_creation_tokens_survive_conversion():
    """Cache writes must not vanish on a LangChain-fronted Anthropic call.

    langchain-core >= 0.3.9 standardizes cache writes as
    ``input_token_details.cache_creation``. Reading only ``cache_read`` left
    every cache-write count at 0.
    """
    usage = UsageMetadata(
        input_tokens=1000,
        output_tokens=20,
        total_tokens=1020,
        input_token_details=InputTokenDetails(cache_read=600, cache_creation=300),
        output_token_details=OutputTokenDetails(reasoning=5),
    )

    response = convert_to_open_responses([HumanMessage(content='hi'), _ai_message_with_usage(usage)])

    details = _usage_of(response)['input_tokens_details']
    assert details['cached_tokens'] == 600
    assert details['cache_creation_tokens'] == 300

    # And the canonical contract picks it up off the converted payload.
    extracted = Usage.extract(_usage_of(response), calls=1)
    assert extracted is not None
    assert extracted.cached_tokens == 600
    assert extracted.cache_creation_tokens == 300
    assert extracted.reasoning_tokens == 5


def test_cache_creation_tokens_sum_across_messages():
    """Multi-turn runs accumulate cache writes rather than keeping the last one."""
    usage = UsageMetadata(
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        input_token_details=InputTokenDetails(cache_read=10, cache_creation=40),
    )

    response = convert_to_open_responses([
        HumanMessage(content='hi'),
        _ai_message_with_usage(usage),
        HumanMessage(content='again'),
        _ai_message_with_usage(usage),
    ])

    details = _usage_of(response)['input_tokens_details']
    assert details['cache_creation_tokens'] == 80
    assert details['cached_tokens'] == 20


def test_cache_creation_tokens_default_to_zero_without_details():
    """A provider that reports no cache detail stays honestly at 0, not absent."""
    usage = UsageMetadata(input_tokens=5, output_tokens=5, total_tokens=10)

    response = convert_to_open_responses([HumanMessage(content='hi'), _ai_message_with_usage(usage)])

    details = _usage_of(response)['input_tokens_details']
    assert details['cache_creation_tokens'] == 0
