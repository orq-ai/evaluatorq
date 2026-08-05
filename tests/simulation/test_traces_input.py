"""Tests for building simulation datapoints from Orq traces."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from evaluatorq.simulation.traces import (
    TraceConversation,
    _conversation_from_spans,
    _messages_from_value,
    _resolve_orq_credentials,
    datapoints_from_traces,
    extend_from_traces,
    fetch_trace_conversations,
)
from evaluatorq.simulation.types import CommunicationStyle, Persona, Scenario


def _make_persona(name: str = "Test Persona") -> Persona:
    return Persona(
        name=name,
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background="A test user.",
    )


def _make_scenario(name: str = "Test Scenario") -> Scenario:
    return Scenario(name=name, goal="Get help", context="Testing")


def _make_conversation(trace_id: str = "t1") -> TraceConversation:
    return TraceConversation(
        trace_id=trace_id,
        messages=[
            {"role": "user", "content": "Where is my order?"},
            {"role": "assistant", "content": "Let me check."},
        ],
    )


# ---------------------------------------------------------------------------
# Message extraction
# ---------------------------------------------------------------------------


def test_messages_from_dict_with_messages() -> None:
    value = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    assert _messages_from_value(value) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_messages_from_choices_output() -> None:
    value = {"choices": [{"message": {"role": "assistant", "content": "answer"}}]}
    assert _messages_from_value(value) == [{"role": "assistant", "content": "answer"}]


def test_messages_from_string_input() -> None:
    assert _messages_from_value("plain question") == [{"role": "user", "content": "plain question"}]


def test_messages_from_string_output_uses_default_role() -> None:
    assert _messages_from_value("the answer", default_role="assistant") == [
        {"role": "assistant", "content": "the answer"}
    ]


def test_messages_from_content_parts() -> None:
    value = [{"role": "user", "content": [{"type": "text", "text": "part one"}, "part two"]}]
    assert _messages_from_value(value) == [{"role": "user", "content": "part one\npart two"}]


def test_messages_from_garbage_is_empty() -> None:
    assert _messages_from_value(None) == []
    assert _messages_from_value(42) == []
    assert _messages_from_value({"foo": "bar"}) == []


def test_conversation_prefers_root_span() -> None:
    spans = [
        {
            "parent_id": "root",
            "type": "ChatCompletion",
            "input": {"messages": [{"role": "user", "content": "child"}]},
        },
        {
            "parent_id": None,
            "type": "Trace",
            "input": {"messages": [{"role": "user", "content": "root question"}]},
            "output": {"role": "assistant", "content": "root answer"},
        },
    ]
    conversation = _conversation_from_spans("t1", spans)
    assert conversation is not None
    assert conversation.messages == [
        {"role": "user", "content": "root question"},
        {"role": "assistant", "content": "root answer"},
    ]
    assert conversation.first_user_message == "root question"


def test_conversation_none_when_no_messages() -> None:
    assert _conversation_from_spans("t1", [{"parent_id": None, "input": {}, "output": {}}]) is None


def test_conversation_string_output_is_assistant_not_user() -> None:
    """A plain-string span output is the assistant's reply — it must never be
    mistaken for the user's opening message (regression: role mislabeling)."""
    spans = [
        {
            "parent_id": None,
            "type": "Trace",
            "input": "what is my balance?",
            "output": "Your balance is $40.",
        }
    ]
    conversation = _conversation_from_spans("t1", spans)
    assert conversation is not None
    assert conversation.messages == [
        {"role": "user", "content": "what is my balance?"},
        {"role": "assistant", "content": "Your balance is $40."},
    ]
    assert conversation.first_user_message == "what is my balance?"


def test_conversation_output_only_yields_no_user_message() -> None:
    """Empty input + string output must not produce a fake first user message."""
    spans = [{"parent_id": None, "type": "Trace", "input": {}, "output": "assistant text"}]
    conversation = _conversation_from_spans("t1", spans)
    assert conversation is not None
    assert conversation.first_user_message is None


def test_conversation_from_gen_ai_attributes() -> None:
    """Real ``/v2/traces/{id}/v3spans`` shape: top-level input/output are null,
    the conversation lives under ``attributes.gen_ai`` as OTel messages whose
    content is a list of typed ``parts``. Non-text parts (tool calls) must not
    leak into the transcript."""
    spans = [
        {
            "parent_id": None,
            "type": "Trace",
            "input": None,
            "output": None,
            "attributes": {
                "gen_ai": {
                    "request": {"model": "gpt-4o", "temperature": 0.7},
                    "response": {"id": "resp_1", "model": "gpt-4o"},
                    "input": {
                        "messages": [
                            {"role": "system", "parts": [{"type": "text", "content": "Be helpful."}]},
                            {"role": "user", "parts": [{"type": "text", "content": "Where is my order?"}]},
                        ]
                    },
                    "output": {
                        "messages": [
                            {
                                "role": "assistant",
                                "parts": [
                                    {"type": "tool_call", "name": "lookup_order", "arguments": {}},
                                    {"type": "text", "content": "It ships tomorrow."},
                                ],
                                "finish_reason": "stop",
                            }
                        ],
                        "type": "text",
                    },
                },
            },
        }
    ]
    conversation = _conversation_from_spans("t1", spans)
    assert conversation is not None
    assert conversation.messages == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Where is my order?"},
        {"role": "assistant", "content": "It ships tomorrow."},
    ]
    assert conversation.first_user_message == "Where is my order?"


def test_conversation_gen_ai_output_single_message() -> None:
    """Live traffic also carries ``gen_ai.output`` as a single message object."""
    spans = [
        {
            "parent_id": None,
            "type": "Trace",
            "input": None,
            "output": None,
            "attributes": {
                "gen_ai": {
                    "input": {"messages": [{"role": "user", "content": "hi"}]},
                    "output": {"role": "assistant", "content": "hello"},
                }
            },
        }
    ]
    conversation = _conversation_from_spans("t1", spans)
    assert conversation is not None
    assert conversation.messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_conversation_top_level_wins_over_gen_ai() -> None:
    spans = [
        {
            "parent_id": None,
            "type": "Trace",
            "input": {"messages": [{"role": "user", "content": "top-level"}]},
            "attributes": {
                "gen_ai": {"input": {"messages": [{"role": "user", "content": "gen_ai"}]}}
            },
        }
    ]
    conversation = _conversation_from_spans("t1", spans)
    assert conversation is not None
    assert conversation.first_user_message == "top-level"


def test_messages_from_prompt_and_completion() -> None:
    """Completion-model spans: ``gen_ai.input.prompt`` / ``gen_ai.output.completion``."""
    assert _messages_from_value({"prompt": "translate this"}) == [
        {"role": "user", "content": "translate this"}
    ]
    assert _messages_from_value({"completion": "voila"}, default_role="assistant") == [
        {"role": "assistant", "content": "voila"}
    ]


def test_messages_from_json_encoded_strings() -> None:
    """Live ``gen_ai`` attributes often carry input/output JSON-encoded as a
    string; the quotes and escapes must not leak into message content."""
    # JSON-encoded bare string (live gen_ai.input shape).
    assert _messages_from_value('"Hi! Just saying hello."') == [
        {"role": "user", "content": "Hi! Just saying hello."}
    ]
    # JSON-encoded message object (live gen_ai.output shape).
    assert _messages_from_value(
        '{"role":"assistant","content":"Hi! How can I help you?","refusal":null}',
        default_role="assistant",
    ) == [{"role": "assistant", "content": "Hi! How can I help you?"}]
    # Escapes decode to real newlines.
    assert _messages_from_value('"line one\\nline two"') == [
        {"role": "user", "content": "line one\nline two"}
    ]
    # Plain text that merely resembles prose stays verbatim.
    assert _messages_from_value('hello "world"') == [{"role": "user", "content": 'hello "world"'}]
    # Malformed JSON falls back to verbatim text.
    assert _messages_from_value('{not json') == [{"role": "user", "content": "{not json"}]


def test_conversation_from_json_encoded_gen_ai() -> None:
    """End-to-end: a span whose gen_ai input/output are JSON-encoded strings."""
    spans = [
        {
            "parent_id": None,
            "type": "trace",
            "input": None,
            "output": None,
            "attributes": {
                "gen_ai": {
                    "input": '"Where is my order?"',
                    "output": '{"role":"assistant","content":"It ships tomorrow."}',
                }
            },
        }
    ]
    conversation = _conversation_from_spans("t1", spans)
    assert conversation is not None
    assert conversation.messages == [
        {"role": "user", "content": "Where is my order?"},
        {"role": "assistant", "content": "It ships tomorrow."},
    ]
    assert conversation.first_user_message == "Where is my order?"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ORQ_API_KEY"):
        _resolve_orq_credentials(None, None)


def test_base_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORQ_BASE_URL", "https://custom.orq.ai/")
    key, host = _resolve_orq_credentials("k", None)
    assert key == "k"
    assert host == "https://custom.orq.ai"


# ---------------------------------------------------------------------------
# Fetching (mocked HTTP)
# ---------------------------------------------------------------------------


def _mock_client(spans_by_trace: dict[str, list[dict[str, Any]]]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/traces/v3oql":
            assert request.headers["Authorization"] == "Bearer test-key"
            data = [{"trace_id": tid} for tid in spans_by_trace]
            return httpx.Response(200, json={"object": "list", "data": data, "has_more": False})
        for tid, spans in spans_by_trace.items():
            if request.url.path == f"/v2/traces/{tid}/v3spans":
                return httpx.Response(200, json=spans)
        return httpx.Response(404, json={"message": "not found"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_fetch_trace_conversations() -> None:
    spans_by_trace = {
        "t1": [
            {
                "parent_id": None,
                "type": "Trace",
                "input": {"messages": [{"role": "user", "content": "hello"}]},
                "output": {"role": "assistant", "content": "hi"},
            }
        ],
        # No user message -> skipped.
        "t2": [{"parent_id": None, "type": "Trace", "input": {}, "output": {}}],
    }
    async with _mock_client(spans_by_trace) as client:
        conversations = await fetch_trace_conversations(
            limit=10, api_key="test-key", base_url="https://my.orq.ai", http_client=client
        )
    assert len(conversations) == 1
    assert conversations[0].trace_id == "t1"
    assert conversations[0].first_user_message == "hello"


@pytest.mark.asyncio
async def test_fetch_pagination_terminates_on_unusable_pages() -> None:
    """Pages with rows lacking trace_id must not loop forever (hard page cap)."""
    from evaluatorq.simulation.traces import _MAX_PAGES

    requests_made = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        requests_made["n"] += 1
        # Non-empty page, has_more forever, but no usable trace_id anywhere.
        return httpx.Response(
            200, json={"object": "list", "data": [{"foo": "bar"}], "has_more": True}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        conversations = await fetch_trace_conversations(
            limit=5, api_key="test-key", http_client=client
        )

    assert conversations == []
    assert requests_made["n"] == _MAX_PAGES


@pytest.mark.asyncio
async def test_fetch_list_error_raises_runtime_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "boom"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="Failed to list Orq traces"):
            await fetch_trace_conversations(limit=5, api_key="test-key", http_client=client)


# ---------------------------------------------------------------------------
# Direct mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_datapoints_from_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.simulation import traces as traces_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())

    async def fake_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", fake_generate_structured)

    conversations = [_make_conversation("abc123")]
    datapoints = await datapoints_from_traces(conversations, client=MagicMock())

    assert len(datapoints) == 1
    dp = datapoints[0]
    assert dp.id == "trace-abc123"
    assert dp.first_message == "Where is my order?"  # verbatim, not LLM-generated
    assert dp.persona.name == "Test Persona"
    assert dp.user_system_prompt  # built from persona + scenario


@pytest.mark.asyncio
async def test_datapoints_from_traces_skips_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.simulation import traces as traces_mod

    calls = {"n": 0}
    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())

    async def flaky_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("LLM down")
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", flaky_generate_structured)

    conversations = [_make_conversation("bad"), _make_conversation("good")]
    datapoints = await datapoints_from_traces(conversations, client=MagicMock())

    assert [dp.id for dp in datapoints] == ["trace-good"]


# ---------------------------------------------------------------------------
# Extension mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extend_from_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.simulation import traces as traces_mod
    from evaluatorq.simulation.generators import datapoint_generator as dpg_mod
    from evaluatorq.simulation.utils.prompt_builders import generate_datapoint

    profile = traces_mod._TrafficProfile(profile="Mostly refund questions, casual tone.")

    async def fake_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        return profile, ""

    monkeypatch.setattr(traces_mod, "generate_structured", fake_generate_structured)

    captured: dict[str, Any] = {}

    class FakeGenerator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def generate_from_description(self, **kwargs: Any) -> list[Any]:
            captured.update(kwargs)
            n = kwargs["num_personas"] * kwargs["num_scenarios"]
            return [generate_datapoint(_make_persona(f"p{i}"), _make_scenario(f"s{i}")) for i in range(n)]

        async def close(self) -> None:
            pass

    monkeypatch.setattr(dpg_mod, "DatapointGenerator", FakeGenerator)

    datapoints = await extend_from_traces(
        [_make_conversation()], num_datapoints=10, client=MagicMock()
    )

    # 10 datapoints -> 3 personas x 4 scenarios grid, truncated to 10.
    assert captured["num_personas"] == 3
    assert captured["num_scenarios"] == 4
    assert len(datapoints) == 10
    assert "Mostly refund questions" in captured["context"]


@pytest.mark.asyncio
async def test_extend_from_traces_requires_conversations() -> None:
    with pytest.raises(ValueError, match="at least one"):
        await extend_from_traces([], num_datapoints=5, client=MagicMock())


def test_normalize_message_role_case_insensitive() -> None:
    """Producer payloads with 'User'/'ASSISTANT' roles still yield a usable conversation."""
    messages = _messages_from_value({"messages": [{"role": "User", "content": "hi"}]})
    assert messages == [{"role": "user", "content": "hi"}]
    conversation = TraceConversation(trace_id="t", messages=messages)
    assert conversation.first_user_message == "hi"


@pytest.mark.asyncio
async def test_fetch_survives_malformed_span_body() -> None:
    """A 200-OK-but-invalid-JSON span body drops that trace, not the batch."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/traces/v3oql":
            data = [{"trace_id": "bad"}, {"trace_id": "good"}]
            return httpx.Response(200, json={"object": "list", "data": data, "has_more": False})
        if request.url.path == "/v2/traces/bad/v3spans":
            return httpx.Response(200, text="<html>not json</html>")
        if request.url.path == "/v2/traces/good/v3spans":
            return httpx.Response(
                200,
                json=[
                    {
                        "parent_id": None,
                        "type": "Trace",
                        "input": {"messages": [{"role": "user", "content": "hello"}]},
                        "output": {"role": "assistant", "content": "hi"},
                    }
                ],
            )
        return httpx.Response(404, json={"message": "not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        conversations = await fetch_trace_conversations(limit=10, api_key="test-key", http_client=client)

    assert [c.trace_id for c in conversations] == ["good"]


@pytest.mark.asyncio
async def test_fetch_skips_non_list_spans_payload() -> None:
    """An error-envelope spans payload ({"message": ...}) drops the trace, not the batch."""
    spans_by_trace: dict[str, Any] = {
        "wrapped": {"message": "internal"},
        "good": [
            {
                "parent_id": None,
                "type": "Trace",
                "input": {"messages": [{"role": "user", "content": "hello"}]},
                "output": {"role": "assistant", "content": "hi"},
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/traces/v3oql":
            data = [{"trace_id": tid} for tid in spans_by_trace]
            return httpx.Response(200, json={"object": "list", "data": data, "has_more": False})
        for tid, spans in spans_by_trace.items():
            if request.url.path == f"/v2/traces/{tid}/v3spans":
                return httpx.Response(200, json=spans)
        return httpx.Response(404, json={"message": "not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        conversations = await fetch_trace_conversations(limit=10, api_key="test-key", http_client=client)

    assert [c.trace_id for c in conversations] == ["good"]


@pytest.mark.asyncio
async def test_datapoints_from_traces_inference_is_bounded_concurrent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference fans out concurrently (not sequentially) but never beyond the cap,
    and the output preserves input order."""
    import asyncio

    from evaluatorq.simulation import traces as traces_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())
    state = {"active": 0, "peak": 0}

    async def tracked_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        state["active"] -= 1
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", tracked_generate_structured)

    conversations = [_make_conversation(f"t{i}") for i in range(12)]
    datapoints = await datapoints_from_traces(conversations, client=MagicMock())

    assert [dp.id for dp in datapoints] == [f"trace-t{i}" for i in range(12)]
    assert state["peak"] > 1  # actually concurrent, not sequential
    assert state["peak"] <= traces_mod._INFER_CONCURRENCY
