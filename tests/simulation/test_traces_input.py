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
    summarize_conversations,
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
    """Pages with rows lacking trace_id must not loop forever: stop on no progress."""
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
    # One page proves the point — a second could not add rows the first didn't.
    assert requests_made["n"] == 1


@pytest.mark.asyncio
async def test_fetch_honours_a_limit_beyond_one_page() -> None:
    """A limit larger than the API page size pages until it is met, not until a page cap."""
    pages_served = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "v3spans" in str(request.url):
            return httpx.Response(200, json=[])
        pages_served["n"] += 1
        start = (pages_served["n"] - 1) * 200
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"trace_id": f"t{i}"} for i in range(start, start + 200)],
                "has_more": True,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await fetch_trace_conversations(limit=5000, api_key="test-key", http_client=client)

    assert pages_served["n"] == 25  # 5000 / 200, not a fixed ceiling of 20


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


def _stub_structured(monkeypatch: pytest.MonkeyPatch, parsed: Any, summary: str = "A short summary.") -> None:
    """Answer both schemas: every direct-mode conversation is summarized before inference."""
    from evaluatorq.simulation import traces as traces_mod

    async def fake_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        if kwargs["response_format"] is traces_mod._ConversationSummary:
            return traces_mod._ConversationSummary(summary=summary), ""
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", fake_generate_structured)


@pytest.mark.asyncio
async def test_datapoints_from_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.simulation import traces as traces_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())
    _stub_structured(monkeypatch, parsed)
    _stub_first_message(monkeypatch, "Hi, chasing an order.")

    conversations = [_make_conversation("abc123")]
    datapoints = await datapoints_from_traces(conversations, client=MagicMock())

    assert len(datapoints) == 1
    dp = datapoints[0]
    assert dp.id == "trace-abc123"
    # Default: written from the inferred persona/scenario, not replayed from the trace.
    assert dp.first_message == "Hi, chasing an order."
    assert dp.persona.name == "Test Persona"
    assert dp.user_system_prompt  # built from persona + scenario


def _stub_first_message(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    """Make FirstMessageGenerator.generate return ``message`` without an LLM call."""
    from evaluatorq.simulation.generators import first_message_generator as fmg_mod

    async def fake_generate(_self: Any, _persona: Any, _scenario: Any) -> str:
        return message

    monkeypatch.setattr(fmg_mod.FirstMessageGenerator, "generate", fake_generate)


@pytest.mark.asyncio
async def test_datapoints_from_traces_can_replay_the_recorded_opening(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-out reproduces a specific recorded case verbatim."""
    from evaluatorq.simulation import traces as traces_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())
    _stub_structured(monkeypatch, parsed)
    _stub_first_message(monkeypatch, "should not be used")

    datapoints = await datapoints_from_traces(
        [_make_conversation("abc123")],
        client=MagicMock(),
        config=traces_mod.TraceAnalysisConfig(generate_first_message=False),
    )

    assert datapoints[0].first_message == "Where is my order?"


@pytest.mark.asyncio
async def test_failed_first_message_generation_falls_back_to_the_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documented degradation: the datapoint survives a dead first-message call."""
    from evaluatorq.simulation import traces as traces_mod
    from evaluatorq.simulation.generators import first_message_generator as fmg_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())
    _stub_structured(monkeypatch, parsed)

    async def boom(_self: Any, _persona: Any, _scenario: Any) -> str:
        raise RuntimeError("first-message LLM down")

    monkeypatch.setattr(fmg_mod.FirstMessageGenerator, "generate", boom)

    datapoints = await datapoints_from_traces([_make_conversation("abc123")], client=MagicMock())

    assert datapoints[0].first_message == "Where is my order?"


@pytest.mark.asyncio
async def test_long_transcript_is_summarized_before_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """The map step: an oversized trace reaches inference as a summary, not raw."""
    from evaluatorq.simulation import traces as traces_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())
    prompts: list[str] = []

    async def fake_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        prompts.append(kwargs["messages"][1]["content"])
        if kwargs["response_format"] is traces_mod._ConversationSummary:
            return traces_mod._ConversationSummary(summary="Impatient user chasing a refund."), ""
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", fake_generate_structured)
    _stub_first_message(monkeypatch, "Where is it?")

    long_conversation = traces_mod.TraceConversation(
        trace_id="long",
        messages=[{"role": "user", "content": "z" * 5000}, {"role": "assistant", "content": "z" * 5000}],
    )
    await datapoints_from_traces(
        [long_conversation],
        client=MagicMock(),
        config=traces_mod.TraceAnalysisConfig(),
    )

    summarize_prompt, infer_prompt = prompts
    assert "z" * 5000 in summarize_prompt  # the map step sees the whole thing
    assert "Impatient user chasing a refund." in infer_prompt
    assert "z" * 5000 not in infer_prompt  # ...the reduce step does not


@pytest.mark.asyncio
async def test_every_conversation_is_summarized_even_a_short_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The map step is unconditional, so one artifact serves both modes."""
    from evaluatorq.simulation import traces as traces_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())
    schemas: list[Any] = []

    async def fake_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        schemas.append(kwargs["response_format"])
        if kwargs["response_format"] is traces_mod._ConversationSummary:
            return traces_mod._ConversationSummary(summary="Short chat about an order."), ""
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", fake_generate_structured)
    _stub_first_message(monkeypatch, "Where is it?")

    await datapoints_from_traces([_make_conversation("abc123")], client=MagicMock())

    assert schemas == [traces_mod._ConversationSummary, traces_mod._InferredPersonaScenario]


@pytest.mark.asyncio
async def test_supplied_summaries_are_not_recomputed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run doing both modes summarizes once: passed-in summaries skip the map call."""
    from evaluatorq.simulation import traces as traces_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())
    schemas: list[Any] = []
    prompts: list[str] = []

    async def fake_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        schemas.append(kwargs["response_format"])
        prompts.append(kwargs["messages"][1]["content"])
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", fake_generate_structured)
    _stub_first_message(monkeypatch, "Where is it?")

    await datapoints_from_traces(
        [_make_conversation("abc123")],
        client=MagicMock(),
        summaries={"abc123": "Already summarized elsewhere."},
    )

    assert schemas == [traces_mod._InferredPersonaScenario]  # no second summarize call
    assert "Already summarized elsewhere." in prompts[0]


@pytest.mark.asyncio
async def test_datapoints_from_traces_skips_summarize_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A conversation whose *summarize* call fails is dropped before inference ever runs."""
    from evaluatorq.simulation import traces as traces_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())

    async def flaky_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        if kwargs["response_format"] is traces_mod._ConversationSummary:
            if "bad" in kwargs["messages"][1]["content"]:
                raise RuntimeError("LLM down")
            return traces_mod._ConversationSummary(summary="good conversation"), ""
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", flaky_generate_structured)
    _stub_first_message(monkeypatch, "Where is it?")

    conversations = [
        TraceConversation(trace_id="bad", messages=[{"role": "user", "content": "bad marker"}]),
        TraceConversation(trace_id="good", messages=[{"role": "user", "content": "fine"}]),
    ]
    datapoints = await datapoints_from_traces(conversations, client=MagicMock())

    assert [dp.id for dp in datapoints] == ["trace-good"]


@pytest.mark.asyncio
async def test_datapoints_from_traces_skips_inference_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the ``except Exception`` branch around the persona/scenario inference call:
    a raising inference call drops only that conversation, not the whole batch."""
    from evaluatorq.simulation import traces as traces_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())

    async def fake_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        if kwargs["response_format"] is traces_mod._ConversationSummary:
            summary = "bad summary" if "bad marker" in kwargs["messages"][1]["content"] else "good summary"
            return traces_mod._ConversationSummary(summary=summary), ""
        # response_format is _InferredPersonaScenario: the inference call itself.
        if "bad summary" in kwargs["messages"][1]["content"]:
            raise RuntimeError("inference LLM down")
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", fake_generate_structured)
    _stub_first_message(monkeypatch, "Where is it?")

    conversations = [
        TraceConversation(trace_id="bad", messages=[{"role": "user", "content": "bad marker"}]),
        TraceConversation(trace_id="good", messages=[{"role": "user", "content": "fine"}]),
    ]
    datapoints = await datapoints_from_traces(conversations, client=MagicMock())

    assert [dp.id for dp in datapoints] == ["trace-good"]


@pytest.mark.asyncio
async def test_datapoints_from_traces_skips_unparseable_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the ``parsed is None`` branch after the inference call: an unparseable
    response drops only that conversation, not the whole batch."""
    from evaluatorq.simulation import traces as traces_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())

    async def fake_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        if kwargs["response_format"] is traces_mod._ConversationSummary:
            summary = "bad summary" if "bad marker" in kwargs["messages"][1]["content"] else "good summary"
            return traces_mod._ConversationSummary(summary=summary), ""
        if "bad summary" in kwargs["messages"][1]["content"]:
            return None, ""
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", fake_generate_structured)
    _stub_first_message(monkeypatch, "Where is it?")

    conversations = [
        TraceConversation(trace_id="bad", messages=[{"role": "user", "content": "bad marker"}]),
        TraceConversation(trace_id="good", messages=[{"role": "user", "content": "fine"}]),
    ]
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
        # Extension mode is map-then-reduce: every conversation is summarized first,
        # so the stub has to answer both schemas.
        if kwargs["response_format"] is traces_mod._ConversationSummary:
            return traces_mod._ConversationSummary(summary="Wants a refund, casual, impatient."), ""
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
        if kwargs["response_format"] is traces_mod._ConversationSummary:
            return traces_mod._ConversationSummary(summary="summary"), ""
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", tracked_generate_structured)
    _stub_first_message(monkeypatch, "Where is it?")

    conversations = [_make_conversation(f"t{i}") for i in range(12)]
    datapoints = await datapoints_from_traces(conversations, client=MagicMock())

    assert [dp.id for dp in datapoints] == [f"trace-t{i}" for i in range(12)]
    assert state["peak"] > 1  # actually concurrent, not sequential
    assert state["peak"] <= traces_mod._INFER_CONCURRENCY


@pytest.mark.asyncio
@pytest.mark.parametrize("redact", [True, False])
async def test_redaction_instruction_follows_the_flag(monkeypatch: pytest.MonkeyPatch, redact: bool) -> None:
    """The knob is the prompt: on, every prompt carries the rule; off, none of them do."""
    from evaluatorq.simulation import traces as traces_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())
    systems: list[str] = []

    async def fake_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        systems.append(kwargs["messages"][0]["content"])
        if kwargs["response_format"] is traces_mod._ConversationSummary:
            return traces_mod._ConversationSummary(summary="Wants a refund."), ""
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", fake_generate_structured)
    _stub_first_message(monkeypatch, "Where is it?")

    long_conversation = traces_mod.TraceConversation(
        trace_id="long",
        messages=[{"role": "user", "content": "z" * 5000}],
    )
    await datapoints_from_traces(
        [long_conversation],
        client=MagicMock(),
        config=traces_mod.TraceAnalysisConfig(redact_pii=redact),
    )

    # Both the summarize prompt and the persona/scenario prompt, since either can
    # copy an order number straight out of the transcript.
    assert len(systems) == 2
    assert all(("[CUSTOMER_NAME]" in s) is redact for s in systems)


# ---------------------------------------------------------------------------
# summarize_conversations (the shared map step, exercised directly)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_conversations_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.simulation import traces as traces_mod

    async def fake_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        transcript = kwargs["messages"][1]["content"]
        trace_id = "one" if "first message" in transcript else "two"
        return traces_mod._ConversationSummary(summary=f"summary for {trace_id}"), ""

    monkeypatch.setattr(traces_mod, "generate_structured", fake_generate_structured)

    conversations = [
        TraceConversation(trace_id="one", messages=[{"role": "user", "content": "first message"}]),
        TraceConversation(trace_id="two", messages=[{"role": "user", "content": "second message"}]),
    ]
    summaries = await summarize_conversations(conversations, client=MagicMock())

    assert summaries == {"one": "summary for one", "two": "summary for two"}


@pytest.mark.asyncio
async def test_summarize_conversations_drops_failures_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from evaluatorq.simulation import traces as traces_mod

    async def flaky_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        if "bad" in kwargs["messages"][1]["content"]:
            raise RuntimeError("LLM down")
        return traces_mod._ConversationSummary(summary="fine"), ""

    monkeypatch.setattr(traces_mod, "generate_structured", flaky_generate_structured)

    conversations = [
        TraceConversation(trace_id="bad", messages=[{"role": "user", "content": "bad marker"}]),
        TraceConversation(trace_id="good", messages=[{"role": "user", "content": "fine"}]),
    ]
    with caplog.at_level("WARNING"):
        summaries = await summarize_conversations(conversations, client=MagicMock())

    assert summaries == {"good": "fine"}
    assert "bad" in caplog.text


@pytest.mark.asyncio
async def test_summarize_conversations_all_fail_returns_empty_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.simulation import traces as traces_mod

    async def always_fails(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        raise RuntimeError("LLM down")

    monkeypatch.setattr(traces_mod, "generate_structured", always_fails)

    conversations = [_make_conversation("t1"), _make_conversation("t2")]
    summaries = await summarize_conversations(conversations, client=MagicMock())

    assert summaries == {}


@pytest.mark.asyncio
async def test_summarize_conversations_is_bounded_concurrent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors ``test_datapoints_from_traces_inference_is_bounded_concurrent`` for the
    map step itself, called directly rather than through a downstream mode."""
    import asyncio

    from evaluatorq.simulation import traces as traces_mod

    state = {"active": 0, "peak": 0}

    async def tracked_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        state["active"] -= 1
        return traces_mod._ConversationSummary(summary="summary"), ""

    monkeypatch.setattr(traces_mod, "generate_structured", tracked_generate_structured)

    conversations = [_make_conversation(f"t{i}") for i in range(12)]
    summaries = await summarize_conversations(conversations, client=MagicMock())

    assert len(summaries) == 12
    assert state["peak"] > 1  # actually concurrent, not sequential
    assert state["peak"] <= traces_mod._INFER_CONCURRENCY


@pytest.mark.asyncio
async def test_supplied_partial_summaries_drop_missing_without_resummarizing_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix: a supplied ``summaries=`` mapping is authoritative in direct mode — a
    trace_id absent from it (already attempted and warned about) is dropped without
    a second summarize call, not silently re-summarized."""
    from evaluatorq.simulation import traces as traces_mod

    parsed = traces_mod._InferredPersonaScenario(persona=_make_persona(), scenario=_make_scenario())
    schemas: list[Any] = []

    async def fake_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        schemas.append(kwargs["response_format"])
        return parsed, ""

    monkeypatch.setattr(traces_mod, "generate_structured", fake_generate_structured)
    _stub_first_message(monkeypatch, "Where is it?")

    conversations = [_make_conversation("present"), _make_conversation("missing")]
    datapoints = await datapoints_from_traces(
        conversations,
        client=MagicMock(),
        summaries={"present": "Already summarized."},
    )

    assert [dp.id for dp in datapoints] == ["trace-present"]
    # Only the inference call for "present" ran — no summarize call for either trace.
    assert schemas == [traces_mod._InferredPersonaScenario]


@pytest.mark.asyncio
async def test_supplied_partial_summaries_drop_missing_without_resummarizing_extend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix: the same authoritative-mapping contract in extension mode — the
    missing trace is dropped from the profile without a second summarize call."""
    from evaluatorq.simulation import traces as traces_mod
    from evaluatorq.simulation.generators import datapoint_generator as dpg_mod
    from evaluatorq.simulation.utils.prompt_builders import generate_datapoint

    profile = traces_mod._TrafficProfile(profile="profile text")
    schemas: list[Any] = []

    async def fake_generate_structured(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        schemas.append(kwargs["response_format"])
        return profile, ""

    monkeypatch.setattr(traces_mod, "generate_structured", fake_generate_structured)

    class FakeGenerator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def generate_from_description(self, **kwargs: Any) -> list[Any]:
            return [generate_datapoint(_make_persona(), _make_scenario())]

        async def close(self) -> None:
            pass

    monkeypatch.setattr(dpg_mod, "DatapointGenerator", FakeGenerator)

    conversations = [_make_conversation("present"), _make_conversation("missing")]
    await extend_from_traces(
        conversations,
        num_datapoints=1,
        client=MagicMock(),
        summaries={"present": "Already summarized."},
    )

    # Only the profile call ran — no summarize call for either trace.
    assert schemas == [traces_mod._TrafficProfile]
