"""Tests for the reverse converters and dataset loader.

Covers the symmetric serialization path:
- :func:`agent_response_to_openresponses` — AgentResponse → ResponseResource dict
- :func:`orchestrator_result_to_openresponses_input` — Turn list → OpenResponses input
- :func:`load_openresponses_dataset` — JSON / JSONL file → StaticDataset
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluatorq.contracts import (
    AgentResponse,
    TextOutputItem,
    ToolCallOutputItem,
)
from evaluatorq.redteam.contracts import (
    AttackerResponse,
    TokenUsage,
    Turn,
)
from evaluatorq.redteam.openresponses_adapter import (
    agent_response_from_openresponses,
    agent_response_to_openresponses,
    load_openresponses_dataset,
    orchestrator_result_to_openresponses_input,
)


class TestAgentResponseToOpenResponses:
    def test_text_only_response_collapses_to_one_message(self):
        resp = AgentResponse(output=[
            TextOutputItem(text="hello ", annotations=[]),
            TextOutputItem(text="world", annotations=[]),
        ])
        result = agent_response_to_openresponses(resp)
        assert result["object"] == "response"
        assert result["output"] == [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello world"}],
        }]

    def test_function_call_breaks_message_boundary(self):
        resp = AgentResponse(output=[
            TextOutputItem(text="checking ", annotations=[]),
            ToolCallOutputItem(name="search", call_id="call_1", arguments='{"q":"x"}'),
            TextOutputItem(text="found it", annotations=[]),
        ])
        result = agent_response_to_openresponses(resp)
        assert [item["type"] for item in result["output"]] == ["message", "function_call", "message"]
        assert result["output"][0]["content"][0]["text"] == "checking "
        assert result["output"][1]["name"] == "search"
        assert result["output"][1]["call_id"] == "call_1"
        assert result["output"][2]["content"][0]["text"] == "found it"

    def test_metadata_fields_carried_through(self):
        resp = AgentResponse(
            output=[TextOutputItem(text="ok", annotations=[])],
            model="agent-id",
            response_id="resp_1",
            finish_reason="completed",
        )
        result = agent_response_to_openresponses(resp)
        assert result["id"] == "resp_1"
        assert result["model"] == "agent-id"
        assert result["status"] == "completed"

    def test_none_metadata_omitted(self):
        resp = AgentResponse(output=[TextOutputItem(text="x", annotations=[])])
        result = agent_response_to_openresponses(resp)
        assert "id" not in result
        assert "model" not in result
        assert "status" not in result

    def test_usage_block_emitted(self):
        resp = AgentResponse(
            output=[TextOutputItem(text="x", annotations=[])],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        result = agent_response_to_openresponses(resp)
        assert result["usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }

    def test_round_trips_through_from_converter(self):
        original = AgentResponse(
            output=[
                TextOutputItem(text="hello", annotations=[]),
                ToolCallOutputItem(name="search", call_id="c1", arguments="{}"),
            ],
            model="agent-id",
            response_id="resp_99",
        )
        wire = agent_response_to_openresponses(original)
        back = agent_response_from_openresponses(wire)
        assert back.text == "hello"
        assert len(back.tool_calls) == 1
        assert back.tool_calls[0].name == "search"
        assert back.response_id == "resp_99"
        assert back.model == "agent-id"


class TestOrchestratorResultToOpenResponsesInput:
    def test_aliases_turns_to_openresponses_input(self):
        turns = [
            Turn(
                attacker=AttackerResponse(generated_prompt="first attack"),
                target=AgentResponse(output=[TextOutputItem(text="agent reply", annotations=[])]),
            ),
        ]
        result = orchestrator_result_to_openresponses_input(turns)
        assert result == [
            {"role": "user", "content": "first attack"},
            {"role": "assistant", "content": "agent reply"},
        ]


class TestLoadOpenResponsesDataset:
    @pytest.fixture
    def _sample_row(self) -> dict:
        return {
            "input": {
                "id": "s1",
                "vulnerability": "prompt_injection",
                "severity": "high",
                "vulnerability_domain": "model",
                "turn_type": "single",
                "source": "test-fixture",
            },
            "openresponses_input": [
                {"role": "user", "content": "adversarial prompt here"},
            ],
        }

    def test_loads_json_object_with_samples_key(self, tmp_path: Path, _sample_row):
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps({"samples": [_sample_row, _sample_row]}))
        ds = load_openresponses_dataset(path)
        assert len(ds.samples) == 2
        assert ds.samples[0].input.id == "s1"
        assert ds.samples[0].messages[0].content == "adversarial prompt here"

    def test_loads_bare_json_list(self, tmp_path: Path, _sample_row):
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps([_sample_row]))
        ds = load_openresponses_dataset(path)
        assert len(ds.samples) == 1

    def test_loads_jsonl(self, tmp_path: Path, _sample_row):
        path = tmp_path / "dataset.jsonl"
        path.write_text("\n".join(json.dumps(_sample_row) for _ in range(3)))
        ds = load_openresponses_dataset(path)
        assert len(ds.samples) == 3

    def test_jsonl_skips_blank_lines(self, tmp_path: Path, _sample_row):
        path = tmp_path / "dataset.jsonl"
        path.write_text(json.dumps(_sample_row) + "\n\n" + json.dumps(_sample_row) + "\n")
        ds = load_openresponses_dataset(path)
        assert len(ds.samples) == 2

    def test_accepts_legacy_input_messages_key(self, tmp_path: Path):
        row = {
            "input": {
                "id": "legacy-1",
                "vulnerability": "prompt_injection",
                "severity": "low",
                "vulnerability_domain": "model",
                "turn_type": "single",
                "source": "legacy",
            },
            "input_messages": [{"role": "user", "content": "x"}],
        }
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps([row]))
        ds = load_openresponses_dataset(path)
        assert ds.samples[0].messages[0].content == "x"

    def test_loads_multi_turn_conversation(self, tmp_path: Path):
        # Matches the ticket's multi-turn example shape.
        row = {
            "input": {
                "id": "mt-1",
                "vulnerability": "prompt_injection",
                "severity": "high",
                "vulnerability_domain": "model",
                "turn_type": "multi",
                "source": "ticket-example",
            },
            "openresponses_input": [
                {"role": "user", "content": "initial attack"},
                {"role": "assistant", "content": "agent response"},
                {"role": "user", "content": "follow-up attack"},
            ],
        }
        path = tmp_path / "mt.json"
        path.write_text(json.dumps([row]))
        ds = load_openresponses_dataset(path)
        assert [m.role for m in ds.samples[0].messages] == ["user", "assistant", "user"]
        assert ds.samples[0].messages[-1].content == "follow-up attack"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_openresponses_dataset(tmp_path / "does-not-exist.json")

    def test_malformed_jsonl_includes_line_number(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"valid": 1}\n{not valid\n')
        with pytest.raises(ValueError, match="line 2"):
            load_openresponses_dataset(path)

    def test_unexpected_shape_raises(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text('"just a string"')
        with pytest.raises(ValueError, match="Unexpected dataset shape"):
            load_openresponses_dataset(path)

    def test_row_missing_input_metadata_raises(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"openresponses_input": []}]))
        with pytest.raises(ValueError, match="missing the 'input' metadata"):
            load_openresponses_dataset(path)

    def test_row_missing_conversation_raises(self, tmp_path: Path, _sample_row):
        row = dict(_sample_row)
        del row["openresponses_input"]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([row]))
        with pytest.raises(ValueError, match="missing 'openresponses_input'"):
            load_openresponses_dataset(path)
