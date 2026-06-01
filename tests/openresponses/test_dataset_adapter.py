"""Tests for the OpenResponses → RedTeamSample dataset adapter.

Datasets stored in OpenResponses format can be loaded into the existing
RedTeamSample schema (which stores attack conversations as OpenAI chat
messages) without re-authoring.
"""

from __future__ import annotations

import json

import pytest

from evaluatorq.openresponses import (
    load_openresponses_dataset,
    messages_from_openresponses_input,
    redteam_sample_from_openresponses,
)
from evaluatorq.redteam.contracts import (
    RedTeamInput,
    Severity,
    TurnType,
    VulnerabilityDomain,
)


def _input() -> RedTeamInput:
    return RedTeamInput(
        id="s1",
        vulnerability="prompt_injection",
        severity=Severity.HIGH,
        vulnerability_domain=VulnerabilityDomain.MODEL,
        turn_type=TurnType.SINGLE,
        source="unit-test",
    )


def _dataset_row() -> dict[str, object]:
    return {
        "input": _input().model_dump(mode="json"),
        "openresponses_input": [{"role": "user", "content": "attack"}],
    }


class TestMessagesFromOpenResponsesInput:
    def test_simple_user_assistant_pair_round_trips(self):
        result = messages_from_openresponses_input([
            {"role": "user", "content": "initial attack"},
            {"role": "assistant", "content": "agent response"},
        ])
        assert result == [
            {"role": "user", "content": "initial attack"},
            {"role": "assistant", "content": "agent response"},
        ]

    def test_system_role_passes_through(self):
        result = messages_from_openresponses_input([
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "x"},
        ])
        assert result == [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "x"},
        ]

    def test_message_item_with_output_text_blocks(self):
        result = messages_from_openresponses_input([
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "first "},
                    {"type": "output_text", "text": "second"},
                ],
            },
        ])
        assert result == [{"role": "assistant", "content": "first second"}]

    def test_message_item_with_input_text_blocks(self):
        result = messages_from_openresponses_input([
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "attack"}],
            },
        ])
        assert result == [{"role": "user", "content": "attack"}]

    def test_function_call_items_are_dropped(self):
        result = messages_from_openresponses_input([
            {"role": "user", "content": "use the search tool"},
            {"type": "function_call", "name": "search", "arguments": "{}", "call_id": "c1"},
            {"role": "user", "content": "follow-up"},
        ])
        assert result == [
            {"role": "user", "content": "use the search tool"},
            {"role": "user", "content": "follow-up"},
        ]

    def test_list_content_is_flattened(self):
        result = messages_from_openresponses_input([
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        ])
        assert result == [{"role": "user", "content": "hello"}]


class TestRedTeamSampleFromOpenResponses:
    def test_builds_valid_sample(self):
        sample = redteam_sample_from_openresponses(
            input=_input(),
            openresponses_input=[
                {"role": "user", "content": "attack prompt"},
            ],
        )
        assert sample.input.id == "s1"
        assert len(sample.messages) == 1
        assert sample.messages[0].role == "user"
        assert sample.messages[0].content == "attack prompt"

    def test_multi_turn_conversation(self):
        sample = redteam_sample_from_openresponses(
            input=_input(),
            openresponses_input=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ],
        )
        assert [m.role for m in sample.messages] == ["user", "assistant", "user"]

    def test_raises_when_no_messages_extracted(self):
        with pytest.raises(ValueError, match="produced no messages"):
            redteam_sample_from_openresponses(
                input=_input(),
                openresponses_input=[
                    {"type": "function_call", "name": "x", "arguments": "{}"},
                ],
            )


class TestLoadOpenResponsesDataset:
    def test_loads_top_level_json_list(self, tmp_path):
        path = tmp_path / "samples.json"
        path.write_text(json.dumps([_dataset_row()]), encoding="utf-8")

        dataset = load_openresponses_dataset(path)

        assert len(dataset.samples) == 1
        assert dataset.samples[0].input.id == "s1"
        assert dataset.samples[0].messages[0].content == "attack"

    def test_loads_samples_wrapper(self, tmp_path):
        path = tmp_path / "samples.json"
        path.write_text(json.dumps({"samples": [_dataset_row()]}), encoding="utf-8")

        dataset = load_openresponses_dataset(path)

        assert len(dataset.samples) == 1

    def test_loads_jsonl_and_skips_blank_lines(self, tmp_path):
        path = tmp_path / "samples.jsonl"
        path.write_text(
            "\n".join([json.dumps(_dataset_row()), "", json.dumps(_dataset_row())]),
            encoding="utf-8",
        )

        dataset = load_openresponses_dataset(path)

        assert len(dataset.samples) == 2

    def test_malformed_jsonl_reports_line_number(self, tmp_path):
        path = tmp_path / "samples.jsonl"
        path.write_text(f"{json.dumps(_dataset_row())}\n{{bad json", encoding="utf-8")

        with pytest.raises(ValueError, match="line 2"):
            load_openresponses_dataset(path)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_openresponses_dataset(tmp_path / "missing.json")

    def test_invalid_row_type_raises_value_error(self, tmp_path):
        path = tmp_path / "samples.json"
        path.write_text(json.dumps([123]), encoding="utf-8")

        with pytest.raises(ValueError, match="row 0 is not an object"):
            load_openresponses_dataset(path)

    def test_invalid_input_metadata_raises_value_error(self, tmp_path):
        row = _dataset_row()
        row["input"] = {"id": "bad"}
        path = tmp_path / "samples.json"
        path.write_text(json.dumps([row]), encoding="utf-8")

        with pytest.raises(ValueError, match="metadata failed validation"):
            load_openresponses_dataset(path)

    def test_accepts_legacy_input_messages(self, tmp_path):
        row = _dataset_row()
        row["input_messages"] = row.pop("openresponses_input")
        path = tmp_path / "samples.json"
        path.write_text(json.dumps([row]), encoding="utf-8")

        dataset = load_openresponses_dataset(path)

        assert dataset.samples[0].messages[0].content == "attack"

    def test_reads_utf8_non_ascii_content(self, tmp_path):
        row = _dataset_row()
        row["openresponses_input"] = [{"role": "user", "content": "olá"}]
        path = tmp_path / "samples.json"
        path.write_text(json.dumps([row], ensure_ascii=False), encoding="utf-8")

        dataset = load_openresponses_dataset(path)

        assert dataset.samples[0].messages[0].content == "olá"
