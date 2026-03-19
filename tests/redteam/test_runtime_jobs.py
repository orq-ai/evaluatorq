"""Unit tests for runtime/jobs.py and runtime/orq_agent_job.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from evaluatorq.redteam.contracts import Message, TokenUsage


# ===========================================================================
# runtime/orq_agent_job.py — _sanitize_job_name
# ===========================================================================


class TestSanitizeJobName:
    """Tests for _sanitize_job_name()."""

    def test_alphanumeric_passthrough(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _sanitize_job_name

        assert _sanitize_job_name("abc123") == "abc123"

    def test_dash_and_underscore_preserved(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _sanitize_job_name

        assert _sanitize_job_name("my-agent_key") == "my-agent_key"

    def test_special_chars_replaced_with_dash(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _sanitize_job_name

        result = _sanitize_job_name("my.agent/key@v2")
        assert result == "my-agent-key-v2"

    def test_leading_trailing_dashes_stripped(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _sanitize_job_name

        result = _sanitize_job_name(".leading-trailing.")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_empty_string_returns_unknown(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _sanitize_job_name

        assert _sanitize_job_name("") == "unknown"

    def test_all_special_chars_becomes_unknown(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _sanitize_job_name

        # All special chars → all dashes → strip → empty → 'unknown'
        result = _sanitize_job_name("@@@")
        assert result == "unknown"

    def test_spaces_replaced(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _sanitize_job_name

        result = _sanitize_job_name("my agent key")
        assert " " not in result

    def test_model_with_slash(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _sanitize_job_name

        result = _sanitize_job_name("azure/gpt-4o-mini")
        assert "/" not in result
        assert "azure" in result
        assert "gpt" in result


# ===========================================================================
# runtime/orq_agent_job.py — _message_content_to_text
# ===========================================================================


class TestMessageContentToText:
    """Tests for _message_content_to_text()."""

    def test_string_content_passthrough(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _message_content_to_text

        assert _message_content_to_text("Hello world") == "Hello world"

    def test_list_of_text_parts_joined(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _message_content_to_text

        parts = [
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ]
        result = _message_content_to_text(parts)
        assert "First part" in result
        assert "Second part" in result

    def test_list_filters_non_text_parts(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _message_content_to_text

        parts = [
            {"type": "image_url", "url": "http://example.com/img.png"},
            {"type": "text", "text": "Only this"},
        ]
        result = _message_content_to_text(parts)
        assert result == "Only this"
        assert "image_url" not in result

    def test_none_content_returns_empty_string(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _message_content_to_text

        assert _message_content_to_text(None) == ""

    def test_other_types_stringified(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _message_content_to_text

        result = _message_content_to_text(42)
        assert result == "42"

    def test_empty_list_returns_empty_string(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _message_content_to_text

        assert _message_content_to_text([]) == ""


# ===========================================================================
# runtime/orq_agent_job.py — _normalize_message
# ===========================================================================


class TestNormalizeMessage:
    """Tests for _normalize_message()."""

    def test_message_object_passthrough(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _normalize_message

        msg = Message(role="user", content="Hello")
        result = _normalize_message(msg)
        assert result is msg

    def test_valid_dict_parsed_to_message(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _normalize_message

        raw = {"role": "user", "content": "Hello"}
        result = _normalize_message(raw)
        assert isinstance(result, Message)
        assert result.role == "user"
        assert result.content == "Hello"

    def test_invalid_dict_kept_as_dict(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _normalize_message

        # Role 'unknown' is not a valid literal → validation fails → returns raw dict
        raw = {"role": "unknown_role", "content": "Something"}
        result = _normalize_message(raw)
        assert isinstance(result, dict)
        assert result["role"] == "unknown_role"

    def test_non_dict_non_message_becomes_user_dict(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _normalize_message

        result = _normalize_message("some plain string")
        assert isinstance(result, dict)
        assert result["role"] == "user"
        assert result["content"] == "some plain string"

    def test_integer_becomes_user_dict(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _normalize_message

        result = _normalize_message(42)
        assert isinstance(result, dict)
        assert result["role"] == "user"
        assert result["content"] == "42"


# ===========================================================================
# runtime/orq_agent_job.py — _extract_agent_response_text
# ===========================================================================


class TestExtractAgentResponseText:
    """Tests for _extract_agent_response_text()."""

    def _make_response(self, items: list[Any]) -> MagicMock:
        """Create a mock response with .output list."""
        response = MagicMock()
        response.output = items
        return response

    def _make_agent_item(self, texts: list[str]) -> MagicMock:
        """Create an output item with role='agent' and text parts."""
        item = MagicMock()
        item.role = "agent"
        parts = []
        for t in texts:
            part = MagicMock()
            part.kind = "text"
            part.text = t
            parts.append(part)
        item.parts = parts
        return item

    def _make_non_agent_item(self, text: str = "tool output") -> MagicMock:
        """Create an output item with role='tool'."""
        item = MagicMock()
        item.role = "tool"
        part = MagicMock()
        part.kind = "text"
        part.text = text
        item.parts = [part]
        return item

    def test_extracts_agent_text_parts(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _extract_agent_response_text

        response = self._make_response([self._make_agent_item(["Hello", " World"])])
        result = _extract_agent_response_text(response)
        assert "Hello" in result
        assert "World" in result

    def test_skips_non_agent_role(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _extract_agent_response_text

        response = self._make_response([self._make_non_agent_item("tool output")])
        result = _extract_agent_response_text(response)
        assert result == ""

    def test_empty_output_returns_empty_string(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _extract_agent_response_text

        response = self._make_response([])
        result = _extract_agent_response_text(response)
        assert result == ""

    def test_multiple_agent_items_joined(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _extract_agent_response_text

        response = self._make_response([
            self._make_agent_item(["Part one"]),
            self._make_agent_item(["Part two"]),
        ])
        result = _extract_agent_response_text(response)
        assert "Part one" in result
        assert "Part two" in result

    def test_ignores_non_text_parts(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _extract_agent_response_text

        item = MagicMock()
        item.role = "agent"
        non_text_part = MagicMock()
        non_text_part.kind = "tool_call"
        non_text_part.text = "should not appear"
        item.parts = [non_text_part]

        response = self._make_response([item])
        result = _extract_agent_response_text(response)
        assert result == ""

    def test_mixed_agent_and_non_agent(self):
        from evaluatorq.redteam.runtime.orq_agent_job import _extract_agent_response_text

        response = self._make_response([
            self._make_non_agent_item("tool result"),
            self._make_agent_item(["Final answer"]),
        ])
        result = _extract_agent_response_text(response)
        assert "Final answer" in result
        assert "tool result" not in result


# ===========================================================================
# runtime/jobs.py — _build_messages
# ===========================================================================


class TestBuildMessages:
    """Tests for _build_messages()."""

    def _make_datapoint(self, messages: list[Any]) -> MagicMock:
        """Create a mock DataPoint with an inputs['messages'] list."""
        dp = MagicMock()
        dp.inputs = {"messages": messages}
        return dp

    def test_from_message_objects(self):
        from evaluatorq.redteam.runtime.jobs import _build_messages

        dp = self._make_datapoint([
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ])
        result = _build_messages(dp)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "assistant"

    def test_from_valid_dicts(self):
        from evaluatorq.redteam.runtime.jobs import _build_messages

        dp = self._make_datapoint([
            {"role": "user", "content": "What is the weather?"},
        ])
        result = _build_messages(dp)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What is the weather?"

    def test_from_invalid_dicts_kept_as_raw(self):
        from evaluatorq.redteam.runtime.jobs import _build_messages

        # Invalid role → validation fails → raw dict kept
        dp = self._make_datapoint([
            {"role": "invalid_role", "content": "Something"},
        ])
        result = _build_messages(dp)
        assert len(result) == 1
        assert result[0]["role"] == "invalid_role"

    def test_from_plain_strings(self):
        from evaluatorq.redteam.runtime.jobs import _build_messages

        dp = self._make_datapoint(["Just a plain string"])
        result = _build_messages(dp)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Just a plain string"

    def test_mixed_types(self):
        from evaluatorq.redteam.runtime.jobs import _build_messages

        dp = self._make_datapoint([
            Message(role="system", content="System"),
            {"role": "user", "content": "Dict user"},
            "plain string",
        ])
        result = _build_messages(dp)
        assert len(result) == 3


# ===========================================================================
# runtime/jobs.py — _extract_deployment_content
# ===========================================================================


class TestExtractDeploymentContent:
    """Tests for _extract_deployment_content()."""

    def _make_completion(self, content) -> MagicMock:
        """Create a mock deployment completion with choices[0].message.content."""
        completion = MagicMock()
        choice = MagicMock()
        message = MagicMock()
        message.content = content
        choice.message = message
        completion.choices = [choice]
        return completion

    def test_string_content_returned_directly(self):
        from evaluatorq.redteam.runtime.jobs import _extract_deployment_content

        completion = self._make_completion("Hello world")
        assert _extract_deployment_content(completion) == "Hello world"

    def test_list_content_text_parts_joined(self):
        from evaluatorq.redteam.runtime.jobs import _extract_deployment_content

        part1 = MagicMock()
        part1.type = "text"
        part1.text = "First"
        part2 = MagicMock()
        part2.type = "text"
        part2.text = "Second"

        completion = self._make_completion([part1, part2])
        result = _extract_deployment_content(completion)
        assert "First" in result
        assert "Second" in result

    def test_list_content_non_text_parts_filtered(self):
        from evaluatorq.redteam.runtime.jobs import _extract_deployment_content

        text_part = MagicMock()
        text_part.type = "text"
        text_part.text = "Visible"
        image_part = MagicMock()
        image_part.type = "image"
        image_part.text = "Hidden"

        completion = self._make_completion([image_part, text_part])
        result = _extract_deployment_content(completion)
        assert "Visible" in result
        assert "Hidden" not in result

    def test_empty_choices_returns_empty_string(self):
        from evaluatorq.redteam.runtime.jobs import _extract_deployment_content

        completion = MagicMock()
        completion.choices = []
        assert _extract_deployment_content(completion) == ""

    def test_none_choices_returns_empty_string(self):
        from evaluatorq.redteam.runtime.jobs import _extract_deployment_content

        completion = MagicMock()
        completion.choices = None
        assert _extract_deployment_content(completion) == ""

    def test_none_content_returns_empty_string(self):
        from evaluatorq.redteam.runtime.jobs import _extract_deployment_content

        completion = self._make_completion(None)
        assert _extract_deployment_content(completion) == ""


# ===========================================================================
# runtime/jobs.py — _normalize_usage
# ===========================================================================


class TestNormalizeUsage:
    """Tests for _normalize_usage()."""

    def test_token_usage_passthrough(self):
        from evaluatorq.redteam.runtime.jobs import _normalize_usage

        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        result = _normalize_usage(usage)
        assert result is usage

    def test_dict_with_standard_keys(self):
        from evaluatorq.redteam.runtime.jobs import _normalize_usage

        raw = {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}
        result = _normalize_usage(raw)
        assert result is not None
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 15
        assert result.total_tokens == 20

    def test_dict_with_short_keys(self):
        from evaluatorq.redteam.runtime.jobs import _normalize_usage

        raw = {"prompt": 3, "completion": 7, "total": 10}
        result = _normalize_usage(raw)
        assert result is not None
        assert result.prompt_tokens == 3
        assert result.completion_tokens == 7
        assert result.total_tokens == 10

    def test_none_returns_none(self):
        from evaluatorq.redteam.runtime.jobs import _normalize_usage

        assert _normalize_usage(None) is None

    def test_non_dict_non_token_usage_returns_none(self):
        from evaluatorq.redteam.runtime.jobs import _normalize_usage

        assert _normalize_usage("not a dict") is None
        assert _normalize_usage(42) is None

    def test_empty_dict_returns_zeros(self):
        from evaluatorq.redteam.runtime.jobs import _normalize_usage

        result = _normalize_usage({})
        assert result is not None
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0


# ===========================================================================
# runtime/jobs.py — create_model_job
# ===========================================================================


class TestCreateModelJob:
    """Tests for create_model_job() factory function."""

    def test_raises_value_error_when_no_params(self):
        from evaluatorq.redteam.runtime.jobs import create_model_job

        with pytest.raises(ValueError, match="Provide one of"):
            create_model_job()
