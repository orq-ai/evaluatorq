"""Unit tests for evaluatorq.deployment content extraction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluatorq.deployment import _extract_content_from_response


def _completion_with_message(message: object) -> object:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class TestExtractContentFromResponse:
    def test_content_type_returns_content(self) -> None:
        completion = _completion_with_message(SimpleNamespace(type="content", content="hello"))
        assert _extract_content_from_response(completion) == "hello"

    def test_tool_calls_type_with_string_content_returns_content(self) -> None:
        """The 'tool_calls' union arm still carries a nullable str content field; surface it."""
        completion = _completion_with_message(
            SimpleNamespace(type="tool_calls", content="here is my answer")
        )
        assert _extract_content_from_response(completion) == "here is my answer"

    def test_unrecognised_type_and_no_content_returns_empty_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        completion = _completion_with_message(SimpleNamespace(type="image", content=None))
        with caplog.at_level("WARNING"):
            result = _extract_content_from_response(completion)
        assert result == ""
        assert "image" in caplog.text
        assert "Unrecognised deployment response" in caplog.text
