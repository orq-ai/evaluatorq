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

    def test_tool_calls_type_with_no_content_warns_without_calling_it_unrecognised(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A pure tool-call turn routinely carries content=None; the type IS recognised,
        so the warning must say the content was empty, not that the type is unrecognised."""
        completion = _completion_with_message(SimpleNamespace(type="tool_calls", content=None))
        with caplog.at_level("WARNING"):
            result = _extract_content_from_response(completion)
        assert result == ""
        assert "tool_calls" in caplog.text
        assert "no text content" in caplog.text
        assert "Unrecognised" not in caplog.text

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("", id="empty-string"),
            pytest.param([], id="empty-list"),
            pytest.param([{"type": "image_url", "image_url": {"url": "x"}}], id="list-with-no-text-parts"),
        ],
    )
    def test_content_type_with_no_text_warns_instead_of_returning_empty_silently(
        self, content: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An empty reply always logs.

        These shapes previously matched the str/list branches, returned '' and
        logged nothing — so an empty deployment reply was indistinguishable from
        a working one, while the sibling content=None case on the same msg_type
        warned. Two adjacent branches must not differ in whether they log.
        """
        completion = _completion_with_message(SimpleNamespace(type="content", content=content))
        with caplog.at_level("WARNING"):
            result = _extract_content_from_response(completion)
        assert result == ""
        assert "no text content" in caplog.text
        assert "Unrecognised" not in caplog.text

    def test_content_type_with_multimodal_text_parts_still_joins_them(self) -> None:
        completion = _completion_with_message(
            SimpleNamespace(
                type="content",
                content=[{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
            )
        )
        assert _extract_content_from_response(completion) == "one\ntwo"
