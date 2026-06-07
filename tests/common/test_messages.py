"""Unit tests for common.messages.coerce_content_text."""

from __future__ import annotations

from evaluatorq.common.messages import coerce_content_text


def test_plain_string_passes_through():
    assert coerce_content_text("hello") == "hello"


def test_none_becomes_empty_string():
    assert coerce_content_text(None) == ""


def test_list_content_surfaces_text_parts():
    content = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert coerce_content_text(content) == "first\nsecond"


def test_list_ignores_non_text_parts():
    content = [
        {"type": "text", "text": "keep"},
        {"type": "image_url", "image_url": {"url": "http://x"}},
        "not-a-dict",
    ]
    assert coerce_content_text(content) == "keep"


def test_empty_list_is_empty_string():
    assert coerce_content_text([]) == ""


def test_non_string_scalar_is_stringified():
    assert coerce_content_text(123) == "123"
