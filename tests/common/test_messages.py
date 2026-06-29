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


def test_list_surfaces_non_text_parts_as_placeholders():
    """Image/file parts are visibly accounted for, not silently dropped."""
    content = [
        {"type": "text", "text": "keep"},
        {"type": "image_url", "image_url": {"url": "http://x"}},
        {"type": "input_file", "file_id": "f-1"},
    ]
    assert coerce_content_text(content) == "keep\n[image]\n[file]"


def test_list_surfaces_unknown_part_types_as_placeholders():
    """Unknown/future part shapes are surfaced, not silently dropped."""
    content = [
        {"type": "text", "text": "keep"},
        {"type": "input_audio", "input_audio": {}},
        {"type": "output_text", "text": "ignored-key"},
        "not-a-dict",
    ]
    assert coerce_content_text(content) == "keep\n[input_audio]\n[output_text]\n[unknown]"


def test_empty_list_is_empty_string():
    assert coerce_content_text([]) == ""


def test_non_string_scalar_is_stringified():
    assert coerce_content_text(123) == "123"
