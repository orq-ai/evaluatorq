"""Unit tests for common.messages.coerce_content_text."""

from __future__ import annotations

from types import SimpleNamespace

from evaluatorq.common.messages import chat_message_reply_text, coerce_content_text


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


def test_empty_list_is_empty_string():
    assert coerce_content_text([]) == ""


def test_non_string_scalar_is_stringified():
    assert coerce_content_text(123) == "123"


def test_chat_message_reply_text_prefers_content_and_falls_back_to_the_refusal():
    assert chat_message_reply_text(SimpleNamespace(content='hello', refusal=None)) == 'hello'
    assert chat_message_reply_text(SimpleNamespace(content='hello', refusal='no')) == 'hello'
    assert chat_message_reply_text(SimpleNamespace(content=None, refusal='I cannot help.')) == 'I cannot help.'
    assert chat_message_reply_text(SimpleNamespace(content='', refusal='I cannot help.')) == 'I cannot help.'
    assert chat_message_reply_text(SimpleNamespace(content=None, refusal=None)) == ''
    assert chat_message_reply_text(SimpleNamespace(content=[{'type': 'text', 'text': 'a'}], refusal=None)) == 'a'
