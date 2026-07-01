"""RES-879: Message supports the developer role and multi-part content."""

from __future__ import annotations

import pytest

from evaluatorq.contracts import ContentPart, Message, content_to_text
from evaluatorq.openresponses.convert_models import (
    InputFileContent,
    InputImageContent,
    InputTextContent,
)


@pytest.fixture
def text_and_image() -> list[ContentPart]:
    """A reusable text + image multi-part content list."""
    return [
        InputTextContent(type="input_text", text="what is this?"),
        InputImageContent(type="input_image", image_url="https://x/y.png"),
    ]


def test_developer_role_accepted() -> None:
    m = Message(role="developer", content="You are a developer assistant.")
    assert m.role == "developer"
    # Round-trips through chat-completions rendering, not just model validation.
    assert m.to_chat_completion() == {"role": "developer", "content": "You are a developer assistant."}


def test_content_accepts_multipart_list(text_and_image: list[ContentPart]) -> None:
    m = Message(role="user", content=text_and_image)
    assert isinstance(m.content, list)
    assert len(m.content) == 2
    assert isinstance(m.content[1], InputImageContent)


def test_text_only_content_still_works() -> None:
    """No regression: the string shorthand still validates."""
    m = Message(role="user", content="hello")
    assert m.content == "hello"
    assert m.to_chat_completion() == {"role": "user", "content": "hello"}


def test_to_chat_completion_renders_multipart() -> None:
    m = Message(
        role="user",
        content=[
            InputTextContent(type="input_text", text="describe"),
            InputImageContent(type="input_image", image_url="https://x/y.png", detail="low"),
        ],
    )
    out = m.to_chat_completion()
    assert out["role"] == "user"
    assert out["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "https://x/y.png", "detail": "low"}},
    ]


def test_to_chat_completion_renders_file_part() -> None:
    """An InputFileContent part serializes to the chat-completions ``file`` block."""
    m = Message(
        role="user",
        content=[
            InputTextContent(type="input_text", text="summarize"),
            InputFileContent(type="input_file", file_id="file-123", filename="doc.pdf"),
        ],
    )
    out = m.to_chat_completion()
    assert out["content"] == [
        {"type": "text", "text": "summarize"},
        {"type": "file", "file": {"file_id": "file-123", "filename": "doc.pdf"}},
    ]


def test_orq_responses_target_passes_multipart_through(text_and_image: list[ContentPart]) -> None:
    """The Responses target serializes multi-part content to input content parts.

    Note: this exercises the private ``_messages_to_input`` directly to assert the
    serialized wire shape without a live call; it is coupled to that implementation
    detail by design.
    """
    from evaluatorq.openresponses.target import OrqResponsesTarget

    m = Message(role="user", content=text_and_image)
    items = OrqResponsesTarget._messages_to_input([m])
    assert items[0]["role"] == "user"
    parts = items[0]["content"]
    assert isinstance(parts, list)
    assert parts[0]["type"] == "input_text"
    assert parts[0]["text"] == "what is this?"
    assert parts[1]["type"] == "input_image"
    assert parts[1]["image_url"] == "https://x/y.png"


def test_to_chat_completion_raises_on_file_id_only_image() -> None:
    """A file_id-only image is Responses-only and must not render an invalid block."""
    m = Message(
        role="user",
        content=[InputImageContent(type="input_image", file_id="file-123")],
    )
    with pytest.raises(NotImplementedError, match="file_id-backed images"):
        m.to_chat_completion()


def test_to_chat_completion_raises_on_file_url_only_file() -> None:
    """A file_url-only file has no chat-completions slot and must fail loudly."""
    m = Message(
        role="user",
        content=[InputFileContent(type="input_file", file_url="https://x/y.pdf")],
    )
    with pytest.raises(NotImplementedError, match="file_url-backed files"):
        m.to_chat_completion()


def test_content_to_text_raises_on_image_part() -> None:
    """Text-only targets refuse non-text content rather than silently dropping it."""
    content: list[ContentPart] = [
        InputTextContent(type="input_text", text="hi"),
        InputImageContent(type="input_image", image_url="https://x/y.png"),
    ]
    with pytest.raises(NotImplementedError, match="accepts only text content"):
        content_to_text(content)


def test_content_to_text_raises_on_file_part() -> None:
    """The raise covers every non-text part, not just images."""
    content: list[ContentPart] = [
        InputTextContent(type="input_text", text="hi"),
        InputFileContent(type="input_file", file_id="file-123"),
    ]
    with pytest.raises(NotImplementedError, match="accepts only text content"):
        content_to_text(content)


def test_content_to_text_text_only() -> None:
    content: list[ContentPart] = [
        InputTextContent(type="input_text", text="a"),
        InputTextContent(type="input_text", text="b"),
    ]
    assert content_to_text(content) == "ab"


def test_content_to_text_empty_list() -> None:
    assert content_to_text([]) == ""
