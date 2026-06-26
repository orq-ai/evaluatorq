"""RES-879: Message supports the developer role and multi-part content."""

from __future__ import annotations

import pytest

from evaluatorq.contracts import ContentPart, Message, content_to_text
from evaluatorq.openresponses.convert_models import InputImageContent, InputTextContent


def test_developer_role_accepted() -> None:
    m = Message(role="developer", content="You are a developer assistant.")
    assert m.role == "developer"


def test_content_accepts_multipart_list() -> None:
    m = Message(
        role="user",
        content=[
            InputTextContent(type="input_text", text="what is this?"),
            InputImageContent(type="input_image", image_url="https://x/y.png"),
        ],
    )
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


def test_orq_responses_target_passes_multipart_through() -> None:
    """The Responses target serializes multi-part content to input content parts."""
    from evaluatorq.openresponses.target import OrqResponsesTarget

    m = Message(
        role="user",
        content=[
            InputTextContent(type="input_text", text="what is this?"),
            InputImageContent(type="input_image", image_url="https://x/y.png"),
        ],
    )
    items = OrqResponsesTarget._messages_to_input([m])
    assert items[0]["role"] == "user"
    parts = items[0]["content"]
    assert isinstance(parts, list)
    assert parts[0]["type"] == "input_text"
    assert parts[0]["text"] == "what is this?"
    assert parts[1]["type"] == "input_image"
    assert parts[1]["image_url"] == "https://x/y.png"


def test_content_to_text_raises_on_image_part() -> None:
    """Text-only targets refuse non-text content rather than silently dropping it."""
    content: list[ContentPart] = [
        InputTextContent(type="input_text", text="hi"),
        InputImageContent(type="input_image", image_url="https://x/y.png"),
    ]
    with pytest.raises(NotImplementedError):
        content_to_text(content)


def test_content_to_text_text_only() -> None:
    content: list[ContentPart] = [
        InputTextContent(type="input_text", text="a"),
        InputTextContent(type="input_text", text="b"),
    ]
    assert content_to_text(content) == "ab"
