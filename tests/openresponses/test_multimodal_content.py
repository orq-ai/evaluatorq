"""BOPS-839: multi-part multi-modal content (image + file) in openresponses models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluatorq.openresponses.convert_models import (
    InputFileContent,
    InputImageContent,
    InputTextContent,
    Message,
    MessageRole,
    MessageStatus,
    OutputTextContent,
)

# Message.content is an invariant list, so build content with the full union type.
ContentList = list[InputTextContent | InputImageContent | InputFileContent | OutputTextContent]


def test_input_image_content_minimal() -> None:
    img = InputImageContent(type="input_image", image_url="https://x/y.png")
    assert img.type == "input_image"
    assert img.image_url == "https://x/y.png"
    assert img.file_id is None
    assert img.detail == "auto"  # default


def test_input_image_content_file_id_only() -> None:
    """image_url is optional — a file_id-only image part is valid (file_id is
    the alternative source in the contract)."""
    img = InputImageContent(type="input_image", file_id="file_123")
    assert img.image_url is None
    assert img.model_dump(mode="json") == {
        "type": "input_image",
        "file_id": "file_123",
        "detail": "auto",
    }


def test_input_image_content_detail_choices() -> None:
    assert InputImageContent(type="input_image", image_url="https://x", detail="low").detail == "low"
    with pytest.raises(ValidationError):
        InputImageContent(type="input_image", image_url="https://x", detail="ultra")  # pyright: ignore[reportArgumentType]


def test_input_file_content_all_optional() -> None:
    f = InputFileContent(type="input_file", file_data="data:application/pdf;base64,AAA", filename="a.pdf")
    assert f.type == "input_file"
    assert f.filename == "a.pdf"
    assert f.file_id is None and f.file_url is None and f.mime_type is None


def test_none_optional_fields_omitted_on_serialization() -> None:
    """Unset optional fields must be absent from the payload, not null — the Orq
    Responses API rejects explicit null on optional content fields."""
    img = InputImageContent(type="input_image", image_url="https://x/y.png")
    assert img.model_dump(mode="json") == {
        "type": "input_image",
        "image_url": "https://x/y.png",
        "detail": "auto",
    }
    f = InputFileContent(type="input_file", file_url="https://x/a.pdf")
    assert f.model_dump(mode="json") == {"type": "input_file", "file_url": "https://x/a.pdf"}


def test_message_accepts_mixed_text_and_image() -> None:
    content: ContentList = [
        InputTextContent(type="input_text", text="what is this?"),
        InputImageContent(type="input_image", image_url="https://x/y.png"),
    ]
    msg = Message(
        type="message",
        id="m1",
        status=MessageStatus.completed,
        role=MessageRole.user,
        content=content,
    )
    dumped = msg.model_dump(mode="json")
    parts = dumped["content"]
    assert parts[0] == {"type": "input_text", "text": "what is this?"}
    assert parts[1]["type"] == "input_image"
    assert parts[1]["image_url"] == "https://x/y.png"
    assert parts[1]["detail"] == "auto"


def test_message_accepts_file_part() -> None:
    content: ContentList = [InputFileContent(type="input_file", file_url="https://x/a.pdf")]
    msg = Message(
        type="message",
        id="m1",
        status=MessageStatus.completed,
        role=MessageRole.user,
        content=content,
    )
    assert msg.model_dump(mode="json")["content"][0]["file_url"] == "https://x/a.pdf"


def test_text_only_message_still_validates() -> None:
    """No regression for existing text-only callers."""
    content: ContentList = [InputTextContent(type="input_text", text="hi")]
    msg = Message(
        type="message",
        id="m1",
        status=MessageStatus.completed,
        role=MessageRole.user,
        content=content,
    )
    assert msg.model_dump(mode="json")["content"] == [{"type": "input_text", "text": "hi"}]


def test_message_round_trips_mixed_content() -> None:
    content: ContentList = [
        InputTextContent(type="input_text", text="caption"),
        InputImageContent(type="input_image", image_url="https://x", detail="high"),
        InputFileContent(type="input_file", file_id="file_123"),
    ]
    msg = Message(
        type="message",
        id="m1",
        status=MessageStatus.completed,
        role=MessageRole.user,
        content=content,
    )
    again = Message.model_validate(msg.model_dump(mode="json"))
    assert again == msg
    assert [type(p).__name__ for p in again.content] == [
        "InputTextContent",
        "InputImageContent",
        "InputFileContent",
    ]
