"""ChatConversation: an agent run as evaluatorq chat messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from evaluatorq.contracts import Message  # noqa: TC001  (pydantic field type)

if TYPE_CHECKING:
    from evaluatorq.formats.responses import ResponsesConversation


class ChatConversation(BaseModel):
    """A run as `contracts.Message` turns. The lossy edge: no reasoning, no per-step metrics, no ids beyond tool calls."""

    model_config = ConfigDict(frozen=True)
    messages: list[Message]

    def to_responses(self) -> ResponsesConversation:
        """Render as Responses input items via `messages_to_responses_input`.

        Lost: tool results with no `tool_call_id` (warned), non-text assistant parts, `Message.name` on
        non-tool rows, and item ids that do not start with `fc_`.
        """
        from evaluatorq.formats import convert_chat_responses

        return convert_chat_responses.chat_to_responses(self)
