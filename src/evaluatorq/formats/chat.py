"""ChatConversation: an agent run as evaluatorq chat messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from evaluatorq.contracts import Message  # noqa: TC001  (pydantic field type)

if TYPE_CHECKING:
    from evaluatorq.formats.atif import AtifTrajectory
    from evaluatorq.formats.otel import OtelTrace
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

    def to_atif(self, *, agent_name: str = 'unknown', agent_version: str = 'unknown') -> AtifTrajectory:
        """Convert to an ATIF trajectory through Responses.

        Lost: everything lost by `ChatConversation.to_responses` and `ResponsesConversation.to_atif`
        (unlinked tool results, non-text assistant parts, `Message.name` on non-tool rows, non-`fc_` item ids,
        file parts as markers). Consecutive assistant turns with no tool result between them merge into one
        agent step.
        """
        return self.to_responses().to_atif(agent_name=agent_name, agent_version=agent_version)

    def to_otel(self, *, agent_name: str = 'unknown', agent_version: str = 'unknown') -> OtelTrace:
        """Convert to one OTel trace through Responses and ATIF.

        Lost: everything lost by `ChatConversation.to_atif` and `AtifTrajectory.to_otel`, including the `fc_`
        item ids (kept in ATIF step `extra`, which OTel drops). Chat has no timing, usage or model names, so the
        spans carry none.
        """
        return self.to_atif(agent_name=agent_name, agent_version=agent_version).to_otel()
