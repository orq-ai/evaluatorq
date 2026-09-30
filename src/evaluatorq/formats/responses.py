"""ResponsesConversation: an agent run as OpenAI Responses items."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from openai.types.responses import Response  # noqa: TC002  (pydantic field type)
from pydantic import BaseModel, ConfigDict, field_validator

from evaluatorq.openresponses.otel_messages import RESPONSES_ITEM_TYPES

if TYPE_CHECKING:
    from evaluatorq.formats.atif import AtifTrajectory
    from evaluatorq.formats.chat import ChatConversation

_KNOWN_ITEM_TYPES = RESPONSES_ITEM_TYPES | {'message'}


class ResponsesConversation(BaseModel):
    """A run as Responses items. `items` are authoritative; `responses` only enrich (model, usage, timing)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    items: list[dict[str, Any]]
    responses: list[Response] | None = None

    @field_validator('items')
    @classmethod
    def _warn_on_unrecognised_items(cls, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for item in items:
            item_type = item.get('type')
            if item_type not in _KNOWN_ITEM_TYPES and 'role' not in item:
                logger.warning('Unrecognised Responses item type {!r}; keeping it.', item_type)
        return items

    def to_chat(self) -> ChatConversation:
        """Render as chat messages.

        Lost: reasoning items (counted and warned once), media that is not text, image or file parts,
        `status`, annotations, and item ids that do not start with `fc_`.
        """
        from evaluatorq.formats import convert_chat_responses

        return convert_chat_responses.responses_to_chat(self)

    def to_atif(self, *, agent_name: str = 'unknown', agent_version: str = 'unknown') -> AtifTrajectory:
        """Convert to an ATIF trajectory: user/system messages become steps, assistant output becomes agent steps.

        A tool result closes its agent step; `responses` enrich agent steps only when there is exactly one per
        agent step (otherwise they are ignored with a warning). Kept in free-form slots: reasoning tokens and
        total tokens in `metrics.extra`; `status`, `error` and `incomplete_details` (non-default only), the
        response id, `fc_` item ids and encrypted-only reasoning in step `extra`; the developer role in
        `extra.original_role` of a system step. Lost: encrypted reasoning content, non-text assistant parts,
        file parts (rendered as `[file: name]` markers), and item types ATIF has no step for (warned).
        """
        from evaluatorq.formats import convert_responses_atif

        return convert_responses_atif.responses_to_atif(self, agent_name=agent_name, agent_version=agent_version)
