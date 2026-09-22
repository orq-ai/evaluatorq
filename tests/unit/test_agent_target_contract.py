"""Tests for the AgentTarget conversation-history capability contract."""

from __future__ import annotations

import pytest

from evaluatorq.contracts import AgentTarget, ConversationHistoryMode, Message
from evaluatorq.openresponses.target import OrqResponsesTarget
from evaluatorq.redteam.backends.orq import ORQAgentTarget

pytest.importorskip('langgraph')
from evaluatorq.integrations.langgraph_integration.target import LangGraphTarget  # noqa: E402

from evaluatorq.integrations.pydantic_ai_integration.target import PydanticAITarget  # noqa: E402

from evaluatorq.integrations.crewai_integration.target import CrewAITarget  # noqa: E402


def test_target_history_modes_are_explicit() -> None:
    assert OrqResponsesTarget.history_mode is ConversationHistoryMode.CALLER
    assert ORQAgentTarget.history_mode is ConversationHistoryMode.TARGET
    assert LangGraphTarget.history_mode is ConversationHistoryMode.TARGET
    assert PydanticAITarget.history_mode is ConversationHistoryMode.TARGET
    assert CrewAITarget.history_mode is ConversationHistoryMode.CALLER


def test_agent_target_defaults_to_caller_owned_history() -> None:
    class _BareTarget(AgentTarget):
        async def respond(self, messages: list[Message]):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def new(self) -> _BareTarget:
            return _BareTarget()

    assert _BareTarget.history_mode is ConversationHistoryMode.CALLER
