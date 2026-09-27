"""Tests for the AgentTarget conversation-history capability contract."""

from __future__ import annotations

import pytest

from evaluatorq.contracts import AgentTarget, ConversationHistoryMode, Message
from evaluatorq.openresponses.target import OrqResponsesTarget
from evaluatorq.redteam.backends.orq import ORQAgentTarget

# Imported per test: a module-level skip would hide the core assertions when one optional extra is missing.


def test_core_target_history_modes_are_explicit() -> None:
    assert OrqResponsesTarget.history_mode is ConversationHistoryMode.CALLER
    assert ORQAgentTarget.history_mode is ConversationHistoryMode.TARGET


def test_langgraph_target_owns_its_history() -> None:
    pytest.importorskip('langgraph')
    from evaluatorq.integrations.langgraph_integration.target import LangGraphTarget

    assert LangGraphTarget.history_mode is ConversationHistoryMode.TARGET


def test_pydantic_ai_target_owns_its_history() -> None:
    pytest.importorskip('pydantic_ai')
    from evaluatorq.integrations.pydantic_ai_integration.target import PydanticAITarget

    assert PydanticAITarget.history_mode is ConversationHistoryMode.TARGET


def test_crewai_target_leaves_history_to_the_caller() -> None:
    pytest.importorskip('crewai')
    from evaluatorq.integrations.crewai_integration.target import CrewAITarget

    assert CrewAITarget.history_mode is ConversationHistoryMode.CALLER


def test_agent_target_defaults_to_caller_owned_history() -> None:
    class _BareTarget(AgentTarget):
        async def respond(self, messages: list[Message]):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def new(self) -> _BareTarget:
            return _BareTarget()

    assert _BareTarget.history_mode is ConversationHistoryMode.CALLER
