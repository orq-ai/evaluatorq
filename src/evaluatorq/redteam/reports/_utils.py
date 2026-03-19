"""Shared helpers for red team report modules."""

from __future__ import annotations

from evaluatorq.redteam.contracts import RedTeamResult


def extract_prompt(result: RedTeamResult) -> str:
    """Extract the first user message content as the attack prompt."""
    for msg in result.messages:
        if msg.role == "user" and msg.content:
            return msg.content
    return ""


def extract_response(result: RedTeamResult) -> str:
    """Extract the agent's response text."""
    if result.response:
        return result.response
    # Fall back to last assistant message
    for msg in reversed(result.messages):
        if msg.role == "assistant" and msg.content:
            return msg.content
    return ""
