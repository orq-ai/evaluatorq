"""AgentResponseError + AgentResponse.error — RES-877."""

from __future__ import annotations

from evaluatorq.contracts import AgentResponse, AgentResponseError


def test_agent_response_error_defaults_to_none():
    resp = AgentResponse(text="ok")
    assert resp.error is None


def test_agent_response_carries_error_and_text():
    err = AgentResponseError(message="[ERROR: boom]", error_type="exception", code="target.crash")
    resp = AgentResponse(text="[ERROR: boom]", error=err)
    assert resp.error is err
    assert resp.error.error_type == "exception"
    assert resp.error.code == "target.crash"
    # .text still returns the human message so the report is unaffected.
    assert resp.text == "[ERROR: boom]"


def test_agent_response_error_is_frozen():
    import pytest
    err = AgentResponseError(message="m", error_type="timeout")
    with pytest.raises(Exception):
        err.message = "changed"  # type: ignore[misc]


def test_agent_response_error_code_optional():
    err = AgentResponseError(message="m", error_type="timeout")
    assert err.code is None
