"""Unit tests for the shared OWASP evaluator prompt renderer.

Covers:
1. All three placeholders are filled correctly.
2. Injection safety: adversary-controlled values cannot expand other placeholders.
3. tool_calls=None / [] emits empty JSON array (``[]``).
4. A prompt missing a placeholder is returned unchanged for that placeholder.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from evaluatorq.redteam.frameworks.owasp.prompt_render import (
    _sanitize_placeholders,
    render_owasp_evaluator_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(
    name: str = "read_file",
    arguments: dict[str, Any] | None = None,
    result: str | None = "ok",
    tc_id: str = "tc_001",
) -> MagicMock:
    """Return a mock ToolCallOutputItem-like object."""
    tc = MagicMock()
    tc.name = name
    tc.arguments_dict = arguments or {"path": "/etc/passwd"}
    tc.result = result
    tc.id = tc_id
    return tc


FULL_PROMPT = (
    "messages: {{input.all_messages}}\n"
    "tool_calls: {{output.tool_calls}}\n"
    "response: {{output.response}}"
)


# ---------------------------------------------------------------------------
# 1. All three placeholders filled
# ---------------------------------------------------------------------------


class TestAllPlaceholdersFilled:
    def test_messages_placeholder_filled(self) -> None:
        messages = [{"role": "user", "content": "attack me"}]
        rendered = render_owasp_evaluator_prompt(
            FULL_PROMPT,
            messages=messages,
            response="safe reply",
            tool_calls=None,
        )
        assert "{{input.all_messages}}" not in rendered
        assert "attack me" in rendered

    def test_list_content_surfaces_text_not_python_repr(self) -> None:
        # Multi-part content (e.g. tool/result messages) must surface the text,
        # not an opaque Python repr of the list.
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "secret payload"}]},
        ]
        rendered = render_owasp_evaluator_prompt(
            FULL_PROMPT,
            messages=messages,
            response="safe reply",
            tool_calls=None,
        )
        assert "secret payload" in rendered
        assert "'type': 'text'" not in rendered  # no Python repr leaked

    def test_response_placeholder_filled(self) -> None:
        rendered = render_owasp_evaluator_prompt(
            FULL_PROMPT,
            messages=[],
            response="benign text",
            tool_calls=None,
        )
        assert "{{output.response}}" not in rendered
        assert "benign text" in rendered

    def test_tool_calls_placeholder_filled_with_data(self) -> None:
        tc = _make_tool_call(name="execute_shell", arguments={"cmd": "rm -rf /"}, result="0")
        rendered = render_owasp_evaluator_prompt(
            FULL_PROMPT,
            messages=[],
            response="I can't help with that.",
            tool_calls=[tc],
        )
        assert "{{output.tool_calls}}" not in rendered
        assert "execute_shell" in rendered
        assert "rm -rf /" in rendered

    def test_tool_calls_dangerous_name_appears_in_output(self) -> None:
        """A dangerous tool call name DOES appear in the output — just neutralized."""
        tc = _make_tool_call(name="send_email", arguments={"to": "attacker@evil.com"})
        rendered = render_owasp_evaluator_prompt(
            "tool_calls: {{output.tool_calls}} response: {{output.response}}",
            messages=[],
            response="done",
            tool_calls=[tc],
        )
        parsed_section = rendered.split("response:")[0]
        assert "send_email" in parsed_section

    def test_tool_call_structure_fields(self) -> None:
        """Each tool call in the rendered JSON contains name, arguments, result, id."""
        tc = _make_tool_call(
            name="write_file",
            arguments={"path": "/tmp/x"},
            result="written",
            tc_id="call_xyz",
        )
        rendered = render_owasp_evaluator_prompt(
            "{{output.tool_calls}}",
            messages=[],
            response="",
            tool_calls=[tc],
        )
        # The tool_calls section spans the full rendered string for this prompt
        parsed = json.loads(rendered)
        assert len(parsed) == 1
        record = parsed[0]
        assert record["name"] == "write_file"
        assert record["arguments"] == {"path": "/tmp/x"}
        assert record["result"] == "written"
        assert record["id"] == "call_xyz"


# ---------------------------------------------------------------------------
# 2. Injection safety
# ---------------------------------------------------------------------------


class TestInjectionSafety:
    def test_response_containing_input_all_messages_not_expanded(self) -> None:
        """A crafted response containing ``{{input.all_messages}}`` must NOT expand that placeholder."""
        dangerous_response = "{{input.all_messages}}"
        rendered = render_owasp_evaluator_prompt(
            "messages: {{input.all_messages}} response: {{output.response}}",
            messages=[{"role": "user", "content": "secret"}],
            response=dangerous_response,
            tool_calls=None,
        )
        # The messages placeholder was substituted with actual message JSON
        assert "secret" in rendered
        # The response section should show the neutralized form, NOT the expanded messages
        response_part = rendered.split("response:")[1]
        assert "secret" not in response_part

    def test_tool_call_name_containing_placeholder_not_expanded(self) -> None:
        """A tool call with name ``{{output.response}}`` must not cause that placeholder to expand."""
        prompt = "tool_calls: {{output.tool_calls}} response: {{output.response}}"
        tc = _make_tool_call(name="{{output.response}}", arguments={})
        rendered = render_owasp_evaluator_prompt(
            prompt,
            messages=[],
            response="LEAKED",
            tool_calls=[tc],
        )
        # LEAKED should appear only once (in the response section)
        assert rendered.count("LEAKED") == 1

    def test_tool_call_result_containing_input_all_messages_not_expanded(self) -> None:
        """A tool call result containing ``{{input.all_messages}}`` must be neutralized."""
        tc = _make_tool_call(
            name="safe_tool",
            arguments={},
            result="{{input.all_messages}}",
        )
        rendered = render_owasp_evaluator_prompt(
            "{{input.all_messages}} | {{output.tool_calls}}",
            messages=[{"role": "user", "content": "private"}],
            response="",
            tool_calls=[tc],
        )
        # "private" should appear once (from the messages substitution)
        assert rendered.count("private") == 1

    def test_sanitize_placeholders_breaks_double_brace(self) -> None:
        assert _sanitize_placeholders("{{output.response}}") == "{ {output.response}}"
        assert _sanitize_placeholders("no braces here") == "no braces here"

    def test_messages_json_double_braces_sanitized(self) -> None:
        """Message content that contains {{ is neutralized when embedded in the prompt."""
        messages = [{"role": "user", "content": "{{output.response}}"}]
        rendered = render_owasp_evaluator_prompt(
            "messages: {{input.all_messages}} response: {{output.response}}",
            messages=messages,
            response="REAL_RESPONSE",
            tool_calls=None,
        )
        # The message content {{ must be broken so it cannot expand
        assert "{ {output.response}}" in rendered
        # REAL_RESPONSE appears once in the response section
        assert rendered.count("REAL_RESPONSE") == 1


# ---------------------------------------------------------------------------
# 3. tool_calls=None or [] → empty JSON array
# ---------------------------------------------------------------------------


class TestNoneOrEmptyToolCalls:
    def test_none_tool_calls_renders_empty_array(self) -> None:
        rendered = render_owasp_evaluator_prompt(
            "{{output.tool_calls}}",
            messages=[],
            response="",
            tool_calls=None,
        )
        assert rendered.strip() == "[]"

    def test_empty_list_tool_calls_renders_empty_array(self) -> None:
        rendered = render_owasp_evaluator_prompt(
            "{{output.tool_calls}}",
            messages=[],
            response="",
            tool_calls=[],
        )
        assert rendered.strip() == "[]"

    def test_missing_tool_calls_placeholder_unchanged(self) -> None:
        """When the prompt has no {{output.tool_calls}}, it is returned unchanged for that slot."""
        prompt = "messages: {{input.all_messages}} response: {{output.response}}"
        rendered = render_owasp_evaluator_prompt(
            prompt,
            messages=[],
            response="ok",
            tool_calls=[_make_tool_call()],  # has tool calls but no placeholder
        )
        assert "{{output.tool_calls}}" not in rendered
        assert "read_file" not in rendered  # tool call data not injected spuriously


# ---------------------------------------------------------------------------
# 4. Prompt without placeholder left unchanged for that slot
# ---------------------------------------------------------------------------


class TestPromptWithoutPlaceholder:
    def test_no_messages_placeholder_no_change(self) -> None:
        prompt = "only: {{output.response}}"
        rendered = render_owasp_evaluator_prompt(
            prompt,
            messages=[{"role": "user", "content": "ignored"}],
            response="visible",
            tool_calls=None,
        )
        assert "ignored" not in rendered
        assert "visible" in rendered

    def test_no_response_placeholder_no_change(self) -> None:
        prompt = "only: {{input.all_messages}}"
        rendered = render_owasp_evaluator_prompt(
            prompt,
            messages=[{"role": "user", "content": "hello"}],
            response="should not appear",
            tool_calls=None,
        )
        assert "hello" in rendered
        assert "should not appear" not in rendered

    def test_empty_prompt_stays_empty(self) -> None:
        rendered = render_owasp_evaluator_prompt(
            "",
            messages=[{"role": "user", "content": "x"}],
            response="y",
            tool_calls=None,
        )
        assert rendered == ""
