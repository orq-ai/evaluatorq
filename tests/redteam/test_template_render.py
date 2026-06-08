"""Integration tests for OWASP evaluator prompt rendering via the new engine.

Tests exercise ``render_template(template, build_eval_replacements(...))`` —
the canonical path used by both dynamic and static judge paths. Pure engine
behaviour (whitelist, nested resolution, etc.) is already parity-tested in
``tests/common/test_template_engine.py`` and is NOT duplicated here.

Covers:
1. All three canonical placeholders fill correctly.
2. Injection safety: adversary-controlled values cannot expand other placeholders
   (single-pass non-rescan defense, NOT brace neutralization).
3. Empty tool_calls -> ``[]``.
4. A prompt with an unresolved placeholder is left intact (verbatim).
"""

from __future__ import annotations

import json

from evaluatorq.common.template_engine import render_template
from evaluatorq.contracts import TextOutputItem, ToolCallOutputItem
from evaluatorq.redteam.judge import build_eval_replacements


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(
    name: str = 'read_file',
    arguments: str | None = None,
    result: str | None = 'ok',
    tc_id: str = 'tc_001',
) -> ToolCallOutputItem:
    """Return a real ToolCallOutputItem (not a MagicMock) for use in tests."""
    return ToolCallOutputItem(
        id=tc_id,
        call_id=tc_id,
        name=name,
        arguments=arguments if arguments is not None else json.dumps({'path': '/etc/passwd'}),
        result=result,
    )


FULL_PROMPT = 'messages: {{input.all_messages}}\ntool_calls: {{output.tools_called}}\nresponse: {{output.response}}'


# ---------------------------------------------------------------------------
# 1. All three placeholders filled
# ---------------------------------------------------------------------------


class TestAllPlaceholdersFilled:
    def test_messages_placeholder_filled(self) -> None:
        input_messages = [{'role': 'user', 'content': 'attack me'}]
        rep = build_eval_replacements(
            input_messages=input_messages,
            output_messages=[TextOutputItem(text='safe reply', annotations=[])],
        )
        rendered = render_template(FULL_PROMPT, rep)
        assert '{{input.all_messages}}' not in rendered
        assert 'attack me' in rendered

    def test_response_placeholder_filled(self) -> None:
        rep = build_eval_replacements(
            input_messages=[],
            output_messages=[TextOutputItem(text='benign text', annotations=[])],
        )
        rendered = render_template(FULL_PROMPT, rep)
        assert '{{output.response}}' not in rendered
        assert 'benign text' in rendered

    def test_tool_calls_placeholder_filled_with_data(self) -> None:
        tc = _make_tool_call(name='execute_shell', arguments=json.dumps({'cmd': 'rm -rf /'}), result='0')
        rep = build_eval_replacements(
            input_messages=[],
            output_messages=[tc, TextOutputItem(text="I can't help with that.", annotations=[])],
        )
        rendered = render_template(FULL_PROMPT, rep)
        assert '{{output.tools_called}}' not in rendered
        assert 'execute_shell' in rendered
        assert 'rm -rf /' in rendered

    def test_tool_calls_dangerous_name_appears_in_output(self) -> None:
        """A dangerous tool call name DOES appear in the output."""
        tc = _make_tool_call(name='send_email', arguments=json.dumps({'to': 'attacker@evil.com'}))
        rep = build_eval_replacements(
            input_messages=[],
            output_messages=[tc, TextOutputItem(text='done', annotations=[])],
        )
        rendered = render_template(
            'tool_calls: {{output.tools_called}} response: {{output.response}}',
            rep,
        )
        parsed_section = rendered.split('response:')[0]
        assert 'send_email' in parsed_section

    def test_tool_call_structure_fields(self) -> None:
        """Each tool call in the rendered JSON contains name, arguments, result, id."""
        tc = _make_tool_call(
            name='write_file',
            arguments=json.dumps({'path': '/tmp/x'}),
            result='written',
            tc_id='call_xyz',
        )
        rep = build_eval_replacements(
            input_messages=[],
            output_messages=[tc],
        )
        rendered = render_template('{{output.tools_called}}', rep)
        parsed = json.loads(rendered)
        assert len(parsed) == 1
        record = parsed[0]
        assert record['name'] == 'write_file'
        assert record['arguments'] == {'path': '/tmp/x'}
        assert record['result'] == 'written'
        assert record['id'] == 'call_xyz'


# ---------------------------------------------------------------------------
# 2. Injection safety (single-pass non-rescan defense)
# ---------------------------------------------------------------------------


class TestInjectionSafety:
    def test_response_containing_input_all_messages_not_expanded(self) -> None:
        """A crafted response containing ``{{input.all_messages}}`` must NOT expand that placeholder."""
        rep = build_eval_replacements(
            input_messages=[{'role': 'user', 'content': 'secret'}],
            output_messages=[TextOutputItem(text='{{input.all_messages}}', annotations=[])],
        )
        rendered = render_template(
            'messages: {{input.all_messages}} response: {{output.response}}',
            rep,
        )
        # The messages placeholder was substituted with actual message JSON
        assert 'secret' in rendered
        # The response section should NOT expand to the messages content
        response_part = rendered.split('response:')[1]
        assert 'secret' not in response_part

    def test_tool_call_name_containing_placeholder_not_expanded(self) -> None:
        """A tool call with name ``{{output.response}}`` must not cause that placeholder to expand."""
        tc = _make_tool_call(name='{{output.response}}', arguments='{}')
        rep = build_eval_replacements(
            input_messages=[],
            output_messages=[tc, TextOutputItem(text='LEAKED', annotations=[])],
        )
        rendered = render_template(
            'tool_calls: {{output.tools_called}} response: {{output.response}}',
            rep,
        )
        # LEAKED should appear only once (in the response section)
        assert rendered.count('LEAKED') == 1

    def test_tool_call_result_containing_input_all_messages_not_expanded(self) -> None:
        """A tool call result containing ``{{input.all_messages}}`` must not expand."""
        tc = _make_tool_call(
            name='safe_tool',
            arguments='{}',
            result='{{input.all_messages}}',
        )
        rep = build_eval_replacements(
            input_messages=[{'role': 'user', 'content': 'private'}],
            output_messages=[tc],
        )
        rendered = render_template('{{input.all_messages}} | {{output.tools_called}}', rep)
        # "private" should appear once (from the messages substitution only)
        assert rendered.count('private') == 1

    def test_injected_placeholder_in_value_is_emitted_verbatim(self) -> None:
        rendered = render_template(
            '{{output.tools_called}}',
            {'output.tools_called': '[{"name": "{{output.response}}"}]', 'output.response': 'SECRET'},
        )
        assert '{{output.response}}' in rendered
        assert 'SECRET' not in rendered


# ---------------------------------------------------------------------------
# 3. tool_calls empty -> empty JSON array
# ---------------------------------------------------------------------------


class TestNoneOrEmptyToolCalls:
    def test_no_tool_call_items_renders_empty_array(self) -> None:
        """With no ToolCallOutputItem in output_messages, tools_called is []."""
        rep = build_eval_replacements(
            input_messages=[],
            output_messages=[TextOutputItem(text='', annotations=[])],
        )
        rendered = render_template('{{output.tools_called}}', rep)
        assert rendered.strip() == '[]'

    def test_empty_output_messages_renders_empty_array(self) -> None:
        rep = build_eval_replacements(
            input_messages=[],
            output_messages=[],
        )
        rendered = render_template('{{output.tools_called}}', rep)
        assert rendered.strip() == '[]'

    def test_missing_tool_calls_placeholder_unchanged(self) -> None:
        """When the prompt has no {{output.tools_called}}, tool call data is not spuriously injected."""
        tc = _make_tool_call()
        rep = build_eval_replacements(
            input_messages=[],
            output_messages=[tc, TextOutputItem(text='ok', annotations=[])],
        )
        prompt = 'messages: {{input.all_messages}} response: {{output.response}}'
        rendered = render_template(prompt, rep)
        assert '{{output.tools_called}}' not in rendered
        assert 'read_file' not in rendered  # tool call data not injected spuriously


# ---------------------------------------------------------------------------
# 4. Prompt without placeholder left unchanged for that slot
# ---------------------------------------------------------------------------


class TestPromptWithoutPlaceholder:
    def test_no_messages_placeholder_no_change(self) -> None:
        rep = build_eval_replacements(
            input_messages=[{'role': 'user', 'content': 'ignored'}],
            output_messages=[TextOutputItem(text='visible', annotations=[])],
        )
        prompt = 'only: {{output.response}}'
        rendered = render_template(prompt, rep)
        assert 'ignored' not in rendered
        assert 'visible' in rendered

    def test_no_response_placeholder_no_change(self) -> None:
        rep = build_eval_replacements(
            input_messages=[{'role': 'user', 'content': 'hello'}],
            output_messages=[TextOutputItem(text='should not appear', annotations=[])],
        )
        prompt = 'only: {{input.all_messages}}'
        rendered = render_template(prompt, rep)
        assert 'hello' in rendered
        assert 'should not appear' not in rendered
