import json
from typing import Any

from evaluatorq.common.judge import build_eval_replacements
from evaluatorq.contracts import ReasoningOutputItem, TextOutputItem, ToolCallOutputItem


def _fixture() -> dict[str, Any]:
    in_msgs = [{'role': 'user', 'content': 'hi'}]
    return dict(input_messages=in_msgs, output_messages=[], expected_output='ref', system_instructions=None)


def test_build_eval_replacements_unchanged_keys():
    r = build_eval_replacements(**_fixture())
    assert isinstance(r['input'], dict) and 'all_messages' in r['input']
    assert r['output']['response'] == ''
    assert 'input.all_messages' in r  # flat key present


def test_build_eval_replacements_prefix_namespaces_everything():
    r = build_eval_replacements(prefix='response_a', **_fixture())
    assert isinstance(r['response_a'], dict)
    assert r['response_a']['output']['response'] == ''
    assert 'response_a.input.all_messages' in r
    assert 'response_a.output.tools_called' in r
    # no un-prefixed leakage
    assert 'input' not in r and 'output' not in r


def test_flat_keys_tolerate_non_serializable():
    class Weird:
        pass

    r = build_eval_replacements(input_messages=[{'role': 'user', 'content': Weird()}], output_messages=[], expected_output=None, system_instructions=None)
    # must not raise; json.dumps with default=str renders the object via str()
    assert 'Weird object at' in json.loads(r['input.all_messages'])[0]['content']


def test_output_error_empty_by_default():
    r = build_eval_replacements(input_messages=[], output_messages=[], expected_output=None, system_instructions=None)
    assert r['output']['error'] == ''


def test_log_namespace_mirrors_input_and_output():
    r = build_eval_replacements(
        input_messages=[{'role': 'user', 'content': 'first'}, {'role': 'user', 'content': 'last'}],
        output_messages=[TextOutputItem(text='answer', annotations=[])],
        expected_output='ref',
        system_instructions=None,
    )
    assert r['log']['input'] == 'last'  # last input message only
    assert r['log']['output'] == 'answer'
    assert r['log']['reference'] == 'ref'
    assert r['log']['expected_output'] == 'ref'
    assert json.loads(r['log.messages'])[0]['content'] == 'first'


def test_reasoning_in_messages_but_not_response():
    r = build_eval_replacements(
        input_messages=[],
        output_messages=[
            ReasoningOutputItem(text='thinking hard'),
            TextOutputItem(text='the answer', annotations=[]),
        ],
        expected_output=None,
        system_instructions=None,
    )
    assert r['output']['response'] == 'the answer'
    assert 'thinking hard' in r['output.messages']


def test_tool_only_output_leaves_response_empty():
    r = build_eval_replacements(
        input_messages=[],
        output_messages=[ToolCallOutputItem(id='t1', name='search', arguments='{"q": "x"}', result='hit')],
        expected_output=None,
        system_instructions=None,
    )
    # A tool-only turn has no assistant text — `output.response` is genuinely empty and
    # a template relying on it alone would show the judge nothing.
    assert r['output']['response'] == ''
    assert r['output']['tools_called'][0]['name'] == 'search'
    assert 'search' in r['output.messages']
