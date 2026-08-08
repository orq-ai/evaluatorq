from evaluatorq.common.template_engine import render_template
from evaluatorq.contracts import AgentResponse, AgentResponseError, ToolCallOutputItem
from evaluatorq.llm_jury import _side_to_namespace


def test_side_namespace_agent_response():
    reps = _side_to_namespace('response_b', AgentResponse(text='ans-b'))
    assert render_template('{{response_b.output.response}}', reps) == 'ans-b'


def test_side_namespace_has_empty_input():
    # A pairwise side carries an answer only — there is no request side to expose.
    reps = _side_to_namespace('response_a', AgentResponse(text='x'))
    assert render_template('{{response_a.input.all_messages}}', reps) == '[]'
    assert render_template('{{response_a.input.expected_output}}', reps) == ''


def test_side_namespace_bare_string_output():
    reps = _side_to_namespace('response_b', 'plain answer')
    assert render_template('{{response_b.output.response}}', reps) == 'plain answer'


def test_side_namespace_dict_output_is_not_special_cased():
    # A dict-shaped Output carrying 'data'/'output' keys is just an Output — it used to
    # be misread as a per-side bundle.
    reps = _side_to_namespace('response_a', {'data': None, 'output': 'inner'})
    assert 'inner' in render_template('{{response_a.output.response}}', reps)


def test_side_namespace_exposes_tool_calls():
    ar = AgentResponse(output=[ToolCallOutputItem(id='t1', name='search', arguments='{"q": "x"}', result='hit')])
    reps = _side_to_namespace('response_a', ar)
    assert 'search' in render_template('{{response_a.output.tools_called}}', reps)


def test_pairwise_side_surfaces_output_error():
    ar = AgentResponse(output=[], error=AgentResponseError(message='timeout', error_type='timeout'))
    reps = _side_to_namespace('response_a', ar)
    assert render_template('{{response_a.output.error}}', reps) == 'timeout'
