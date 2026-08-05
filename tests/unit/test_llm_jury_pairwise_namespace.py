from evaluatorq.common.template_engine import render_template
from evaluatorq.contracts import AgentResponse
from evaluatorq.llm_jury import _side_to_namespace
from evaluatorq.types import DataPoint


def test_side_namespace_output_and_input():
    side = {
        'data': DataPoint(inputs={'input': 'prompt-a'}, expected_output=None),
        'output': AgentResponse(text='ans-a'),
    }
    reps = _side_to_namespace('response_a', side)
    assert render_template('{{response_a.output.response}}', reps) == 'ans-a'
    assert 'prompt-a' in render_template('{{response_a.input.all_messages}}', reps)


def test_side_namespace_output_only():
    reps = _side_to_namespace('response_b', AgentResponse(text='ans-b'))
    assert render_template('{{response_b.output.response}}', reps) == 'ans-b'
    assert render_template('{{response_b.input.all_messages}}', reps) == '[]'


def test_side_namespace_bare_string_output():
    reps = _side_to_namespace('response_b', 'plain answer')
    assert render_template('{{response_b.output.response}}', reps) == 'plain answer'
