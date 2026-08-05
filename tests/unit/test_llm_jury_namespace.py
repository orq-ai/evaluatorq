from evaluatorq.llm_jury import _build_replacements
from evaluatorq.contracts import AgentResponse, AgentResponseError
from evaluatorq.common.template_engine import render_template
from evaluatorq.types import DataPoint


def test_pointwise_replacements_expose_redteam_namespace():
    data = DataPoint(inputs={'input': 'ping'}, expected_output='pong')
    reps = _build_replacements(data=data, output=AgentResponse(text='pong!'), criteria='crit')
    assert render_template('{{output.response}}', reps) == 'pong!'
    assert 'ping' in render_template('{{input.all_messages}}', reps)
    assert render_template('{{criteria}}', reps) == 'crit'
    # bare vars gone
    assert render_template('{{output}}', reps) == '{{output}}'


def test_pointwise_surfaces_output_error():
    # AgentResponseError.error_type is required (no default) in the real API, unlike
    # the brief's fixture which only set `message` — adapted to include error_type.
    ar = AgentResponse(output=[], error=AgentResponseError(message='rate limited', error_type='rate_limit'))
    reps = _build_replacements(data=DataPoint(inputs={}, expected_output=None), output=ar, criteria='c')
    assert render_template('{{output.error}}', reps) == 'rate limited'
