from evaluatorq.llm_jury import _build_replacements
from evaluatorq.contracts import AgentResponse
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
