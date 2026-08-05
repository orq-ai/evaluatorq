from evaluatorq.contracts import AgentResponse
from evaluatorq.redteam.reports.converters import _coerce_job_output_text


def test_coerce_text_from_agentresponse():
    assert _coerce_job_output_text(AgentResponse(text='the reply')) == 'the reply'
