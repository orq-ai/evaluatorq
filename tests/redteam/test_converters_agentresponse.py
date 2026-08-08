from evaluatorq.contracts import AgentResponse, AgentResponseError, TokenUsage
from evaluatorq.redteam.reports.converters import _coerce_job_output_payload, _coerce_job_output_text


def test_coerce_text_from_agentresponse():
    assert _coerce_job_output_text(AgentResponse(text='the reply')) == 'the reply'


def test_coerce_payload_from_agentresponse():
    # AgentResponse.output is a list[OutputMessage], which does not validate against
    # JobOutputPayload.output: str | None — without an explicit branch the
    # ValidationError is swallowed and the report renders an empty payload.
    payload = _coerce_job_output_payload(AgentResponse(text='the reply'))
    assert payload.final_response == 'the reply'
    assert payload.response == 'the reply'
    assert [m.content for m in payload.conversation] == ['the reply']


def test_coerce_payload_from_agentresponse_carries_error_and_usage():
    ar = AgentResponse(
        output=[],
        error=AgentResponseError(message='timeout', error_type='timeout'),
        usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    payload = _coerce_job_output_payload(ar)
    assert payload.error == 'timeout'
    assert payload.token_usage is not None
    assert payload.token_usage.total_tokens == 3
    assert payload.conversation == []
