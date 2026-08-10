from evaluatorq.contracts import AgentResponse
from evaluatorq.types import JobResult


def test_jobresult_serializes_agentresponse_output():
    jr = JobResult(job_name='j', output=AgentResponse(text='hello'))
    # Pin the union resolves to AgentResponse, not a dict-coerced degrade (the
    # dict member precedes AgentResponse under left_to_right).
    assert isinstance(jr.output, AgentResponse)
    dumped = jr.model_dump(mode='json')
    # Serialized output must be JSON-able (dict), not a dropped-through model / repr.
    assert isinstance(dumped['output'], dict)
    assert 'output' in dumped['output']  # AgentResponse.output list is present


def test_plain_dict_output_stays_dict():
    jr = JobResult(job_name='j', output={'k': 'v'})
    assert isinstance(jr.output, dict)  # left_to_right keeps dicts as dicts, no coercion
