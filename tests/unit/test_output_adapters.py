from evaluatorq.common.output_adapters import output_to_text
from evaluatorq.contracts import AgentResponse


def test_output_to_text_agentresponse_returns_text():
    assert output_to_text(AgentResponse(text='the answer')) == 'the answer'


def test_output_to_text_str_passthrough():
    assert output_to_text('plain') == 'plain'


def test_output_to_text_none_is_empty():
    assert output_to_text(None) == ''


def test_output_to_text_dict_non_response_json():
    assert output_to_text({'k': 'v'}) == "{'k': 'v'}" or output_to_text({'k': 'v'}).strip().startswith('{')


def test_output_to_text_never_raises_on_weird():
    class Weird:
        def __str__(self):
            raise RuntimeError('boom')
    # Weird isn't a valid Output, but the helper must be defensive.
    assert isinstance(output_to_text(Weird()), str)  # degrades, no raise
