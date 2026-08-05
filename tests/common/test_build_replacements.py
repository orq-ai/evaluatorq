import json

from evaluatorq.common.judge import build_eval_replacements, build_side_replacements


def _fixture():
    in_msgs = [{'role': 'user', 'content': 'hi'}]
    return dict(input_messages=in_msgs, output_messages=[], expected_output='ref', system_instructions=None)


def test_build_eval_replacements_unchanged_keys():
    r = build_eval_replacements(**_fixture())
    assert isinstance(r['input'], dict) and 'all_messages' in r['input']
    assert r['output']['response'] == ''
    assert 'input.all_messages' in r  # flat key present


def test_build_side_replacements_prefixes_everything():
    r = build_side_replacements('response_a', **_fixture())
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
    # must not raise; json.dumps with default=str
    json.loads(r['input.all_messages']) if False else None  # smoke: key exists and is a str
    assert isinstance(r['input.all_messages'], str)


def test_output_error_empty_by_default():
    r = build_eval_replacements(input_messages=[], output_messages=[], expected_output=None, system_instructions=None)
    assert r['output']['error'] == ''
