"""Tests for extract_json utility."""

from evaluatorq.simulation.utils.extract_json import extract_json_from_response


def test_extract_from_code_block():
    content = '```json\n[{"name": "test"}]\n```'
    result = extract_json_from_response(content)
    assert result == '[{"name": "test"}]'


def test_extract_from_code_block_no_lang():
    content = '```\n[{"name": "test"}]\n```'
    result = extract_json_from_response(content)
    assert result == '[{"name": "test"}]'


def test_extract_plain_array():
    content = 'Here is the result: [{"name": "test"}]'
    result = extract_json_from_response(content)
    assert result == '[{"name": "test"}]'


def test_extract_plain_object():
    content = 'The result: {"name": "test"}'
    result = extract_json_from_response(content)
    assert result == '{"name": "test"}'


def test_extract_empty_string():
    assert extract_json_from_response("") == ""


def test_extract_no_json():
    result = extract_json_from_response("just some text")
    assert result == "just some text"


def test_extract_nested_brackets():
    content = '[{"criteria": [{"type": "must_happen"}]}]'
    result = extract_json_from_response(content)
    assert '"must_happen"' in result


def test_extract_with_surrounding_text():
    content = (
        'Here are the personas:\n```json\n[{"name": "Alice"}]\n```\nHope that helps!'
    )
    result = extract_json_from_response(content)
    assert result == '[{"name": "Alice"}]'


def test_prefers_code_block_over_bare():
    content = '{"bare": true}\n```json\n{"block": true}\n```'
    result = extract_json_from_response(content)
    assert result == '{"block": true}'


def test_handles_escaped_quotes():
    content = '{"key": "value with \\"quotes\\""}'
    result = extract_json_from_response(content)
    assert '"key"' in result
    assert "quotes" in result


def test_returns_trimmed_fallback():
    result = extract_json_from_response("  no json here  ")
    assert result == "no json here"


def test_extracts_first_valid_json():
    content = 'start {"a": 1} middle {"b": 2} end'
    result = extract_json_from_response(content)
    assert result == '{"a": 1}'
