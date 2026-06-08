"""Parity tests for evaluatorq.common.template_engine (port of Orq's {{...}} engine)."""

from __future__ import annotations

from evaluatorq.common.template_engine import is_valid_template_path, render_template


class TestRenderBasics:
    def test_flat_exact_match_wins(self) -> None:
        out = render_template('{{a.b}}', {'a.b': 'FLAT', 'a': {'b': 'NESTED'}})
        assert out == 'FLAT'

    def test_nested_fallback(self) -> None:
        out = render_template('{{a.b}}', {'a': {'b': 'NESTED'}})
        assert out == 'NESTED'

    def test_unresolved_left_intact(self) -> None:
        assert render_template('{{missing.key}}', {}) == '{{missing.key}}'

    def test_jinja_whitespace_tolerated(self) -> None:
        assert render_template('{{ a }}', {'a': 'X'}) == 'X'

    def test_internal_whitespace_rejected(self) -> None:
        assert render_template('{{a b}}', {'a b': 'X'}) == '{{a b}}'

    def test_multiple_placeholders(self) -> None:
        assert render_template('{{a}} and {{b}}', {'a': 'X', 'b': 'Y'}) == 'X and Y'


class TestNestedTraversal:
    def test_bracket_index(self) -> None:
        assert render_template('{{a[0]}}', {'a': ['first', 'second']}) == 'first'

    def test_negative_index(self) -> None:
        assert render_template('{{a[-1]}}', {'a': ['x', 'y', 'z']}) == 'z'

    def test_out_of_range_index_intact(self) -> None:
        assert render_template('{{a[99]}}', {'a': ['x']}) == '{{a[99]}}'
        assert render_template('{{a[-99]}}', {'a': ['x']}) == '{{a[-99]}}'

    def test_dotted_numeric_is_string_key(self) -> None:
        assert render_template('{{data.0}}', {'data': {'0': 'zero'}}) == 'zero'

    def test_nested_after_bracket(self) -> None:
        data = {'a': {'b': [{'c': 'DEEP'}]}}
        assert render_template('{{a.b[0].c}}', data) == 'DEEP'


class TestFormatting:
    def test_dict_is_json(self) -> None:
        assert render_template('{{a}}', {'a': {'k': 1}}) == '{\n  "k": 1\n}'

    def test_list_is_json(self) -> None:
        assert render_template('{{a}}', {'a': [1, 2]}) == '[\n  1,\n  2\n]'

    def test_str_passthrough(self) -> None:
        assert render_template('{{a}}', {'a': 'raw'}) == 'raw'

    def test_falsy_values_render_via_str(self) -> None:
        assert render_template('{{a}}', {'a': False}) == 'False'
        assert render_template('{{a}}', {'a': None}) == 'None'
        assert render_template('{{a}}', {'a': 0}) == '0'
        assert render_template('{{a}}', {'a': ''}) == ''
        assert render_template('{{a}}', {'a': {}}) == '{}'
        assert render_template('{{a}}', {'a': []}) == '[]'

    def test_backslash_in_value_survives_verbatim(self) -> None:
        assert render_template('{{a}}', {'a': r'\g<0> and \1'}) == r'\g<0> and \1'


class TestSecurityWhitelist:
    def test_function_call_rejected(self) -> None:
        assert render_template('{{eval(x)}}', {'eval(x)': 'X'}) == '{{eval(x)}}'

    def test_semicolon_rejected(self) -> None:
        assert render_template('{{a;b}}', {'a;b': 'X'}) == '{{a;b}}'

    def test_injected_placeholder_in_value_not_re_expanded(self) -> None:
        out = render_template('{{tool}}', {'tool': '{{output.response}}', 'output.response': 'SECRET'})
        assert out == '{{output.response}}'

    def test_is_valid_template_path(self) -> None:
        assert is_valid_template_path('a.b[0].c')
        assert is_valid_template_path('messages[-1]')
        assert not is_valid_template_path('eval(x)')
        assert not is_valid_template_path('a;b')
        assert not is_valid_template_path('a b')

    def test_is_valid_template_path_edges(self) -> None:
        assert not is_valid_template_path('')
        assert is_valid_template_path('_private')
        assert not is_valid_template_path('0')
