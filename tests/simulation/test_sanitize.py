"""Tests for sanitize utility."""

from evaluatorq.simulation.utils.sanitize import delimit


def test_delimit_basic():
    assert delimit("hello") == "<data>hello</data>"


def test_delimit_escapes_ampersand():
    result = delimit("a & b")
    assert "&amp;" in result
    assert "<data>a &amp; b</data>" == result


def test_delimit_escapes_data_tags():
    result = delimit("test <data>injection</data> here")
    assert "<data>" not in result.replace("<data>", "", 1).replace("</data>", "", 1)
    assert "&lt;data&gt;" in result
    assert "&lt;/data&gt;" in result


def test_delimit_case_insensitive():
    result = delimit("<DATA>test</DATA>")
    assert "&lt;data&gt;" in result
    assert "&lt;/data&gt;" in result


def test_delimit_empty_string():
    assert delimit("") == "<data></data>"


def test_preserves_non_data_tags():
    result = delimit("<b>bold</b>")
    assert "<b>bold</b>" in result
