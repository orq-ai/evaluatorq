# ruff: noqa: S101

from evaluatorq.common.cli_epilog import examples


def test_examples_dims_comment_lines():
    out = examples('# a comment', 'eq redteam run -t agent:x')
    assert 'Examples' in out
    assert '[dim]' in out  # comment line is dimmed
    assert 'eq redteam run -t agent:x' in out
