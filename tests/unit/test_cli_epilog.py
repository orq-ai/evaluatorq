# ruff: noqa: S101

from __future__ import annotations

from typer.testing import CliRunner

from evaluatorq.cli import app
from evaluatorq.common.cli_epilog import examples


def test_examples_dims_comment_lines():
    out = examples('# a comment', 'eq redteam run -t agent:x')
    assert 'Examples' in out
    assert '[dim]' in out  # comment line is dimmed
    assert 'eq redteam run -t agent:x' in out


def test_root_help_links_to_documentation_site():
    result = CliRunner().invoke(app, ['--help'])

    assert result.exit_code == 0
    assert 'https://orq-ai.github.io/evaluatorq/' in result.output
