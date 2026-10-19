# ruff: noqa: S101, SLF001

import subprocess
import sys
import textwrap

import typer
from typer.testing import CliRunner

from evaluatorq import cli as cli_module


def test_subapps_are_registered():
    app = typer.Typer()
    cli_module._register_subapps(app)
    result = CliRunner().invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'redteam' in result.output
    assert 'sim' in result.output


def test_subapps_register_without_optional_insights_numerical_dependencies():
    script = textwrap.dedent(
        '''
        import importlib.abc
        import sys
        import typer

        class BlockOptionalNumerics(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'numpy', 'scipy', 'umap'}:
                    raise ModuleNotFoundError('blocked optional dependency: ' + fullname, name=fullname)

        sys.meta_path.insert(0, BlockOptionalNumerics())

        from evaluatorq import cli
        import evaluatorq.insights.cli

        assert 'evaluatorq.insights.pipeline' not in sys.modules

        app = typer.Typer()
        cli._register_subapps(app)
        from typer.testing import CliRunner

        result = CliRunner().invoke(app, ['--help'])
        assert result.exit_code == 0, result.exception
        assert 'insights' in result.output
        assert not any(name.split('.')[0] in {'numpy', 'scipy', 'umap'} for name in sys.modules)
        '''
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
