"""Config overrides follow parameter provenance across Typer's Click implementations."""

from __future__ import annotations

from typing import Annotated

import pytest
import typer
from typer.testing import CliRunner

from evaluatorq.common.cli_config import explicitly_set


@pytest.mark.parametrize(
    ('args', 'default_map', 'environment', 'expected'),
    [
        ([], None, None, False),
        ([], {'value': 'mapped'}, None, False),
        ([], None, 'from-env', True),
        (['--value', 'default'], None, None, True),
    ],
)
def test_explicit_flag_sources(
    args: list[str],
    default_map: dict[str, str] | None,
    environment: str | None,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('EQ_TEST_VALUE', raising=False)
    if environment is not None:
        monkeypatch.setenv('EQ_TEST_VALUE', environment)
    app = typer.Typer()
    observed: list[bool] = []

    @app.command()
    def command(
        ctx: typer.Context,
        value: Annotated[str, typer.Option(envvar='EQ_TEST_VALUE')] = 'default',
    ) -> None:
        observed.append(explicitly_set(ctx, 'value'))

    result = CliRunner().invoke(app, args, default_map=default_map)

    assert result.exit_code == 0, result.output
    assert observed == [expected]
