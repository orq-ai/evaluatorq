# ruff: noqa: S101

from __future__ import annotations

import json

from typer.testing import CliRunner

from evaluatorq.redteam.adaptive.strategy_registry import known_strategy_names
from evaluatorq.redteam.cli import app as redteam_app


def test_strategies_json_lists_every_name_the_strategy_flag_accepts():
    result = CliRunner().invoke(redteam_app, ['strategies', '--json'])

    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    assert sorted(row['name'] for row in rows) == sorted(known_strategy_names())
    assert {row['vulnerability'] for row in rows if row['name'] == 'tool_output_hijack'} == {'goal_hijacking'}


def test_strategies_table_keeps_every_name_whole_on_a_narrow_terminal():
    result = CliRunner().invoke(redteam_app, ['strategies'], env={'COLUMNS': '80'})

    assert result.exit_code == 0
    assert all(name in result.stdout for name in known_strategy_names())
