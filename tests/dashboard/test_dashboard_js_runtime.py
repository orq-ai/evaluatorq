"""Exercise dashboard event behavior with Node's built-in JavaScript runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

import tinycss2


def test_dashboard_events() -> None:
    script = Path(__file__).with_name('dashboard_js_runtime.cjs')
    result = subprocess.run(['node', '--test', str(script)], capture_output=True, text=True, check=False, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_drawer_closing_class_has_exit_animation() -> None:
    from evaluatorq.dashboard.styles import DASHBOARD_CSS

    rules = tinycss2.parse_stylesheet(DASHBOARD_CSS, skip_whitespace=True, skip_comments=True)
    declarations = [
        declaration
        for rule in rules
        if rule.type == 'qualified-rule'
        and '.sim-report .sim-entity-dialog--closing' in tinycss2.serialize(rule.prelude)
        for declaration in tinycss2.parse_declaration_list(rule.content, skip_whitespace=True, skip_comments=True)
        if declaration.type == 'declaration' and declaration.name == 'animation'
    ]
    assert any('sim-drawer-out' in tinycss2.serialize(declaration.value) for declaration in declarations)
