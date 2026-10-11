"""Exercise dashboard event behavior with Node's built-in JavaScript runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

import tinycss2


def test_dashboard_events() -> None:
    script = Path(__file__).with_name('dashboard_js_runtime.cjs')
    result = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False, timeout=60)
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


def test_insights_run_form_controller() -> None:
    import json
    from dataclasses import replace
    from datetime import datetime, timezone

    from evaluatorq.dashboard.insights_run_form import RunFormValues, render_run_form, render_run_page
    from evaluatorq.dashboard.insights_sessions_views import encode_ref, render_session_results
    from evaluatorq.local_sessions import SessionSearchResult, SessionSummary

    def summary(source: str, path: str) -> SessionSummary:
        when = datetime(2026, 10, 2, 10, tzinfo=timezone.utc)
        return SessionSummary.model_validate({
            'source': source,
            'path': path,
            'session_id': path,
            'title': 'A session',
            'started_at': when,
            'updated_at': when,
            'size_bytes': 2048,
        })

    sessions = SessionSearchResult(
        sessions=(
            summary('claude-code', '/home/me/.claude/projects/p/a.jsonl'),
            summary('codex', '/home/me/.codex/sessions/2026/10/02/b.jsonl'),
        ),
        scanned_files=2,
        candidate_files=2,
        complete=True,
    )
    values = RunFormValues.defaults()
    fixtures = {
        'form': render_run_form(values, csrf='token'),
        'rejected': render_run_page(
            replace(values, source='snapshot'), csrf='token', error='Browse to choose a trace file first.'
        ),
        'sessions': render_session_results(sessions, selected=frozenset(), roots_found=True),
        # Refs carry the platform's path form (backslashes on Windows), so the harness compares against these.
        'session_refs': [encode_ref(s) for s in sessions.sessions],
    }
    script = Path(__file__).with_name('insights_run_form.cjs')
    # Fixtures go in on stdin; insights_run_form.cjs explains why.
    result = subprocess.run(
        ['node', str(script)],
        input=json.dumps(fixtures),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,

    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_insights_review_signal_details() -> None:
    script = Path(__file__).with_name('insights_review_signals.cjs')
    result = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
