"""Tests for `redteam_transcripts.render_attack_fragment` (Task 13)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.dashboard.redteam_transcripts import render_attack_fragment

if TYPE_CHECKING:
    from evaluatorq.redteam.contracts import RedTeamResult


def test_fragment_vulnerable_verdict(rt_result_vuln: RedTeamResult) -> None:
    html = render_attack_fragment(rt_result_vuln)
    assert 'Evaluator verdict' in html
    assert 'orange-500' in html or 'rt-verdict-vuln' in html


def test_fragment_resistant_verdict(rt_result_safe: RedTeamResult) -> None:
    html = render_attack_fragment(rt_result_safe)
    assert 'green' in html.lower()


def test_fragment_error_row(rt_result_error: RedTeamResult) -> None:
    html = render_attack_fragment(rt_result_error)
    assert 'error' in html.lower()


def test_fragment_bubble_transcript(rt_result_vuln: RedTeamResult) -> None:
    html = render_attack_fragment(rt_result_vuln)
    assert 'rt-msg-avatar' in html  # bubble avatar span
