"""Tests for `redteam_transcripts.render_attack_fragment` (Task 13) and the
`/r/{rid}/redteam/attack` fragment endpoint (Task 14)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.dashboard.redteam_transcripts import _jury_panel_health, render_attack_fragment

if TYPE_CHECKING:
    from starlette.testclient import TestClient

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


def _with_jury(result: RedTeamResult, jury: object) -> RedTeamResult:
    """Attach a JuryResult to a fixture result (both models are frozen)."""
    return result.model_copy(update={'evaluation': result.evaluation.model_copy(update={'jury': jury})})


def test_fragment_renders_jury_panel_health(rt_result_vuln: RedTeamResult) -> None:
    from evaluatorq.redteam.contracts import JuryResult, JuryVote

    # 3 configured (one failed mechanically) + 1 replacement stand-in; 2 decisive.
    jury = JuryResult(
        judges_configured=3,
        judges_succeeded=2,
        judges_failed=1,
        replacements_used=1,
        votes=[
            JuryVote(model='openai/gpt-5', success=True, value=True),
            JuryVote(model='anthropic/claude', success=True, abstained=True),
            JuryVote(model='mistral/large', success=False, error='timeout'),
            JuryVote(model='google/gemini', success=True, value=True, replacement=True),
        ],
    )
    html = render_attack_fragment(_with_jury(rt_result_vuln, jury))

    assert 'Jury panel health' in html
    assert '2/3 judges' in html
    assert '1 failed' in html
    assert '1 replacement' in html
    # Per-judge rows: model tail shown, one <li> per vote, status surfaced.
    assert html.count('class="rt-jury-vote"') == 4
    for tail in ('gpt-5', 'claude', 'large', 'gemini'):
        assert tail in html
    assert 'abstained' in html
    assert 'stand-in' in html  # the replacement judge
    assert 'error' in html.lower()


def test_fragment_omits_jury_for_single_judge(rt_result_vuln: RedTeamResult) -> None:
    # The fixture carries no jury (single-judge run): no panel-health section.
    assert 'Jury panel health' not in render_attack_fragment(rt_result_vuln)


def test_jury_panel_health_is_empty_without_a_jury() -> None:
    assert _jury_panel_health(None) == ''


def test_jury_panel_health_flags_tie() -> None:
    from evaluatorq.redteam.contracts import JuryResult, JuryVote

    jury = JuryResult(
        judges_configured=2,
        judges_succeeded=2,
        judges_failed=0,
        tie=True,
        votes=[JuryVote(model='a', success=True, value=True), JuryVote(model='b', success=True, value=False)],
    )
    assert 'tie' in _jury_panel_health(jury).lower()


def test_jury_panel_health_flags_inconclusive_and_failed_repetitions() -> None:
    from evaluatorq.redteam.contracts import JuryResult, JuryVote

    # A decisive vote whose underlying repetitions partly failed, on an
    # inconclusive panel.
    jury = JuryResult(
        judges_configured=1,
        judges_succeeded=1,
        judges_failed=0,
        inconclusive=True,
        votes=[JuryVote(model='m0', success=True, value=False, repetitions_failed=2)],
    )
    html = _jury_panel_health(jury)
    assert 'inconclusive' in html
    assert '2 failed reps' in html


def test_attack_endpoint_honors_filtered_index(
    client_with_rt_fixture: tuple[TestClient, str],
) -> None:
    # filter to Vulnerable, idx=0 -> first vulnerable result's fragment
    client, rid = client_with_rt_fixture
    resp = client.get(f'/r/{rid}/redteam/attack?idx=0&result=Vulnerable')
    assert resp.status_code == 200
    assert 'Evaluator verdict' in resp.text
