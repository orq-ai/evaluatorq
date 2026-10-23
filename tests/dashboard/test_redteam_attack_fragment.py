"""Tests for `redteam_transcripts.render_attack_fragment` (Task 13) and the
`/r/{rid}/redteam/attack` fragment endpoint (Task 14)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from evaluatorq.dashboard.redteam_transcripts import _jury_panel_health, _jury_vote_label, render_attack_fragment

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
    assert '>2/4 decisive votes</span>' in html
    assert '>1 failed</span>' in html
    assert '>1 replacement</span>' in html
    # Per-judge rows: model tail shown, one <li> per vote, status surfaced.
    assert html.count('class="rt-jury-vote"') == 4
    for tail in ('gpt-5', 'claude', 'large', 'gemini'):
        assert f'<b>{tail}</b>' in html
    assert 'abstained' in html
    assert 'stand-in' in html  # the replacement judge
    assert '<div class="rt-jury-error">timeout</div>' in html
    assert html.count('abstained') == 1
    assert '&rarr; RESISTANT' in html
    assert '&rarr; True' not in html


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


@pytest.mark.parametrize(('value', 'label'), [(True, 'RESISTANT'), (False, 'VULNERABLE'), (0.75, '0.75'), ('custom', 'custom')])
def test_jury_vote_labels_preserve_verdict_meaning(value: bool | float | str, label: str) -> None:
    from evaluatorq.contracts import JuryVote

    assert _jury_vote_label(JuryVote(model='judge', success=True, value=value)) == label


def test_jury_vote_missing_verdict_is_not_displayed_as_none() -> None:
    from evaluatorq.contracts import JuryVote

    # Normal validation rejects this state. Defend against a model_copy or
    # model_construct caller that bypasses that boundary.
    vote = JuryVote.model_construct(model='judge', success=True, value=None)
    assert _jury_vote_label(vote) == 'no verdict'


def test_jury_panel_escapes_model_verdict_and_failure_text() -> None:
    from evaluatorq.contracts import JuryResult, JuryVote

    jury = JuryResult(judges_configured=2, judges_succeeded=1, judges_failed=1, votes=[
        JuryVote(model='provider/<img src=x onerror=alert(1)>', success=True, value='<script>verdict</script>'),
        JuryVote(model='failed', success=False, error='<script>failure</script>'),
    ])
    html = _jury_panel_health(jury)
    assert '<script>' not in html
    assert '<img ' not in html
    assert '&lt;script&gt;verdict&lt;/script&gt;' in html
    assert '&lt;script&gt;failure&lt;/script&gt;' in html


def test_jury_replacements_can_produce_more_decisive_votes_than_configured() -> None:
    from evaluatorq.contracts import JuryResult, JuryVote

    jury = JuryResult(judges_configured=1, judges_succeeded=2, judges_failed=0, replacements_used=1, votes=[
        JuryVote(model='configured', success=True, value=True),
        JuryVote(model='replacement', success=True, value=False, replacement=True),
    ])
    assert '>2/2 decisive votes</span>' in _jury_panel_health(jury)


def test_saved_jury_reaches_attack_endpoint(tmp_path, rt_report_multi) -> None:
    from starlette.testclient import TestClient

    from evaluatorq.contracts import JuryResult, JuryVote
    from evaluatorq.dashboard.app import build_app
    from evaluatorq.dashboard.library import report_id

    jury = JuryResult(judges_configured=1, judges_succeeded=0, judges_failed=1, replacements_used=2, inconclusive=True, votes=[
        JuryVote(model='configured', success=False, error='timeout'),
        JuryVote(model='stand-in-a', success=False, replacement=True, error='unavailable'),
        JuryVote(model='stand-in-b', success=True, replacement=True, abstained=True, repetitions_failed=1),
    ])
    rt_report_multi.results[0].evaluation = rt_report_multi.results[0].evaluation.model_copy(update={'jury': jury})
    path = tmp_path / 'jury.json'
    path.write_text(rt_report_multi.model_dump_json())
    with TestClient(build_app(roots=[tmp_path])) as client:
        response = client.get(f'/r/{report_id(path)}/redteam/attack?idx=0')
    assert response.status_code == 200
    assert '>0/3 decisive votes</span>' in response.text
    assert '>2 replacements</span>' in response.text
    assert '>1 failed</span>' in response.text  # configured failures only
    assert '>1 failed rep</span>' in response.text
    assert '>inconclusive</span>' in response.text
    assert '<div class="rt-jury-error">unavailable</div>' in response.text
