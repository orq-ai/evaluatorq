"""Apply-suggestions dashboard flow for agent simulation (RES-1143).

Pins the sim side of the shared apply UI: the Recommendations tab renders the
apply bar + per-suggestion buttons for orq-agent runs, the sim preview/confirm
routes drive the shared drawer, and confirm records applied suggestions on the
run JSON under ``applied_suggestions``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

import evaluatorq.dashboard.apply_ui as apply_mod
from evaluatorq.common.apply import ApplyRecommendationsResult

if TYPE_CHECKING:
    from starlette.testclient import TestClient  # noqa: F401


def _sim_run(target_kind: str = 'orq_agent', target: str | None = 'agent-sim'):
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.types import (
        SimulationRecommendation,
        SimulationResult,
        SimulationRun,
        TerminatedBy,
    )

    result = SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason='done',
        goal_achieved=False,
        goal_completion_score=0.0,
        rules_broken=[],
        turn_count=2,
        turn_metrics=[],
        token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        metadata={'persona': 'alice', 'scenario': 'billing'},
    )
    return SimulationRun(
        run_name='sim-apply-fixture',
        created_at=datetime.now(tz=timezone.utc),
        mode='run',
        target_kind=target_kind,
        target=target,
        evaluator_names=['goal_achieved'],
        total_results=1,
        scorer_averages={'goal_achieved': 0.0},
        results=[result],
        recommendations=[
            SimulationRecommendation(
                result_index=0,
                persona='alice',
                scenario='billing dispute',
                triggers=['agent looped on refund policy'],
                suggestions=['Clarify the refund policy up front', 'Escalate after two failed attempts'],
            )
        ],
    )


@pytest.fixture
def sim_apply_client(tmp_path):
    """(client, rid, run_path) for a sim run targeting an orq agent."""
    from starlette.testclient import TestClient as _TC

    from evaluatorq.dashboard.app import build_app
    from evaluatorq.dashboard.library import report_id

    run_dir = tmp_path / 'runs'
    run_dir.mkdir()
    run_path = run_dir / 'sim_fixture.json'
    run_path.write_text(_sim_run().model_dump_json())
    client = _TC(build_app(roots=[run_dir]), raise_server_exceptions=True)
    return client, report_id(run_path), run_path


class TestSimRecommendationsTab:
    def test_tab_carries_apply_bar_and_suggestion_buttons(self, sim_apply_client) -> None:
        client, rid, _path = sim_apply_client
        html = client.get(f'/r/{rid}').text
        assert 'Recommendations' in html
        assert 'recommendation(s) ready to apply' in html
        assert f'/r/{rid}/sim/apply/preview' in html
        assert 'id="rt-apply-drawer"' in html
        assert 'Clarify the refund policy up front' in html
        assert html.count('rt-focus-rec-apply') >= 2  # one per pending suggestion

    def test_non_agent_target_renders_bullets_without_apply_ui(self, tmp_path) -> None:
        from starlette.testclient import TestClient as _TC

        from evaluatorq.dashboard.app import build_app
        from evaluatorq.dashboard.library import report_id

        run_dir = tmp_path / 'runs'
        run_dir.mkdir()
        run_path = run_dir / 'sim_fixture.json'
        run_path.write_text(_sim_run(target_kind='openai_model', target='gpt-5-mini').model_dump_json())
        client = _TC(build_app(roots=[run_dir]), raise_server_exceptions=True)
        html = client.get(f'/r/{report_id(run_path)}').text
        assert 'Clarify the refund policy up front' in html
        assert 'class="rt-apply-bar"' not in html
        assert 'class="rt-focus-rec-apply"' not in html

    def test_applied_suggestion_shows_tick(self, sim_apply_client) -> None:
        client, rid, path = sim_apply_client
        raw = json.loads(path.read_text())
        raw['applied_suggestions'] = ['Clarify the refund policy up front']
        path.write_text(json.dumps(raw, default=str))
        html = client.get(f'/r/{rid}').text
        assert '✓ applied' in html
        assert '1 recommendation(s) ready to apply' in html


class TestSimPreview:
    def test_preview_applies_fenced_edits_response(self, sim_apply_client, monkeypatch: pytest.MonkeyPatch) -> None:
        client, rid, path = sim_apply_client

        class FakeAgents:
            def retrieve(self, agent_key):
                return SimpleNamespace(instructions='Old rules.')

        fake_orq = SimpleNamespace(agents=FakeAgents())
        monkeypatch.setattr(apply_mod, '_build_clients', lambda: (fake_orq, object(), 'm'))

        from evaluatorq.common import apply as common_apply

        chat = AsyncMock(
            return_value=(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='```json\n{"edits": [{"find": "Old rules.", "replace": "Old rules.\\nRefund policy first."}]}\n```'
                            )
                        )
                    ]
                ),
                None,
            )
        )
        monkeypatch.setattr(common_apply, 'execute_chat_completion', chat)

        r = client.post(
            f'/r/{rid}/sim/apply/preview',
            data={apply_mod.CSRF_FIELD: apply_mod._CSRF_TOKEN, 'agent_key': 'agent-sim'},
        )

        assert r.status_code == 200
        assert 'Refund policy first.' in r.text
        assert 'rt-diff' in r.text
        assert chat.await_count == 1
        assert json.loads(path.read_text()).get('applied_suggestions', []) == []

    def test_preview_renders_diff_and_sim_confirm(self, sim_apply_client, monkeypatch: pytest.MonkeyPatch) -> None:
        client, rid, path = sim_apply_client
        monkeypatch.setattr(apply_mod, '_build_clients', lambda: (object(), object(), 'm'))

        async def fake_apply(*args, **kwargs):
            return ApplyRecommendationsResult(
                agent_key='agent-sim',
                recommendations=['Clarify the refund policy up front'],
                original_instructions='Old rules.',
                new_instructions='Old rules.\nRefund policy first.',
                diff='+Refund policy first.\n',
                applied=False,
            )

        import evaluatorq.simulation.reports.apply as source_mod

        monkeypatch.setattr(source_mod, 'apply_suggestions', fake_apply)
        r = client.post(
            f'/r/{rid}/sim/apply/preview', data={apply_mod.CSRF_FIELD: apply_mod._CSRF_TOKEN, 'agent_key': 'agent-sim'}
        )
        assert r.status_code == 200
        assert 'rt-diff' in r.text
        assert 'Read this diff before applying.' in r.text
        assert f'/r/{rid}/sim/apply/confirm' in r.text
        # Preview must not have touched the run file.
        assert json.loads(path.read_text()).get('applied_suggestions', []) == []

    def test_single_suggestion_preview_narrows_and_shows_breakdown(
        self, sim_apply_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, rid, _path = sim_apply_client
        monkeypatch.setattr(apply_mod, '_build_clients', lambda: (object(), object(), 'm'))
        seen: list = []

        async def fake_apply(recs, agent_key, *args, **kwargs):
            seen.append(recs)
            return ApplyRecommendationsResult(
                agent_key=agent_key,
                recommendations=['Clarify the refund policy up front'],
                original_instructions='Old.',
                new_instructions='Old.\nGuard.',
                diff='+Guard.\n',
                applied=False,
            )

        import evaluatorq.simulation.reports.apply as source_mod

        monkeypatch.setattr(source_mod, 'apply_suggestions', fake_apply)
        r = client.post(
            f'/r/{rid}/sim/apply/preview',
            data={
                apply_mod.CSRF_FIELD: apply_mod._CSRF_TOKEN,
                'agent_key': 'agent-sim',
                'rec': 'Clarify the refund policy up front',
                'result_index': '0',
            },
        )
        assert r.status_code == 200
        # The merge saw exactly one card holding exactly the chosen suggestion.
        assert len(seen[0]) == 1
        assert seen[0][0].suggestions == ['Clarify the refund policy up front']
        # The drawer carries the persona/scenario breakdown block.
        assert 'Simulation finding' in r.text
        assert 'alice' in r.text
        assert 'billing dispute' in r.text

    def test_single_suggestion_preview_rejects_unknown(self, sim_apply_client) -> None:
        client, rid, _path = sim_apply_client
        r = client.post(
            f'/r/{rid}/sim/apply/preview',
            data={
                apply_mod.CSRF_FIELD: apply_mod._CSRF_TOKEN,
                'agent_key': 'agent-sim',
                'rec': 'made up',
                'result_index': '0',
            },
        )
        assert 'no longer on the report' in r.text

    def test_non_agent_run_preview_is_rejected(self, tmp_path) -> None:
        from starlette.testclient import TestClient as _TC

        from evaluatorq.dashboard.app import build_app
        from evaluatorq.dashboard.library import report_id

        run_dir = tmp_path / 'runs'
        run_dir.mkdir()
        run_path = run_dir / 'sim_fixture.json'
        run_path.write_text(_sim_run(target_kind='openai_model', target='gpt-5-mini').model_dump_json())
        client = _TC(build_app(roots=[run_dir]), raise_server_exceptions=True)
        r = client.post(
            f'/r/{report_id(run_path)}/sim/apply/preview',
            data={apply_mod.CSRF_FIELD: apply_mod._CSRF_TOKEN, 'agent_key': 'x'},
        )
        assert 'rt-drawer-error' in r.text
        assert 'orq agent' in r.text


class TestSimConfirm:
    def test_confirm_updates_agent_and_records_applied_suggestions(
        self, sim_apply_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, rid, path = sim_apply_client
        calls: list[dict] = []

        class FakeAgents:
            def retrieve(self, agent_key):
                return SimpleNamespace(instructions='Old rules.')

            def update(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(version='2.1.0')

        fake_orq = SimpleNamespace(agents=FakeAgents())
        monkeypatch.setattr(apply_mod, '_build_clients', lambda: (fake_orq, object(), 'm'))
        token = apply_mod._store_preview({
            'rid': rid,
            'surface': 'sim',
            'agent_key': 'agent-sim',
            'original_instructions': 'Old rules.',
            'new_instructions': 'Old rules.\nRefund policy first.',
            'recommendations': ['Clarify the refund policy up front'],
        })
        r = client.post(
            f'/r/{rid}/sim/apply/confirm',
            data={apply_mod.CSRF_FIELD: apply_mod._CSRF_TOKEN, 'confirm_token': token},
        )
        assert r.status_code == 200
        assert 'Applied 1 recommendation(s)' in r.text
        assert calls[0]['agent_key'] == 'agent-sim'
        assert 'simulation' in calls[0]['version_description']

        raw = json.loads(path.read_text())
        assert raw['applied_suggestions'] == ['Clarify the refund policy up front']
        assert raw.get('applied_recommendations') in (None, [])

        # The re-rendered report shows one suggestion pending, one applied.
        html = client.get(f'/r/{rid}').text
        assert '✓ applied' in html
        assert '1 recommendation(s) ready to apply' in html

    def test_confirm_rejects_a_token_minted_for_the_other_surface(
        self, sim_apply_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A red-team preview token must not authorize a sim write (surface is
        part of the stored entry, not the URL)."""
        client, rid, path = sim_apply_client
        calls: list[dict] = []

        class FakeAgents:
            def retrieve(self, agent_key):
                return SimpleNamespace(instructions='Old rules.')

            def update(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(version='2.1.0')

        monkeypatch.setattr(apply_mod, '_build_clients', lambda: (SimpleNamespace(agents=FakeAgents()), object(), 'm'))
        token = apply_mod._store_preview({
            'rid': rid,
            'surface': 'redteam',
            'agent_key': 'agent-sim',
            'original_instructions': 'Old rules.',
            'new_instructions': 'X',
            'recommendations': ['Clarify the refund policy up front'],
        })
        r = client.post(
            f'/r/{rid}/sim/apply/confirm',
            data={apply_mod.CSRF_FIELD: apply_mod._CSRF_TOKEN, 'confirm_token': token},
        )
        assert 'different report' in r.text
        assert calls == []
        assert json.loads(path.read_text()).get('applied_suggestions', []) == []
