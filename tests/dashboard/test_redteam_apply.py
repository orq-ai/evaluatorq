"""Apply-recommendations dashboard flow (RES-1143).

Pins the contract of the Focus-areas apply bar, the preview/confirm drawer
routes, and the report write-back: preview never writes, confirm writes the
agent AND records the applied recommendations on the report JSON, and every
failure path degrades to an error drawer instead of a 500.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

import evaluatorq.dashboard.apply_ui as apply_mod
from evaluatorq.dashboard.apply_ui import record_applied_on_report, render_preview_drawer
from evaluatorq.redteam.reports.apply import ApplyRecommendationsResult

if TYPE_CHECKING:
    from pathlib import Path

    from starlette.testclient import TestClient

    from evaluatorq.redteam.contracts import RedTeamReport


@pytest.fixture
def apply_client(tmp_path, rt_report_multi):
    """(client, rid, report_path): like client_with_rt_fixture but the report
    path is exposed so write-back assertions can read the file."""
    from starlette.testclient import TestClient as _TC

    from evaluatorq.dashboard.app import build_app
    from evaluatorq.dashboard.library import report_id

    rt_dir = tmp_path / 'runs'
    rt_dir.mkdir()
    rt_path = rt_dir / 'rt_fixture.json'
    # Apply is a single-agent feature; narrow the multi-agent fixture to one.
    single = rt_report_multi.model_copy(update={'tested_agents': ['agent-a']})
    rt_path.write_text(single.model_dump_json())
    client = _TC(build_app(roots=[rt_dir]), raise_server_exceptions=True)
    return client, report_id(rt_path), rt_path


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------


class TestApplyPanel:
    def test_focus_tab_carries_apply_bar_and_drawer_mount(self, apply_client) -> None:
        client, rid, _path = apply_client
        html = client.get(f'/r/{rid}').text
        assert 'recommendation(s) ready to apply' in html
        assert f'/r/{rid}/redteam/apply/preview' in html
        assert 'id="rt-apply-drawer"' in html
        assert 'Preview &amp; apply' in html

    def test_recommendation_bullets_render_on_focus_cards(self, apply_client) -> None:
        client, rid, _path = apply_client
        html = client.get(f'/r/{rid}').text
        assert 'Add stricter goal-boundary checks' in html
        assert 'Log and alert on objective drift' in html


# ---------------------------------------------------------------------------
# Preview route
# ---------------------------------------------------------------------------


class TestPreview:
    def test_preview_without_api_key_shows_error_drawer(self, apply_client, monkeypatch: pytest.MonkeyPatch) -> None:
        client, rid, path = apply_client
        monkeypatch.delenv('ORQ_API_KEY', raising=False)
        r = client.post(f'/r/{rid}/redteam/apply/preview', data={'agent_key': 'agent-a'})
        assert r.status_code == 200
        assert 'ORQ_API_KEY' in r.text
        assert 'rt-drawer-error' in r.text

    def test_preview_without_agent_key_shows_error_drawer(self, apply_client) -> None:
        client, rid, _path = apply_client
        r = client.post(f'/r/{rid}/redteam/apply/preview', data={})
        assert 'rt-drawer-error' in r.text

    def test_preview_renders_diff_and_confirm(self, apply_client, monkeypatch: pytest.MonkeyPatch) -> None:
        client, rid, path = apply_client
        monkeypatch.setattr(apply_mod, '_build_clients', lambda: (object(), object(), 'm'))

        async def fake_apply(*args, **kwargs):
            return ApplyRecommendationsResult(
                agent_key='agent-a',
                recommendations=['Add stricter goal-boundary checks'],
                original_instructions='Old rules.',
                new_instructions='Old rules.\nNew guard.',
                diff='--- instructions (current)\n+++ instructions (proposed)\n+New guard.\n',
                applied=False,
            )

        import evaluatorq.redteam.reports.apply as source_mod

        monkeypatch.setattr(source_mod, 'apply_recommendations', fake_apply)
        r = client.post(f'/r/{rid}/redteam/apply/preview', data={'agent_key': 'agent-a'})
        assert r.status_code == 200
        assert 'rt-diff' in r.text
        assert 'New guard.' in r.text
        assert f'/r/{rid}/redteam/apply/confirm' in r.text
        assert 'Apply to agent' in r.text
        # Preview must not have touched the report file.
        raw = json.loads(path.read_text())
        assert raw.get('applied_recommendations', []) == []

    def test_preview_error_becomes_drawer_not_500(self, apply_client, monkeypatch: pytest.MonkeyPatch) -> None:
        client, rid, path = apply_client
        monkeypatch.setattr(apply_mod, '_build_clients', lambda: (object(), object(), 'm'))

        async def boom(*args, **kwargs):
            raise RuntimeError('agent not found')

        import evaluatorq.redteam.reports.apply as source_mod

        monkeypatch.setattr(source_mod, 'apply_recommendations', boom)
        r = client.post(f'/r/{rid}/redteam/apply/preview', data={'agent_key': 'agent-a'})
        assert r.status_code == 200
        assert 'rt-drawer-error' in r.text
        assert 'agent not found' in r.text


# ---------------------------------------------------------------------------
# Confirm route
# ---------------------------------------------------------------------------


class TestConfirm:
    def _payload(self) -> str:
        return json.dumps({
            'agent_key': 'agent-a',
            'new_instructions': 'Old rules.\nNew guard.',
            'recommendations': ['Add stricter goal-boundary checks'],
        })

    def test_confirm_updates_agent_and_records_on_report(self, apply_client, monkeypatch: pytest.MonkeyPatch) -> None:
        client, rid, path = apply_client
        calls: list[dict] = []

        class FakeAgents:
            def update(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(version='1.3.0')

        fake_orq = SimpleNamespace(agents=FakeAgents())
        monkeypatch.setattr(apply_mod, '_build_clients', lambda: (fake_orq, object(), 'm'))

        r = client.post(f'/r/{rid}/redteam/apply/confirm', data={'payload': self._payload()})
        assert r.status_code == 200
        assert 'Applied 1 recommendation(s)' in r.text
        assert '1.3.0' in r.text
        assert calls[0]['agent_key'] == 'agent-a'
        assert calls[0]['instructions'] == 'Old rules.\nNew guard.'
        assert calls[0]['version_increment'] == 'minor'

        raw = json.loads(path.read_text())
        assert raw['applied_recommendations'] == ['Add stricter goal-boundary checks']

        # The re-rendered report shows the applied state: one rec left pending.
        html = client.get(f'/r/{rid}').text
        assert '✓ applied' in html
        assert '1 recommendation(s) ready to apply' in html

    def test_confirm_agent_update_failure_becomes_error_drawer(
        self, apply_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, rid, path = apply_client

        class FakeAgents:
            def update(self, **kwargs):
                raise RuntimeError('403 from platform')

        fake_orq = SimpleNamespace(agents=FakeAgents())
        monkeypatch.setattr(apply_mod, '_build_clients', lambda: (fake_orq, object(), 'm'))
        r = client.post(f'/r/{rid}/redteam/apply/confirm', data={'payload': self._payload()})
        assert 'rt-drawer-error' in r.text
        assert '403' in r.text
        raw = json.loads(path.read_text())
        assert raw.get('applied_recommendations', []) == []

    def test_confirm_malformed_payload_is_an_error_drawer(self, apply_client) -> None:
        client, rid, _path = apply_client
        r = client.post(f'/r/{rid}/redteam/apply/confirm', data={'payload': 'not json'})
        assert 'rt-drawer-error' in r.text

    def test_confirm_rejects_payload_for_a_different_agent(self, apply_client, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stale/hand-crafted payload naming another agent is refused: the
        confirm handler trusts the reloaded report, not the posted agent_key."""
        client, rid, path = apply_client
        called = {'update': False}

        class FakeAgents:
            def update(self, **kwargs):
                called['update'] = True
                return SimpleNamespace(version='9.9.9')

        monkeypatch.setattr(apply_mod, '_build_clients', lambda: (SimpleNamespace(agents=FakeAgents()), object(), 'm'))
        payload = json.dumps({
            'agent_key': 'some-other-agent',
            'new_instructions': 'X',
            'recommendations': ['Add stricter goal-boundary checks'],
        })
        r = client.post(f'/r/{rid}/redteam/apply/confirm', data={'payload': payload})
        assert 'rt-drawer-error' in r.text
        assert called['update'] is False
        assert json.loads(path.read_text()).get('applied_recommendations', []) == []

    def test_confirm_skips_already_applied_recommendations(self, apply_client, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bullets already recorded as applied are dropped before the write, so a
        replayed confirm cannot re-merge them."""
        client, rid, path = apply_client
        # Pre-record the bullet the payload will carry.
        raw = json.loads(path.read_text())
        raw['applied_recommendations'] = ['Add stricter goal-boundary checks']
        path.write_text(json.dumps(raw))
        called = {'update': False}

        class FakeAgents:
            def update(self, **kwargs):
                called['update'] = True
                return SimpleNamespace(version='9.9.9')

        monkeypatch.setattr(apply_mod, '_build_clients', lambda: (SimpleNamespace(agents=FakeAgents()), object(), 'm'))
        r = client.post(f'/r/{rid}/redteam/apply/confirm', data={'payload': self._payload()})
        assert 'rt-drawer-error' in r.text
        assert 'already applied' in r.text
        assert called['update'] is False


# ---------------------------------------------------------------------------
# Fragments + write-back units
# ---------------------------------------------------------------------------


class TestUnits:
    def test_dismiss_returns_empty(self, apply_client) -> None:
        client, _rid, _path = apply_client
        r = client.get('/apply/dismiss')
        assert r.status_code == 200
        assert r.text == ''

    def test_no_change_preview_has_no_confirm_button(self) -> None:
        html = render_preview_drawer(
            'rid',
            ApplyRecommendationsResult(
                agent_key='a',
                recommendations=['r1'],
                original_instructions='same',
                new_instructions='same',
                diff='',
                applied=False,
            ),
        )
        assert 'no change' in html
        assert 'apply/confirm' not in html

    def test_record_applied_deduplicates(self, tmp_path) -> None:
        path = tmp_path / 'r.json'
        path.write_text(json.dumps({'applied_recommendations': ['a']}))
        record_applied_on_report(path, ['a', 'b', 'b '])
        assert json.loads(path.read_text())['applied_recommendations'] == ['a', 'b']


# ---------------------------------------------------------------------------
# Per-recommendation apply (breakdown drawer)
# ---------------------------------------------------------------------------


class TestPerRecommendationApply:
    def test_pending_bullets_carry_apply_buttons(self, apply_client) -> None:
        client, rid, _path = apply_client
        html = client.get(f'/r/{rid}').text
        assert 'name="rec"' in html
        assert html.count('rt-focus-rec-apply') >= 2  # one per pending bullet

    def test_applied_bullet_has_tick_and_no_button(self, apply_client) -> None:
        client, rid, path = apply_client
        raw = json.loads(path.read_text())
        raw['applied_recommendations'] = ['Add stricter goal-boundary checks']
        path.write_text(json.dumps(raw))
        html = client.get(f'/r/{rid}').text
        # The applied bullet gets a tick; the other stays a button.
        assert '✓ applied' in html
        assert 'rt-focus-rec-apply' in html

    def test_single_rec_preview_narrows_merge_and_shows_area_breakdown(
        self, apply_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, rid, _path = apply_client
        monkeypatch.setattr(apply_mod, '_build_clients', lambda: (object(), object(), 'm'))
        seen_areas = []

        async def fake_apply(areas, agent_key, *args, **kwargs):
            seen_areas.append(areas)
            return ApplyRecommendationsResult(
                agent_key=agent_key,
                recommendations=['Add stricter goal-boundary checks'],
                original_instructions='Old.',
                new_instructions='Old.\nGuard.',
                diff='+Guard.\n',
                applied=False,
            )

        import evaluatorq.redteam.reports.apply as source_mod

        monkeypatch.setattr(source_mod, 'apply_recommendations', fake_apply)
        r = client.post(
            f'/r/{rid}/redteam/apply/preview',
            data={
                'agent_key': 'agent-a',
                'rec': 'Add stricter goal-boundary checks',
                'category': 'ASI01',
            },
        )
        assert r.status_code == 200
        # The merge saw exactly one area holding exactly the chosen rec.
        assert len(seen_areas[0]) == 1
        assert seen_areas[0][0].recommendations == ['Add stricter goal-boundary checks']
        # The drawer carries the focus-area breakdown block.
        assert 'Focus area' in r.text
        assert 'Goal Hijacking' in r.text
        assert 'failed trace(s) analyzed' in r.text
        assert 'Recommendation to fold in' in r.text

    def test_single_rec_preview_rejects_already_applied(self, apply_client) -> None:
        client, rid, path = apply_client
        raw = json.loads(path.read_text())
        raw['applied_recommendations'] = ['Add stricter goal-boundary checks']
        path.write_text(json.dumps(raw))
        r = client.post(
            f'/r/{rid}/redteam/apply/preview',
            data={
                'agent_key': 'agent-a',
                'rec': 'Add stricter goal-boundary checks',
                'category': 'ASI01',
            },
        )
        assert 'already applied' in r.text
        assert 'rt-drawer-error' in r.text

    def test_single_rec_preview_rejects_unknown_rec(self, apply_client) -> None:
        client, rid, _path = apply_client
        r = client.post(
            f'/r/{rid}/redteam/apply/preview',
            data={'agent_key': 'agent-a', 'rec': 'made up', 'category': 'ASI01'},
        )
        assert 'no longer on the report' in r.text


# ---------------------------------------------------------------------------
# Single-agent gating (review decision: multi-agent runs are for comparison)
# ---------------------------------------------------------------------------


class TestMultiAgentGating:
    @pytest.fixture
    def multi_client(self, tmp_path, rt_report_multi):
        from starlette.testclient import TestClient as _TC

        from evaluatorq.dashboard.app import build_app
        from evaluatorq.dashboard.library import report_id

        rt_dir = tmp_path / 'runs'
        rt_dir.mkdir()
        rt_path = rt_dir / 'rt_fixture.json'
        rt_path.write_text(rt_report_multi.model_dump_json())
        client = _TC(build_app(roots=[rt_dir]), raise_server_exceptions=True)
        return client, report_id(rt_path)

    def test_multi_agent_report_has_no_apply_ui(self, multi_client) -> None:
        client, rid = multi_client
        html = client.get(f'/r/{rid}').text
        assert 'class="rt-apply-bar"' not in html
        assert 'class="rt-focus-rec-apply' not in html
        # The recommendations themselves still render as plain bullets.
        assert 'Add stricter goal-boundary checks' in html

    def test_multi_agent_preview_is_rejected(self, multi_client) -> None:
        client, rid = multi_client
        r = client.post(f'/r/{rid}/redteam/apply/preview', data={'agent_key': 'agent-a'})
        assert 'single-agent runs' in r.text
        assert 'rt-drawer-error' in r.text


# ---------------------------------------------------------------------------
# Apply-model config setting
# ---------------------------------------------------------------------------


class TestApplyModelSetting:
    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(apply_mod.APPLY_MODEL_ENV, raising=False)
        assert apply_mod.apply_model() == 'gpt-5.6-luna'

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(apply_mod.APPLY_MODEL_ENV, 'openai/gpt-6')
        assert apply_mod.apply_model() == 'openai/gpt-6'
        monkeypatch.setenv(apply_mod.APPLY_MODEL_ENV, '   ')
        assert apply_mod.apply_model() == 'gpt-5.6-luna'

    def test_settings_page_shows_the_model(self, apply_client, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _rid, _path = apply_client
        monkeypatch.delenv(apply_mod.APPLY_MODEL_ENV, raising=False)
        html = client.get('/settings').text
        assert 'Apply-recommendations model' in html
        assert 'gpt-5.6-luna (default)' in html

    def test_settings_page_shows_the_override_source(self, apply_client, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _rid, _path = apply_client
        monkeypatch.setenv(apply_mod.APPLY_MODEL_ENV, 'openai/gpt-6')
        html = client.get('/settings').text
        assert 'openai/gpt-6 (EVALUATORQ_APPLY_MODEL)' in html
