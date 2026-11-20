"""`evaluatorq.models`: card parsing, dominance, the frontier, one-per-model, the CLI and the dashboard page."""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from evaluatorq import models as models_mod
from evaluatorq.common.model_frontier import ScoredModel, scored_model_from_card
from evaluatorq.models import (
    dominated_by,
    dominates,
    find_model,
    one_per_model,
    pareto,
    pareto_frontier,
    score_of,
    x_of,
)
from tests.common.garden_fakes import card


def model(router_id: str, index: float, input_price: float, output_price: float, **fields: Any) -> ScoredModel:
    return ScoredModel(
        router_id=router_id,
        intelligence_index=index,
        input_price=input_price,
        output_price=output_price,
        provider=router_id.split('/')[0],
        enabled=True,
        **fields,
    )


# Blended prices: strong 8.0, mid 2.0, cheap 0.5, bad 4.0 (dominated by mid), free 0.
STRONG = model('a/strong', 50.0, 5.0, 17.0)
MID = model('b/mid', 40.0, 1.0, 5.0)
CHEAP = model('c/cheap', 20.0, 0.25, 1.25)
BAD = model('d/bad', 35.0, 2.0, 10.0)
FREE = model('e/free', 60.0, 0.0, 0.0)


class TestCardParsing:
    def test_reads_categories_speed_and_estimate_flag(self) -> None:
        v2 = {
            'intelligence_index': 40.0,
            'aa_slug': 'm',
            'aa_index_is_estimated': True,
            'aa_evaluations': {
                'artificial_analysis_intelligence_index': 40.0,
                'artificial_analysis_legal_index': 33.5,
                'artificial_analysis_coding_index': None,
                'artificial_analysis_economics_index': 'n/a',
            },
            'aa_median_output_tokens_per_second': 120.0,
            'aa_median_time_to_first_token_seconds': 0.8,
        }
        parsed = scored_model_from_card(
            card('openai', 'm', model_developer='openai', display_name='M', metadata={'autorouter': {'v2': v2}})
        )
        assert parsed is not None
        assert parsed.category_indices == {'legal': 33.5}
        assert parsed.is_estimated is True
        assert parsed.output_tokens_per_second == 120.0
        assert parsed.time_to_first_token_seconds == 0.8
        assert (parsed.developer, parsed.display_name, parsed.aa_slug) == ('openai', 'M', 'm')

    def test_malformed_evaluations_read_as_absent(self) -> None:
        v2 = {'intelligence_index': 40.0, 'aa_evaluations': ['not', 'a', 'mapping']}
        parsed = scored_model_from_card(card('openai', 'm', metadata={'autorouter': {'v2': v2}}))
        assert parsed is not None
        assert parsed.category_indices == {}
        assert parsed.is_estimated is False


class TestDominance:
    def test_cheaper_and_smarter_dominates(self) -> None:
        assert dominates(MID, BAD)
        assert not dominates(BAD, MID)

    def test_trade_off_is_not_dominance(self) -> None:
        assert not dominates(STRONG, MID)
        assert not dominates(MID, STRONG)

    def test_a_model_never_dominates_itself(self) -> None:
        assert not dominates(MID, MID)

    def test_free_models_have_no_place_on_price(self) -> None:
        assert x_of(FREE) is None
        assert not dominates(FREE, CHEAP)

    def test_higher_speed_is_better(self) -> None:
        fast = model('a/fast', 30.0, 1.0, 1.0, output_tokens_per_second=200.0)
        slow = model('b/slow', 30.0, 0.1, 0.1, output_tokens_per_second=50.0)
        assert dominates(fast, slow, x='output_speed')
        assert dominates(slow, fast, x='price')


class TestPareto:
    def test_frontier_and_nearest_dominator(self) -> None:
        result = pareto([STRONG, MID, CHEAP, BAD, FREE])
        assert [p.model.router_id for p in result.frontier] == ['a/strong', 'b/mid', 'c/cheap']
        [bad] = result.dominated
        assert bad.dominated_by == 'b/mid'
        assert bad.dominators == ['b/mid']
        assert [e.router_id for e in result.excluded] == ['e/free']

    def test_pareto_score_is_score_per_dollar(self) -> None:
        point = next(p for p in pareto([MID]).points)
        assert point.pareto_score == pytest.approx(40.0 / 2.0)

    def test_nearest_dominator_is_the_cheapest_frontier_model_at_least_as_smart(self) -> None:
        weak = model('z/weak', 10.0, 10.0, 10.0)
        assert dominated_by(weak, [STRONG, MID, CHEAP]) == CHEAP

    def test_dominated_dominator_is_not_named(self) -> None:
        # BAD dominates WORSE, but MID dominates BAD: the answer is the frontier model.
        worse = model('z/worse', 30.0, 3.0, 11.0)
        assert dominated_by(worse, [MID, BAD]) == MID

    def test_effort_reads_that_rung_and_excludes_models_without_it(self) -> None:
        laddered = model(
            'a/l', 30.0, 1.0, 1.0, effort_parameter=True, default_effort='low', effort_indices={'low': 30.0, 'high': 45.0}
        )
        flat = model('b/f', 40.0, 1.0, 1.0)
        assert score_of(laddered) == 30.0
        assert score_of(laddered, effort='high') == 45.0
        result = pareto([laddered, flat], effort='high')
        assert [p.model.router_id for p in result.points] == ['a/l']
        assert result.excluded[0].reason == 'no intelligence score at effort high'

    def test_domain_index_axis(self) -> None:
        legal = model('a/l', 30.0, 1.0, 1.0, category_indices={'legal': 50.0})
        result = pareto([legal, MID], index='legal')
        assert [p.score for p in result.points] == [50.0]
        assert result.excluded[0].router_id == 'b/mid'

    def test_domain_index_refuses_an_effort(self) -> None:
        with pytest.raises(ValueError, match='default reasoning effort'):
            pareto([MID], index='legal', effort='high')

    def test_unknown_axis_is_refused(self) -> None:
        with pytest.raises(ValueError, match='unknown x axis'):
            pareto([MID], x='vibes')  # ty: ignore[invalid-argument-type]

    def test_measured_only_drops_estimates(self) -> None:
        guess = model('z/guess', 99.0, 0.1, 0.1, is_estimated=True)
        result = pareto([MID, guess], measured_only=True)
        assert [p.model.router_id for p in result.points] == ['b/mid']
        assert result.excluded[0].reason == 'estimated score'
        assert pareto_frontier([MID, guess]) == [guess]


class TestOnePerModel:
    def test_prefers_the_developer_listing(self) -> None:
        own = model('openai/gpt-x', 40.0, 2.0, 8.0, developer='openai', aa_slug='gpt-x')
        azure = model('azure/gpt-x', 40.0, 1.0, 4.0, developer='openai', aa_slug='gpt-x')
        assert one_per_model([azure, own]) == [own]

    def test_falls_back_to_the_cheapest_host(self) -> None:
        a = model('h1/kimi', 40.0, 2.0, 8.0, developer='moonshotai', aa_slug='kimi')
        b = model('h2/kimi', 40.0, 1.0, 4.0, developer='moonshotai', aa_slug='kimi')
        assert one_per_model([a, b]) == [b]

    def test_google_ai_counts_as_google(self) -> None:
        own = model('google-ai/gemini', 40.0, 2.0, 8.0, developer='google', aa_slug='g')
        other = model('vertex/gemini', 40.0, 1.0, 4.0, developer='google', aa_slug='g')
        assert one_per_model([other, own]) == [own]


class TestFindModel:
    def test_exact_then_unique_bare_id(self) -> None:
        pool = [MID, model('x/mid', 1.0, 1.0, 1.0), CHEAP]
        assert find_model(pool, 'b/mid') == MID
        assert find_model(pool, 'cheap') == CHEAP
        assert find_model(pool, 'mid') is None  # two hosts share the bare id
        assert find_model(pool, 'nope') is None


@pytest.fixture
def fake_catalogue(monkeypatch: pytest.MonkeyPatch) -> list[ScoredModel]:
    pool = [STRONG, MID, CHEAP, BAD, FREE]

    async def fake_load(client: object = None, *, include_disabled: bool = False, all_hosts: bool = False) -> list[ScoredModel]:
        del client, include_disabled, all_hosts
        return pool

    monkeypatch.setattr(models_mod, 'load_models', fake_load)
    return pool


class TestCli:
    def _run(self, *args: str) -> Any:
        from evaluatorq.cli import _register_subapps, app

        _register_subapps(app)
        return CliRunner().invoke(app, ['models', 'pareto', *args])

    def test_json_frontier(self, fake_catalogue: list[ScoredModel]) -> None:
        result = self._run('--json')
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert [p['router_id'] for p in payload['points']] == ['a/strong', 'b/mid', 'c/cheap']
        assert payload['excluded'] == [{'router_id': 'e/free', 'reason': 'free (no list price)'}]

    def test_dominated_table_names_the_winner(self, fake_catalogue: list[ScoredModel]) -> None:
        result = self._run('--dominated')
        assert result.exit_code == 0, result.output
        assert 'd/bad' in result.stdout
        assert 'b/mid' in result.stdout

    def test_model_filter_shows_a_dominated_model_without_the_flag(self, fake_catalogue: list[ScoredModel]) -> None:
        payload = json.loads(self._run('--model', 'bad', '--json').stdout)
        assert [(p['router_id'], p['dominated_by']) for p in payload['points']] == [('d/bad', 'b/mid')]

    def test_unknown_model_is_a_usage_error(self, fake_catalogue: list[ScoredModel]) -> None:
        result = self._run('--model', 'nope')
        assert result.exit_code == 2

    def test_bad_index_is_a_usage_error(self, fake_catalogue: list[ScoredModel]) -> None:
        assert self._run('--index', 'vibes').exit_code == 2


class TestDashboardPage:
    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
        from starlette.testclient import TestClient

        from evaluatorq.dashboard import app as app_module
        from evaluatorq.dashboard.app import build_app

        monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'dashboard-settings.json'))
        monkeypatch.setenv('EVALUATORQ_USER_SETTINGS', str(tmp_path / 'user-settings.json'))
        monkeypatch.setattr(app_module, 'list_oauth_sessions', lambda: ())
        return TestClient(build_app(roots=[tmp_path]), follow_redirects=False)

    def test_without_a_key_the_page_says_how_to_connect(self, client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('ORQ_API_KEY', raising=False)
        response = client.get('/models')
        assert response.status_code == 200
        assert 'models-alert' in response.text
        assert 'ORQ_API_KEY is not set' in response.text
        assert 'data-needs-orq="1"' in response.text

    def test_renders_chart_table_and_dominator(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            return fake_catalogue

        monkeypatch.setattr(models_views, 'load_models', fake_load)
        response = client.get('/models?sort=pareto')
        assert response.status_code == 200
        text = response.text
        assert 'data-vega-for="models-pareto"' in text
        assert '3 on the frontier, 1 dominated, 1 not plotted.' in text
        assert '<td>b/mid</td>' in text  # d/bad's nearest dominator
        assert 'e/free' in text  # listed under "not plotted"

    def test_domain_index_disables_the_effort_select(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            return fake_catalogue

        monkeypatch.setattr(models_views, 'load_models', fake_load)
        text = client.get('/models?index=legal&effort=high').text
        assert 'name="effort" onchange="this.form.submit()" disabled' in text
        assert '0 on the frontier, 0 dominated, 5 not plotted.' in text
