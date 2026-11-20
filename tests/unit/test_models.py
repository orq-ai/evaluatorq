"""`evaluatorq.models`: card parsing, dominance, the frontier, one-per-model, the CLI and the dashboard page."""

from __future__ import annotations

import json
import random
from typing import Any, get_args

import pytest
from typer.testing import CliRunner

from evaluatorq import models as models_mod
from evaluatorq.common import model_frontier
from evaluatorq.common.model_frontier import (
    INDEX_AXES,
    X_AXIS_SPECS,
    ScoredModel,
    available_efforts,
    available_indices,
    matches_vendor,
    model_vendor,
    scored_model_from_card,
)
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

    def test_lower_time_to_first_token_is_better(self) -> None:
        quick = model('a/quick', 30.0, 9.0, 9.0, time_to_first_token_seconds=0.4)
        laggy = model('b/laggy', 30.0, 0.1, 0.1, time_to_first_token_seconds=2.5)
        unmeasured = model('c/unmeasured', 90.0, 0.1, 0.1)
        assert dominates(quick, laggy, x='time_to_first_token')
        assert not dominates(laggy, quick, x='time_to_first_token')
        result = pareto([quick, laggy, unmeasured], x='time_to_first_token')
        assert [p.model.router_id for p in result.frontier] == ['a/quick']
        assert result.points[1].dominated_by == 'a/quick'
        assert result.points[1].pareto_score is None
        assert [(e.router_id, e.reason) for e in result.excluded] == [('c/unmeasured', 'no time to first token data')]

    def test_unknown_index_is_refused_for_a_pair(self) -> None:
        with pytest.raises(ValueError, match='unknown index'):
            dominates(MID, BAD, index='vibes')  # ty: ignore[invalid-argument-type]


def _old_dominates(a: ScoredModel, b: ScoredModel) -> bool:
    """`ScoredModel.dominates` as it was before it delegated to the shared kernel."""
    if a.free or b.free:
        return False
    at_least_as_good = a.default_index >= b.default_index and a.price <= b.price
    return at_least_as_good and (a.default_index > b.default_index or a.price < b.price)


def _old_frontier(models: list[ScoredModel]) -> list[ScoredModel]:
    """`common.model_frontier.pareto_frontier` as it was before it read the shared kernel."""
    comparable = [m for m in models if not m.free]
    frontier = [m for m in comparable if not any(_old_dominates(o, m) for o in comparable)]
    return sorted(frontier, key=lambda m: -m.default_index)


class TestKernelAgreement:
    """Seating (router pools, juries, in-family upgrades) reads one kernel; it must decide as the old rule did."""

    @pytest.mark.parametrize('seed', range(5))
    def test_seating_rule_matches_the_old_one(self, seed: int) -> None:
        rng = random.Random(seed)
        garden = []
        for i in range(40):
            # A coarse grid forces ties on both axes; a few free cards and effort ladders ride along.
            index = rng.choice([10.0, 20.0, 20.0, 30.0, 40.0])
            ladder = {'low': index - 5, 'high': index + 5} if rng.random() < 0.3 else {}
            garden.append(
                model(
                    f'p{i % 4}/m{i}',
                    index,
                    rng.choice([0.0, 0.5, 1.0, 1.0, 2.0]),
                    rng.choice([0.0, 1.0, 2.0, 2.0, 4.0]),
                    effort_parameter=bool(ladder),
                    default_effort=rng.choice(['low', 'high']) if ladder else None,
                    effort_indices=ladder,
                )
            )
        for a in garden:
            for b in garden:
                assert a.dominates(b) is _old_dominates(a, b)
                assert dominates(a, b) is _old_dominates(a, b)
        assert model_frontier.pareto_frontier(garden) == _old_frontier(garden)
        assert {m.router_id for m in pareto_frontier(garden)} == {p.model.router_id for p in pareto(garden).frontier}


class TestAxisTable:
    def test_one_row_per_literal_value(self) -> None:
        assert tuple(INDEX_AXES) == get_args(models_mod.Index)
        assert tuple(X_AXIS_SPECS) == get_args(models_mod.XAxis)

    def test_direction_and_format(self) -> None:
        assert [k for k, s in X_AXIS_SPECS.items() if s.lower_is_better] == ['price', 'time_to_first_token']
        assert X_AXIS_SPECS['price'].format(0.4375) == '0.438'
        assert X_AXIS_SPECS['output_speed'].format(None) == '—'

    def test_available_indices_and_efforts_come_from_the_models(self) -> None:
        laddered = model('a/l', 30.0, 1.0, 1.0, effort_indices={'high': 40.0, 'low': 20.0}, category_indices={'legal': 3.0})
        flat = model('b/f', 30.0, 1.0, 1.0, headline_effort='medium')
        assert available_indices([laddered, flat]) == ['intelligence', 'legal']
        assert available_indices([]) == ['intelligence']
        assert available_efforts([laddered, flat]) == ['low', 'medium', 'high']


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

    def test_default_floor_excludes_low_scorers_with_the_reason(self) -> None:
        junk = model('z/junk', 2.7, 0.05, 0.05)
        result = pareto([STRONG, MID, CHEAP, junk])
        assert result.min_score == model_frontier.MIN_SEAT_INDEX == 15.0
        assert [p.model.router_id for p in result.frontier] == ['a/strong', 'b/mid', 'c/cheap']
        [gone] = result.excluded
        assert gone.router_id == 'z/junk'
        assert gone.reason == 'score 2.7 is below min_score 15'
        assert gone.below_min_score is True
        assert result.below_min_score == [gone]

    def test_floor_zero_keeps_every_model(self) -> None:
        junk = model('z/junk', 2.7, 0.05, 0.05)
        result = pareto([STRONG, MID, CHEAP, junk], min_score=0)
        assert [p.model.router_id for p in result.frontier] == ['a/strong', 'b/mid', 'c/cheap', 'z/junk']
        assert result.excluded == []

    def test_floor_applies_to_the_chosen_domain_index(self) -> None:
        weak_legal = model('a/wl', 50.0, 1.0, 1.0, category_indices={'legal': 9.0})
        strong_legal = model('b/sl', 10.0, 1.0, 1.0, category_indices={'legal': 30.0})
        result = pareto([weak_legal, strong_legal], index='legal')
        assert [p.model.router_id for p in result.points] == ['b/sl']
        assert result.excluded[0].reason == 'score 9 is below min_score 15'

    def test_missing_axis_reason_wins_over_the_floor(self) -> None:
        free_junk = model('z/free', 2.0, 0.0, 0.0)
        [gone] = pareto([free_junk]).excluded
        assert (gone.reason, gone.below_min_score) == ('free (no list price)', False)

    def test_dominated_by_ignores_the_floor(self) -> None:
        junk = model('z/junk', 2.7, 0.5, 0.5)
        dearer_junk = model('z/dear', 2.0, 9.0, 9.0)
        assert dominated_by(dearer_junk, [junk, STRONG]) == junk


class TestVendor:
    def test_one_key_for_host_and_developer_spellings(self) -> None:
        gemini = model('google-ai/gemini', 40.0, 1.0, 1.0, developer='Google')
        assert model_vendor(gemini) == 'google'
        assert matches_vendor(gemini, ['google'])
        assert matches_vendor(gemini, ['google-ai'])
        assert not matches_vendor(gemini, ['openai', ''])

    def test_host_or_developer_matches(self) -> None:
        azure_gpt = model('azure/gpt-x', 40.0, 1.0, 1.0, developer='openai')
        assert matches_vendor(azure_gpt, ['azure'])
        assert matches_vendor(azure_gpt, ['OpenAI'])


class TestSummary:
    def test_summary_is_the_model_and_the_point(self) -> None:
        [point] = pareto([MID]).points
        row = point.summary()
        assert set(ScoredModel.model_fields) <= set(row)
        for key in ('price', 'score', 'x', 'pareto_score', 'on_frontier', 'dominated_by', 'dominators'):
            assert key in row
        assert 'model' not in row
        assert row['price'] == MID.price
        assert row['pareto_score'] == pytest.approx(20.0)
        json.dumps(row)


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


def _flat(output: str) -> str:
    """Rich wraps a usage error inside a box; read it as one line."""
    return ' '.join(output.replace('│', ' ').split())


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
        assert payload['min_score'] == 15.0
        assert payload['excluded'] == [{'router_id': 'e/free', 'reason': 'free (no list price)', 'below_min_score': False}]

    def test_min_score_hides_and_zero_shows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        junk = model('z/junk', 2.7, 0.05, 0.05)

        async def fake_load(client: object = None, *, include_disabled: bool = False, all_hosts: bool = False) -> list[ScoredModel]:
            del client, include_disabled, all_hosts
            return [STRONG, MID, junk]

        monkeypatch.setattr(models_mod, 'load_models', fake_load)
        table = self._run()
        assert table.exit_code == 0, table.output
        assert 'z/junk' not in table.stdout
        assert '1 model(s) hidden for scoring below --min-score 15. Pass --min-score 0 to show every model.' in _flat(table.stdout)
        assert 'not compared' not in table.stdout
        shown = json.loads(self._run('--min-score', '0', '--json').stdout)
        assert 'z/junk' in [p['router_id'] for p in shown['points']]
        assert shown['excluded'] == []
        named = self._run('--model', 'junk')
        assert 'z/junk is not plotted: score 2.7 is below min_score 15.' in _flat(named.output)

    def test_negative_min_score_is_a_usage_error(self, fake_catalogue: list[ScoredModel]) -> None:
        assert self._run('--min-score', '-1').exit_code == 2

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

    def test_index_no_loaded_model_has_is_a_usage_error(self, fake_catalogue: list[ScoredModel]) -> None:
        result = self._run('--index', 'coding')
        assert result.exit_code == 2
        assert 'no model in the catalogue has a coding score' in _flat(result.output)

    def test_effort_no_loaded_model_has_is_a_usage_error(self, fake_catalogue: list[ScoredModel]) -> None:
        result = self._run('--effort', 'turbo')
        assert result.exit_code == 2
        assert "no model in the catalogue has a score at effort 'turbo'" in _flat(result.output)

    def test_effort_a_loaded_model_has_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        laddered = model('a/l', 30.0, 1.0, 1.0, effort_indices={'high': 45.0})

        async def fake_load(client: object = None, *, include_disabled: bool = False, all_hosts: bool = False) -> list[ScoredModel]:
            del client, include_disabled, all_hosts
            return [laddered, MID]

        monkeypatch.setattr(models_mod, 'load_models', fake_load)
        payload = json.loads(self._run('--effort', 'high', '--json').stdout)
        assert [(p['router_id'], p['score']) for p in payload['points']] == [('a/l', 45.0)]

    def test_provider_filter_matches_host_or_developer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        azure_gpt = model('azure/gpt-x', 40.0, 1.0, 1.0, developer='openai')
        gemini = model('google-ai/gemini', 40.0, 1.0, 1.0, developer='google')

        async def fake_load(client: object = None, *, include_disabled: bool = False, all_hosts: bool = False) -> list[ScoredModel]:
            del client, include_disabled, all_hosts
            return [azure_gpt, gemini, MID]

        monkeypatch.setattr(models_mod, 'load_models', fake_load)
        payload = json.loads(self._run('--provider', 'openai', '--provider', 'google', '--json').stdout)
        assert sorted(p['router_id'] for p in payload['points']) == ['azure/gpt-x', 'google-ai/gemini']

    def test_table_goes_to_stdout_with_the_axis_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        timed = model('a/t', 30.0, 1.0, 1.0, time_to_first_token_seconds=0.75)

        async def fake_load(client: object = None, *, include_disabled: bool = False, all_hosts: bool = False) -> list[ScoredModel]:
            del client, include_disabled, all_hosts
            return [timed, MID]

        monkeypatch.setattr(models_mod, 'load_models', fake_load)
        result = self._run('--x', 'time_to_first_token')
        assert result.exit_code == 0, result.output
        assert 'TTFT s' in result.stdout
        assert '0.75' in result.stdout
        assert 'a/t' in result.stdout


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

        fake_catalogue.append(model('f/legal', 30.0, 1.0, 1.0, category_indices={'legal': 50.0}))
        monkeypatch.setattr(models_views, 'load_models', fake_load)
        text = client.get('/models?index=legal&effort=high').text
        assert 'name="effort" onchange="this.form.submit()" disabled' in text
        assert '1 on the frontier, 0 dominated, 5 not plotted.' in text

    def test_min_score_control_hides_and_zero_shows(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            return fake_catalogue

        fake_catalogue.append(model('z/junk', 2.7, 0.05, 0.05))
        monkeypatch.setattr(models_views, 'load_models', fake_load)
        text = client.get('/models').text
        assert 'name="min_score" min="0" step="any" value="15"' in text
        assert '3 on the frontier, 1 dominated, 2 not plotted (1 below the minimum score of 15).' in text
        assert 'score 2.7 is below min_score 15' in text
        text = client.get('/models?min_score=0').text
        assert 'value="0"' in text
        assert '4 on the frontier, 1 dominated, 1 not plotted.' in text
        assert 'value="15"' in client.get('/models?min_score=nope').text

    def test_index_dropdown_lists_only_indices_a_model_has(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            return fake_catalogue

        fake_catalogue.append(model('f/legal', 30.0, 1.0, 1.0, category_indices={'legal': 50.0}))
        monkeypatch.setattr(models_views, 'load_models', fake_load)
        text = client.get('/models?index=coding').text
        assert '<option value="legal">Legal</option>' in text
        assert 'value="coding"' not in text
        assert 'value="economics"' not in text
        assert '<option value="intelligence" selected>' in text  # an empty index falls back to the default
