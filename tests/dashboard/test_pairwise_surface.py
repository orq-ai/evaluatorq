"""Tests for the pairwise dashboard surface.

Covers the persisted run artifact, the file-scanner sniff, the surface
adapter, and the rendered sections.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id, scan, sniff_kind
from evaluatorq.dashboard.surfaces import ADAPTERS
from evaluatorq.pairwise import PairwiseComparison, PairwiseVote
from evaluatorq.pairwise_reports.export_html import render_report_body
from evaluatorq.pairwise_reports.sections import build_report_sections
from evaluatorq.pairwise_run import PairwiseRun, new_run

JUDGES = ['judge-one', 'judge-two', 'judge-three']


def _vote(
    model: str,
    value: str | None,
    *,
    flipped: bool = False,
    replacement: bool = False,
    completed: bool = True,
) -> PairwiseVote:
    return PairwiseVote(
        model=model,
        vote=value,
        flipped=flipped,
        completed=completed,
        replacement=replacement,
        explanation=f'{model} says {value}',
    )


def _comparison(values: list[str | None], winner: str, **kw: bool) -> PairwiseComparison:
    return PairwiseComparison(
        winner=winner,
        votes=[_vote(m, v, **kw) for m, v in zip(JUDGES, values)],
    )


@pytest.fixture
def run() -> PairwiseRun:
    """A run with a decided pair, a split pair, a tie and an inconclusive pair."""
    r = new_run(
        run_name='v2 vs v3',
        label_a='prompt-v2',
        label_b='prompt-v3',
        judges=JUDGES,
        criteria='accuracy',
    )
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='q1', response_a='a1', response_b='b1')
    r.add(_comparison(['A', 'B', 'tie'], 'tie'), question='q2', response_a='a2', response_b='b2')
    r.add(_comparison(['B', 'B', 'B'], 'B'), question='q3', response_a='a3', response_b='b3')
    r.add(
        _comparison([None, None, None], 'inconclusive', flipped=True), question='q4', response_a='a4', response_b='b4'
    )
    return r


@pytest.fixture
def saved(run: PairwiseRun, tmp_path: Path) -> tuple[Path, Path]:
    """The run written into a runs dir; returns (runs_dir, file)."""
    runs_dir = tmp_path / 'pairwise-runs'
    runs_dir.mkdir()
    return runs_dir, run.save(runs_dir / 'run.json')


# ---------------------------------------------------------------------------
# Run artifact
# ---------------------------------------------------------------------------


def test_save_round_trips_through_json(saved: tuple[Path, Path], run: PairwiseRun) -> None:
    _, path = saved
    loaded = PairwiseRun.load(path)
    assert loaded.run_name == run.run_name
    assert loaded.label_a == 'prompt-v2'
    assert loaded.label_b == 'prompt-v3'
    assert len(loaded.entries) == 4
    assert loaded.entries[0].question == 'q1'
    assert loaded.entries[0].comparison.winner == 'A'


def test_save_computes_and_stores_the_rollup(saved: tuple[Path, Path]) -> None:
    _, path = saved
    loaded = PairwiseRun.load(path)
    assert loaded.report is not None
    assert loaded.report.comparisons == 4
    # Two decided comparisons: one A, one B.
    assert loaded.report.a_win_rate == pytest.approx(0.5)
    assert loaded.report.b_win_rate == pytest.approx(0.5)
    assert loaded.report.tie_rate == pytest.approx(0.25)
    assert loaded.report.inconclusive_rate == pytest.approx(0.25)


def test_label_defaults_are_the_slot_letters() -> None:
    r = new_run(run_name='unnamed sides')
    assert (r.label_a, r.label_b) == ('A', 'B')


def test_rollup_recomputes_when_no_report_is_stored(run: PairwiseRun) -> None:
    assert run.report is None
    assert run.rollup().comparisons == 4


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def test_default_save_path_does_not_overwrite_a_same_second_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp only resolves to the second, so two same-named runs collide."""
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    first = new_run(run_name='same name')
    first.add(_comparison(['A', 'A', 'A'], 'A'), question='q1', response_a='a', response_b='b')
    second = new_run(run_name='same name')
    second.add(_comparison(['B', 'B', 'B'], 'B'), question='q2', response_a='a', response_b='b')
    second.created_at = first.created_at

    p1 = first.save()
    p2 = second.save()
    assert p1 != p2
    assert PairwiseRun.load(p1).entries[0].question == 'q1'
    assert PairwiseRun.load(p2).entries[0].question == 'q2'


def test_sniff_identifies_a_pairwise_file() -> None:
    assert sniff_kind({'judging': 'pairwise'}) == 'pairwise'


def test_sniff_leaves_the_existing_surfaces_alone() -> None:
    assert sniff_kind({'mode': 'run'}) == 'sim'
    assert sniff_kind({'pipeline': {}}) == 'redteam'
    assert sniff_kind({'something': 'else'}) is None


def test_scan_lists_the_run_with_a_comparison_headline(saved: tuple[Path, Path]) -> None:
    runs_dir, path = saved
    cards = [c for c in scan([runs_dir]) if c.surface == 'pairwise']
    assert len(cards) == 1
    assert cards[0].name == 'v2 vs v3'
    assert cards[0].headline == '4 comparisons'
    assert cards[0].error is None
    assert cards[0].id == report_id(path)


def test_scan_flags_a_run_missing_its_entries(tmp_path: Path) -> None:
    runs_dir = tmp_path / 'pairwise-runs'
    runs_dir.mkdir()
    (runs_dir / 'broken.json').write_text(json.dumps({'judging': 'pairwise', 'run_name': 'broken'}))
    card = next(c for c in scan([runs_dir]) if c.surface == 'pairwise')
    assert card.error == 'missing required field: entries'


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def test_adapter_reads_name_and_timestamp(saved: tuple[Path, Path], run: PairwiseRun) -> None:
    _, path = saved
    adapter = ADAPTERS['pairwise']
    loaded = adapter.load(path)
    assert adapter.name(loaded) == 'v2 vs v3'
    assert adapter.created_at(loaded) == run.created_at


def test_adapter_rows_resolve_every_vote_to_a_label(run: PairwiseRun) -> None:
    adapter = ADAPTERS['pairwise']
    assert adapter.rows is not None
    rows = adapter.rows(run, run.entries)
    assert len(rows) == 4
    assert rows[0]['Question'] == 'q1'
    assert rows[0]['Winner'] == 'prompt-v2'
    assert rows[0]['judge-one'] == 'prompt-v2'
    assert rows[2]['Winner'] == 'prompt-v3'
    assert rows[3]['Winner'] == 'inconclusive'
    assert rows[3]['judge-one'] == 'abstained'


def test_adapter_rows_keep_identical_keys_when_a_judge_joins_late() -> None:
    """The CSV export takes its fieldnames from the first row, so a replacement
    judge promoted on a later comparison must not widen that row's key set."""
    r = new_run(run_name='late judge', judges=['j1'])
    r.add(
        PairwiseComparison(winner='A', votes=[_vote('j1', 'A')]),
        question='q1',
        response_a='a',
        response_b='b',
    )
    r.add(
        PairwiseComparison(winner='A', votes=[_vote('j1', 'A'), _vote('stand-in', 'A', replacement=True)]),
        question='q2',
        response_a='a',
        response_b='b',
    )
    adapter = ADAPTERS['pairwise']
    assert adapter.rows is not None
    rows = adapter.rows(r, r.entries)
    assert [sorted(x) for x in rows] == [sorted(rows[0])] * 2
    assert rows[0]['stand-in'] == ''


def test_csv_export_survives_a_judge_promoted_mid_run(tmp_path: Path) -> None:
    r = new_run(run_name='late judge', judges=['j1'])
    r.add(PairwiseComparison(winner='A', votes=[_vote('j1', 'A')]), question='q1', response_a='a', response_b='b')
    r.add(
        PairwiseComparison(winner='A', votes=[_vote('j1', 'A'), _vote('stand-in', 'A', replacement=True)]),
        question='q2',
        response_a='a',
        response_b='b',
    )
    runs_dir = tmp_path / 'pairwise-runs'
    runs_dir.mkdir()
    path = r.save(runs_dir / 'run.json')
    c = TestClient(build_app(roots=[runs_dir]), raise_server_exceptions=True)
    resp = c.get(f'/r/{report_id(path)}/export.csv')
    assert resp.status_code == 200
    assert 'stand-in' in resp.text


def test_adapter_body_from_results_rescopes_the_rollup(run: PairwiseRun) -> None:
    """A filtered body must not report the full run's numbers."""
    adapter = ADAPTERS['pairwise']
    body = adapter.body_from_results(run, run.entries[:1])
    assert 'the 1 of 1 comparisons' in body


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def test_sections_are_built_in_display_order(run: PairwiseRun) -> None:
    kinds = [s.kind for s in build_report_sections(run)]
    assert kinds == ['pairwise_consensus', 'pairwise_judges', 'pairwise_comparisons']


def test_consensus_reports_the_decided_count_not_the_total(run: PairwiseRun) -> None:
    consensus = build_report_sections(run)[0]
    assert consensus.data['comparisons'] == 4
    assert consensus.data['decided'] == 2


def test_decided_count_is_exact_when_the_rates_do_not_divide_evenly() -> None:
    """Counted from the entries, so thirds cannot round the caption off by one."""
    r = new_run(run_name='thirds', judges=JUDGES)
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='q1', response_a='a', response_b='b')
    r.add(_comparison(['A', 'B', 'tie'], 'tie'), question='q2', response_a='a', response_b='b')
    r.add(_comparison([None, None, None], 'inconclusive'), question='q3', response_a='a', response_b='b')
    consensus = build_report_sections(r)[0]
    assert consensus.data['comparisons'] == 3
    assert consensus.data['decided'] == 1


def test_split_and_inconclusive_comparisons_are_marked(run: PairwiseRun) -> None:
    rows = build_report_sections(run)[2].data['rows']
    assert [r['split'] for r in rows] == [False, True, False, True]


def test_judge_columns_keep_a_fixed_order_across_rows(run: PairwiseRun) -> None:
    section = build_report_sections(run)[2]
    assert section.data['judges'] == JUDGES
    for row in section.data['rows']:
        assert [v['model'] for v in row['votes']] == JUDGES


def test_position_bias_is_unavailable_when_the_run_did_not_swap() -> None:
    r = new_run(run_name='single ordering', judges=JUDGES, swap=False)
    # completed=False throughout: pairwise.py cannot mark a pair complete when
    # only one ordering ran, so a swap=False run never carries a completed vote.
    r.add(_comparison(['A', 'A', 'A'], 'A', completed=False), question='q', response_a='a', response_b='b')
    rows = build_report_sections(r)[1].data['rows']
    # None, not 0.0: nothing was flippable, so there is no measurement to show.
    assert all(row['position_bias'] is None for row in rows)
    assert all(row['biased'] is False for row in rows)


def test_position_bias_is_unavailable_when_no_pair_was_ever_flippable() -> None:
    """swap=True is not enough: both orderings still have to have landed.

    position_bias divides flips by completed pairs and falls back to 0.0 on an
    empty denominator, which would read as a perfectly steady judge.
    """
    r = new_run(run_name='swap on, nothing completed', judges=['j1'], swap=True)
    r.add(
        PairwiseComparison(
            winner='A',
            votes=[PairwiseVote(model='j1', vote='A', completed=False, explanation='')],
        ),
        question='q',
        response_a='a',
        response_b='b',
    )
    row = build_report_sections(r)[1].data['rows'][0]
    assert row['position_bias'] is None
    assert row['biased'] is False
    assert 'n/a' in render_report_body(r)


def test_position_bias_is_flagged_above_the_threshold() -> None:
    r = new_run(run_name='biased panel', judges=['flipper'])
    for i in range(4):
        flipped = i < 2
        r.add(
            PairwiseComparison(winner='A', votes=[_vote('flipper', 'A', flipped=flipped)]),
            question=f'q{i}',
            response_a='a',
            response_b='b',
        )
    row = build_report_sections(r)[1].data['rows'][0]
    assert row['position_bias'] == pytest.approx(0.5)
    assert row['biased'] is True


def test_replacement_judges_are_tagged() -> None:
    r = new_run(run_name='with stand-in', judges=['primary'])
    r.add(
        PairwiseComparison(winner='A', votes=[_vote('stand-in', 'A', replacement=True)]),
        question='q',
        response_a='a',
        response_b='b',
    )
    row = build_report_sections(r)[1].data['rows'][0]
    assert row['replacement'] is True


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_body_uses_the_side_labels_not_the_slot_letters(run: PairwiseRun) -> None:
    body = render_report_body(run)
    assert 'prompt-v2 rate' in body
    assert 'prompt-v3 rate' in body
    assert '>A rate<' not in body


def test_body_renders_every_section(run: PairwiseRun) -> None:
    body = render_report_body(run)
    for anchor in ('pairwise_consensus', 'pairwise_judges', 'pairwise_comparisons'):
        assert f'id="{anchor}"' in body


def test_detail_includes_both_responses_and_every_explanation(run: PairwiseRun) -> None:
    body = render_report_body(run)
    assert 'a2' in body
    assert 'b2' in body
    for judge in JUDGES:
        assert f'{judge} says' in body


def test_abstention_renders_distinctly_from_a_decided_vote(run: PairwiseRun) -> None:
    body = render_report_body(run)
    assert 'pw-chip--abstain' in body


def test_single_ordering_run_says_bias_was_not_measured() -> None:
    r = new_run(run_name='single ordering', judges=JUDGES, swap=False)
    r.add(_comparison(['A', 'A', 'A'], 'A', completed=False), question='q', response_a='a', response_b='b')
    body = render_report_body(r)
    assert 'n/a' in body
    assert 'position bias was not measured' in body


def test_empty_run_renders_without_error() -> None:
    body = render_report_body(new_run(run_name='nothing here'))
    assert 'No comparisons in this run.' in body


def test_fully_inconclusive_run_renders_without_error() -> None:
    r = new_run(run_name='all noise', judges=JUDGES)
    r.add(_comparison([None, None, None], 'inconclusive'), question='q', response_a='a', response_b='b')
    body = render_report_body(r)
    # No decided comparison, so both win rates are None and render as a dash.
    assert 'the 0 of 1 comparisons' in body


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def client(saved: tuple[Path, Path]) -> tuple[TestClient, str]:
    runs_dir, path = saved
    return TestClient(build_app(roots=[runs_dir]), raise_server_exceptions=True), report_id(path)


def test_report_route_renders_the_run(client: tuple[TestClient, str]) -> None:
    c, rid = client
    r = c.get(f'/r/{rid}')
    assert r.status_code == 200
    assert 'prompt-v2' in r.text
    assert 'Comparisons' in r.text


def test_run_list_route_renders(client: tuple[TestClient, str]) -> None:
    c, _ = client
    r = c.get('/?surface=pairwise')
    assert r.status_code == 200
    assert 'v2 vs v3' in r.text


def test_run_list_route_is_titled_pairwise(client: tuple[TestClient, str]) -> None:
    # Regression: view.SURFACE_LABELS lacked a 'pairwise' entry, so the page
    # fell back to the generic 'Reports' title.
    c, _ = client
    r = c.get('/?surface=pairwise')
    assert '<h1 class="app-title">Pairwise</h1>' in r.text
    assert '<title>Pairwise' in r.text


def test_empty_pairwise_list_names_the_surface(tmp_path: Path) -> None:
    c = TestClient(build_app(roots=[tmp_path]), raise_server_exceptions=True)
    r = c.get('/?surface=pairwise')
    assert r.status_code == 200
    assert 'Run a pairwise job to generate one.' in r.text


def test_pairwise_type_cell_has_a_glyph(client: tuple[TestClient, str]) -> None:
    # Regression: _SURFACE_ICONS lacked a 'pairwise' entry, so the landing
    # "Recent runs" Type cell rendered the label with no icon.
    c, _ = client
    r = c.get('/')
    assert '<span class="type-cell pairwise"><svg' in r.text
    assert 'Pairwise' in r.text


def test_empty_landing_names_pairwise(tmp_path: Path) -> None:
    c = TestClient(build_app(roots=[tmp_path]), raise_server_exceptions=True)
    r = c.get('/')
    assert 'Run a red team, simulation, or pairwise job to generate reports.' in r.text


def test_export_routes_all_succeed(client: tuple[TestClient, str]) -> None:
    c, rid = client
    for suffix in ('export.html', 'export.json', 'export.csv'):
        resp = c.get(f'/r/{rid}/{suffix}')
        assert resp.status_code == 200, suffix
        assert resp.text


def test_back_link_returns_to_the_pairwise_run_list(client: tuple[TestClient, str]) -> None:
    c, rid = client
    r = c.get(f'/r/{rid}')
    assert 'href="/?surface=pairwise"' in r.text
    assert 'Pairwise runs' in r.text


def test_landing_omits_a_kind_with_no_runs(saved: tuple[Path, Path]) -> None:
    """by_kind drops zeros, matching tokens_by_kind / cost_by_kind."""
    from evaluatorq.dashboard import metrics

    runs_dir, _ = saved
    landing = metrics.landing([runs_dir])
    # Only pairwise runs exist in this root, so it is the only bar.
    assert landing.by_kind == [('Pairwise', 1)]


def test_no_filter_rail_is_rendered_for_pairwise(client: tuple[TestClient, str]) -> None:
    """Pairwise has no dimensions, so an empty 'Filters' panel must not appear."""
    c, rid = client
    r = c.get(f'/r/{rid}')
    # 'filter-rail' also appears in the always-inlined stylesheet, so assert on
    # the rail's markup and its POST wiring rather than the bare class name.
    assert 'class="filter-rail"' not in r.text
    assert 'hx-post' not in r.text


def test_csv_export_neutralises_formula_injection(tmp_path: Path) -> None:
    """A question starting with '=' must not execute when the CSV opens in Excel."""
    r = new_run(run_name='injection', judges=JUDGES)
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='=cmd|/c calc', response_a='a', response_b='b')
    runs_dir = tmp_path / 'pairwise-runs'
    runs_dir.mkdir()
    path = r.save(runs_dir / 'run.json')
    c = TestClient(build_app(roots=[runs_dir]), raise_server_exceptions=True)

    body = c.get(f'/r/{report_id(path)}/export.csv').text
    assert "'=cmd|/c calc" in body
    assert ',=cmd' not in body


def test_csv_export_leaves_the_na_placeholder_alone() -> None:
    """'-' and signed numbers are not formulas; quoting them would corrupt exports."""
    from evaluatorq.dashboard.app import _csv_safe

    assert _csv_safe('-') == '-'
    assert _csv_safe('-1.5') == '-1.5'
    assert _csv_safe('+7') == '+7'
    assert _csv_safe('normal text') == 'normal text'
    assert _csv_safe(3) == 3
    assert _csv_safe('@SUM(A1)') == "'@SUM(A1)"


# ---------------------------------------------------------------------------
# Swap is derived from the votes, not trusted from the flag
# ---------------------------------------------------------------------------


def _incomplete(models: list[str]) -> PairwiseComparison:
    """A comparison where no judge completed both orderings."""
    return PairwiseComparison(
        winner='inconclusive',
        votes=[
            PairwiseVote(model=m, vote=None, flipped=False, completed=False, replacement=False, explanation='')
            for m in models
        ],
    )


def test_swap_flag_is_qualified_when_never_observed() -> None:
    """A run saved with the default swap=True but executed single-ordering."""
    from evaluatorq.pairwise_reports.sections import observed_swap

    r = new_run(run_name='mislabelled', judges=JUDGES, swap=True)
    r.add(_incomplete(JUDGES), question='q', response_a='a', response_b='b')

    assert observed_swap(r) is False
    html = render_report_body(r)
    # Header must not claim a plain "on" the votes never bore out.
    assert 'never observed' in html
    # And the judges table must blame the run, not each judge in turn.
    assert 'No pair completed both orderings' in html
    assert 'This judge never completed' not in html


def test_swap_observed_renders_plain_on() -> None:
    from evaluatorq.pairwise_reports.sections import observed_swap

    r = new_run(run_name='swapped', judges=JUDGES, swap=True)
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='q', response_a='a', response_b='b')

    assert observed_swap(r) is True
    html = render_report_body(r)
    assert 'never observed' not in html


def test_single_judge_unmeasured_blames_the_judge_not_the_run() -> None:
    """One judge completed a pair, another did not: that is a per-judge fact."""
    mixed = PairwiseComparison(
        winner='A',
        votes=[
            PairwiseVote(
                model='judge-one', vote='A', flipped=False, completed=True, replacement=False, explanation='x'
            ),
            PairwiseVote(
                model='judge-two', vote='A', flipped=False, completed=False, replacement=False, explanation='y'
            ),
        ],
    )
    r = new_run(run_name='mixed', judges=['judge-one', 'judge-two'], swap=True)
    r.add(mixed, question='q', response_a='a', response_b='b')

    html = render_report_body(r)
    assert 'This judge never completed' in html
    assert 'No pair completed both orderings' not in html


def test_bias_is_gated_on_the_votes_not_the_flag() -> None:
    """swap=False leaves every vote incomplete, so bias is n/a either way."""
    from evaluatorq.pairwise_reports.sections import build_report_sections

    r = new_run(run_name='single-ordering', judges=JUDGES, swap=False)
    r.add(_incomplete(JUDGES), question='q', response_a='a', response_b='b')

    judges = next(s for s in build_report_sections(r) if s.kind == 'pairwise_judges')
    assert judges.data['observed_swap'] is False
    assert all(row['position_bias'] is None for row in judges.data['rows'])
    assert all(row['biased'] is False for row in judges.data['rows'])
    assert 'single ordering' in render_report_body(r)


# ---------------------------------------------------------------------------
# Landing: score metric, column labelling, stable colours
# ---------------------------------------------------------------------------


def test_pairwise_score_is_the_decided_share(saved: tuple[Path, Path]) -> None:
    """Not mean_agreement: that is quantized by panel size against a fixed 0.8."""
    from evaluatorq.dashboard import metrics

    runs_dir, _ = saved
    (row,) = metrics.run_rows([runs_dir])
    assert row.surface == 'pairwise'
    # Fixture: 4 comparisons, 1 inconclusive.
    assert row.score == pytest.approx(0.75)


def test_score_column_names_its_metric_per_surface(client: tuple[TestClient, str]) -> None:
    from evaluatorq.dashboard.view import _score_title

    c, _ = client
    assert 'Decided rate' in _score_title('pairwise')
    assert 'Resistance rate' in _score_title('redteam')
    assert _score_title('unknown') == ''
    assert 'Decided rate' in c.get('/?surface=pairwise').text


def test_kind_colours_are_stable_when_kinds_are_absent() -> None:
    """by_kind drops zeros, so positional colours would shift per workspace."""
    from evaluatorq.dashboard.view import _kind_colors

    both = _kind_colors([('Red team', 1), ('Pairwise', 1)])
    alone = _kind_colors([('Pairwise', 1)])
    assert alone == [both[1]]
    assert both[0] != both[1]


# ---------------------------------------------------------------------------
# Smaller findings
# ---------------------------------------------------------------------------


def test_csv_header_escaping_covers_a_hostile_judge_name(tmp_path: Path) -> None:
    """Judge names become column headers, so they need the same guard as cells."""
    evil = '=evil'
    r = new_run(run_name='hostile-judge', judges=[evil])
    r.add(
        PairwiseComparison(
            winner='A',
            votes=[
                PairwiseVote(model=evil, vote='A', flipped=False, completed=True, replacement=False, explanation='x')
            ],
        ),
        question='q',
        response_a='a',
        response_b='b',
    )
    runs_dir = tmp_path / 'pairwise-runs'
    runs_dir.mkdir()
    path = r.save(runs_dir / 'run.json')
    c = TestClient(build_app(roots=[runs_dir]), raise_server_exceptions=True)

    header = c.get(f'/r/{report_id(path)}/export.csv').text.splitlines()[0]
    assert "'=evil" in header
    assert ',=evil' not in header


def test_filtered_view_rolls_up_once(saved: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """The header and two section builders each call rollup()."""
    import evaluatorq.pairwise as pw_mod
    from evaluatorq.dashboard.surfaces import ADAPTERS

    _, path = saved
    run = ADAPTERS['pairwise'].load(path)

    calls = 0
    real = pw_mod.build_report

    def counting(comparisons: list[PairwiseComparison]) -> object:
        nonlocal calls
        calls += 1
        return real(comparisons)

    monkeypatch.setattr(pw_mod, 'build_report', counting)
    body = ADAPTERS['pairwise'].body_from_results(run, list(run.entries[:2]))

    assert calls == 1, f'rolled up {calls} times'
    assert body


def test_long_run_name_stays_within_the_filename_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A long name must not push the filename past the filesystem's limit.

    ``save()`` runs after the judging is already paid for, so a name that fails
    with ENAMETOOLONG would lose a completed run.
    """
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    # 'ab ' repeats so the 64-byte cut lands exactly on a joining hyphen, which
    # is the case the trailing-dash strip exists for.
    r = new_run(run_name='ab ' * 40, judges=JUDGES)
    r.add(
        PairwiseComparison(winner='A', votes=[_vote(m, 'A') for m in JUDGES]),
        question='q',
        response_a='a',
        response_b='b',
    )
    path = r.save()
    assert len(path.name.encode()) <= 255
    assert not path.stem.endswith('-')
    assert PairwiseRun.load(path).run_name == r.run_name


def test_a_multibyte_run_name_is_clamped_on_bytes_not_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The filesystem limit counts bytes, and isalnum() admits multibyte text.

    A 64-*character* clamp lets four-byte alphanumerics reach 256 bytes, which
    is the ENAMETOOLONG this guard exists to prevent.
    """
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    r = new_run(run_name='𠮷' * 80, judges=JUDGES)
    r.add(
        PairwiseComparison(winner='A', votes=[_vote(m, 'A') for m in JUDGES]),
        question='q',
        response_a='a',
        response_b='b',
    )
    path = r.save()
    assert len(path.name.encode()) <= 255
    # Cut on a character boundary, not mid-codepoint.
    assert '�' not in path.stem


def test_a_long_question_is_readable_and_written_once() -> None:
    """The summary line wraps instead of clipping, so no second copy is needed.

    Repeating it in the detail panel put two identical lines a row apart for the
    common case of a short question, which reads as a rendering bug.
    """
    run = new_run(run_name='r', label_a='v1', label_b='v2', judges=JUDGES)
    question = 'Why ' + 'very ' * 60 + 'long?'
    run.add(
        PairwiseComparison(winner='A', votes=[_vote(m, 'A') for m in JUDGES]),
        question=question,
        response_a='a',
        response_b='b',
    )
    html = render_report_body(run)
    assert html.count(question) == 1
    # Pins the rule on this class rather than searching the whole stylesheet for
    # an ellipsis, which would break on any unrelated rule that legitimately
    # clips and would pass if the clip moved to another class name.
    assert '.pw-cmp__q{flex:1;min-width:0;overflow-wrap:anywhere}' in html
