"""Pairwise comparisons list and lazy detail routes."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard import trace_links
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id
from evaluatorq.pairwise import PairwiseComparison, PairwiseVote, RepetitionObservation
from evaluatorq.pairwise_run import PairwiseRun, new_run

if TYPE_CHECKING:
    from pathlib import Path

JUDGES = ['judge-one', 'judge-two', 'judge-three']


def _vote(model: str, value: str | None, *, flipped: bool = False) -> PairwiseVote:
    return PairwiseVote(model=model, vote=value, flipped=flipped, explanation=f'{model} says {value}')


def _comparison(values: list[str | None], winner: str, **kw: bool) -> PairwiseComparison:
    return PairwiseComparison(winner=winner, votes=[_vote(m, v, **kw) for m, v in zip(JUDGES, values)])


def _fixture_run() -> PairwiseRun:
    r = new_run(run_name='v2 vs v3', label_a='prompt-v2', label_b='prompt-v3', judges=JUDGES, criteria='accuracy')
    r.add(_comparison(['B', 'B', 'B'], 'B'), question='q1', response_a='a1', response_b='b1')
    r.add(_comparison(['A', 'B', 'tie'], 'tie'), question='q2', response_a='a2', response_b='b2')
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='q3', response_a='a3', response_b='b3')
    r.add(
        _comparison([None, None, None], 'inconclusive', flipped=True), question='q4', response_a='a4', response_b='b4'
    )
    return r


def _client(run: PairwiseRun, tmp_path: Path) -> tuple[TestClient, str]:
    runs_dir = tmp_path / 'pairwise-runs'
    runs_dir.mkdir()
    path = run.save(runs_dir / 'run.json')
    return TestClient(build_app(roots=[runs_dir])), report_id(path)


@pytest.fixture
def client(tmp_path: Path) -> tuple[TestClient, str]:
    return _client(_fixture_run(), tmp_path)


def _numbers(html: str) -> list[int]:
    return [int(n) for n in re.findall(r'<tbody class="pw-cmp-group" id="cmp-(\d+)"', html)]


def test_row_list_renders_one_group_per_comparison(client: tuple[TestClient, str]) -> None:
    c, rid = client
    resp = c.get(f'/r/{rid}/pairwise/row-list')
    assert resp.status_code == 200
    assert _numbers(resp.text) == [1, 2, 3, 4]
    assert '4 comparisons' in resp.text


def test_split_filter_narrows_the_list_and_keeps_numbering(client: tuple[TestClient, str]) -> None:
    c, rid = client
    # Comparison 2 has judges disagreeing; comparison 4 has a vote thrown out.
    assert _numbers(c.get(f'/r/{rid}/pairwise/row-list?split=yes').text) == [2, 4]


def test_sort_by_winner_orders_a_b_tie_inconclusive(client: tuple[TestClient, str]) -> None:
    c, rid = client
    # winners by original index: 1=B, 2=tie, 3=A, 4=inconclusive
    assert _numbers(c.get(f'/r/{rid}/pairwise/row-list?sort=winner').text) == [3, 1, 2, 4]
    assert _numbers(c.get(f'/r/{rid}/pairwise/row-list?sort=winner&dir=desc').text) == [4, 2, 1, 3]


def test_invalid_sort_and_direction_fall_back(client: tuple[TestClient, str]) -> None:
    c, rid = client
    assert _numbers(c.get(f'/r/{rid}/pairwise/row-list?sort=nope&dir=sideways').text) == [1, 2, 3, 4]


def _big_run() -> PairwiseRun:
    r = new_run(run_name='big', judges=JUDGES)
    for i in range(30):
        r.add(_comparison(['A', 'A', 'A'], 'A'), question=f'q{i}', response_a='a', response_b='b')
    return r


def test_second_page_keeps_original_numbers(tmp_path: Path) -> None:
    c, rid = _client(_big_run(), tmp_path)
    resp = c.get(f'/r/{rid}/pairwise/row-list?page=2')
    assert _numbers(resp.text) == [26, 27, 28, 29, 30]
    assert 'Page 2 of 2 · 30 comparisons' in resp.text


def test_page_is_clamped(tmp_path: Path) -> None:
    c, rid = _client(_big_run(), tmp_path)
    assert _numbers(c.get(f'/r/{rid}/pairwise/row-list?page=99').text) == [26, 27, 28, 29, 30]
    assert _numbers(c.get(f'/r/{rid}/pairwise/row-list?page=-3').text) == list(range(1, 26))
    assert _numbers(c.get(f'/r/{rid}/pairwise/row-list?page=abc').text) == list(range(1, 26))


def test_detail_shows_both_responses_every_judge_and_flip_tag(client: tuple[TestClient, str]) -> None:
    c, rid = client
    resp = c.get(f'/r/{rid}/pairwise/comparison?idx=3')
    assert resp.status_code == 200
    assert 'a4' in resp.text
    assert 'b4' in resp.text
    for judge in JUDGES:
        assert judge in resp.text
    assert '⚖ flipped when sides swapped, vote thrown out' in resp.text
    assert '⚖' not in c.get(f'/r/{rid}/pairwise/comparison?idx=0').text


def test_detail_lists_each_pass_with_ordering_labels(tmp_path: Path) -> None:
    r = new_run(run_name='obs', label_a='old', label_b='new', judges=['j'])
    vote = PairwiseVote(
        model='j',
        vote='B',
        explanation='reconciled',
        observations=[
            RepetitionObservation(ordering='ab', repetition=0, verdict='B', explanation='first look'),
            RepetitionObservation(ordering='ba', repetition=0, verdict='B', explanation='second look'),
            RepetitionObservation(ordering='ba', repetition=1, verdict='A', explanation='third look'),
        ],
    )
    r.add(PairwiseComparison(winner='B', votes=[vote]), question='q', response_a='a', response_b='b')
    c, rid = _client(r, tmp_path)
    text = c.get(f'/r/{rid}/pairwise/comparison?idx=0').text
    assert 'original order' in text
    assert 'swapped order' in text
    assert 'pass 2' in text
    assert 'first look' in text
    assert 'second look' in text
    assert 'third look' in text


@pytest.mark.parametrize('idx', ['abc', '-1', '99', ''])
def test_bad_idx_is_a_200_empty_state(client: tuple[TestClient, str], idx: str) -> None:
    c, rid = client
    resp = c.get(f'/r/{rid}/pairwise/comparison?idx={idx}')
    assert resp.status_code == 200
    assert 'No comparison at that index.' in resp.text


def test_unknown_rid_is_404(client: tuple[TestClient, str]) -> None:
    c, _ = client
    assert c.get('/r/does-not-exist/pairwise/row-list').status_code == 404
    assert c.get('/r/does-not-exist/pairwise/comparison?idx=0').status_code == 404


def test_non_pairwise_rid_is_404(tmp_path: Path) -> None:
    runs_dir = tmp_path / 'sim-runs'
    runs_dir.mkdir()
    path = runs_dir / 'sim.json'
    path.write_text('{"mode": "simulation"}')
    c = TestClient(build_app(roots=[runs_dir]))
    rid = report_id(path)
    assert c.get(f'/r/{rid}/pairwise/row-list').status_code == 404
    assert c.get(f'/r/{rid}/pairwise/comparison?idx=0').status_code == 404


def _traced_run() -> PairwiseRun:
    r = new_run(run_name='traced', judges=['j'])
    r.add(
        PairwiseComparison(winner='A', votes=[_vote('j', 'A')], trace_id='abc123def456'),
        question='q',
        response_a='a',
        response_b='b',
    )
    return r


def test_trace_link_appears_with_a_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_WORKSPACE', 'acme')
    c, rid = _client(_traced_run(), tmp_path)
    assert 'abc123def456' in c.get(f'/r/{rid}/pairwise/row-list').text


def test_no_trace_link_without_a_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trace_links, 'workspace_slug', lambda: None)
    c, rid = _client(_traced_run(), tmp_path)
    text = c.get(f'/r/{rid}/pairwise/row-list').text
    assert 'abc123def456' not in text
    assert 'trace-link' not in text


def test_response_markup_is_escaped(tmp_path: Path) -> None:
    r = new_run(run_name='xss', judges=['j'])
    r.add(
        PairwiseComparison(winner='A', votes=[_vote('j', 'A')]),
        question='<script>q</script>',
        response_a='<img onerror=alert(1)>',
        response_b='ok',
    )
    c, rid = _client(r, tmp_path)
    detail = c.get(f'/r/{rid}/pairwise/comparison?idx=0').text
    assert '<img' not in detail
    assert '&lt;img' in detail
    assert '<script>' not in c.get(f'/r/{rid}/pairwise/row-list').text


# ---------------------------------------------------------------------------
# Row focus (comparison links), CLI scope, corrupt files
# ---------------------------------------------------------------------------


class _Rows(HTMLParser):
    """Collects each comparison group's number and whether its details element is open."""

    def __init__(self) -> None:
        super().__init__()
        self.open_by_number: dict[int, bool] = {}
        self._current: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get('id') or ''
        if tag == 'tbody' and element_id.startswith('cmp-'):
            self._current = int(element_id.removeprefix('cmp-'))
            self.open_by_number[self._current] = False
        elif tag == 'details' and self._current is not None:
            self.open_by_number[self._current] = 'open' in attributes

    def handle_endtag(self, tag: str) -> None:
        if tag == 'tbody':
            self._current = None


def _rows(html: str) -> dict[int, bool]:
    parser = _Rows()
    parser.feed(html)
    return parser.open_by_number


def test_focus_renders_the_page_holding_the_row_with_it_open(tmp_path: Path) -> None:
    c, rid = _client(_big_run(), tmp_path)
    html = c.get(f'/r/{rid}/pairwise/row-list?focus=30').text
    rows = _rows(html)
    assert list(rows) == [26, 27, 28, 29, 30]
    assert [n for n, is_open in rows.items() if is_open] == [30]
    # The open row already carries its detail: a parsed ``open`` toggles before htmx can lazy-load it.
    assert c.get(f'/r/{rid}/pairwise/comparison?idx=29').text in html


def test_focus_overrides_the_page_parameter(tmp_path: Path) -> None:
    c, rid = _client(_big_run(), tmp_path)
    assert list(_rows(c.get(f'/r/{rid}/pairwise/row-list?page=1&focus=30').text)) == [26, 27, 28, 29, 30]


def test_focus_follows_the_sort_order(tmp_path: Path) -> None:
    c, rid = _client(_big_run(), tmp_path)
    # Descending, comparison 3 sits at position 28 of 30, so it is on page 2.
    rows = _rows(c.get(f'/r/{rid}/pairwise/row-list?dir=desc&focus=3').text)
    assert list(rows) == [5, 4, 3, 2, 1]
    assert rows[3] is True


@pytest.mark.parametrize('focus', ['0', '31', '-1', 'abc', ''])
def test_invalid_focus_falls_back_to_the_requested_page(tmp_path: Path, focus: str) -> None:
    c, rid = _client(_big_run(), tmp_path)
    rows = _rows(c.get(f'/r/{rid}/pairwise/row-list?page=2&focus={focus}').text)
    assert list(rows) == [26, 27, 28, 29, 30]
    assert not any(rows.values())


def test_focus_on_a_filtered_out_row_is_ignored(client: tuple[TestClient, str]) -> None:
    c, rid = client
    # Comparison 1 is unanimous, so the split filter drops it.
    rows = _rows(c.get(f'/r/{rid}/pairwise/row-list?split=yes&focus=1').text)
    assert list(rows) == [2, 4]
    assert not any(rows.values())


def test_rows_are_closed_without_focus(client: tuple[TestClient, str]) -> None:
    c, rid = client
    assert not any(_rows(c.get(f'/r/{rid}/pairwise/row-list').text).values())


class _Dots(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hx_get: dict[int, str] = {}
        self.href: dict[int, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == 'a' and 'pw-dot' in (attributes.get('class') or '').split():
            href = attributes.get('href') or ''
            number = int(href.rsplit('#cmp-', 1)[1])
            self.href[number] = href
            self.hx_get[number] = attributes.get('hx-get') or ''


def test_dots_fetch_the_row_list_focused_on_their_comparison(tmp_path: Path) -> None:
    c, rid = _client(_big_run(), tmp_path)
    dots = _Dots()
    dots.feed(c.get(f'/r/{rid}').text)
    assert sorted(dots.hx_get) == list(range(1, 31))
    # Without HTMX, following the ordinary link must load and open the later row.
    fallback_rows = _rows(c.get(dots.href[30]).text)
    assert fallback_rows[30] is True
    # The dot's own request must return the page holding that comparison, open.
    rows = _rows(c.get(dots.hx_get[30]).text)
    assert list(rows) == [26, 27, 28, 29, 30]
    assert rows[30] is True


class _Tabs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.radios: list[tuple[str, bool]] = []
        self.labels: dict[str, str] = {}
        self._label_for: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == 'input' and attributes.get('name') == 'pwtab':
            self.radios.append((attributes.get('id') or '', 'checked' in attributes))
        elif tag == 'label' and 'for' in attributes:
            target = attributes['for'] or ''
            self._label_for = target
            self.labels[target] = ''

    def handle_data(self, data: str) -> None:
        if self._label_for is not None:
            self.labels[self._label_for] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == 'label':
            self._label_for = None


def _checked_tab(html: str) -> str:
    tabs = _Tabs()
    tabs.feed(html)
    checked = [tab_id for tab_id, is_checked in tabs.radios if is_checked]
    assert len(checked) == 1
    return tabs.labels[checked[0]]


def test_report_cmp_parameter_opens_the_comparisons_tab_on_that_row(tmp_path: Path) -> None:
    c, rid = _client(_big_run(), tmp_path)
    html = c.get(f'/r/{rid}?cmp=30').text
    assert _checked_tab(html).startswith('Comparisons')
    rows = _rows(html)
    assert rows[30] is True
    assert list(rows) == [26, 27, 28, 29, 30]


def test_report_opens_on_overview_without_cmp(tmp_path: Path) -> None:
    c, rid = _client(_big_run(), tmp_path)
    html = c.get(f'/r/{rid}').text
    assert _checked_tab(html).startswith('Overview')
    assert not any(_rows(html).values())


@pytest.mark.parametrize('cmp', ['0', '31', 'abc', ''])
def test_report_ignores_an_invalid_cmp_parameter(tmp_path: Path, cmp: str) -> None:
    c, rid = _client(_big_run(), tmp_path)
    html = c.get(f'/r/{rid}?cmp={cmp}').text
    assert _checked_tab(html).startswith('Overview')
    assert list(_rows(html)) == list(range(1, 26))


def test_row_list_discovers_the_cli_workspace_once_per_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard import orq_workspace
    from evaluatorq.dashboard.orq_scope import OrqScope

    for variable in ('ORQ_WORKSPACE', 'ORQ_WORKSPACE_SLUG', 'ORQ_BASE_URL', 'ORQ_API_KEY'):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'empty-settings.json'))
    monkeypatch.setattr(orq_workspace.shutil, 'which', lambda _name: '/usr/bin/orq')
    calls = 0

    def discover(_profile: str | None, *, use_cli_session: bool = False) -> OrqScope:
        nonlocal calls
        calls += 1
        return OrqScope(workspace_key='acme')

    monkeypatch.setattr('evaluatorq.dashboard.orq_scope.discover_orq_scope', discover)
    run = new_run(run_name='traced', judges=['j'])
    for i in range(10):
        run.add(
            PairwiseComparison(winner='A', votes=[_vote('j', 'A')], trace_id=f'trace{i:08d}'),
            question=f'q{i}',
            response_a='a',
            response_b='b',
        )
    c, rid = _client(run, tmp_path)
    text = c.get(f'/r/{rid}/pairwise/row-list').text
    assert text.count('trace00000003') >= 1
    assert calls == 1


def test_a_pairwise_file_that_fails_validation_is_404_not_500(tmp_path: Path) -> None:
    runs_dir = tmp_path / 'pairwise-runs'
    runs_dir.mkdir()
    bad = runs_dir / 'bad.json'
    bad.write_text('{"judging": "pairwise", "entries": "not-a-list"}')
    c = TestClient(build_app(roots=[runs_dir]), raise_server_exceptions=False)
    rid = report_id(bad)
    assert c.get(f'/r/{rid}/pairwise/row-list').status_code == 404
    assert c.get(f'/r/{rid}/pairwise/comparison?idx=0').status_code == 404
