"""The drift report against hand-built gardens (RES-1346). No network."""

from datetime import datetime, timezone

import pytest

from evaluatorq.common.model_frontier import ScoredModel
from evaluatorq.jury_presets import MAX_SEAT_AGE_DAYS, PRESETS, all_router_ids
from evaluatorq.preset_drift import DriftReport, Finding, drift_report, earliest_listing, render_summary, snapshot_presets

UTC = timezone.utc
DAY_MS = 86_400_000
NOW = datetime(2026, 11, 12, tzinfo=UTC)


def _card(router_id: str, index: float = 40.0, input_price: float = 1.0, output_price: float = 4.0) -> ScoredModel:
    return ScoredModel(
        router_id=router_id,
        intelligence_index=index,
        input_price=input_price,
        output_price=output_price,
        headline_effort=None,
    )


def _garden() -> dict[str, ScoredModel]:
    return {router_id: _card(router_id) for router_id in all_router_ids()}


def _everything_listed(age_days: int = 10) -> dict[str, int | None]:
    stamp = int(NOW.timestamp() * 1000) - age_days * DAY_MS
    return {router_id: stamp for router_id in all_router_ids()}


def test_a_retired_seat_is_reported_with_its_reserve_as_the_option():
    listed = _everything_listed()
    seat = PRESETS['balanced-trio'].judges[0]
    del listed[seat]

    report = drift_report(_garden(), listed, now=NOW)

    finding = next(f for f in report.findings if f.kind == 'not_served' and f.subject == seat)
    assert 'balanced-trio' in finding.held_by
    assert any(PRESETS['balanced-trio'].reserve_judges[0] in option for option in finding.options)
    assert not finding.needs_human


def test_a_retired_seat_with_no_reserve_needs_a_human(monkeypatch):
    listed = _everything_listed()
    seat = PRESETS['eu-region'].judges[0]
    del listed[seat]
    monkeypatch.setattr(PRESETS['eu-region'], 'reserve_judges', ())

    report = drift_report(_garden(), listed, now=NOW)

    finding = next(f for f in report.findings if f.kind == 'not_served' and f.subject == seat)
    assert finding.needs_human
    assert report.needs_human


def test_an_old_seat_without_a_waiver_is_reported_with_the_waiver_as_an_option():
    listed = _everything_listed(age_days=MAX_SEAT_AGE_DAYS + 30)

    report = drift_report(_garden(), listed, pools={}, now=NOW)

    aged = [f for f in report.findings if f.kind == 'aged_seat']
    assert aged
    assert all(any('STALE_SEAT_WAIVERS' in option for option in f.options) for f in aged)


def test_young_seats_raise_no_age_finding():
    report = drift_report(_garden(), _everything_listed(age_days=10), now=NOW)

    assert not [f for f in report.findings if f.kind == 'aged_seat']


def test_the_report_serialises_to_json_an_agent_can_read():
    report = drift_report(_garden(), _everything_listed(), now=NOW)

    assert DriftReport.model_validate_json(report.model_dump_json()) == report


def test_an_undated_listing_does_not_crash():
    listed = _everything_listed()
    listed[PRESETS['strong-jury'].judges[0]] = None

    drift_report(_garden(), listed, now=NOW)


@pytest.mark.parametrize(('stamp', 'expected'), [(1_700_000_000, 1_700_000_000_000), (1_700_000_000_000, 1_700_000_000_000)])
def test_seconds_and_milliseconds_stamps_date_the_same(stamp, expected):
    assert earliest_listing({'openai/gpt-6-luna': stamp}) == {'gpt-6-luna': expected}


def test_a_waived_unscored_seat_raises_no_finding(monkeypatch):
    garden = _garden()
    del garden['deepseek/deepseek-flash']
    monkeypatch.setattr('evaluatorq.preset_drift.UNSCORED_SEAT_WAIVERS', {'deepseek/deepseek-flash': 'kept for cost'})

    report = drift_report(garden, _everything_listed(), now=NOW)

    assert not [f for f in report.findings if f.kind == 'unscored_seat']


def test_an_unscored_seat_without_a_waiver_needs_a_human(monkeypatch):
    garden = _garden()
    del garden['deepseek/deepseek-flash']
    monkeypatch.setattr('evaluatorq.preset_drift.UNSCORED_SEAT_WAIVERS', {})

    report = drift_report(garden, _everything_listed(), now=NOW)

    finding = next(f for f in report.findings if f.kind == 'unscored_seat')
    assert finding.subject == 'deepseek/deepseek-flash'
    assert finding.needs_human


def test_a_waiver_for_a_seat_that_is_now_scored_is_reported_as_stale(monkeypatch):
    monkeypatch.setattr('evaluatorq.preset_drift.UNSCORED_SEAT_WAIVERS', {'deepseek/deepseek-flash': 'kept for cost'})
    report = drift_report(_garden(), _everything_listed(), now=NOW)

    stale = [f for f in report.findings if f.kind == 'stale_waiver' and f.subject == 'deepseek/deepseek-flash']
    assert stale
    assert any('UNSCORED_SEAT_WAIVERS' in option for option in stale[0].options)


def _snapshot(judges, reserves=('r/reserve',), cost=1.0):
    return {'trio': {'judges': list(judges), 'reserve_judges': list(reserves), 'estimated_cost_per_1k': cost}}


def test_the_summary_strikes_a_replaced_seat_beside_its_successor():
    before = _snapshot(['a/one', 'b/two', 'c/three'], cost=4.45)
    after = _snapshot(['a/one', 'd/four', 'c/three'], cost=3.75)

    summary = render_summary(before, after, DriftReport())

    assert '~~`b/two`~~ → **`d/four`**' in summary
    assert '4.45 → **3.75**' in summary
    assert 'xychart-beta' in summary


def test_a_seat_that_only_moved_reads_as_kept():
    before = _snapshot(['a/one', 'b/two', 'c/three'])
    after = _snapshot(['a/one', 'c/three', 'b/two'])

    summary = render_summary(before, after, DriftReport())

    assert '~~' not in summary
    assert 'xychart-beta' not in summary, 'no chart when no estimate moved'


def test_a_decision_the_branch_answered_is_ticked_and_an_open_one_is_not():
    before = _snapshot(['a/one', 'b/two', 'c/three'])
    after = _snapshot(['a/one', 'd/four', 'c/three'])
    report = DriftReport(
        findings=[
            Finding(kind='unscored_seat', subject='b/two', held_by=['trio'], detail='gone', needs_human=True),
            Finding(kind='unscored_seat', subject='b/two', held_by=['other'], detail='gone', needs_human=True),
            Finding(kind='unseatable', subject='c/three', held_by=['trio'], detail='disabled', needs_human=True),
        ]
    )

    summary = render_summary(before, after, report)

    assert '- [x] ~~**`b/two`** in `trio`, `other`' in summary
    assert '- [ ] **`c/three`** in `trio`: disabled' in summary
    assert '`c/three` ⛔' in summary


def test_the_snapshot_round_trips_every_shipped_preset():
    snapshot = snapshot_presets()

    assert set(snapshot) == set(PRESETS)
    assert render_summary(snapshot, snapshot, DriftReport()).count('🔁') == 1, 'only the legend line marks a change'
