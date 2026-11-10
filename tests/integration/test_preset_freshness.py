"""Presets rot even when nobody edits them (RES-1346).

A jury is hand-picked; the garden and the price list are not. Ways a published
preset becomes a lie without a single line changing:

  1. a judge is retired, so the preset names a model nobody can call
  2. a provider reprices, so the published $/1k is wrong
  3. a newer generation ships and the preset quietly stays a generation behind
  4. a lineage stops shipping and a seat ages with nothing flagging it

These read the live garden: `/v2/models` for scores and rates, the public
`/v2/model-catalog` for the roster and first-listed dates. They are marked
`integration` and run weekly with a real key
(`.github/workflows/preset-drift-weekly.yml`), not on every PR, so a gateway
outage or our own billing state cannot block unrelated work. The checks of the
checks run offline on hand-built cards, in the normal suite.
"""

import asyncio
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import pytest

from evaluatorq.common.model_catalogue import model_cards
from evaluatorq.common.model_frontier import (
    NEVER_SEAT,
    UNLISTED_ROUTES,
    UNROUTABLE,
    ScoredModel,
    catalog_scope,
    headline_contradicts_its_own_rung,
    in_family_upgrades,
    load_garden_sync,
    public_listings,
    rank_cheapest_top,
    unseatable_reason,
    with_aa_fills,
    with_unlisted_routes,
)
from evaluatorq.jury_presets import (
    AGED_SEATS,
    HELD_SUPERSEDED,
    MAX_SEAT_AGE_DAYS,
    PRESETS,
    REVIEWED_SUCCESSORS,
    REVIEWED_UNBENCHMARKED_SEATS,
    all_router_ids,
    judge_family,
    model_identity,
)
from evaluatorq.router_pools import get_pools

UTC = timezone.utc


def live(test):  # noqa: ANN001, ANN201 - a decorator over tests and test classes alike
    """Reads the live catalogue: integration-only, and exempt from the network guard."""
    return pytest.mark.allow_network(pytest.mark.integration(test))


@pytest.fixture(scope='module')
def garden() -> dict[str, ScoredModel]:
    return load_garden_sync()


@pytest.fixture(scope='module')
def listed() -> dict[str, int | None]:
    """Router ID to first-listed epoch milliseconds, from the public catalogue."""
    return asyncio.run(public_listings())


@live
def test_the_garden_is_read_with_a_workspace_scoped_key():
    """A project-scoped key hides every model the workspace has not enabled, so
    the frontier and every check below would run on a truncated catalogue."""
    assert catalog_scope(asyncio.run(model_cards())) == 'workspace', (
        '/v2/models returned no disabled cards: the key is project-scoped. Run the drift '
        'suite with a workspace-scoped ORQ_API_KEY.'
    )


@live
@pytest.mark.parametrize('router_id', sorted(all_router_ids()))
def test_every_shipped_judge_is_still_served(listed, router_id):
    """A retired judge is dead weight the user picks without being able to see."""
    # UNLISTED_ROUTES are callable but absent from the listing.
    assert router_id in listed or router_id in UNLISTED_ROUTES, (
        f'{router_id} is named by a preset but no longer served by the garden. '
        'Promote a reserve rather than deleting the seat.'
    )


@live
@pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
def test_published_cost_matches_the_live_card_rates(garden, preset):
    """Every published $/1k recomputed from today's card rates; a repricing must break it loudly.

    Card rates are not always what the router bills (zai/glm-5.3-flash has been
    billed at $0.15/$0.50 against a $0.075/$0.25 card); the card leads, and the
    gap is disclosed beside Cheap Aggregate in docs/jury-presets.md.
    """
    assert preset.cost_per_1k(garden) == preset.estimated_cost_per_1k, (
        f'{preset.name} publishes ${preset.estimated_cost_per_1k} per 1k but the live cards price it at '
        f'${preset.cost_per_1k(garden)}. Update estimated_cost_per_1k and the docs table, or reseat.'
    )


# moonshotai/kimi-k2.6 was recorded here until the live pull of 2026-10-26, whose card
# agrees with itself (headline 27.0 at `reasoning`, rung 27.0).
CONTRADICTED_CARDS: dict[str, str] = {}
"""Headline-ranked cards whose own ladder contradicts that headline, keyed by router ID."""


def _contradicted(garden: dict[str, ScoredModel]) -> dict[str, str]:
    return {
        router_id: f'headline {model.intelligence_index} against rungs {model.effort_indices}'
        for router_id, model in sorted(garden.items())
        if headline_contradicts_its_own_rung(model)
    }


def _assert_no_card_scores_its_own_rung_twice(garden: dict[str, ScoredModel]) -> None:
    contradicted = {k: v for k, v in _contradicted(garden).items() if k not in CONTRADICTED_CARDS}

    assert not contradicted, (
        f'these cards disagree with themselves about one rung: {contradicted}. '
        f'Record the card in CONTRADICTED_CARDS. Nothing may be seated on a number '
        f'the card contradicts.'
    )


@live
def test_no_card_scores_its_own_rung_twice(garden):
    """A card ranked on its headline must not publish a different score for the headline's rung."""
    _assert_no_card_scores_its_own_rung_twice(garden)


@live
def test_a_contradicted_card_holds_no_seat():
    """The register records a bad card; it does not license seating one."""
    seats = {j for preset in PRESETS.values() for j in (*preset.judges, *preset.reserve_judges)}
    seats |= {m for pool in get_pools().values() for m in (*pool.models, *pool.reserve_models)}
    seated = sorted(seats & set(CONTRADICTED_CARDS))

    assert not seated, f'{seated} are seated on a card that contradicts itself. Reseat.'


@live
def test_contradicted_entries_are_dropped_once_the_card_agrees_with_itself(garden):
    """An entry whose card no longer contradicts itself is stale."""
    resolved = sorted(set(CONTRADICTED_CARDS) & set(garden) - set(_contradicted(garden)))

    assert not resolved, f'{resolved} now agree with themselves. Drop the entries.'


class TestContradictionFires:
    """The contradiction rule against cards built to break it, and to pass it."""

    @staticmethod
    def _garden(**card: Any) -> dict[str, ScoredModel]:
        base: dict[str, Any] = {
            'intelligence_index': 44.2,
            'input_price': 1.0,
            'output_price': 4.0,
            'headline_effort': None,
        }
        return {'moonshotai/unregistered': ScoredModel(router_id='moonshotai/unregistered', **(base | card))}

    def test_a_headline_ranked_card_that_scores_its_rung_twice_is_refused(self):
        with pytest.raises(AssertionError, match='disagree with themselves'):
            _assert_no_card_scores_its_own_rung_twice(self._garden(effort_indices={'none': 34.6}))

    def test_a_card_ranked_on_its_ladder_is_left_alone(self):
        _assert_no_card_scores_its_own_rung_twice(
            self._garden(effort_indices={'none': 34.6}, effort_parameter=True, default_effort='none')
        )

    def test_a_card_publishing_one_number_is_left_alone(self):
        _assert_no_card_scores_its_own_rung_twice(self._garden(effort_indices={}))


@pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
def test_every_preset_names_at_least_one_reserve(preset):
    """A preset with no reserve has no answer to a retirement."""
    assert preset.reserve_judges


def _assert_every_judge_and_reserve_is_seatable(garden: dict[str, ScoredModel]) -> None:
    refused = {}
    for preset in PRESETS.values():
        for router_id in (*preset.judges, *preset.reserve_judges):
            # An unlisted route is validated from its UNLISTED_ROUTES fields, never skipped.
            model = garden.get(router_id) or with_unlisted_routes({}).get(router_id)
            if model is None:
                # Collected, not asserted: one missing card must not hide every other refusal.
                refused[f'{preset.name}: {router_id}'] = 'unknown route: no card in the garden'
                continue
            model = with_aa_fills({router_id: model})[router_id]
            reason = unseatable_reason(model)
            if reason and router_id in HELD_SUPERSEDED and reason.startswith('superseded'):
                continue
            if reason:
                reviewed = REVIEWED_UNBENCHMARKED_SEATS.get(router_id)
                if (
                    reviewed is not None
                    and reason.startswith('unrankable: no benchmark')
                    and model.default_effort == reviewed[0]
                    and tuple(sorted(model.effort_indices)) == reviewed[1]
                    and model.enabled is not False
                    and model.router_id not in UNROUTABLE
                    and not any(family in model.router_id for family in NEVER_SEAT)
                ):
                    continue
                refused[f'{preset.name}: {router_id}'] = reason

    assert not refused, f'presets recommend unseatable models: {refused}'


@live
def test_every_judge_and_reserve_is_seatable(garden):
    """The floor, NEVER_SEAT, and ranking at the catalogue default bind the juries too.

    A judge with no benchmark at the effort it runs at is reported here rather
    than ranked on a rung it never runs at.
    """
    _assert_every_judge_and_reserve_is_seatable(garden)


@live
def test_unbenchmarked_holds_expire_when_the_card_changes(garden):
    """A held default is a data blocker, never a licence to ignore new data."""
    reachable = with_unlisted_routes(garden)
    for router_id, (default, rungs, reason) in REVIEWED_UNBENCHMARKED_SEATS.items():
        assert router_id in all_router_ids(), f'{router_id} is no longer seated; drop its hold'
        model = reachable[router_id]
        assert model.default_index is None, f'{router_id} now has a default score; review and drop its hold'
        assert model.default_effort == default, f'{router_id} default changed; review its hold'
        assert tuple(sorted(model.effort_indices)) == rungs, f'{router_id} measured rungs changed; review its hold'
        assert reason.strip()


@live
@pytest.mark.parametrize(
    'changes',
    [
        {'enabled': False},
        {'input_price': 0.0},
        # An effort no held card measures, so the hold's fingerprint cannot match.
        {'default_effort': 'minimal'},
        {'effort_indices': {'none': 10.0}},
    ],
)
def test_an_unbenchmarked_hold_does_not_mask_other_failures(garden, changes):
    # The first hold with a live card: an unlisted route's fields are fixed in code.
    held = next(r for r in REVIEWED_UNBENCHMARKED_SEATS if r in garden)
    updated = garden | {held: garden[held].model_copy(update=changes)}
    with pytest.raises(AssertionError, match=held):
        _assert_every_judge_and_reserve_is_seatable(updated)


def _earliest_listing(listed: Mapping[str, int | None]) -> dict[str, int]:
    """Earliest listing per model identity, in epoch milliseconds, across every host of it.

    A second host relisting the same weights carries its own `created` stamp,
    which is younger than the model: the EU Bedrock haiku listing is 125 days
    newer than the Anthropic one for a model that shipped once, and Bedrock
    relisting GPT-5.4 in 2026-10 does not make GPT-5.4 newer than GPT-6.
    """
    earliest: dict[str, int] = {}
    for router_id, stamp in listed.items():
        if stamp is None:
            continue
        # Some cards stamp milliseconds, some seconds.
        ms = stamp if stamp > 1e11 else stamp * 1000
        identity = model_identity(router_id)
        earliest[identity] = min(earliest.get(identity, ms), ms)
    return earliest


def _unreviewed_successors(listed: Mapping[str, int | None]) -> dict[str, str]:
    """Garden models newer than everything a preset seats from their own lineage.

    Compared against the newest seat per lineage rather than per judge: a panel
    that already seats GPT-5.6 does not need to answer for GPT-5.1 carrying a
    later catalog timestamp than gpt-oss-20b. Identity-normalised and dated by
    the identity's first listing, so the same weights on a second host are
    neither a second decision nor a newer model.
    """
    seated_identities = {model_identity(r) for r in all_router_ids()}
    earliest = _earliest_listing(listed)

    newest_seated: dict[str, int] = {}
    for router_id in all_router_ids():
        if router_id not in listed:
            continue
        # A card with no `created` stamp cannot date its lineage.
        if (stamp := earliest.get(model_identity(router_id))) is None:
            continue
        family = judge_family(router_id)
        newest_seated[family] = max(newest_seated.get(family, 0), stamp)

    candidates: dict[str, str] = {}
    for model_id in sorted(listed):
        # Research-workspace endpoints are not generally served, and image
        # models are not judges.
        if model_id.startswith('orq-research@') or 'image' in model_id:
            continue
        identity = model_identity(model_id)
        if identity in seated_identities or identity in candidates:
            continue
        try:
            family = judge_family(model_id)
        except ValueError:
            continue  # unclassifiable lineage cannot be a successor to ours
        listed_at = earliest.get(identity)
        if listed_at is None:
            continue  # undated: cannot be shown newer than a seat
        if family in newest_seated and listed_at > newest_seated[family]:
            candidates[identity] = model_id

    return candidates


class TestSuccessorDating:
    """Offline: a relisting is not a release."""

    DAY = 86_400_000

    @pytest.mark.parametrize(
        ('router_id', 'identity'),
        [
            ('aws/us-east-1.openai.gpt-5.4', 'gpt-5.4'),
            ('aws/eu-central-2.anthropic.claude-haiku-5-5', 'claude-haiku-5-5'),
            ('aws/openai.gpt-5.5', 'gpt-5.5'),
            ('openai/gpt-5.6-terra', 'gpt-5.6-terra'),
            ('munito/zai-org/glm-5.3-flash@eu', 'glm-5.3-flash'),
        ],
    )
    def test_bedrock_region_and_vendor_are_stripped(self, router_id, identity):
        assert model_identity(router_id) == identity

    def test_an_old_model_relisted_on_a_new_host_is_not_a_successor(self):
        seat = 'openai/gpt-6-luna'
        listed = {seat: 100 * self.DAY, 'openai/gpt-5.4': 10 * self.DAY, 'aws/us-east-1.openai.gpt-5.4': 200 * self.DAY}
        assert 'gpt-5.4' not in _unreviewed_successors(listed)

    def test_a_model_first_listed_after_the_seat_is_a_successor(self):
        listed = {'openai/gpt-6-luna': 100 * self.DAY, 'openai/gpt-7-luna': 200 * self.DAY}
        assert _unreviewed_successors(listed)['gpt-7-luna'] == 'openai/gpt-7-luna'

    def test_an_undated_listing_is_skipped_not_a_crash(self):
        """`public_listings` returns None for a card with no `created`; one must not fail the weekly run."""
        listed = {'openai/gpt-6-luna': None, 'openai/gpt-7-luna': None, 'openai/gpt-6-sol': 100 * self.DAY}
        assert 'gpt-7-luna' not in _unreviewed_successors(listed)

    def test_seconds_and_milliseconds_stamps_date_the_same(self):
        assert _earliest_listing({'a/x': 1_700_000_000, 'b/x': 1_700_000_000_000}) == {'x': 1_700_000_000_000}


@live
def test_no_newer_model_is_silently_ignored(listed):
    """A newer sibling must be seated or dismissed in writing, never just missed.

    This is the failure the other tests cannot see: every judge still served,
    every price still correct, and the panel quietly a generation behind.
    """
    unreviewed = {
        identity: model_id
        for identity, model_id in _unreviewed_successors(listed).items()
        if identity not in REVIEWED_SUCCESSORS
    }

    assert not unreviewed, (
        'the garden has newer models in a lineage a preset seats: '
        f'{sorted(unreviewed.values())}. Seat one, or add its identity to '
        'REVIEWED_SUCCESSORS with the reason it stays out.'
    )


def test_reviewed_successors_are_not_carried_after_they_are_seated():
    """A dismissal that outlives its subject is stale documentation."""
    seated_identities = {model_identity(r) for r in all_router_ids()}

    still_out = seated_identities & set(REVIEWED_SUCCESSORS)

    assert not still_out, f'{sorted(still_out)} are seated but still listed as reviewed-and-rejected'


@live
def test_reviewed_successors_are_dropped_once_they_stop_being_successors(listed):
    """A dismissal survives the seat it was written about, and then reads as fact.

    Reseating Cheap Aggregate off qwen left three entries explaining why various
    qwen models were not worth taking `the seated 35b-a3b`'s place, when no qwen
    held a seat at all. Every word was still true and none of it was about anything.
    """
    stale = sorted(set(REVIEWED_SUCCESSORS) - set(_unreviewed_successors(listed)))

    assert not stale, (
        f'{stale} are recorded as reviewed-and-rejected successors but no seat '
        f'in their lineage makes them a successor to anything. Drop the entries.'
    )


DOMINANCE_EXEMPTIONS: dict[str, str] = {}
"""Seats knowingly kept despite a computed in-family upgrade.

Same disposition-register contract as REVIEWED_SUCCESSORS: seat it or write down
why not. An empty register means the arithmetic and the shipped presets agree.
"""


@live
class TestFrontier:
    """The upgrade rule, as arithmetic rather than taste (RES-1347)."""

    def test_no_seat_has_an_unreviewed_in_family_upgrade(self, garden):
        """A judge must be the best buy available within its own lineage.

        Scoped to the lineage on purpose: a jury buys family diversity, and a
        global frontier check would push all three seats onto whichever vendor
        is cheapest, recreating the correlated errors the panel exists to cancel.
        """
        # Unlisted seats (reviewed routes with no card) are incumbents too, or a
        # same-family model could dominate one without this check noticing.
        reachable = with_aa_fills(with_unlisted_routes(garden))
        models = list(reachable.values())

        stale = {}
        for preset in PRESETS.values():
            for judge in preset.judges:
                incumbent = reachable.get(judge)
                if incumbent is None:
                    continue
                upgrades = in_family_upgrades(incumbent, models, judge_family, model_identity)
                if upgrades and judge not in DOMINANCE_EXEMPTIONS:
                    stale[judge] = upgrades[0].router_id

        assert not stale, (
            f'These seats are beaten on both index and price by a same-family, '
            f'same-region model: {stale}. Either reseat, or record the reason in '
            f'DOMINANCE_EXEMPTIONS.'
        )

    def test_exemptions_are_dropped_once_they_stop_applying(self, garden):
        """An exemption whose upgrade has gone away is stale reasoning."""
        reachable = with_aa_fills(with_unlisted_routes(garden))
        models = list(reachable.values())

        pointless = [
            judge
            for judge in DOMINANCE_EXEMPTIONS
            if judge in reachable and not in_family_upgrades(reachable[judge], models, judge_family, model_identity)
        ]

        assert not pointless, (
            f'{pointless} are exempted from a dominance check that no longer flags them. Drop the entries.'
        )

    def test_every_seat_is_scored(self, garden):
        """An unscored seat is one the upgrade check silently skips."""
        unscored = all_router_ids() - set(garden) - set(UNLISTED_ROUTES)
        assert not unscored, (
            f'{unscored} carry no autorouter intelligence/price pair, so no upgrade check runs against them.'
        )


def _first_listed(listed: dict[str, int | None]) -> dict[str, datetime]:
    """`_earliest_listing` as datetimes, for the age check."""
    return {identity: datetime.fromtimestamp(ms / 1000, UTC) for identity, ms in _earliest_listing(listed).items()}


def _all_seats() -> dict[str, list[str]]:
    """Every seat in the library, jury and pool alike, with where it sits."""
    seats: dict[str, list[str]] = {}
    for preset in PRESETS.values():
        for judge in preset.judges:
            seats.setdefault(judge, []).append(f'jury {preset.name}')
    for pool in get_pools().values():
        for member in pool.models:
            seats.setdefault(member, []).append(f'pool {pool.name}')
    return seats


@live
def test_no_seat_ages_out_unargued(listed):
    """A lineage that stops shipping is the blind spot in every check above.

    The upgrade and successor tests both need a newer sibling to exist before
    they can say anything. If a vendor ships nothing, they stay silent forever
    and the seat keeps getting older. This is the only test that fires on absence.
    """
    earliest = _first_listed(listed)
    now = datetime.now(UTC)

    aged = {}
    for router_id, held in _all_seats().items():
        first = earliest.get(model_identity(router_id))
        if first is None:
            continue
        days = (now - first).days
        if days > MAX_SEAT_AGE_DAYS and model_identity(router_id) not in AGED_SEATS:
            aged[router_id] = f'{days}d, held by {", ".join(held)}'

    assert not aged, (
        f'these seats were listed more than {MAX_SEAT_AGE_DAYS} days ago and no '
        f'successor has been flagged, so nothing else will ever ask about them: '
        f'{aged}. Reseat, or record the reason in AGED_SEATS.'
    )


@live
def test_aged_seat_entries_are_dropped_once_the_seat_is_young_again(listed):
    """A vendor shipping again should retire the excuse, not leave it standing."""
    earliest = _first_listed(listed)
    now = datetime.now(UTC)
    seated = {model_identity(r) for r in _all_seats()}

    pointless = sorted(
        identity
        for identity in AGED_SEATS
        if identity not in seated or (identity in earliest and (now - earliest[identity]).days <= MAX_SEAT_AGE_DAYS)
    )

    assert not pointless, (
        f'{pointless} are excused for age but are either no longer seated or no longer old. Drop the entries.'
    )


@live
def test_cheap_aggregate_matches_the_live_derivation(garden):
    """Cheap Aggregate is the live top three families under MAX_BLEND, reserve the fourth.

    Fails with both lists when the ranking moves, so a reseat is a deliberate,
    reviewed edit to the preset and never a silent drift.
    """
    ranked = rank_cheapest_top(with_aa_fills(with_unlisted_routes(garden)), judge_family)
    derived = [m.router_id for m in ranked]
    preset = PRESETS['Cheap Aggregate']

    table = '\n'.join(f'  {m.router_id}  index {m.default_index}  blend {m.price:.3f}' for m in ranked[:6])
    assert (derived[:3], derived[3:4]) == (list(preset.judges), list(preset.reserve_judges)), (
        f'live ranking moved.\nderived seats {derived[:3]} reserve {derived[3:4]}\n'
        f'preset seats {list(preset.judges)} reserve {list(preset.reserve_judges)}\n'
        f'top of the ranking:\n{table}\n'
        'Reseat Cheap Aggregate deliberately: update judges, reserve, estimate and docs/jury-presets.md.'
    )
