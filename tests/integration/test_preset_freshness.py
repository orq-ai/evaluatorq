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
    MANUAL_MODEL_CARDS,
    ScoredModel,
    catalog_scope,
    headline_contradicts_its_own_rung,
    in_family_upgrades,
    load_garden_sync,
    public_listings,
    rank_cheapest_top,
    unseatable_reason,
    with_aa_fills,
    with_manual_cards,
)
from evaluatorq.jury_presets import (
    DOMINANCE_EXEMPTIONS,
    MAX_SEAT_AGE_DAYS,
    STALE_SEAT_WAIVERS,
    UNSCORED_SEAT_WAIVERS,
    PRESETS,
    REVIEWED_SUCCESSORS,
    all_router_ids,
    judge_family,
    model_identity,
)
from evaluatorq.preset_drift import (
    earliest_listing as _earliest_listing,
    first_listed as _first_listed,
    seat_holders,
    unreviewed_successors as _unreviewed_successors,
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
    # MANUAL_MODEL_CARDS are callable but absent from the listing.
    assert router_id in listed or router_id in MANUAL_MODEL_CARDS, (
        f'{router_id} is named by a preset but no longer served by the garden. '
        'Promote a reserve rather than deleting the seat.'
    )


@live
@pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
def test_published_cost_matches_the_live_card_rates(garden, preset):
    """Every published $/1k recomputed from today's card rates; a repricing must break it loudly.

    Card rates are not always what the router bills (zai/glm-5.3-flash has been
    billed at $0.15/$0.50 against a $0.075/$0.25 card); the card leads, and the
    gap is disclosed beside cheap-aggregate in docs/jury-presets.md.
    """
    assert preset.cost_per_1k(garden) == preset.estimated_cost_per_1k, (
        f'{preset.name} publishes ${preset.estimated_cost_per_1k} per 1k but the live cards price it at '
        f'${preset.cost_per_1k(garden)}. Update estimated_cost_per_1k and the docs table, or reseat.'
    )


def _contradicted(garden: dict[str, ScoredModel]) -> dict[str, str]:
    return {
        router_id: f'headline {model.intelligence_index} against rungs {model.effort_indices}'
        for router_id, model in sorted(garden.items())
        if headline_contradicts_its_own_rung(model)
    }


def _assert_no_card_scores_its_own_rung_twice(garden: dict[str, ScoredModel]) -> None:
    contradicted = _contradicted(garden)

    assert not contradicted, (
        f'these cards disagree with themselves about one rung: {contradicted}. '
        f'Nothing may be seated on a number the card contradicts.'
    )


@live
def test_no_card_scores_its_own_rung_twice(garden):
    """A card ranked on its headline must not publish a different score for the headline's rung."""
    _assert_no_card_scores_its_own_rung_twice(garden)


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
            # A model without a live card is validated from its MANUAL_MODEL_CARDS fields, never skipped.
            model = garden.get(router_id) or with_manual_cards({}).get(router_id)
            if model is None and router_id in UNSCORED_SEAT_WAIVERS:
                continue
            if model is None:
                # Collected, not asserted: one missing card must not hide every other refusal.
                refused[f'{preset.name}: {router_id}'] = 'unknown route: no card in the garden'
                continue
            model = with_aa_fills({router_id: model})[router_id]
            reason = unseatable_reason(model)
            if reason:
                refused[f'{preset.name}: {router_id}'] = reason

    assert not refused, f'presets recommend unseatable models: {refused}'


@live
def test_every_judge_and_reserve_is_seatable(garden):
    """The floor, NEVER_SEAT, SUPERSEDED and UNROUTABLE bind the juries too."""
    _assert_every_judge_and_reserve_is_seatable(garden)


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

    Reseating cheap-aggregate off qwen left three entries explaining why various
    qwen models were not worth taking `the seated 35b-a3b`'s place, when no qwen
    held a seat at all. Every word was still true and none of it was about anything.
    """
    stale = sorted(set(REVIEWED_SUCCESSORS) - set(_unreviewed_successors(listed)))

    assert not stale, (
        f'{stale} are recorded as reviewed-and-rejected successors but no seat '
        f'in their lineage makes them a successor to anything. Drop the entries.'
    )


@live
class TestFrontier:
    """The upgrade rule, as arithmetic rather than taste (RES-1347)."""

    def test_no_seat_has_an_unreviewed_in_family_upgrade(self, garden):
        """A judge must be the best buy available within its own lineage.

        Scoped to the lineage on purpose: a jury buys family diversity, and a
        global frontier check would push all three seats onto whichever vendor
        is cheapest, recreating the correlated errors the panel exists to cancel.
        """
        # Seats with only a hand-written card are incumbents too, or a
        # same-family model could dominate one without this check noticing.
        reachable = with_aa_fills(with_manual_cards(garden))
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
        reachable = with_aa_fills(with_manual_cards(garden))
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
        unscored = all_router_ids() - set(garden) - set(MANUAL_MODEL_CARDS) - set(UNSCORED_SEAT_WAIVERS)
        assert not unscored, (
            f'{unscored} carry no autorouter intelligence/price pair, so no upgrade check runs against them.'
        )


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
    for router_id, held in seat_holders({name: pool.models for name, pool in get_pools().items()}).items():
        first = earliest.get(model_identity(router_id))
        if first is None:
            continue
        days = (now - first).days
        if days > MAX_SEAT_AGE_DAYS and model_identity(router_id) not in STALE_SEAT_WAIVERS:
            aged[router_id] = f'{days}d, held by {", ".join(held)}'

    assert not aged, (
        f'these seats were listed more than {MAX_SEAT_AGE_DAYS} days ago and no '
        f'successor has been flagged, so nothing else will ever ask about them: '
        f'{aged}. Reseat, or record the reason in STALE_SEAT_WAIVERS.'
    )


@live
def test_stale_seat_waivers_are_dropped_once_the_seat_is_young_again(listed):
    """A vendor shipping again should retire the excuse, not leave it standing."""
    earliest = _first_listed(listed)
    now = datetime.now(UTC)
    seated = {model_identity(r) for r in seat_holders({name: pool.models for name, pool in get_pools().items()})}

    pointless = sorted(
        identity
        for identity in STALE_SEAT_WAIVERS
        if identity not in seated or (identity in earliest and (now - earliest[identity]).days <= MAX_SEAT_AGE_DAYS)
    )

    assert not pointless, (
        f'{pointless} are excused for age but are either no longer seated or no longer old. Drop the entries.'
    )


@live
def test_unscored_seat_waivers_are_dropped_once_the_seat_is_scored(garden):
    """A card that gains a score should retire the excuse, not leave it standing."""
    pointless = sorted(
        router_id
        for router_id in UNSCORED_SEAT_WAIVERS
        if router_id not in all_router_ids() or router_id in garden or router_id in MANUAL_MODEL_CARDS
    )

    assert not pointless, f'{pointless} are excused for having no score but are unseated or now scored. Drop the entries.'


@live
def test_cheap_aggregate_matches_the_live_derivation(garden):
    """cheap-aggregate is the live top three families under MAX_BLEND, reserve the fourth.

    Fails with both lists when the ranking moves, so a reseat is a deliberate,
    reviewed edit to the preset and never a silent drift.
    """
    ranked = rank_cheapest_top(with_aa_fills(with_manual_cards(garden)), judge_family)
    derived = [m.router_id for m in ranked]
    preset = PRESETS['cheap-aggregate']

    table = '\n'.join(f'  {m.router_id}  index {m.default_index}  blend {m.price:.3f}' for m in ranked[:6])
    assert (derived[:3], derived[3:4]) == (list(preset.judges), list(preset.reserve_judges)), (
        f'live ranking moved.\nderived seats {derived[:3]} reserve {derived[3:4]}\n'
        f'preset seats {list(preset.judges)} reserve {list(preset.reserve_judges)}\n'
        f'top of the ranking:\n{table}\n'
        'Reseat cheap-aggregate deliberately: update judges, reserve, estimate and docs/jury-presets.md.'
    )
