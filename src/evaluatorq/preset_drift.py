"""Deterministic drift report for jury presets and router pools (RES-1346).

The weekly check says *that* something drifted. This module says *what* and lists
the options that arithmetic allows: which reserve can replace a retired seat,
which same-lineage model beats a seat, which newer model nobody has reviewed.
It decides nothing. A person or an agent still picks, and must confirm on the web
that a candidate really is newer or superseded, because the catalogue knows
listing dates, not release notes.

`drift_report()` is pure over a garden and a listing map, so it runs offline on
hand-built cards. `python -m evaluatorq.preset_drift` reads the live garden and
prints the report as JSON (needs a workspace-scoped `ORQ_API_KEY`).
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from evaluatorq.common.model_frontier import (
    MANUAL_MODEL_CARDS,
    ScoredModel,
    headline_contradicts_its_own_rung,
    in_family_upgrades,
    public_listings,
    rank_cheapest_top,
    unseatable_reason,
    with_aa_fills,
    with_manual_cards,
)
from evaluatorq.jury_presets import (
    DOMINANCE_EXEMPTIONS,
    MAX_SEAT_AGE_DAYS,
    PRESETS,
    REVIEWED_SUCCESSORS,
    STALE_SEAT_WAIVERS,
    all_router_ids,
    judge_family,
    model_identity,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

UTC = timezone.utc

FindingKind = Literal[
    'not_served',
    'cost_drift',
    'unreviewed_successor',
    'in_family_upgrade',
    'aged_seat',
    'cheap_aggregate_moved',
    'unseatable',
    'contradicted_card',
    'unscored_seat',
    'stale_waiver',
    'stale_exemption',
    'stale_reviewed_successor',
    'pool_underivable',
]


class Finding(BaseModel):
    kind: FindingKind
    subject: str = Field(description='The seat, model, preset or register entry the finding is about.')
    held_by: list[str] = Field(default_factory=list, description='Presets or pools that seat the subject.')
    detail: str
    options: list[str] = Field(
        default_factory=list,
        description='Candidates the arithmetic allows. Each is a suggestion, not a decision.',
    )
    needs_human: bool = Field(
        default=False,
        description='True when the code found no option, so an agent cannot propose a fix on its own.',
    )


class DriftReport(BaseModel):
    findings: list[Finding] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def needs_human(self) -> bool:
        return any(f.needs_human for f in self.findings)


def earliest_listing(listed: Mapping[str, int | None]) -> dict[str, int]:
    """Earliest listing per model identity, in epoch milliseconds, across every host of it.

    A second host relisting the same weights carries its own `created` stamp,
    which is younger than the model, so only the earliest counts.
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


def first_listed(listed: Mapping[str, int | None]) -> dict[str, datetime]:
    """`earliest_listing` as datetimes, for the age check."""
    return {identity: datetime.fromtimestamp(ms / 1000, UTC) for identity, ms in earliest_listing(listed).items()}


def unreviewed_successors(listed: Mapping[str, int | None]) -> dict[str, str]:
    """Garden models newer than everything a preset seats from their own lineage.

    Compared against the newest seat per lineage, identity-normalised and dated by
    the identity's first listing, so the same weights on a second host are
    neither a second decision nor a newer model. Maps identity to a router id.
    """
    seated_identities = {model_identity(r) for r in all_router_ids()}
    earliest = earliest_listing(listed)

    newest_seated: dict[str, int] = {}
    for router_id in all_router_ids():
        if router_id not in listed:
            continue
        if (stamp := earliest.get(model_identity(router_id))) is None:
            continue
        family = judge_family(router_id)
        newest_seated[family] = max(newest_seated.get(family, 0), stamp)

    candidates: dict[str, str] = {}
    for model_id in sorted(listed):
        # Research-workspace endpoints are not generally served, and image models are not judges.
        if model_id.startswith('orq-research@') or 'image' in model_id:
            continue
        identity = model_identity(model_id)
        if identity in seated_identities or identity in candidates:
            continue
        try:
            family = judge_family(model_id)
        except ValueError:
            continue  # an unclassifiable lineage cannot be a successor to ours
        listed_at = earliest.get(identity)
        if listed_at is None:
            continue  # undated: cannot be shown newer than a seat
        if family in newest_seated and listed_at > newest_seated[family]:
            candidates[identity] = model_id
    return candidates


def seat_holders(pools: Mapping[str, tuple[str, ...]] | None = None) -> dict[str, list[str]]:
    """Every seat in the library, jury and pool alike, with where it sits."""
    seats: dict[str, list[str]] = {}
    for preset in PRESETS.values():
        for judge in preset.judges:
            seats.setdefault(judge, []).append(f'jury {preset.name}')
    for pool_name, members in (pools or {}).items():
        for member in members:
            seats.setdefault(member, []).append(f'pool {pool_name}')
    return seats


def _describe(model: ScoredModel) -> str:
    return f'{model.router_id} (index {model.default_index}, blend ${model.price:.3f})'


@dataclass
class _Ctx:
    garden: dict[str, ScoredModel]
    listed: Mapping[str, int | None]
    now: datetime
    reachable: dict[str, ScoredModel]
    models: list[ScoredModel]
    holders: dict[str, list[str]]
    successors: dict[str, str]
    earliest: dict[str, datetime]


def _not_served(ctx: _Ctx) -> list[Finding]:
    """A seat the catalogue no longer serves."""
    found: list[Finding] = []
    for router_id in sorted(all_router_ids()):
        if router_id in ctx.listed or router_id in MANUAL_MODEL_CARDS:
            continue
        presets = [p for p in PRESETS.values() if router_id in (*p.judges, *p.reserve_judges)]
        reserves = [r for p in presets if router_id in p.judges for r in p.reserve_judges]
        found.append(
            Finding(
                kind='not_served',
                subject=router_id,
                held_by=[p.name for p in presets],
                detail='Named by a preset but no longer listed in the public catalogue.',
                options=[f'promote reserve {r}' for r in reserves],
                needs_human=not reserves,
            )
        )
    return found


def _cost_drift(ctx: _Ctx) -> list[Finding]:
    """A published price that no longer matches the live cards."""
    found: list[Finding] = []
    for preset in PRESETS.values():
        try:
            live = preset.cost_per_1k(ctx.reachable)
        except KeyError:
            continue  # a seat with no card is reported by _bad_cards, not as a price
        if live != preset.estimated_cost_per_1k:
            found.append(
                Finding(
                    kind='cost_drift',
                    subject=preset.name,
                    held_by=[preset.name],
                    detail=f'Publishes ${preset.estimated_cost_per_1k} per 1k; live cards price it at ${live}.',
                    options=[f'set estimated_cost_per_1k={live} and update the docs table'],
                )
            )
    return found


def _bad_cards(ctx: _Ctx) -> list[Finding]:
    """Cards that cannot be seated, contradict themselves, or carry no score."""
    found: list[Finding] = []
    for preset in PRESETS.values():
        for router_id in (*preset.judges, *preset.reserve_judges):
            model = ctx.reachable.get(router_id)
            if model is None:
                found.append(
                    Finding(
                        kind='unscored_seat',
                        subject=router_id,
                        held_by=[preset.name],
                        detail='No card with an intelligence and price pair, so no upgrade check runs against it.',
                        needs_human=True,
                    )
                )
            elif reason := unseatable_reason(model):
                found.append(
                    Finding(
                        kind='unseatable', subject=router_id, held_by=[preset.name], detail=reason, needs_human=True
                    )
                )
    found.extend(
        Finding(
            kind='contradicted_card',
            subject=router_id,
            detail=f'Headline {model.intelligence_index} against rungs {model.effort_indices}.',
            needs_human=True,
        )
        for router_id, model in sorted(ctx.garden.items())
        if headline_contradicts_its_own_rung(model)
    )
    return found


def _unreviewed(ctx: _Ctx) -> list[Finding]:
    """Newer models in a seated lineage nobody has reviewed."""
    found: list[Finding] = []
    for identity, router_id in sorted(ctx.successors.items()):
        if identity in REVIEWED_SUCCESSORS:
            continue
        card = ctx.reachable.get(router_id)
        seat = f'seat {_describe(card)}' if card else f'seat {router_id} (no card scored)'
        found.append(
            Finding(
                kind='unreviewed_successor',
                subject=router_id,
                detail=(
                    'First listed after the newest seat in its lineage. Confirm on the vendor site that it is '
                    'a successor and not a variant or a relisting.'
                ),
                options=[seat, f'add {identity!r} to REVIEWED_SUCCESSORS with the reason it stays out'],
            )
        )
    return found


def _upgrades(ctx: _Ctx) -> list[Finding]:
    """A seat a same-lineage model beats on index and price."""
    found: list[Finding] = []
    for preset in PRESETS.values():
        for judge in preset.judges:
            incumbent = ctx.reachable.get(judge)
            if incumbent is None or judge in DOMINANCE_EXEMPTIONS:
                continue
            upgrades = in_family_upgrades(incumbent, ctx.models, judge_family, model_identity)
            if upgrades:
                found.append(
                    Finding(
                        kind='in_family_upgrade',
                        subject=judge,
                        held_by=[preset.name],
                        detail=f'Beaten on index and price by a same-lineage, same-region model: {_describe(incumbent)}.',
                        options=[
                            *(f'reseat to {_describe(u)}' for u in upgrades[:3]),
                            f'add {judge!r} to DOMINANCE_EXEMPTIONS with the reason it stays',
                        ],
                    )
                )
    return found


def _aged(ctx: _Ctx) -> list[Finding]:
    """A seat older than the age limit with no written waiver."""
    found: list[Finding] = []
    for router_id, held in sorted(ctx.holders.items()):
        first = ctx.earliest.get(model_identity(router_id))
        if first is None:
            continue
        days = (ctx.now - first).days
        if days > MAX_SEAT_AGE_DAYS and model_identity(router_id) not in STALE_SEAT_WAIVERS:
            family = judge_family(router_id)
            newer = [r for r in ctx.successors.values() if judge_family(r) == family]
            found.append(
                Finding(
                    kind='aged_seat',
                    subject=router_id,
                    held_by=held,
                    detail=f'First listed {days} days ago; the limit is {MAX_SEAT_AGE_DAYS}.',
                    options=[
                        *(f'reseat to {r}' for r in newer),
                        f'add {model_identity(router_id)!r} to STALE_SEAT_WAIVERS with the reason it stays',
                    ],
                )
            )
    return found


def _cheap_aggregate(ctx: _Ctx) -> list[Finding]:
    """cheap-aggregate is derived, so it must equal the live ranking."""
    derived = [m.router_id for m in rank_cheapest_top(ctx.reachable, judge_family)]
    cheap = PRESETS['cheap-aggregate']
    if (derived[:3], derived[3:4]) == (list(cheap.judges), list(cheap.reserve_judges)):
        return []
    return [
        Finding(
            kind='cheap_aggregate_moved',
            subject='cheap-aggregate',
            held_by=['cheap-aggregate'],
            detail=(
                f'Preset seats {list(cheap.judges)} reserve {list(cheap.reserve_judges)}; '
                f'live ranking gives seats {derived[:3]} reserve {derived[3:4]}.'
            ),
            options=[f'set judges={derived[:3]}, reserve_judges={derived[3:4]}, then the estimate and the docs'],
        )
    ]


def _stale_registers(ctx: _Ctx) -> list[Finding]:
    """Register entries that no longer apply."""
    found: list[Finding] = [
        Finding(
            kind='stale_reviewed_successor',
            subject=identity,
            detail='Recorded as a reviewed successor but no seat makes it a successor to anything.',
            options=[f'delete {identity!r} from REVIEWED_SUCCESSORS'],
        )
        for identity in sorted(set(REVIEWED_SUCCESSORS) - set(ctx.successors))
    ]
    seated = {model_identity(r) for r in ctx.holders}
    for identity in sorted(STALE_SEAT_WAIVERS):
        young = identity in ctx.earliest and (ctx.now - ctx.earliest[identity]).days <= MAX_SEAT_AGE_DAYS
        if identity not in seated or young:
            found.append(
                Finding(
                    kind='stale_waiver',
                    subject=identity,
                    detail='Excused for age but no longer seated, or no longer old.',
                    options=[f'delete {identity!r} from STALE_SEAT_WAIVERS'],
                )
            )
    found.extend(
        Finding(
            kind='stale_exemption',
            subject=judge,
            detail='Exempt from a dominance check that no longer flags it.',
            options=[f'delete {judge!r} from DOMINANCE_EXEMPTIONS'],
        )
        for judge in sorted(DOMINANCE_EXEMPTIONS)
        if judge in ctx.reachable
        and not in_family_upgrades(ctx.reachable[judge], ctx.models, judge_family, model_identity)
    )
    return found


def drift_report(
    garden: dict[str, ScoredModel],
    listed: Mapping[str, int | None],
    *,
    pools: Mapping[str, tuple[str, ...]] | None = None,
    underivable_pools: Mapping[str, str] | None = None,
    reviewed_underivable: Mapping[str, object] | None = None,
    now: datetime | None = None,
) -> DriftReport:
    """Every way the shipped presets and pools disagree with the live garden, with options."""
    reachable = with_aa_fills(with_manual_cards(garden))
    ctx = _Ctx(
        garden=garden,
        listed=listed,
        now=now or datetime.now(UTC),
        reachable=reachable,
        models=list(reachable.values()),
        holders=seat_holders(pools),
        successors=unreviewed_successors(listed),
        earliest=first_listed(listed),
    )
    findings: list[Finding] = []
    for check in (
        _not_served,
        _cost_drift,
        _bad_cards,
        _unreviewed,
        _upgrades,
        _aged,
        _cheap_aggregate,
        _stale_registers,
    ):
        findings.extend(check(ctx))
    # A pool shape that cannot be derived and nobody has reviewed.
    findings.extend(
        Finding(
            kind='pool_underivable',
            subject=name,
            detail=why,
            options=['add it to REVIEWED_UNDERIVABLE_POOLS with the reason, or add a dropped-pool reason'],
            needs_human=True,
        )
        for name, why in sorted((underivable_pools or {}).items())
        if name not in (reviewed_underivable or {})
    )
    return DriftReport(findings=findings)


def live_report() -> DriftReport:
    """Read the live garden and the public catalogue, and report on them."""
    from evaluatorq.common.model_frontier import load_garden_sync
    from evaluatorq.router_pools import REVIEWED_UNDERIVABLE_POOLS, get_pools, get_underivable_pools

    garden = load_garden_sync()
    listed = asyncio.run(public_listings())
    pools = {name: pool.models for name, pool in get_pools().items()}
    return drift_report(
        garden,
        listed,
        pools=pools,
        underivable_pools=get_underivable_pools(),
        reviewed_underivable=REVIEWED_UNDERIVABLE_POOLS,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Report how jury presets and router pools have drifted from the live garden.'
    )
    parser.add_argument('--output', help='Write the JSON report here instead of stdout.')
    args = parser.parse_args(argv)
    report = live_report()
    text = report.model_dump_json(indent=2)
    if args.output:
        pathlib.Path(args.output).write_text(text + '\n', encoding='utf-8')
    else:
        sys.stdout.write(text + '\n')
    return 0 if report.clean else 1


if __name__ == '__main__':
    raise SystemExit(main())
