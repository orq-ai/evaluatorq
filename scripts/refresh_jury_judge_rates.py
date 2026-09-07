#!/usr/bin/env python3
"""Re-capture `common/data/jury_judge_rates.json` from the live Orq catalogue.

The jury presets publish a $/1k figure per panel, recomputed on every run from
the billing rates in that table. The table is committed rather than fetched,
because a preset ships to customers on their own contracts and must not reprice
because our gateway is down. Committed is not the same as frozen: a provider
reprices and the published figure silently becomes wrong, which is the failure
mode this script exists to catch.

What it refreshes, and what it deliberately does not:

* `input_rate` / `output_rate` are read from `GET /v2/models` via
  `common.model_catalogue`, so there is one parser for that payload rather than
  two. This is the only endpoint that publishes both the rates and the accepted
  reasoning-effort values; `/v2/model-catalog` is unauthenticated and lists more
  models but carries neither.
* `seated_effort` and `priced_at_ceiling` are **carried over untouched**. Both
  come from the intelligence/price frontier derivation, which lives in the
  research repo and is rerun and argued there. Guessing them here would put a
  number in front of customers that nothing derived.

Nothing is decided automatically. A repricing invalidates a published
`estimated_cost_per_1k` and a retirement means promoting a reserve; both are
printed as decisions for a human to act on.

RES-1528 will replace this with `orq/auto` / `orq/frontier`, which compute the
same selection server-side and update on every catalog sync. Until those
endpoints are usable from here, this is how the table stays honest.

Usage:

```bash
ORQ_API_KEY=... uv run python scripts/refresh_jury_judge_rates.py
ORQ_API_KEY=... uv run python scripts/refresh_jury_judge_rates.py --check
```

`--check` writes nothing and exits 1 if the table would change, which is what
the monthly workflow runs before it opens a pull request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.model_catalogue import get_model_info

if TYPE_CHECKING:
    from evaluatorq.common.model_catalogue import ModelInfo

RATES_PATH = Path(__file__).resolve().parent.parent / 'src/evaluatorq/common/data/jury_judge_rates.json'

# Fields the frontier derivation owns. Carried over from the previous capture
# rather than recomputed, because the derivation is not in this repository.
DERIVED_FIELDS = ('seated_effort', 'priced_at_ceiling')


def refreshed_row(previous: dict[str, Any], info: ModelInfo) -> dict[str, Any]:
    """One judge's row: live rates, carried-over derivation, original key order.

    `/v2/models` quotes per 1000 tokens and the table is per 1M, which is the
    unit the published $/1k arithmetic and every seat comment are written in.
    """
    row = {
        'input_rate': round(info.input_cost_per_1k * 1000, 6),
        'output_rate': round(info.output_cost_per_1k * 1000, 6),
    }
    row.update({field: previous[field] for field in DERIVED_FIELDS if field in previous})
    return row


def decisions(
    previous: dict[str, Any], current: dict[str, Any], efforts: dict[str, frozenset[str] | None]
) -> list[str]:
    """What a human has to act on before this capture can be committed."""
    lines: list[str] = []
    for router_id, before in previous.items():
        after = current.get(router_id)
        if after is None:
            lines.append(
                f'NOT SERVED: {router_id} is named by a preset but absent from /v2/models. '
                'Retired, or not enabled for this key - check before promoting a reserve.'
            )
            continue
        if (before['input_rate'], before['output_rate']) != (after['input_rate'], after['output_rate']):
            lines.append(
                f'REPRICED: {router_id} {before["input_rate"]}/{before["output_rate"]} -> '
                f'{after["input_rate"]}/{after["output_rate"]} per 1M. Every panel seating it '
                'publishes a stale $/1k until the preset figure is recomputed.'
            )

    # The check that would have caught `reasoning` being sent as an effort: the
    # catalogue publishes the values a model accepts, and a seat ranked at a rung
    # outside that set cannot be called where it was costed.
    for router_id, row in current.items():
        seated, accepted = row.get('seated_effort'), efforts.get(router_id)
        if seated is None or accepted is None or seated in accepted:
            continue
        lines.append(
            f'EFFORT NOT ACCEPTED: {router_id} is seated at {seated!r}, which is not among the '
            f'values /v2/models says it takes ({sorted(accepted)}). Either the seat is priced at a '
            'rung it cannot be called at, or the rung is a card dialect rather than a sendable value.'
        )
    return lines


async def capture(previous: dict[str, Any]) -> tuple[dict[str, Any], dict[str, frozenset[str] | None]]:
    """Live rates for every judge in the table, plus the efforts each one accepts."""
    if not os.environ.get('ORQ_API_KEY'):
        raise SystemExit('ORQ_API_KEY is required: /v2/models is authenticated and project-scoped.')
    resolved = resolve_llm_client(require_orq=True)
    current: dict[str, Any] = {}
    efforts: dict[str, frozenset[str] | None] = {}
    try:
        for router_id, row in previous.items():
            info = await get_model_info(router_id, resolved.client)
            if info is None:
                continue
            current[router_id] = refreshed_row(row, info)
            efforts[router_id] = info.reasoning_efforts
    finally:
        if resolved.owned:
            await resolved.client.close()

    # A rejected key, an outage, or a project with no models enabled all reach
    # here as "every judge is missing", which reads identically to fifteen
    # simultaneous retirements. That is never the real answer, and reporting it
    # as one would hand a reviewer fifteen decisions to make about nothing.
    if not current:
        raise SystemExit(
            'No judge in the table resolved against /v2/models. That is an access or availability '
            'failure, not a catalogue of retirements - check the key and the project scope. '
            'Run with EVALUATORQ_LOG_LEVEL=DEBUG to see the underlying response.'
        )
    return current, efforts


def main() -> int:
    parser = argparse.ArgumentParser(description='Re-capture the jury judge rate table.')
    parser.add_argument('--out', type=Path, default=RATES_PATH)
    parser.add_argument('--check', action='store_true', help='write nothing; exit 1 if the table would change')
    args = parser.parse_args()

    table = json.loads(args.out.read_text())
    previous = table['judges']
    current, efforts = asyncio.run(capture(previous))

    pending = decisions(previous, current, efforts)
    for line in pending:
        print(line)

    # A judge that vanished from the catalogue keeps its committed row: dropping
    # it would make the preset name a seat with no price and take the published
    # figure down with it. The decision above is the signal; the data stays put
    # until a human promotes the reserve.
    merged = {router_id: current.get(router_id, row) for router_id, row in previous.items()}
    unchanged = merged == previous

    if args.check:
        # Green means nothing needs a human. A decision with an unchanged table
        # still needs one: a retired judge leaves its committed rates in place.
        print('nothing to act on' if unchanged and not pending else 'the table needs review')
        return 0 if unchanged and not pending else 1

    if unchanged:
        print('no rate changes; leaving captured_at alone so an unchanged table stays a no-op diff')
        return 0

    args.out.write_text(
        json.dumps({**table, 'captured_at': datetime.now(timezone.utc).isoformat(), 'judges': merged}, indent=2) + '\n'
    )
    print(f'wrote {args.out} ({len(merged)} judges)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
