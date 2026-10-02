#!/usr/bin/env python3
"""Re-capture `common/data/jury_judge_rates.json` from the live Orq catalogue.

The jury presets publish a $/1k figure per panel, recomputed on every run from
the billing rates in that table. The table is committed rather than fetched,
because a preset ships to customers on their own contracts and must not reprice
because our gateway is down. Committed is not the same as frozen: a provider
reprices and the published figure silently becomes wrong, which is the failure
mode this script exists to catch.

What it refreshes, all from `GET /v2/models` via `common.model_catalogue`, so
there is one parser for that payload rather than two. This is the only
endpoint that publishes both the rates and the reasoning-effort parameter;
`/v2/model-catalog` is unauthenticated and lists more models but carries
neither.

* `input_rate` / `output_rate`, the per-token billing rates.
* `default_reasoning_effort`, the effort each seat runs at, since presets call
  every judge without one.

Nothing is decided automatically. A repricing invalidates a published
`estimated_cost_per_1k`, a changed default changes how a seat judges, and a
retirement means promoting a reserve; all three are printed as decisions for a
human to act on.

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


def refreshed_row(info: ModelInfo) -> dict[str, Any] | None:
    """One judge's row: live rates and default effort, or None when the catalogue lists no price.

    `/v2/models` quotes per 1000 tokens and the table is per 1M, which is the
    unit the published $/1k arithmetic and every seat comment are written in.
    """
    if info.input_cost_per_1k is None or info.output_cost_per_1k is None:
        return None
    return {
        'input_rate': round(info.input_cost_per_1k * 1000, 6),
        'output_rate': round(info.output_cost_per_1k * 1000, 6),
        'default_reasoning_effort': info.default_reasoning_effort,
    }


def decisions(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """What a human has to act on before this capture can be committed."""
    lines: list[str] = []
    for router_id, before in previous.items():
        after = current.get(router_id)
        if after is None:
            lines.append(
                f'NOT SERVED: {router_id} is named by a preset but absent from /v2/models or listed '
                'without a price. Retired, repriced to nothing, or not enabled for this key - check '
                'before promoting a reserve.'
            )
            continue
        if (before['input_rate'], before['output_rate']) != (after['input_rate'], after['output_rate']):
            lines.append(
                f'REPRICED: {router_id} {before["input_rate"]}/{before["output_rate"]} -> '
                f'{after["input_rate"]}/{after["output_rate"]} per 1M. Every panel seating it '
                'publishes a stale $/1k until the preset figure is recomputed.'
            )
        if before.get('default_reasoning_effort') != after['default_reasoning_effort']:
            lines.append(
                f'DEFAULT EFFORT CHANGED: {router_id} {before.get("default_reasoning_effort")!r} -> '
                f'{after["default_reasoning_effort"]!r}. Every panel seating it now judges at a different effort.'
            )
    return lines


async def capture(previous: dict[str, Any]) -> dict[str, Any]:
    """Live rates and default effort for every judge in the table."""
    if not os.environ.get('ORQ_API_KEY'):
        raise SystemExit('ORQ_API_KEY is required: /v2/models is authenticated and project-scoped.')
    resolved = resolve_llm_client(require_orq=True)
    current: dict[str, Any] = {}
    try:
        for router_id in previous:
            info = await get_model_info(router_id, resolved.client)
            row = refreshed_row(info) if info is not None else None
            if row is not None:
                current[router_id] = row
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
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description='Re-capture the jury judge rate table.')
    parser.add_argument('--out', type=Path, default=RATES_PATH)
    parser.add_argument('--check', action='store_true', help='write nothing; exit 1 if the table would change')
    args = parser.parse_args()

    table = json.loads(args.out.read_text())
    previous = table['judges']
    current = asyncio.run(capture(previous))

    pending = decisions(previous, current)
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
