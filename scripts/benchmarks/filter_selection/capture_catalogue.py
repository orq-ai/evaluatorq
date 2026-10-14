"""Capture a live facet catalogue snapshot for the filter-selection benchmark.

Usage: uv run python scripts/benchmarks/filter_selection/capture_catalogue.py [--profile research-bauke]

The API key is read in-process from the `orq` CLI profile and never printed or passed on argv.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from evaluatorq.common.orq_client import (
    DEFAULT_ORQ_BASE_URL,
    close_orq_client,
    list_orq_profiles,
    resolve_orq_client,
)
from evaluatorq.trace_finder.facets import load_facet_catalogue

OUT = Path(__file__).with_name('catalogue.json')
LIMIT = 50
WINDOW_DAYS = 30


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


async def capture(profile_name: str) -> dict:
    profile = next((p for p in list_orq_profiles() if p.name == profile_name), None)
    if profile is None:
        raise SystemExit(f'orq profile {profile_name!r} not found')
    orq = resolve_orq_client(profile.api_key, base_url=profile.server or DEFAULT_ORQ_BASE_URL)
    try:
        end = datetime.now(UTC).replace(microsecond=0)
        start = end - timedelta(days=WINDOW_DAYS)
        catalogue = await load_facet_catalogue(orq, start=start, end=end, limit=LIMIT)
    finally:
        await close_orq_client(orq)
    return {
        'profile': profile_name,
        'captured_at': _iso(end),
        'window_start': _iso(start),
        'window_end': _iso(end),
        'limit': LIMIT,
        'catalogue': catalogue.model_dump(mode='json'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default='research-bauke')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    snapshot = asyncio.run(capture(args.profile))
    args.out.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    cat = snapshot['catalogue']
    for facet in ('project', 'status', 'product', 'trace_type', 'model', 'provider', 'agent_name', 'tool_name'):
        print(f'{facet}: {len(cat[facet])} values')
    print('truncated:', sorted(cat['truncated_facets']))
    print('captured_at:', snapshot['captured_at'])


if __name__ == '__main__':
    main()
