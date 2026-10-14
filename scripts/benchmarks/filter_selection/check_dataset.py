"""Self-check dataset.jsonl against catalogue.json."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
FACETS = ('project', 'status', 'product', 'trace_type', 'model', 'provider', 'agent_name', 'tool_name')
NUMERIC = {'tokens_min', 'tokens_max', 'duration_ms_min', 'duration_ms_max'}
CATEGORIES = {
    'facet', 'multi_facet', 'status', 'numeric', 'time', 'combo', 'semantic_only', 'aggregate',
    'implied_trap', 'paraphrase', 'outside_top20',
}  # fmt: skip


def parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace('Z', '+00:00')).astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=HERE / 'dataset.jsonl')
    parser.add_argument('--catalogue', type=Path, default=HERE / 'catalogue.json')
    args = parser.parse_args()
    snapshot = json.loads(args.catalogue.read_text())
    cat = snapshot['catalogue']
    captured = snapshot['captured_at']
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    errors: list[str] = []
    ids = [r['id'] for r in rows]
    if ids != [f'q{i:02d}' for i in range(1, len(rows) + 1)]:
        errors.append(f'ids not q01..q{len(rows):02d} in order: {ids}')
    now = parse(captured)
    floor = parse(snapshot['window_start'])
    for r in rows:
        rid, exp = r['id'], r['expected']
        if r['now'] != captured:
            errors.append(f'{rid}: now != captured_at')
        if r['category'] not in CATEGORIES:
            errors.append(f'{rid}: bad category {r["category"]}')
        for facet, values in exp['facets'].items():
            if facet not in FACETS or not values:
                errors.append(f'{rid}: bad/empty facet {facet}')
                continue
            for value in values:
                if value not in cat[facet]:
                    errors.append(f'{rid}: {facet}={value!r} not in catalogue')
                elif r['category'] == 'outside_top20' and not 20 <= cat[facet].index(value) < 50:
                    errors.append(f'{rid}: {value!r} rank {cat[facet].index(value) + 1} not in 21..50')
                elif r['category'] != 'outside_top20' and cat[facet].index(value) >= 20:
                    errors.append(f'{rid}: {value!r} rank {cat[facet].index(value) + 1} is outside top 20')
        if r['category'] == 'outside_top20' and not exp['facets']:
            errors.append(f'{rid}: outside_top20 without facets')
        for key, value in exp['numeric'].items():
            if key not in NUMERIC or not isinstance(value, int):
                errors.append(f'{rid}: bad numeric {key}={value!r}')
        lo, hi = exp['numeric'].get('tokens_min'), exp['numeric'].get('tokens_max')
        if lo is not None and hi is not None and lo > hi:
            errors.append(f'{rid}: tokens_min > tokens_max')
        if exp['time'] is not None:
            start = parse(exp['time']['start']) if exp['time']['start'] else floor
            end = parse(exp['time']['end']) if exp['time']['end'] else now
            if not (floor <= start < end <= now):
                errors.append(f'{rid}: time {exp["time"]} outside retention [{floor}, {now}] or start>=end')
        if exp['aggregate'] and (exp['facets'] or exp['numeric'] or exp['time'] or exp['needs_judgment']):
            errors.append(f'{rid}: aggregate row has filters/judgment')
    print('rows:', len(rows))
    print('categories:', dict(Counter(r['category'] for r in rows)))
    print('values per facet:', {f: len(cat[f]) for f in FACETS})
    print('errors:', errors or 'none')
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
