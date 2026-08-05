#!/usr/bin/env python3
"""Resolve the three newest stable minor releases of the Orq AI SDK."""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from typing import Any

PYPI_URL = 'https://pypi.org/pypi/orq-ai-sdk/json'
VERSION_PATTERN = re.compile(r'^(\d+)\.(\d+)\.(\d+)$')


def _resolve_versions(payload: dict[str, Any]) -> list[str]:
    """Return the newest patch for each of the three newest stable minors."""
    releases = payload.get('releases')
    if not isinstance(releases, dict):
        raise TypeError('PyPI response does not contain a releases object')

    newest_by_minor: dict[tuple[int, int], tuple[int, str]] = {}
    for version in releases:
        match = VERSION_PATTERN.fullmatch(version)
        if match is None:
            continue

        major, minor, patch = (int(part) for part in match.groups())
        minor_key = (major, minor)
        current = newest_by_minor.get(minor_key)
        if current is None or patch > current[0]:
            newest_by_minor[minor_key] = (patch, version)

    newest_minors = sorted(newest_by_minor, reverse=True)
    if len(newest_minors) < 3:
        raise ValueError(f'PyPI response contains only {len(newest_minors)} stable SDK minors; need at least 3')

    return [newest_by_minor[minor][1] for minor in newest_minors[:3]]


def _self_check() -> None:
    """Check numeric sorting, patch selection, and pre-release filtering."""
    sample_payload = {
        'releases': {
            '4.9.0': [],
            '4.10.0': [],
            '4.10.1': [],
            '4.11.2': [],
            '4.11.10': [],
            '4.12.9': [],
            '4.12.17': [],
            '4.13.0rc3': [],
        }
    }
    assert _resolve_versions(sample_payload) == ['4.12.17', '4.11.10', '4.10.1']  # noqa: S101


def main() -> int:
    _self_check()

    try:
        with urllib.request.urlopen(PYPI_URL, timeout=30) as response:
            payload = json.load(response)
    except (OSError, TimeoutError) as exc:
        print(f'Error: failed to fetch {PYPI_URL}: {exc}', file=sys.stderr)
        return 1
    except (UnicodeError, json.JSONDecodeError) as exc:
        print(f'Error: received invalid JSON from {PYPI_URL}: {exc}', file=sys.stderr)
        return 1

    try:
        if not isinstance(payload, dict):
            raise TypeError('PyPI response is not a JSON object')
        versions = _resolve_versions(payload)
    except (TypeError, ValueError) as exc:
        print(f'Error: could not resolve a stable SDK matrix: {exc}', file=sys.stderr)
        return 1

    print(json.dumps(versions, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
