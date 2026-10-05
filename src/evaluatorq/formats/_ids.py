"""Deterministic ids for the format converters (SHA-256, never uuid4)."""

from __future__ import annotations

import hashlib
import json


def stable_hex(*parts: str, length: int) -> str:
    """Return the first `length` hex chars of SHA-256 over `parts` joined by NUL."""
    digest = hashlib.sha256('\x00'.join(parts).encode('utf-8')).hexdigest()
    return digest[:length]


def content_seed(value: object) -> str:
    """Hash canonical JSON of `value`; the fallback seed when no session or trajectory id exists."""
    text = json.dumps(value, sort_keys=True, default=str, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
