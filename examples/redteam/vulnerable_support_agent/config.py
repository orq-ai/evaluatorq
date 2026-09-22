"""Shared config for the vulnerable support agent demo."""

from __future__ import annotations

import os

# Model served through the orq router. Mirrors the crypto_stealing_demo default.
MODEL = os.environ.get("DEMO_MODEL", "openai/gpt-5.4-mini")
