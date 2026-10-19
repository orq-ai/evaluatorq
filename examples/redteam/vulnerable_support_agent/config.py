"""Shared config for the vulnerable support agent demo."""

from __future__ import annotations

import os

# Attacker and judge model, served through the orq router. A capable model here
# writes strong attacks and scores them reliably.
MODEL = os.environ.get("DEMO_MODEL", "openai/gpt-5.4-mini")

# The target (Ava) model, separate so the agent under test follows its own weak
# instructions while the attacker and judge stay strong. A safety-heavy frontier
# model resists the attacks on its own training regardless of Ava's prompt, which
# hides the agent-level vulnerability this demo exists to show; a more
# instruction-following model exposes it. With gpt-4o-mini as the target, the
# recorded run lost 2 of 3 goal-hijacking attacks and 0 of 3 prompt-injection
# attacks; dynamic runs vary. Override with DEMO_TARGET_MODEL to try another target.
TARGET_MODEL = os.environ.get("DEMO_TARGET_MODEL", "openai/gpt-4o-mini")
