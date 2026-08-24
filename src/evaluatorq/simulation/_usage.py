"""Per-instance token accounting shared by the simulation agents and generators."""

from __future__ import annotations

from evaluatorq.contracts import TokenUsage


class UsageTracking:
    """Cumulative `TokenUsage` for one agent or generator instance.

    Generation and simulation both run before any run object exists, so this
    accumulator is the only place a caller can read what they cost (RES-1295).

    Subclasses call `reset_usage()` from their own `__init__` to initialise the
    counter, and `_accumulate()` after every billed call — including the fallback
    rungs `generate_structured` burned on the way to an answer, not just the rung
    that answered.
    """

    # Shared zero, never mutated: `reset_usage` and `_accumulate` both rebind
    # onto the instance, so concurrent instances never touch this one.
    _usage: TokenUsage = TokenUsage()

    def get_usage(self) -> TokenUsage:
        """Token usage accumulated across every call on this instance."""
        return self._usage.model_copy()

    def reset_usage(self) -> None:
        """Zero the accumulator."""
        self._usage = TokenUsage()

    def _accumulate(self, delta: TokenUsage | None) -> None:
        """Fold one call's usage in. `None` — an unbilled or unreported call — is a no-op."""
        if delta is not None:
            self._usage = self._usage + delta
