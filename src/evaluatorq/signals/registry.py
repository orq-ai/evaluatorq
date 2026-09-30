"""The signal registry and `compute_signals`.

Each group module (`structure`, `tools`, `autonomy`, `tags`) exports one explicit dict of `name -> (group, fn)`.
Adding a group is one import and one entry in the `_merge(...)` call below; there is no decorator registration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.signals.config import SignalsConfig
from evaluatorq.signals.models import Group, SignalFn, SignalReport, SignalResult

if TYPE_CHECKING:
    from evaluatorq.formats.atif import AtifTrajectory

SignalTable = Mapping[str, tuple[Group, SignalFn]]


def _merge(*tables: SignalTable) -> SignalTable:
    """Combine the group tables, refusing a name two groups both define, and freeze the result."""
    merged: dict[str, tuple[Group, SignalFn]] = {}
    for table in tables:
        for name, entry in table.items():
            if name in merged:
                msg = f'Signal {name!r} is registered twice'
                raise ValueError(msg)
            merged[name] = entry
    return MappingProxyType(merged)


# Later tasks add: `from evaluatorq.signals import structure, tools, autonomy, tags` and
# `_merge(structure.SIGNALS, tools.SIGNALS, autonomy.SIGNALS, tags.SIGNALS)`.
SIGNALS: SignalTable = _merge()
SIGNAL_NAMES: tuple[str, ...] = tuple(SIGNALS)


def compute_signals(
    trajectory: AtifTrajectory, config: SignalsConfig | None = None, only: Iterable[str] | None = None
) -> SignalReport:
    """Run the signals on a trajectory, in registry order.

    A signal that raises is reported as no-basis with the exception type and message (and logged), so one broken
    rule does not hide the others.

    Args:
        trajectory: The trajectory to measure.
        config: Rule knobs; defaults to `SignalsConfig()`.
        only: Signal names to run; defaults to all of them.

    Raises:
        ValueError: `only` names a signal that is not registered.
    """
    config = config or SignalsConfig()
    wanted = None if only is None else set(only)
    if wanted is not None and (unknown := sorted(wanted - set(SIGNALS))):
        msg = f'Unknown signal names: {", ".join(unknown)}'
        raise ValueError(msg)
    results: dict[str, SignalResult] = {}
    for name, (group, fn) in SIGNALS.items():
        if wanted is not None and name not in wanted:
            continue
        try:
            results[name] = fn(trajectory, config)
        except Exception as exc:  # noqa: BLE001 - failure isolation: one broken signal must not hide the others
            logger.warning('Signal {} failed: {}: {}', name, type(exc).__name__, exc)
            results[name] = SignalResult(name=name, group=group, no_basis=f'signal raised {type(exc).__name__}: {exc}')
    return SignalReport(trajectory_id=trajectory.trajectory_id, results=results, config_version=config.version)
