"""The signal registry and `compute_signals`.

Each group module (`structure`, `tools`, `autonomy`, `tags`) exports one explicit dict of `name -> (group, fn)`.
Adding a group is one import and one entry in the `_merge(...)` call below; there is no decorator registration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.signals import autonomy, structure, tags, tools
from evaluatorq.signals.config import SignalsConfig
from evaluatorq.signals.models import Group, SignalFn, SignalReport, SignalResult
from evaluatorq.signals.walk import SignalContext

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


SIGNALS: SignalTable = _merge(structure.SIGNALS, tools.SIGNALS, autonomy.SIGNALS, tags.SIGNALS)
SIGNAL_NAMES: tuple[str, ...] = tuple(SIGNALS)


def compute_signals(
    trajectory: AtifTrajectory, config: SignalsConfig | None = None, only: Iterable[str] | None = None
) -> SignalReport:
    """Run the signals on a trajectory, in registry order.

    A signal that raises is reported as no-basis with the exception type and message (and logged), so one broken
    rule does not hide the others. The trajectory is walked once into a `SignalContext` that every signal shares;
    `ctx.results` fills in registry order as signals finish.

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
    if wanted is not None:
        pending = list(wanted)
        while pending:
            name = pending.pop()
            pending.extend(dependency for dependency in tags.TAG_DEPENDENCIES.get(name, ()) if dependency not in wanted)
            wanted.update(tags.TAG_DEPENDENCIES.get(name, ()))
    selected = {name: entry for name, entry in SIGNALS.items() if wanted is None or name in wanted}
    report = partial(SignalReport, trajectory_id=trajectory.trajectory_id, config_version=config.version)
    results: dict[str, SignalResult] = {}
    try:
        ctx = SignalContext.build(trajectory, config)
    except Exception as exc:  # noqa: BLE001 - failure isolation: a walk that fails must still yield a report
        logger.warning('Walking the trajectory failed: {}: {}', type(exc).__name__, exc)
        why = f'walk failed: {type(exc).__name__}: {exc}'
        return report(results={n: SignalResult(name=n, group=g, no_basis=why) for n, (g, _) in selected.items()})
    for name, (group, fn) in selected.items():
        try:
            results[name] = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - failure isolation: one broken signal must not hide the others
            logger.warning('Signal {} failed: {}: {}', name, type(exc).__name__, exc)
            results[name] = SignalResult(name=name, group=group, no_basis=f'signal raised {type(exc).__name__}: {exc}')
        ctx.results[name] = results[name]
    return report(results=results)
