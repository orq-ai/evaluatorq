"""Result types shared by every signal: a value, its evidence, and what the trajectory did or did not provide."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from evaluatorq.formats.atif import AtifTrajectory
    from evaluatorq.signals.config import SignalsConfig

Group = Literal['A', 'B', 'C', 'D']
"""ADR-25 groups: A structure (BOPS-1207), B tools (BOPS-1208), C autonomy (BOPS-1261), D tags (BOPS-1262)."""

_FROZEN = ConfigDict(frozen=True)


class Evidence(BaseModel):
    """A step (and optionally a tool call) that contributed to a signal's value.

    A step is addressed by `step_id` plus `agent_path`, the `trajectory_id`s from the root down to the trajectory
    holding it (the root is `()`), because step ids restart in every subagent trajectory. `related` links other
    steps involved (the original a duplicate repeats, the failed call a retry follows, the other members of a
    consecutive run). `reason` is only set when it says something the signal name does not. `subgroup` refines
    `reason` for the second level of the tool-count breakdown.

    Call-based signals also name the tool call: an agent step can hold several parallel calls, so `step_id`
    alone cannot say which one is the duplicate. `call_id` is the call this item is about, `related_call_ids`
    the calls behind `related` (the original, the failed attempt, the rest of a run, the spawning call).
    """

    model_config = _FROZEN
    step_id: int
    agent_path: tuple[str, ...] = ()
    call_id: str | None = None
    related: list[tuple[tuple[str, ...], int]] = Field(default_factory=list)
    related_call_ids: list[str] = Field(default_factory=list)
    reason: str = ''
    subgroup: str | None = None


class Precondition(BaseModel):
    """What a signal needs from the trajectory, and whether this trajectory provides it.

    `met` is True, False or `'partial'` (some steps qualify, some do not; `detail` says which). A `required`
    precondition that is not met makes the signal no-basis. A non-required one only qualifies the value
    (coverage, approximation).
    """

    model_config = _FROZEN
    name: str
    met: bool | Literal['partial']
    detail: str = ''
    required: bool = True


class SignalResult(BaseModel):
    """The value of one signal on one trajectory, with the steps that produced it.

    `no_basis` says why the signal cannot be computed; when it is set `value` is None, so a signal never
    reports a number it had no basis for.
    """

    model_config = _FROZEN
    name: str
    group: Group
    value: Any = None
    evidence: list[Evidence] = Field(default_factory=list)
    approximate: bool = False
    no_basis: str | None = None
    preconditions: list[Precondition] = Field(default_factory=list)
    reason: str | None = None
    """Trajectory tags (group D): which rule clauses fired."""
    rule_version: str | None = None
    """Trajectory tags (group D): version of the thresholds that produced `value`."""

    @model_validator(mode='after')
    def _no_basis_has_no_value(self) -> SignalResult:
        if self.no_basis is not None and self.value is not None:
            msg = f'Signal {self.name!r} has no_basis set, so its value must be None (got {self.value!r})'
            raise ValueError(msg)
        return self


class SignalReport(BaseModel):
    """Every computed signal for one trajectory."""

    model_config = _FROZEN
    trajectory_id: str | None
    results: dict[str, SignalResult]
    config_version: str

    def values(self) -> dict[str, Any]:
        """Name to value for every signal with a basis; no-basis signals are omitted, not zero-filled (ADR-25)."""
        return {name: result.value for name, result in self.results.items() if result.no_basis is None}


SignalFn = Callable[['AtifTrajectory', 'SignalsConfig'], SignalResult]
"""A signal: a pure function of the trajectory and the config."""


def result(
    name: str,
    group: Group,
    value: Any,
    evidence: list[Evidence] | None = None,
    preconditions: list[Precondition] | None = None,
    *,
    approximate: bool = False,
    reason: str | None = None,
    rule_version: str | None = None,
) -> SignalResult:
    """Build a result; any required precondition that is not met makes it no-basis.

    Evidence is sorted by `(agent_path, step_id)` so the order is deterministic (a stable sort, so evidence for
    one step keeps the order the signal produced it in).
    """
    preconditions = preconditions or []
    failed = [p for p in preconditions if p.required and p.met is False]
    if failed:
        why = '; '.join(f'{p.name}: {p.detail}' for p in failed)
        return SignalResult(
            name=name,
            group=group,
            approximate=approximate,
            no_basis=why,
            preconditions=preconditions,
        )
    ordered = sorted(evidence or [], key=lambda e: (e.agent_path, e.step_id))
    return SignalResult(
        name=name,
        group=group,
        value=value,
        evidence=ordered,
        approximate=approximate,
        preconditions=preconditions,
        reason=reason,
        rule_version=rule_version,
    )
