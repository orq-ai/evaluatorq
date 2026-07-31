"""Persisted pairwise (preference) judging runs.

:mod:`evaluatorq.pairwise` returns a :class:`~evaluatorq.pairwise.PairwiseComparison`
per A-vs-B pair and rolls a set of them up with ``build_report()``, but keeps
none of it. A comparison also drops the inputs that produced it, so on its own
it has no identity: nothing to label a table row with and no text to show
beside a judge's explanation.

``PairwiseRun`` is the persisted artifact that closes both gaps. It pairs each
comparison with the question and the two responses that produced it, records
the panel configuration, and stores the rollup computed once at save time. It
is what the dashboard's ``pairwise`` surface loads.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from evaluatorq.common.run_store_dir import get_store_dir
from evaluatorq.pairwise import PairwiseComparison, PairwiseReport, build_report

if TYPE_CHECKING:
    from collections.abc import Sequence


def get_pairwise_runs_dir() -> Path:
    """Return the pairwise runs directory (``<store>/pairwise-runs``)."""
    return get_store_dir('pairwise-runs')


class PairwiseEntry(BaseModel):
    """One comparison together with the inputs that produced it."""

    question: str = Field(description='The question or task both responses answer')
    response_a: str = Field(description='The response held in slot A')
    response_b: str = Field(description='The response held in slot B')
    comparison: PairwiseComparison = Field(description="The panel's reconciled result for this pair")


class PairwiseRun(BaseModel):
    """A saved pairwise run: panel configuration, every comparison, and the rollup."""

    # Surface discriminator for the dashboard file scanner. Named for what it
    # distinguishes rather than reusing 'mode', which simulation already claims
    # with different values.
    judging: Literal['pairwise'] = 'pairwise'

    run_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # "A" and "B" are slot letters, not names. Nothing in the judging data says
    # what was compared, so a reader of the dashboard cannot tell what winning
    # side A even means. These name the two systems under comparison and every
    # part of the view uses them in place of the bare letters.
    label_a: str = 'A'
    label_b: str = 'B'

    judges: list[str] = Field(default_factory=list, description='Configured judge panel')
    criteria: str = Field(default='', description='What "better" meant for this run')
    # Recorded because JudgeStats.position_bias falls back to 0.0 when no pair
    # was flippable. Without this flag a single-ordering run is indistinguishable
    # from a panel with no position bias, which is the flattering reading and the
    # wrong one.
    swap: bool = True

    entries: list[PairwiseEntry] = Field(default_factory=list)
    report: PairwiseReport | None = Field(
        default=None,
        description='Rollup over ``entries``, computed at save time. A cached projection, not an input.',
    )

    def add(
        self,
        comparison: PairwiseComparison,
        *,
        question: str,
        response_a: str,
        response_b: str,
    ) -> None:
        """Append one judged comparison and the inputs it came from."""
        self.entries.append(
            PairwiseEntry(
                question=question,
                response_a=response_a,
                response_b=response_b,
                comparison=comparison,
            )
        )

    def comparisons(self) -> list[PairwiseComparison]:
        """The comparisons alone, in entry order."""
        return [e.comparison for e in self.entries]

    def rollup(self) -> PairwiseReport:
        """Roll the entries up, preferring the stored report when present."""
        return self.report if self.report is not None else build_report(self.comparisons())

    def save(self, path: Path | str | None = None) -> Path:
        """Compute the rollup and write the run to JSON.

        The rollup is computed here rather than on every dashboard render, so
        the stored report always matches the entries it was built from.
        """
        self.report = build_report(self.comparisons())
        if path is None:
            runs_dir = get_pairwise_runs_dir()
            runs_dir.mkdir(parents=True, exist_ok=True)
            stamp = self.created_at.strftime('%Y%m%d_%H%M%S')
            path = runs_dir / f'{stamp}_{_slug(self.run_name)}.json'
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding='utf-8')
        return target

    @classmethod
    def load(cls, path: Path | str) -> PairwiseRun:
        """Read a saved run back."""
        return cls.model_validate(json.loads(Path(path).read_text(encoding='utf-8')))


def _slug(text: str) -> str:
    cleaned = ''.join(c if c.isalnum() else '-' for c in text.lower())
    return '-'.join(part for part in cleaned.split('-') if part) or 'run'


def new_run(
    *,
    run_name: str,
    label_a: str = 'A',
    label_b: str = 'B',
    judges: Sequence[str] = (),
    criteria: str = '',
    swap: bool = True,
) -> PairwiseRun:
    """Start an empty run. Accumulate with ``add()``, then ``save()``."""
    return PairwiseRun(
        run_name=run_name,
        label_a=label_a,
        label_b=label_b,
        judges=list(judges),
        criteria=criteria,
        swap=swap,
    )
