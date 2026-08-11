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

_PAIRWISE_AGGREGATIONS = frozenset({'plurality', 'bt-sigma'})

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

    def rollup(self, *, aggregation: str = 'plurality') -> PairwiseReport:
        """Roll the entries up, preferring the stored report when present.

        A stored report is only reused when it satisfies the requested
        aggregation, so asking for ``'bt-sigma'`` on a run saved with plurality
        recomputes instead of silently returning a report without the block.
        """
        _validate_aggregation(aggregation)
        if self.report is not None and (aggregation == 'plurality' or self.report.bt_sigma is not None):
            return _annotate_bt_sigma_scope(self.report, self.entries)
        report = build_report(self.comparisons(), aggregation=aggregation)
        return _annotate_bt_sigma_scope(report, self.entries)

    def save(self, path: Path | str | None = None, *, aggregation: str = 'plurality') -> Path:
        """Compute the rollup and write the run to JSON.

        The rollup is computed here rather than on every dashboard render, so
        the stored report always matches the entries it was built from. Pass
        ``aggregation='bt-sigma'`` to persist the reliability-weighted block
        alongside the plurality headline.
        """
        _validate_aggregation(aggregation)
        self.report = _annotate_bt_sigma_scope(build_report(self.comparisons(), aggregation=aggregation), self.entries)
        payload = self.model_dump_json(indent=2)
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            _ = target.write_text(payload, encoding='utf-8')
            return target

        runs_dir = get_pairwise_runs_dir()
        runs_dir.mkdir(parents=True, exist_ok=True)
        base = f'{self.created_at.strftime("%Y%m%d_%H%M%S")}_{_slug(self.run_name)}'
        # Exclusive-create, matching the sim run store: the stamp only resolves
        # to the second, so two runs of the same name in the same second would
        # otherwise silently overwrite each other. 'x' also avoids the TOCTOU
        # race an exists() check would leave open.
        target = runs_dir / f'{base}.json'
        for counter in range(1000):
            try:
                with target.open('x', encoding='utf-8') as fh:
                    _ = fh.write(payload)
            except FileExistsError:  # noqa: PERF203 — exclusive-create retry is the point
                target = runs_dir / f'{base}_{counter + 1:03d}.json'
            except OSError:
                # 'x' created the file before the write failed (e.g. disk full);
                # don't leave an empty orphan behind.
                target.unlink(missing_ok=True)
                raise
            else:
                return target
        raise RuntimeError(f'could not find a free filename for {base} after 1000 attempts')

    @classmethod
    def load(cls, path: Path | str) -> PairwiseRun:
        """Read a saved run back."""
        return cls.model_validate(json.loads(Path(path).read_text(encoding='utf-8')))


def _slug(text: str) -> str:
    cleaned = ''.join(c if c.isalnum() else '-' for c in text.lower())
    joined = '-'.join(part for part in cleaned.split('-') if part)
    # Clamped because save() runs after the judging is already paid for, so a
    # name long enough to push the filename past the filesystem's limit would
    # lose a completed run to ENAMETOOLONG.
    #
    # On bytes rather than characters, which is what the limit is actually
    # counted in. The sim run store clamps 64 characters, but it first forces
    # ASCII, so there the two are the same number; ``isalnum()`` is
    # Unicode-aware and keeps non-Latin names readable, where 64 characters can
    # be up to 256 bytes. Decoding with ``ignore`` drops a character the cut
    # landed mid-way through.
    clamped = joined.encode()[:64].decode(errors='ignore')
    # The cut can land on a joining hyphen, which would leave a trailing dash
    # against the '.json' suffix.
    return clamped.rstrip('-') or 'run'


def _validate_aggregation(aggregation: str) -> None:
    if aggregation not in _PAIRWISE_AGGREGATIONS:
        msg = f"unknown aggregation {aggregation!r}; expected 'plurality' or 'bt-sigma'"
        raise ValueError(msg)


def _annotate_bt_sigma_scope(report: PairwiseReport, entries: Sequence[PairwiseEntry]) -> PairwiseReport:
    """Warn when a run's BT-sigma fit pools different task-level datapoints."""
    block = report.bt_sigma
    if block is None:
        return report
    input_pairs = {(entry.question, entry.response_a, entry.response_b) for entry in entries}
    if len(input_pairs) > 1:
        warning = (
            f'run contains {len(input_pairs)} distinct question/response pairs; '
            'judge sigma measures global agreement across tasks, not isolated judge variance'
        )
        if warning not in block.fit_warnings:
            block.fit_warnings.append(warning)
    return report


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
