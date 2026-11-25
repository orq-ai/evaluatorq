"""Saved ``evaluatorq()`` experiment runs.

``evaluatorq(..., save=SaveMode.FINAL)`` writes one `ExperimentRun` per run to
``.evaluatorq/experiment-runs/`` (or the caller's ``output_dir``). The file holds
the full in-memory result list, judge token usage included, which the Orq upload
strips, plus the run's metadata. `load_experiment_run` reads one back, and the
dashboard's ``experiment`` surface renders it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from evaluatorq.common.run_store_dir import get_store_dir, sanitise_run_name, write_unique_json
from evaluatorq.contracts import RUN_SUMMARY_TOTAL_KEY
from evaluatorq.types import DataPointResult  # noqa: TC001 — runtime-required by the pydantic field below

if TYPE_CHECKING:
    from evaluatorq.contracts import RunSummary, Usage
    from evaluatorq.types import EvaluatorScore

EXPERIMENT_RUNS_SUBDIR = 'experiment-runs'
EXPERIMENT_RUN_VERSION = 1
"""Format version of a saved `ExperimentRun`; the dashboard sniffs the surface by this key."""


def get_experiment_runs_dir() -> Path:
    """Return the experiment runs directory (``<store>/experiment-runs``)."""
    return get_store_dir(EXPERIMENT_RUNS_SUBDIR)


@dataclass(frozen=True)
class EvaluatorStats:
    """One evaluator's scores across a run.

    Attributes:
        name: The evaluator name.
        scored: Scores with a numeric value (a bool counts as 1 or 0).
        average: Mean of the numeric values; ``None`` when there are none.
        pass_set: Scores that set ``pass``.
        passed: Scores with ``pass`` set to True.
        non_numeric: Scores whose value is a string, a dict or a cell, which no average can include.
        errors: Scores whose evaluator raised.
    """

    name: str
    scored: int
    average: float | None
    pass_set: int
    passed: int
    non_numeric: int
    errors: int

    @property
    def pass_rate(self) -> float | None:
        """Share of the scores that set ``pass`` that passed; ``None`` when none set it."""
        return self.passed / self.pass_set if self.pass_set else None


@dataclass(frozen=True)
class RowCounts:
    """How many rows a run has and how many of them errored.

    A row is one job result, or one datapoint that has no job results. A row
    errored when its job raised or its datapoint did; this is the one errored
    rule the report, the run list and the manifest summary share.

    Attributes:
        rows: Job results, plus one per datapoint with none.
        job_results: Job results across every datapoint.
        errored: Rows whose job or datapoint raised.
    """

    rows: int
    job_results: int
    errored: int

    @property
    def all_errored(self) -> bool:
        """True when the run has rows and every one of them errored."""
        return self.rows > 0 and self.errored == self.rows


def combined_pass_rate(stats: list[EvaluatorStats]) -> float | None:
    """Share of the scores that set ``pass`` which passed, across *stats*; ``None`` when none set it."""
    pass_set = sum(s.pass_set for s in stats)
    return sum(s.passed for s in stats) / pass_set if pass_set else None


def numeric_score(value: object) -> float | None:
    """The value as a number for averaging, or ``None`` when it is not one.

    A bool counts as 1 or 0. NaN and infinity count as not numeric, because one
    of them would turn the evaluator's average into NaN.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


@dataclass
class _Tally:
    """Running counts for one evaluator while `ExperimentRun.evaluator_stats` walks the results."""

    values: list[float] = field(default_factory=list)
    pass_set: int = 0
    passed: int = 0
    non_numeric: int = 0
    errors: int = 0

    def add(self, score: EvaluatorScore) -> None:
        if score.error is not None:
            self.errors += 1
            return
        number = numeric_score(score.score.value)
        if number is None:
            self.non_numeric += 1
        else:
            self.values.append(number)
        if score.score.pass_ is not None:
            self.pass_set += 1
            self.passed += int(score.score.pass_)

    def stats(self, name: str) -> EvaluatorStats:
        return EvaluatorStats(
            name=name,
            scored=len(self.values),
            average=math.fsum(self.values) / len(self.values) if self.values else None,
            pass_set=self.pass_set,
            passed=self.passed,
            non_numeric=self.non_numeric,
            errors=self.errors,
        )


class ExperimentRun(BaseModel):
    """A saved ``evaluatorq()`` run: its metadata and every datapoint result.

    ``results`` is the list ``evaluatorq()`` returned, unchanged, so evaluator
    ``token_usage`` and ``raw_output`` are kept, and so is the ``usage`` of a job
    that returns an `AgentResponse`. Nothing here totals that target usage: the
    run statistics cover the evaluators' calls only.
    """

    experiment_run_version: int = Field(
        default=EXPERIMENT_RUN_VERSION,
        ge=1,
        le=EXPERIMENT_RUN_VERSION,
        description='Format version; a file written by a newer evaluatorq fails to load instead of misreading',
    )
    run_id: str = Field(description='Unique id of this run; also the id of its run manifest')
    run_name: str = Field(description='The name passed to evaluatorq()')
    description: str | None = None
    dataset_id: str | None = Field(default=None, description='The Orq dataset id when data was a DatasetIdInput')
    started_at: datetime
    ended_at: datetime
    duration_seconds: float = Field(ge=0.0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc), description='When this file was first written'
    )
    evaluatorq_version: str
    evaluator_names: list[str] = Field(default_factory=list, description='Configured evaluators, in order')
    experiment_url: str | None = Field(default=None, description='The Orq experiment URL when results were uploaded')
    results: list[DataPointResult] = Field(default_factory=list)

    def evaluator_stats(self) -> list[EvaluatorStats]:
        """Per-evaluator score counts, average and pass rate, in configured then first-seen order."""
        tallies = {name: _Tally() for name in self.evaluator_names}
        for result in self.results:
            for job in result.job_results or []:
                for score in job.evaluator_scores or []:
                    tallies.setdefault(score.evaluator_name, _Tally()).add(score)
        return [tally.stats(name) for name, tally in tallies.items()]

    def pass_rate(self) -> float | None:
        """Share of all scores that set ``pass`` which passed, across evaluators; ``None`` when none set it."""
        return combined_pass_rate(self.evaluator_stats())

    def row_counts(self) -> RowCounts:
        """Rows, job results and errored rows, by the rule `RowCounts` describes."""
        rows = jobs = errored = 0
        for result in self.results:
            job_results = result.job_results or []
            if not job_results:
                rows += 1
                errored += result.error is not None
                continue
            rows += len(job_results)
            jobs += len(job_results)
            errored += sum(1 for job in job_results if job.error is not None or result.error is not None)
        return RowCounts(rows=rows, job_results=jobs, errored=errored)

    def judge_usages(self) -> list[Usage]:
        """Every evaluator score's ``token_usage``, in result order; scores without one are skipped."""
        return [
            score.score.token_usage
            for result in self.results
            for job in result.job_results or []
            for score in job.evaluator_scores or []
            if score.score.token_usage is not None
        ]

    def judge_usage(self) -> tuple[int, float | None]:
        """``(total tokens, total cost)`` of the evaluators' own LLM calls; cost ``None`` when none was reported."""
        usages = self.judge_usages()
        costs = [u.total_cost for u in usages if u.total_cost is not None]
        return sum(u.total_tokens for u in usages), (math.fsum(costs) if costs else None)

    def manifest_summary(self) -> RunSummary:
        """Compact headline stats for the run manifest, which a run-list card reads without opening the run file."""
        stats = self.evaluator_stats()
        return {
            RUN_SUMMARY_TOTAL_KEY: len(self.results),
            'scorer_averages': {s.name: s.average for s in stats if s.average is not None},
            'pass_rate': combined_pass_rate(stats),
            'errored_rows': self.row_counts().errored,
            'judge_tokens': self.judge_usage()[0],
        }

    def to_json(self) -> str:
        """Serialize the run so `load_experiment_run` reads back the same values.

        Not ``model_dump_json``: the result types serialize a dict output or a
        dict score to a JSON string for the Orq upload, and a saved run must keep
        the dict.
        """
        return json.dumps(to_jsonable_python(self.model_dump(mode='python'), fallback=str), indent=2)


def save_experiment_run(run: ExperimentRun, *, output_dir: Path | str | None = None) -> Path:
    """Write *run* to ``<output_dir or .evaluatorq/experiment-runs>/<name>_<timestamp>.json``.

    Returns:
        The path written. Two runs of one name in the same second get distinct files.

    Raises:
        OSError: The folder or file could not be written.
    """
    runs_dir = Path(output_dir) if output_dir is not None else get_experiment_runs_dir()
    base = f'{sanitise_run_name(run.run_name, default="experiment")}_{run.started_at.strftime("%Y%m%d-%H%M%S")}'
    return write_unique_json(runs_dir, base=base, payload=run.to_json())


def load_experiment_run(path: Path | str) -> ExperimentRun:
    """Read a run saved by ``evaluatorq(..., save=SaveMode.FINAL)``.

    Example:
    ```python
    from evaluatorq import load_experiment_run

    run = load_experiment_run(".evaluatorq/experiment-runs/my-eval_20261120-101500.json")
    for stats in run.evaluator_stats():
        print(stats.name, stats.average, stats.pass_rate)
    ```

    Raises:
        OSError: The file could not be read.
        pydantic.ValidationError: The file is not a saved experiment run.
    """
    return ExperimentRun.model_validate_json(Path(path).read_text(encoding='utf-8'))
