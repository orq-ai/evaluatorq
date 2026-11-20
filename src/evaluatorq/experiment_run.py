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
from dataclasses import dataclass
from datetime import datetime  # noqa: TC003 — runtime-required by the pydantic fields below
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from evaluatorq.common.run_store_dir import get_store_dir, sanitise_run_name, write_unique_json
from evaluatorq.contracts import RUN_SUMMARY_TOTAL_KEY
from evaluatorq.types import DataPointResult  # noqa: TC001 — runtime-required by the pydantic field below

if TYPE_CHECKING:
    from evaluatorq.contracts import RunSummary

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


class ExperimentRun(BaseModel):
    """A saved ``evaluatorq()`` run: its metadata and every datapoint result.

    ``results`` is the list ``evaluatorq()`` returned, unchanged, so evaluator
    ``token_usage`` and ``raw_output`` are kept. The target's own cost is not
    recorded, because a job returns only its output.
    """

    experiment_run_version: int = EXPERIMENT_RUN_VERSION
    run_id: str = Field(description='Unique id of this run; also the id of its run manifest')
    run_name: str = Field(description='The name passed to evaluatorq()')
    description: str | None = None
    dataset_id: str | None = Field(default=None, description='The Orq dataset id when data was a DatasetIdInput')
    started_at: datetime
    ended_at: datetime
    duration_seconds: float = Field(ge=0.0)
    evaluatorq_version: str
    evaluator_names: list[str] = Field(default_factory=list, description='Configured evaluators, in order')
    experiment_url: str | None = Field(default=None, description='The Orq experiment URL when results were uploaded')
    results: list[DataPointResult] = Field(default_factory=list)

    def evaluator_stats(self) -> list[EvaluatorStats]:
        """Per-evaluator score counts, average and pass rate, in configured then first-seen order."""
        order = list(self.evaluator_names)
        scores: dict[str, list[float]] = {}
        counts: dict[str, dict[str, int]] = {}
        for result in self.results:
            for job in result.job_results or []:
                for score in job.evaluator_scores or []:
                    name = score.evaluator_name
                    if name not in order:
                        order.append(name)
                    c = counts.setdefault(name, {'pass_set': 0, 'passed': 0, 'non_numeric': 0, 'errors': 0})
                    if score.error is not None:
                        c['errors'] += 1
                        continue
                    number = numeric_score(score.score.value)
                    if number is None:
                        c['non_numeric'] += 1
                    else:
                        scores.setdefault(name, []).append(number)
                    if score.score.pass_ is not None:
                        c['pass_set'] += 1
                        c['passed'] += int(score.score.pass_)
        stats: list[EvaluatorStats] = []
        for name in order:
            values = scores.get(name, [])
            c = counts.get(name, {'pass_set': 0, 'passed': 0, 'non_numeric': 0, 'errors': 0})
            stats.append(
                EvaluatorStats(
                    name=name,
                    scored=len(values),
                    average=math.fsum(values) / len(values) if values else None,
                    pass_set=c['pass_set'],
                    passed=c['passed'],
                    non_numeric=c['non_numeric'],
                    errors=c['errors'],
                )
            )
        return stats

    def manifest_summary(self) -> RunSummary:
        """Compact headline stats for the run manifest, so a run list never reads the full file."""
        averages = {s.name: s.average for s in self.evaluator_stats() if s.average is not None}
        return {RUN_SUMMARY_TOTAL_KEY: len(self.results), 'scorer_averages': averages}

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
    return write_unique_json(runs_dir, base, run.to_json())


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
