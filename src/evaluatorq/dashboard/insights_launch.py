"""Validate dashboard Insights requests and launch an independent run process."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from evaluatorq.common.run_manifest import start_manifest
from evaluatorq.insights.models import DimensionName, InsightsPopulation, LabelSpec
from evaluatorq.insights.presets import LABEL_PRESETS
from evaluatorq.insights.progress import stage_plan
from evaluatorq.trace_finder.export import RunExport
from evaluatorq.trace_finder.models import FacetSelection

Source = Literal['recent', 'query', 'finder']
Preset = Literal['sentiment', 'customer_satisfaction']
_REQUEST_ENV = 'EVALUATORQ_INSIGHTS_LAUNCH_REQUEST'


class InsightsLaunchSpec(BaseModel):
    """Small, allow-listed subset of the Insights CLI for the dashboard wizard."""

    name: str = Field(default='', max_length=80)
    source: Source = 'recent'
    query: str = Field(default='', max_length=500)
    finder_export: str = Field(default='', max_length=4096)
    window_days: int = Field(default=7, ge=1, le=90)
    limit: int = Field(default=100, ge=1, le=5000)
    facets: FacetSelection = FacetSelection()
    parallelism: int = Field(default=20, ge=1, le=200)
    labels: list[Preset] = Field(default_factory=list)
    dimensions: list[DimensionName] = Field(default_factory=lambda: ['intent'])

    @model_validator(mode='after')
    def validate_source(self) -> Self:
        if self.source == 'query' and not self.query.strip():
            raise ValueError('Enter a question to find matching traces.')
        if self.source == 'finder':
            if self.facets != FacetSelection():
                raise ValueError('A Finder export already fixes the trace population; remove the facet filters.')
            if not self.finder_export.strip():
                raise ValueError('Enter the path to a Finder JSON export.')
            path = Path(self.finder_export).expanduser()
            try:
                RunExport.model_validate_json(path.read_text(encoding='utf-8'))
            except (OSError, ValueError) as exc:
                raise ValueError(f'Could not read a valid Finder export: {exc}') from exc
        if not self.labels and not self.dimensions:
            raise ValueError('Select at least one label or dimension.')
        if len(set(self.labels)) != len(self.labels) or len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError('Select each label and dimension once.')
        return self

    def population(self) -> InsightsPopulation:
        if self.source == 'finder':
            return InsightsPopulation.from_finder_export(Path(self.finder_export).expanduser())
        return InsightsPopulation(
            query=self.query.strip() if self.source == 'query' else None,
            facets=self.facets,
            window_days=self.window_days,
            limit=self.limit,
        )

    def label_specs(self) -> list[LabelSpec]:
        return [LABEL_PRESETS[name] for name in self.labels]

    def dimension_names(self) -> list[DimensionName]:
        return list(self.dimensions)


class InsightsLaunchPayload(BaseModel):
    run_id: str
    run_name: str
    runs_dir: Path
    spec: InsightsLaunchSpec


def launch_insights(spec: InsightsLaunchSpec, runs_dir: Path) -> str:
    """Create a visible manifest, then spawn a worker that survives dashboard reloads."""
    run_id = str(uuid.uuid4())
    run_name = spec.name.strip() or f'Insights {datetime.now().astimezone():%Y-%m-%d %H:%M}'
    plan = stage_plan(spec.population(), spec.label_specs(), spec.dimension_names())
    writer = start_manifest(
        run_id=run_id,
        surface='insights',
        run_name=run_name,
        runs_dir=runs_dir,
        planned_stages=[name for name, _ in plan],
        stage_labels=dict(plan),
    )
    payload = InsightsLaunchPayload(run_id=run_id, run_name=run_name, runs_dir=runs_dir, spec=spec)
    log_dir = runs_dir / '.logs'
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f'{run_id}.log').open('a', encoding='utf-8') as log:
            subprocess.Popen(
                [sys.executable, '-m', 'evaluatorq.dashboard.insights_worker'],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, _REQUEST_ENV: payload.model_dump_json()},
                start_new_session=True,
            )
    except OSError as exc:
        writer.fail(f'Could not start Insights worker: {exc}', stage='start')
    return run_id


def read_launch_payload() -> InsightsLaunchPayload:
    raw = os.environ.pop(_REQUEST_ENV, None)
    if raw is None:
        raise ValueError('Missing Insights launch request')
    return InsightsLaunchPayload.model_validate_json(raw)
