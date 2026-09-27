"""Validate dashboard Insights requests and launch an independent run process."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing_extensions import Self

from evaluatorq.common.orq_client import DEFAULT_ORQ_BASE_URL, OrqProfile
from evaluatorq.common.run_manifest import start_manifest
from evaluatorq.common.run_store_dir import get_store_dir
from evaluatorq.insights.models import DimensionName, InsightsPopulation, LabelSpec
from evaluatorq.insights.presets import LABEL_PRESETS
from evaluatorq.insights.progress import stage_plan
from evaluatorq.trace_finder.export import RunExport
from evaluatorq.trace_finder.models import FacetSelection

Source = Literal['recent', 'query', 'finder']
Preset = Literal['sentiment', 'customer_satisfaction']
_REQUEST_ENV = 'EVALUATORQ_INSIGHTS_LAUNCH_REQUEST'
MAX_FINDER_EXPORT_BYTES = 10 * 1024 * 1024


def get_finder_exports_dir() -> Path:
    """Return the only directory from which dashboard runs may read Finder exports."""
    return get_store_dir('finder-exports')


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
    _finder_export_snapshot: str | None = PrivateAttr(default=None)

    def validated_finder_export_snapshot(self) -> str | None:
        """Return the bounded Finder JSON captured during source validation."""
        return self._finder_export_snapshot

    @model_validator(mode='after')
    def validate_source(self) -> Self:
        if self.source == 'query' and not self.query.strip():
            raise ValueError('Enter a question to find matching traces.')
        if self.source == 'finder':
            if self.facets != FacetSelection():
                raise ValueError('A Finder export already fixes the trace population; remove the facet filters.')
            if not self.finder_export.strip():
                raise ValueError('Enter the path to a Finder JSON export.')
            try:
                root = get_finder_exports_dir().resolve()
                requested = Path(self.finder_export).expanduser()
                path = (requested if requested.is_absolute() else root / requested).resolve()
            except (OSError, RuntimeError) as exc:
                raise ValueError(f'Could not resolve Finder export path: {exc}') from exc
            if path.parent != root:
                raise ValueError(f'Finder exports must be in {root}.')
            try:
                if not path.is_file():
                    raise ValueError('Finder export must be a regular file.')
                with path.open('rb') as export_file:
                    raw = export_file.read(MAX_FINDER_EXPORT_BYTES + 1)
                if len(raw) > MAX_FINDER_EXPORT_BYTES:
                    raise ValueError(
                        f'Finder export exceeds the {MAX_FINDER_EXPORT_BYTES // (1024 * 1024)} MiB size limit.'
                    )
                RunExport.model_validate_json(raw)
                self._finder_export_snapshot = raw.decode('utf-8')
            except (OSError, ValueError) as exc:
                raise ValueError(f'Could not read a valid Finder export: {exc}') from exc
            self.finder_export = str(path)
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
    finder_export_snapshot: Path | None = None


def launch_insights(spec: InsightsLaunchSpec, runs_dir: Path, *, profile: OrqProfile | None = None) -> str:
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
    snapshot_path: Path | None = None
    try:
        finder_snapshot = spec.validated_finder_export_snapshot()
        if finder_snapshot is not None:
            snapshot_directory = Path(tempfile.mkdtemp(prefix='evaluatorq-finder-snapshot-'))
            snapshot_path = snapshot_directory / 'finder-export.json'
            descriptor = os.open(snapshot_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'w', encoding='utf-8') as snapshot_file:
                snapshot_file.write(finder_snapshot)
        # Serialize the already validated spec without asking Pydantic to validate
        # the mutable Finder path a second time.
        worker_request = json.dumps({
            'run_id': run_id,
            'run_name': run_name,
            'runs_dir': str(runs_dir),
            'spec': spec.model_dump(mode='json'),
            'finder_export_snapshot': str(snapshot_path) if snapshot_path is not None else None,
        })
        worker_env = {**os.environ, _REQUEST_ENV: worker_request}
        if profile is not None:
            worker_env.update(ORQ_API_KEY=profile.api_key, ORQ_BASE_URL=profile.server or DEFAULT_ORQ_BASE_URL)
        log_dir = runs_dir / '.logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f'{run_id}.log').open('a', encoding='utf-8') as log:
            subprocess.Popen(
                [sys.executable, '-m', 'evaluatorq.dashboard.insights_worker'],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=worker_env,
                start_new_session=True,
            )
    except OSError as exc:
        writer.fail(f'Could not start Insights worker: {exc}', stage='start')
        if snapshot_path is not None:
            shutil.rmtree(snapshot_path.parent, ignore_errors=True)
    return run_id


def read_launch_payload() -> InsightsLaunchPayload:
    raw = os.environ.pop(_REQUEST_ENV, None)
    if raw is None:
        raise ValueError('Missing Insights launch request')
    data = json.loads(raw)
    spec_data = dict(data['spec'])
    finder_source = spec_data.get('source') == 'finder'
    finder_path = spec_data.get('finder_export', '')
    if finder_source:
        # Validate all user-selected options while neutralizing the original
        # pathname, which may have changed since launch validation.
        spec_data['source'] = 'recent'
        spec_data['finder_export'] = ''
    data['spec'] = spec_data
    payload = InsightsLaunchPayload.model_validate(data)
    if finder_source:
        payload.spec = payload.spec.model_copy(update={'source': 'finder', 'finder_export': finder_path})
    return payload
