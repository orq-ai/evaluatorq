"""Trace intelligence for evaluatorq: cluster and label a trace population, reviewed in the dashboard.

`insights`/`insights_sync` (the pipeline entry points) are exposed lazily via `__getattr__`
so importing this package never pulls in `numpy`/`scipy`/`umap-learn` — those are behind
the `insights` extra and only imported once a run is actually started.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import presets
from .models import (
    Cluster,
    ClusterAssignment,
    DimensionName,
    DimensionResult,
    InsightsConfig,
    InsightsPopulation,
    InsightsRun,
    LabelAnswer,
    LabelResult,
    LabelSpec,
    PriorityPoint,
    StageFailure,
    TraceInsight,
    TraceSummary,
)

if TYPE_CHECKING:
    from .pipeline import insights, insights_sync

__all__ = [
    'Cluster',
    'ClusterAssignment',
    'DimensionName',
    'DimensionResult',
    'InsightsConfig',
    'InsightsPopulation',
    'InsightsRun',
    'LabelAnswer',
    'LabelResult',
    'LabelSpec',
    'PriorityPoint',
    'StageFailure',
    'TraceInsight',
    'TraceSummary',
    'insights',
    'insights_sync',
    'presets',
]

_LAZY_PIPELINE_NAMES = frozenset({'insights', 'insights_sync'})  # noqa: RUF067  # lookup table backing __getattr__ below


def __getattr__(name: str) -> Any:
    if name in _LAZY_PIPELINE_NAMES:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
