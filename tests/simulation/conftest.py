"""Shared fixtures for simulation tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.types import SimulationResult, TerminatedBy


class CollectingExporter(SpanExporter):
    """Minimal in-memory exporter that collects finished spans."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def new_collector() -> tuple[CollectingExporter, TracerProvider, Any]:
    """Return (exporter, provider, tracer) wired to an in-memory exporter."""
    exporter = CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider, provider.get_tracer('evaluatorq-simulation-test')


def span_attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def find_span(exporter: CollectingExporter, prefix: str) -> ReadableSpan:
    for span in exporter.spans:
        if span.name.startswith(prefix):
            return span
    raise AssertionError(f'no span starting {prefix!r}; got {[s.name for s in exporter.spans]}')


@pytest.fixture
def sim_result_factory():
    def _make(
        *,
        goal_achieved: bool = True,
        persona: str = "p",
        scenario: str = "s",
        turn_count: int = 2,
        error: str | None = None,
    ) -> SimulationResult:
        meta: dict[str, object] = {"persona": persona, "scenario": scenario}
        if error is not None:
            meta["error"] = error
        return SimulationResult(
            messages=[],
            terminated_by=TerminatedBy.error if error else TerminatedBy.judge,
            reason="done",
            goal_achieved=goal_achieved,
            goal_completion_score=1.0 if goal_achieved else 0.0,
            rules_broken=[],
            turn_count=turn_count,
            turn_metrics=[],
            token_usage=TokenUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
            metadata=meta,
        )

    return _make
