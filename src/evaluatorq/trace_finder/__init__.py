"""Core contracts for finding and classifying recent Orq traces."""

from __future__ import annotations

from .compiler import CompiledPlan, CompileError, classification_legend, compile_query
from .facets import load_facet_catalogue
from .filter_selector import NO_FILTER_LABEL, FilterSelectionError, select_filters
from .jev import build_datapoint, build_jev_evaluator, matches_selection, parse_datapoint_result, run_jev
from .models import (
    FACET_NAMES,
    CompiledQuery,
    FacetCatalogue,
    FacetName,
    FacetOption,
    FacetSelection,
    JevProjection,
    LegendItem,
    NumericFilters,
    PopulationRequest,
    RunRequest,
    RunSnapshot,
    RunState,
    SelectionRule,
    Snapshot,
    ThresholdSelection,
    TraceClassification,
    TraceDetail,
    TraceRecord,
    ValueSelection,
    validate_compiled_query,
)
from .orq_source import OrqTraceSource, build_oql
from .projection import MAX_TOKEN_BUDGET, OMISSION_MARKER, estimate_tokens, project_trace, serialize_projection

__all__ = [
    'FACET_NAMES',
    'MAX_TOKEN_BUDGET',
    'NO_FILTER_LABEL',
    'OMISSION_MARKER',
    'CompileError',
    'CompiledPlan',
    'CompiledQuery',
    'FacetCatalogue',
    'FacetName',
    'FacetOption',
    'FacetSelection',
    'FilterSelectionError',
    'JevProjection',
    'LegendItem',
    'NumericFilters',
    'OrqTraceSource',
    'PopulationRequest',
    'RunRequest',
    'RunSnapshot',
    'RunState',
    'SelectionRule',
    'Snapshot',
    'ThresholdSelection',
    'TraceClassification',
    'TraceDetail',
    'TraceRecord',
    'ValueSelection',
    'build_datapoint',
    'build_jev_evaluator',
    'build_oql',
    'classification_legend',
    'compile_query',
    'estimate_tokens',
    'load_facet_catalogue',
    'matches_selection',
    'parse_datapoint_result',
    'project_trace',
    'run_jev',
    'select_filters',
    'serialize_projection',
    'validate_compiled_query',
]
