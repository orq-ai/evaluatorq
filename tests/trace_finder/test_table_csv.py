from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from evaluatorq.trace_finder.models import DimensionAnswer, TraceClassification
from evaluatorq.trace_finder.rows import TraceRow
from evaluatorq.trace_finder.table_csv import export_table_csv


def _read(value: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(value, newline='')))


def test_exports_raw_selected_values_with_csv_quoting_and_unicode() -> None:
    rows = (
        TraceRow(
            trace_id='trace-α',
            name='first, "quoted"\nnext',
            started_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
            duration_ms=1250,
            cost_total=0.031,
        ),
        TraceRow(trace_id='trace-β', name=None, started_at=None, duration_ms=None, cost_total=None),
    )

    table = _read(export_table_csv(rows, ('name', 'started', 'duration', 'cost')))

    assert table == [
        ['Name', 'Time', 'Duration', 'Cost'],
        ['first, "quoted"\nnext', '2026-09-30T00:00:00+00:00', '1250', '0.031'],
        ['', '', '', ''],
    ]


def test_ai_match_exports_one_column_per_active_dimension() -> None:
    rows = (TraceRow(trace_id='t1'), TraceRow(trace_id='t2'))
    results = {
        't1': TraceClassification(
            trace_id='t1',
            span_id='s',
            matched=True,
            answers=(DimensionAnswer(value=True), DimensionAnswer(value='café')),
        ),
    }

    table = _read(export_table_csv(rows, ('trace_id', 'match'), dimensions=('safety', 'quality'), results=results))

    assert table == [['Trace ID', 'safety', 'quality'], ['t1', 'True', 'café'], ['t2', '', '']]


def test_ai_match_column_is_omitted_without_classifier_dimensions() -> None:
    table = _read(export_table_csv((TraceRow(trace_id='t1'),), ('trace_id', 'match')))

    assert table == [['Trace ID'], ['t1']]


def test_formula_like_text_is_escaped_but_negative_numbers_remain_numeric() -> None:
    rows = (TraceRow(trace_id='=HYPERLINK("bad")', name='  +SUM(A1:A2)', duration_ms=-5),)

    table = _read(export_table_csv(rows, ('trace_id', 'name', 'duration')))

    assert table == [['Trace ID', 'Name', 'Duration'], ['\'=HYPERLINK("bad")', "'  +SUM(A1:A2)", '-5']]


def test_export_preserves_supplied_sort_and_filtered_row_order_across_all_rows() -> None:
    rows = tuple(TraceRow(trace_id=f't{i}', duration_ms=i) for i in range(205, -1, -1))

    table = _read(export_table_csv(rows, ('trace_id', 'duration')))

    assert len(table) == 207
    assert table[1] == ['t205', '205']
    assert table[-1] == ['t0', '0']
