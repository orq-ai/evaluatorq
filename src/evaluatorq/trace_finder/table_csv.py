"""CSV export for the trace explorer's selected table columns."""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import TYPE_CHECKING

from evaluatorq.trace_finder.columns import MATCH, resolve_columns

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from evaluatorq.trace_finder.columns import Column
    from evaluatorq.trace_finder.models import TraceClassification
    from evaluatorq.trace_finder.rows import TraceRow


def export_table_csv(
    rows: Sequence[TraceRow],
    column_keys: Sequence[str] | None,
    *,
    dimensions: Sequence[str] = (),
    results: Mapping[str, TraceClassification] | None = None,
) -> str:
    """Serialize the supplied visible rows in order using the selected table columns.

    Values come from each column's raw value accessor. The renderer is never called, so markup
    and display-only formatting cannot leak into the download.
    """
    columns = resolve_columns(column_keys)
    headers: list[str] = []
    selectors: list[tuple[Column, int | None]] = []
    for column in columns:
        if column.key == MATCH:
            if dimensions:
                for index, name in enumerate(dimensions):
                    headers.append(name)
                    selectors.append((column, index))
            continue
        headers.append(column.label)
        selectors.append((column, None))

    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\r\n')
    writer.writerow([_safe_text(header) for header in headers])
    for row in rows:
        result = results.get(row.trace_id) if results is not None else None
        values: list[str | int | float] = []
        for column, dimension_index in selectors:
            if dimension_index is not None:
                answer = (
                    result.answers[dimension_index]
                    if result and not result.error and dimension_index < len(result.answers)
                    else None
                )
                value = answer.value if answer is not None and not answer.error else None
            else:
                value = column.value(row)
            values.append(_csv_value(value))
        writer.writerow(values)
    return output.getvalue()


def _csv_value(value: object) -> str | int | float:
    if value is None:
        return ''
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return _safe_text(', '.join(str(item) for item in value))
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, int | float):
        return value
    return str(value)


def _safe_text(value: str) -> str:
    """Keep spreadsheet programs from treating trace-controlled text as a formula."""
    if value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value
