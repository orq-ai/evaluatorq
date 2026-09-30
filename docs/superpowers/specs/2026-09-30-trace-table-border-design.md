# Trace table border design

## Decision

Remove the grey perimeter border around the trace table. Keep the pale grey header band, the header underline, and the thin horizontal separators between rows. This is option A, “Quiet dividers,” selected in the visual brainstorming companion.

## Scope

Change only the outer frame visible around the table in the trace table view. Preserve the existing header and row separation, sticky headers and first column, sorting, row selection, hover states, and error and match indicators. Do not change the Trajectories view, toolbar borders, or table spacing.

## Acceptance criteria

- The trace table no longer has a visible grey perimeter border.
- The header band and its bottom divider remain.
- Thin horizontal row dividers remain.
- Sticky behavior and existing row states continue to render as before.
