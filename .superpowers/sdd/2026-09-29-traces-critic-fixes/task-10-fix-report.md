# Task 10 focused fixes

Date: 2026-09-30. Applied the three confirmed browser findings from the Task 10 walkthrough.

Completed classifier matches now lead the default All view in a stable order. The selected All view and Table or Trajectories mode remain unchanged, and an explicit column sort still controls the row order.

At widths up to 1000px, `/traces` example chips wrap onto additional lines and remain fully readable. The rule is scoped to the traces command strip; `/find` keeps its legacy examples layout.

Span rows now use shrinkable grid tracks and truncate long kind, name, and token labels inside their cells instead of letting labels overlap. Kind and name retain the complete escaped text in native `title` attributes and in the DOM text.

The walkthrough did not reveal any prose explanation returned by JEV. The current classifier result is a probability distribution; `run_classify` synthesizes score text such as `noul=0.93 (threshold 0.5)`, which the UI intentionally does not present as a reason. Per-row prose therefore remains unavailable under the existing Task 4 cost and data contract; no explanation was invented and no additional model request was made.

Browser evidence: `.context/critics/after-task-10-fix-900.png` shows the three full example chips in two rows; `.context/critics/after-task-10-fix-spans.png` and `.context/critics/after-task-10-fix-spans-900.png` show the span drawer at 1280px and 900px. At 900px the two span rows have non-overlapping column bounds and no horizontal overflow.

Final live Ask AI browser proof on commit `d0b88547`: asked “Did any customers get frustrated?” on the loaded traces, selected Trajectories before submitting, and waited for completion. The run matched 21 of 194 traces; Trajectories remained selected after completion. After switching to Table, All remained selected and the first matched row preceded the first nonmatch (row indexes 0 and 21). The inspected screenshot `.context/critics/after-task-10-fix-ai-order.png` shows a match immediately followed by a nonmatch while the completed result summary and selected All/Table controls remain visible. This was the second and final Ask AI submission for Task 10.

Checks: `uv run pytest tests/dashboard/test_explorer.py -q` (97 passed), `uv run ruff check src`, `uv run ruff format --check src`, `uv run basedpyright` (0 errors, 0 warnings, 0 notes), and `git diff --check` all passed. No docs pages changed, so the strict docs build was not needed.
