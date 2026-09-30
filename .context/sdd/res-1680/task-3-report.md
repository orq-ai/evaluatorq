# RES-1680 Task 3 report

Implemented Group A structure and token signals on the shared ATIF walk and registered all 22 signals in `evaluatorq.signals.registry`. Evidence identifies each contributing `step_id`, `agent_path`, and tool `call_id` where applicable. The signals respect skipped compaction and copied-context steps, use configured tool roles, parse shell command families, and apply the ATIF token mapping with cached prompt tokens removed from input totals. Added coverage for counts, turns, usage and cache ratios, tool families, custom roles, and subagent linkage and depth.

Checks: `uv run ruff check src` passed; `uv run ruff format --check src` passed; `uv run pytest -m 'not integration' tests/signals tests/formats` passed with 249 passed and 4 skipped.

`uv run basedpyright` remains red with 30 findings, all in the gitignored reference copy under `.context/signals-src/src/signals_explorer/`. It reports no findings in the Task 3 source or tests. The command scans this local reference copy, whose imports and types are outside the package and not part of the implementation.
