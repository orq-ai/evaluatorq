# Task 7: Wire long-lived container mode into `CodingAgentTarget`

## Status

DONE. Container mode is integrated in `CodingAgentTarget`; the focused container tests and backend suite pass.

## RED

Wrote `tests/backends/test_coding_agent_container.py` from the Task 7 brief before changing production code. Ran `uv run pytest tests/backends/test_coding_agent_container.py -q`. The unmodified host-only target did not reach a normal assertion failure: it launched the host `claude` path and waited for the default 300-second idle timeout. I interrupted that run after 67.52 seconds. Its traceback showed `respond()` entering the host `_run()`/`read_stdout()` path, confirming container mode was not wired. This RED attempt was inconclusive as a completed pytest run.

The initial test execution also exposed a polling race in the supplied fixture helper: it read the fake Docker log before the first subprocess created the file. The helper now returns an empty list until that file exists.

## GREEN

- `uv run pytest tests/backends/test_coding_agent_container.py -q` — **16 passed in 34.17s**. Includes one-container/two-turn reuse, env argv secrecy, image/start errors, exit mapping, timeout cleanup, cancellation cleanup, startup cancellation, restart, concurrent clones, kept workdir, copied skills, unsafe mount warning, and orphan sweep ordering.
- `uv run pytest tests/backends -q` — **123 passed, 1 skipped in 54.59s**.
- `uv run ruff check src` — **All checks passed**.
- `uv run ruff format --check src` — **227 files already formatted**.
- `uv run basedpyright` — **0 errors, 0 warnings, 0 notes**.
- `git diff --check` — clean.

An earlier backend run had one failure in `test_descendant_holding_stderr_open_still_hits_idle_limit` because the subprocess emitted no output before its short idle deadline. The exact test passed when rerun alone, and the final full backend run passed.

## Files

- Modified `src/evaluatorq/backends/coding_agent.py` to allocate container mount directories; copy skills into container workdirs; resolve forwarded environment names once; start, inspect, reuse, restart, unregister, and remove containers; add span attributes and container error mapping; and wait for the bounded `docker run` result before cancellation cleanup can remove the container.
- Created `tests/backends/test_coding_agent_container.py` with the fake Docker binary coverage from the brief. The log polling helper now handles the interval before the fake Docker process creates its log.

## Self-review

- The Docker exec argv contains bare `-e KEY` entries. Secret values are supplied only in the Docker exec client's process environment through `_run`.
- Container names are unregistered before removal, including close, early turn termination, start failure, and restart cleanup paths.
- Docker control calls use a 60-second subprocess timeout. If cancellation arrives during `docker run`, `_docker` waits for the bounded subprocess result; `respond()` then unregisters and removes the name before propagating cancellation. This prevents a delayed background `docker run` from creating a container after cleanup has finished.
- The container finalizer uses the shared owned-name list; explicit close detaches it after removing the current container. Lease directory removal is part of close, including when the workdir is kept.
- Changes are limited to the Task 7 production and test files plus this requested report.

## Concerns

The first RED pytest attempt had to be interrupted instead of completing because the old host path waits for the 300-second idle limit. The final implementation has complete GREEN coverage; no known outstanding concern remains.

## Commit

`feat(backends): wire long-lived coding agent containers (RES-1628)`.
