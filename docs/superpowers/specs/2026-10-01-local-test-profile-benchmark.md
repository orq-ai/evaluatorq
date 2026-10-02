# Local pytest profile benchmark

This benchmark measures evaluatorq's quick local pytest profile against the complete non-integration profile on one identified machine.

Use these results to decide whether the quick profile meets this change's acceptance gates. Do not use them as a universal performance claim: laptop load, operating-system state, and filesystem caches can move an individual sample.

## Result

Under the disclosed fixed-order protocol, the quick profile passed both gates on implementation commit `69fae17d4581730cea5125520f51eae45b693185`.

| Profile | Selected outcome in every sample | Median wall time | Median peak resident memory |
|---|---:|---:|---:|
| Quick | 6,818 passed, 6 skipped, 39 deselected | 88.85 s | 552,828,928 bytes (527.22 MiB) |
| Full non-integration | 6,834 passed, 7 skipped, 22 deselected | 127.37 s | 566,525,952 bytes (540.28 MiB) |

The wall-time reduction is `(127.37 - 88.85) / 127.37 = 30.24%`, above the required 20%. This is faster feedback from omitting tests that deliberately wait on real time, not a material compute-load reduction: median user CPU time is effectively unchanged at 55.97 seconds for quick and 56.14 seconds for full. The quick median peak resident memory is only `(566,525,952 - 552,828,928) / 566,525,952 = 2.42%` below the full median. Both acceptance gates pass, but the resource case for the wider migration rests on the checker improvement and removal of unnecessary fixture setup rather than this pytest profile's CPU or memory difference.

## Method

The six measured commands ran serially in a clean worktree at the named commit, with no benchmark samples overlapping. The order was quick samples 1–3 followed by full samples 1–3. There was no dedicated warm-up and no cache reset between samples; the failed pre-fix run described below happened before all six valid samples. `PYTEST_ADDOPTS` was unset. These are the commands as run:

Because every quick sample ran before every full sample, profile and run order are confounded: cache warming, thermal behavior, or background-load drift could contribute to the observed gap. The 30.24% result is the difference between medians under this disclosed protocol, not a randomized or alternating estimate of the pure profile effect.

```bash
/usr/bin/time -lp uv run pytest
/usr/bin/time -lp uv run pytest -m 'not integration'
```

The host was a MacBook Pro `Mac15,6` with an Apple M3 Pro (12 cores: 6 performance and 6 efficiency), 18 GB memory, and macOS 26.6.2 build 25G83. The toolchain was Darwin 25.6.0 arm64, uv 0.11.29, Python 3.13.2, and pytest 9.0.3.

The complete collection contained 6,863 cases. The quick profile selected 6,824 cases and deselected 39; the full non-integration profile selected 6,841 and deselected 22. The 17-case difference is exactly the non-integration `slow` selection listed below.

## Deliberately slow cases

The `slow` marker is reserved for tests that deliberately spend real elapsed time exercising subprocess lifecycle, timeout, watchdog, or scheduler behavior. This command collected the cases selected by the full profile and excluded by the quick profile:

```bash
uv run pytest --collect-only -q -m 'slow and not integration'
```

```text
tests/backends/test_coding_agent_container.py::test_idle_timeout_removes_container
tests/backends/test_coding_agent_container.py::test_cancellation_removes_container_before_propagating
tests/backends/test_coding_agent_container.py::test_cancel_during_run_still_cleans_up
tests/backends/test_coding_agent_container.py::test_cancel_during_restart_removal_waits_until_old_container_is_removed
tests/backends/test_coding_agent_respond.py::test_timeout_kills_and_is_non_retryable
tests/backends/test_coding_agent_stream.py::test_steady_output_outlives_idle_limit
tests/backends/test_coding_agent_stream.py::test_silence_past_idle_limit_is_idle_timeout
tests/backends/test_coding_agent_stream.py::test_open_child_after_stdout_eof_still_hits_idle_limit
tests/backends/test_coding_agent_stream.py::test_descendant_holding_stderr_open_still_hits_idle_limit
tests/backends/test_coding_agent_stream.py::test_hard_cap_fires_on_steady_output
tests/backends/test_coding_agent_stream.py::test_sustained_stderr_resets_idle_limit
tests/backends/test_coding_agent_stream.py::test_idle_timeout_after_stderr_stops
tests/backends/test_coding_agent_stream.py::test_partial_line_counts_as_output
tests/backends/test_coding_agent_stream.py::test_cancellation_logs_cancelled
tests/backends/test_container.py::test_watchdog_script_lifecycle[sh]
tests/backends/test_container.py::test_watchdog_script_lifecycle[busybox]
tests/trace_finder/test_cli.py::test_find_debug_progress_skips_unchanged_polls

17/6863 tests collected (6846 deselected) in 3.34s
```

## Raw samples

Each block preserves the pytest terminal summary and the complete `/usr/bin/time -lp` record from that sample.

### Quick sample 1

```text
======================== 6818 passed, 6 skipped, 39 deselected, 41 warnings in 86.42s (0:01:26) ========================
real 88.85
user 56.46
sys 9.73
           552828928  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             1938360  page reclaims
               11110  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                2374  messages sent
                2340  messages received
                 332  signals received
               15329  voluntary context switches
              106869  involuntary context switches
           379687459  instructions retired
           150284318  cycles elapsed
            20808160  peak memory footprint
```

### Quick sample 2

```text
======================== 6818 passed, 6 skipped, 39 deselected, 41 warnings in 84.56s (0:01:24) ========================
real 86.92
user 55.97
sys 9.92
           560136192  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             1925463  page reclaims
                9534  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                2374  messages sent
                2332  messages received
                 342  signals received
               15380  voluntary context switches
              104021  involuntary context switches
           294536898  instructions retired
           106625368  cycles elapsed
            20808160  peak memory footprint
```

### Quick sample 3

```text
======================== 6818 passed, 6 skipped, 39 deselected, 41 warnings in 87.87s (0:01:27) ========================
real 90.20
user 55.89
sys 11.51
           550289408  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             1926785  page reclaims
                9120  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                2374  messages sent
                2337  messages received
                 353  signals received
               19279  voluntary context switches
              109059  involuntary context switches
           296317183  instructions retired
           100200638  cycles elapsed
            20906464  peak memory footprint
```

### Full non-integration sample 1

```text
======================= 6834 passed, 7 skipped, 22 deselected, 41 warnings in 124.08s (0:02:04) ========================
real 126.23
user 55.98
sys 10.34
           565149696  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             2030136  page reclaims
                9371  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                2431  messages sent
                2381  messages received
                 346  signals received
               12065  voluntary context switches
              100579  involuntary context switches
           296700326  instructions retired
           100484612  cycles elapsed
            20824568  peak memory footprint
```

### Full non-integration sample 2

```text
======================= 6834 passed, 7 skipped, 22 deselected, 41 warnings in 125.68s (0:02:05) ========================
real 128.02
user 56.14
sys 10.51
           566525952  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             2054505  page reclaims
               10830  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                2431  messages sent
                2388  messages received
                 383  signals received
               17400  voluntary context switches
              103994  involuntary context switches
           379175581  instructions retired
           154821087  cycles elapsed
            20791800  peak memory footprint
```

### Full non-integration sample 3

```text
======================= 6834 passed, 7 skipped, 22 deselected, 41 warnings in 124.96s (0:02:04) ========================
real 127.37
user 56.24
sys 10.62
           568016896  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             2052444  page reclaims
               10782  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                2431  messages sent
                2380  messages received
                 372  signals received
               16294  voluntary context switches
              104248  involuntary context switches
           392392404  instructions retired
           162917363  cycles elapsed
            20824568  peak memory footprint
```

## Pre-change reference

Commit `480e4b286` had one previously measured full non-integration run on this laptop: 6,813 passed, 7 skipped, and 22 deselected; pytest reported 444.88 seconds; `/usr/bin/time -lp` reported 458.51 seconds wall, 138.31 seconds user, 69.92 seconds system, and 409 MB peak resident memory.

This is a single noisy laptop sample, not a range and not a controlled before/after estimate. It is retained as context only; the acceptance result uses the two three-sample medians from commit `69fae17d4`.

## Failure mode observed before measurement

The first attempted quick run at commit `3d3fcf9f7` failed `tests/simulation/test_cli.py::test_run_report_and_autosave_both_written`: the global isolation fixture redirected `EVALUATORQ_DIR`, while the test still asserted the previous implicit store location. That run ended with 1 failed, 6,817 passed, 6 skipped, and 39 deselected in 88.50 seconds wall with 548,159,488 bytes peak resident memory.

The regression was fixed at `69fae17d4` by making the test choose the store it asserts. The failed pre-fix run is not one of the benchmark samples. This is the named measurement failure mode: a fast timing from a red suite is invalid evidence, even when its duration looks plausible.
