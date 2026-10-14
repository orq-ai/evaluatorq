# Filter-selection benchmark

The active production comparison is exactly two modes: `llm+jev` and `llm`.
Both invoke `evaluatorq.trace_finder.planner.plan_query`, use the same labelled
queries (including datetime expectations), source bounds, and fixed 50-value
catalogue snapshot injected at the production catalogue-cache loader. The
snapshot supplies a stable workload; the planner and cache remain production
code. Historical raw runs containing experimental `merged` or model-pinned
arms remain untouched and are not pooled into these workloads.

`llm+jev` uses the configured compiler fast role and configured classifier
role. `llm` uses the same compiler role and the configured fast filter task
role. Model settings are resolved as the application does; explicit CLI model
overrides are available, but there are no benchmark-pinned model defaults.
The metadata records the source of each role/override, a hash and path for the
settings snapshot (never the settings contents or credentials), exact requested
and served models, usage, retries/ladder legs, and cache telemetry. Raw call
rows include per-attempt parsed structured output for compiler/filter bounds
(not provider transcripts). Unknown or partial usage/cost remains explicitly
unknown/partial.

## Reproduce

Run the main 30-query workload and supplement 15-query workload as separate
experiments. Each query is measured three times after discarded warmups,
concurrency is one, and mode order alternates deterministically per repeat.
Both runs retain all scored query repeats; summaries aggregate repeats within
each query before paired row-level comparisons.

```bash
uv run python scripts/benchmarks/filter_selection/bench.py \
  --dataset scripts/benchmarks/filter_selection/dataset.jsonl \
  --catalogue scripts/benchmarks/filter_selection/catalogue.json \
  --arms llm+jev,llm --repeats 3 --warmup 2 --arm-concurrency 1
uv run python scripts/benchmarks/filter_selection/bench.py \
  --dataset scripts/benchmarks/filter_selection/dataset_research-fn.jsonl \
  --catalogue scripts/benchmarks/filter_selection/catalogue_research-fn.json \
  --profile research-fn --arms llm+jev,llm --repeats 3 --warmup 2 --arm-concurrency 1
```

Use the resulting two run directories separately:

```bash
uv run python scripts/benchmarks/filter_selection/report.py <main-run> \
  --supplement <supplement-run> --analysis scripts/benchmarks/filter_selection/analysis.md
```

The Orq profile is resolved in-process; keys are never printed or put on argv.
Each output directory contains immutable `raw.jsonl`, derived `scored.jsonl`,
`meta.json`, and summaries. Existing directories are not overwritten: provide a
new `--out` path when rerunning a workload.

For model overrides, use `--compiler-model`, `--classifier-model`, or
`--fast-filter-model`. Role resolution otherwise uses the configured model
roles/tasks, and refuses a settings-file model unless `--allow-settings-models`
is explicitly supplied. Record/config isolation is captured by settings path
and SHA-256 only.

## Measurement and interpretation

- Each query has three measured repeats per mode; discarded warmups are never
  scored. The reported query-level rates preserve the pair between modes.
  Repeats are not treated as independent query samples; category and overall
  scores use per-query repeat rates/majority votes, and paired tests operate on
  labelled rows.
- Main and supplement are separate workloads and must never be pooled.
  Literal-family queries remain separate labelled rows.
- The datetime label and source time bounds are identical for both modes.
  `llm+jev` no longer receives a free pass on time: its compiler's generated
  datetime filter is scored alongside `llm`'s.
- Facet offers include all 50 available catalogue values plus confirmed named
  candidates, using the production schema/selection logic. Diagnostics describe
  offered-catalogue coverage, not a top-20 cutoff.
- Every production cache-loader event is reported, including hit/miss, elapsed
  time, and source; multi-load planner calls retain per-event telemetry. Cold
  and warm cache end-to-end scenarios are measured separately against the live
  catalogue cache. A captured snapshot alone is not a cache-hit measurement.
  No sampled catalogue latency is added to model calls to manufacture a
  purported measured e2e latency.
- Latency is measured at concurrency one, with deterministic alternating mode
  order. Provider-side routing, rate limits, cache state beyond the recorded
  production cache state, and co-tenant load are uncontrolled and remain
  caveats. Do not run other model workloads during latency-grade collection.
- Reports include token usage and per-call costs, plus missing/partial pricing
  states. Cost is not silently treated as zero.
- `analysis.md` is reserved for future Opus analysis; generated reports include
  it only when supplied.

Historical runs can still be rendered for audit, but their experimental
`merged`, model-pinned, top-20, and modeled-latency claims are historical only
and are not comparable to the active two-mode workloads.
