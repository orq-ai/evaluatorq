# CLI Reference

Both `evaluatorq` and `eq` are aliases for the same entry point:

```toml
# pyproject.toml [project.scripts]
evaluatorq = "evaluatorq.cli:main"
eq         = "evaluatorq.cli:main"
```

Subcommands are registered at startup. `eq redteam` requires the `redteam` extra; `eq sim` requires the `simulation` extra.

!!! note "Primary UI — `eq dashboard`"
    The recommended way to browse saved runs is the multi-run FastHTML dashboard,
    `eq dashboard`. The canonical invocation scans a run directory — `eq dashboard`
    browses both default stores (red team + simulation), and `eq dashboard
    .evaluatorq/sim-runs` scopes to simulation. Passing a single JSON report file
    is an optional direct deep-link. The legacy `eq redteam ui` / `eq sim ui`
    Streamlit commands remain callable but are deprecated. See
    [Dashboard](../dashboard.md) and [Simulation](simulation.md).

Two command groups have their own pages:

- **[Red Teaming](redteam.md)** — adversarial testing (`eq redteam`).
- **[Simulation](simulation.md)** — multi-turn user simulation (`eq sim`; `sim` is shorthand).

## Canonical flag names

!!! note "Simulation I/O flags name the artifact they read or write"
    Commands that read a datapoints file use `--input` / `-i` (`simulate`,
    `export`, `upload-dataset`). Output flags are named for the artifact each
    command writes:

    | Command | Output flag(s) | Writes |
    |---|---|---|
    | `sim generate` | `--datapoints` / `-d` | generated datapoints JSONL |
    | `sim simulate` | `--results` / `-r` | simulation results JSONL |
    | `sim run` | `--datapoints` / `-d`, `--results` / `-r` | generated inputs, and the results |
    | `sim export` | `--output` / `-o` | OpenResponses payload JSON |

    The generic `--output` / `-o` was **removed** from `generate` / `simulate` /
    `run` — it wrote a different artifact per command. `sim export` keeps it, as
    it has a single output. On `sim simulate`, the input file is `--input` / `-i`
    (there is no `--datapoints` input alias).

    Other historical migrations are:

    | Historical | Current | Command(s) |
    |---|---|---|
    | `--report-output` | `--report` | `sim simulate`, `sim run` |
    | `--save-datapoints` | `--datapoints` | `sim run` |
    | `--export-md` / `--export-html` | `--report-md` / `--report-html` | `sim simulate`/`run`, `redteam run` |
    | `--save-report` | `--report` | `redteam run` |
    | `--output-dir` | `--artifacts-dir` | `redteam run` |

    Unchanged: `sim export --output`, `--no-save`, `--dataset-format`, `redteam --save`.

    **Removed aliases** — these no longer work; calling them raises an error:

    - SDK `simulate(run_output=...)` / `generate_and_simulate(run_output=...)` — removed, use `report=...` (raises `TypeError`)
    - SDK `red_team(output_dir=...)` — removed, use `artifacts_dir=...` (raises `TypeError`)
    - CLI `redteam run --output-dir` — removed, use `--artifacts-dir` (no such option)

!!! note "Simulation validation"
    Use `eq sim validate --input PATH`. The older `eq sim validate-dataset PATH`
    command remains as a compatibility alias (see [Simulation](simulation.md)).

## Top-level options

`eq --version` prints the installed version (e.g. `evaluatorq 1.3.2`) and exits. Running `eq` with no arguments prints help and exits.

## Recipes

```bash
# CI smoke run — one strategy per category, no LLM-generated strategies, quiet
eq redteam run -t agent:my-agent --max-per-category 1 --no-generate-strategies -q

# Save full per-stage artifacts to a directory
eq redteam run -t agent:my-agent --save detail --artifacts-dir ./runs

# Quick simulation — two personas, two scenarios
eq sim run --target agent:my-agent --num-personas 2 --num-scenarios 2
```

## Where to next

- **[Agent Simulation](../guides/agent-simulation.md)** — the `eq sim` workflow in depth.
- **[Red Teaming](../guides/red-teaming.md)** — the `eq redteam` workflow in depth.
- **[Getting Started](../guides/getting-started.md)** — run your first evaluation end-to-end.
