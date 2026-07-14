# Simulation (`eq sim`)

Agent simulation subcommand group. Registered only when `evaluatorq[simulation]` is installed. `sim` is shorthand for convenience — the feature is **agent simulation**.

Three main verbs: `generate` (datapoints only), `simulate` (run against pre-built datapoints), `run` (generate then simulate in one shot).

!!! note "Primary UI — `eq dashboard`"
    The recommended way to browse saved simulation runs is the multi-run FastHTML
    dashboard, `eq dashboard .evaluatorq/sim-runs` (scopes to simulation) or
    `eq dashboard` (both stores). Passing a single JSON report file is an optional
    direct deep-link. The legacy `eq sim ui` Streamlit command remains callable
    but is deprecated (see below).

## `eq sim run`

Generate personas and scenarios, then run simulations.

```bash
eq sim run --agent-description "..." --openai-model gpt-4o-mini
eq sim run --target agent:<key>
```

Targets — provide **exactly one**:

| Flag | Description |
|---|---|
| `--target` | `agent:<key>` or `deployment:<key>`. Bare values default to `agent:<key>`. |
| `--vercel-url` | Vercel AI SDK HTTP endpoint URL. |
| `--openai-model` | OpenAI-compatible model name. Provider resolved from env: `ORQ_API_KEY` → Orq AI Router; `OPENAI_API_KEY` → OpenAI-compatible. |

| Flag | Type / Default | Description |
|---|---|---|
| `--agent-description` | `str \| None` / `None` | Free-text description of the agent. May be omitted when `--target` is an Orq agent (fetched automatically). |
| `--name` / `-n` | `str` / `sim` | Run name for the run-store entry. |
| `--sim-model` | `str` / `openai/gpt-5.4-mini` | Model for user-simulator, judge, and generation. |
| `--max-turns` | `int` / `10` | Maximum conversation turns. |
| `--parallelism` | `int` / `5` | Concurrent simulations. |
| `--num-personas` | `int` / `5` | Number of personas to generate. |
| `--num-scenarios` | `int` / `5` | Number of scenarios to generate. |
| `--evaluator` | `str` (repeatable) / API defaults | Evaluator name(s). Repeatable. |
| `--no-save` | `bool` / `False` | Skip writing to `.evaluatorq/sim-runs/`. |
| `--datapoints` | `Path \| None` / `None` | Write generated datapoints to JSONL for reproducible re-runs. |
| `--output` / `-o` | `Path \| None` / `None` | Path to write results JSONL. (`--results` was removed; use `--output`.) |
| `--report` | `Path \| None` / `None` | Path to write full SimulationRun report JSON. |
| `--report-md` | `Path \| None` / `None` | Directory for an auto-named Markdown report. |
| `--report-html` | `Path \| None` / `None` | Directory for an auto-named HTML report. |
| `--executive-summary` / `--no-executive-summary` | `bool` / `True` | Generate an LLM narrative executive summary in the report. |
| `--yes` / `-y` | `bool` / `False` | Skip interactive confirmation prompt. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. `-v` info; `-vv` debug. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |

---

## `eq sim simulate`

Run simulations from a pre-built datapoints JSONL file.

```bash
eq sim simulate --input dp.jsonl --target agent:<key>
```

Targets — same three flags as `eq sim run`. Provide exactly one of `--input` (`-i`)
and `--dataset-id`; the latter fetches the datapoints from an Orq dataset.

| Flag | Type / Default | Description |
|---|---|---|
| `--input` / `-i` | `Path \| None` | Path to datapoints JSONL file. Mutually exclusive with `--dataset-id`. (`--datapoints` / `-d` were removed; use `--input`.) |
| `--dataset-id` | `str \| None` | Fetch datapoints from an Orq dataset instead of a local file. Requires `ORQ_API_KEY`. |
| `--name` / `-n` | `str` / `sim` | Run name for the run-store entry. |
| `--sim-model` | `str` / `openai/gpt-5.4-mini` | Model for user-simulator and judge. |
| `--max-turns` | `int` / `10` | Maximum conversation turns. |
| `--parallelism` | `int` / `5` | Concurrent simulations. |
| `--evaluator` | `str` (repeatable) / API defaults | Evaluator name(s). Repeatable. |
| `--no-save` | `bool` / `False` | Skip writing to `.evaluatorq/sim-runs/`. |
| `--output` / `-o` | `Path \| None` / `None` | Path to write results JSONL. (`--results` was removed; use `--output`.) |
| `--report` | `Path \| None` / `None` | Path to write full SimulationRun report JSON. |
| `--report-md` | `Path \| None` / `None` | Directory for an auto-named Markdown report. |
| `--report-html` | `Path \| None` / `None` | Directory for an auto-named HTML report. |
| `--executive-summary` / `--no-executive-summary` | `bool` / `True` | Generate an LLM narrative executive summary in the report. |
| `--yes` / `-y` | `bool` / `False` | Skip interactive confirmation prompt. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |

---

## `eq sim upload-dataset`

Upload simulation datapoints to an Orq dataset, or append them to an existing
dataset.

```bash
eq sim upload-dataset -i cases.jsonl -n "Support simulation set"
eq sim upload-dataset -i more.jsonl --dataset-id <id>
```

Persona and scenario objects are JSON-stringified because the Orq dataset API
accepts scalar `inputs` values. The simulation reader restores them when the
dataset is used with `eq sim simulate --dataset-id`.

| Flag | Type / Default | Description |
|---|---|---|
| `--input` / `-i` | `Path` (required) | Raw `sim generate` JSONL or a `--dataset-format` JSONL file. |
| `--name` / `-n` | `str \| None` | Display name for a new dataset; required unless `--dataset-id` is provided. |
| `--path` | `str` / `Default` | Orq folder path for a new dataset. |
| `--dataset-id` | `str \| None` | Append to this existing dataset instead of creating one. |

---

## `eq sim generate`

Generate simulation datapoints only — no simulation is run.

```bash
eq sim generate --output dp.jsonl --agent-description "..."
```

| Flag | Type / Default | Description |
|---|---|---|
| `--output` / `-o` | `Path` (required) | Path to write generated datapoints JSONL. (`--datapoints` was removed; use `--output`.) |
| `--agent-description` | `str \| None` / `None` | Free-text description of the agent. |
| `--target` | `str \| None` / `None` | Agent target used to fetch the description when `--agent-description` is omitted. Accepts `agent:<key>`. |
| `--sim-model` | `str` / `openai/gpt-5.4-mini` | Model for persona/scenario/first-message generation. |
| `--num-personas` | `int` / `5` | Number of personas to generate. |
| `--num-scenarios` | `int` / `5` | Number of scenarios to generate. |
| `--dataset-format` | `bool` / `False` | Write Orq dataset-row envelopes instead of raw simulation datapoints. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |

---

## `eq sim export`

Convert simulation results JSONL to OpenResponses payload JSON.

```bash
eq sim export --input results.jsonl --output payload.json
```

| Flag | Type / Default | Description |
|---|---|---|
| `--input` / `-i` | `Path` (required) | Path to results JSONL file. |
| `--output` / `-o` | `Path` (required) | Path to write OpenResponses payload JSON. |

---

## `eq sim validate`

Validate a simulation datapoints JSONL file.

```bash
eq sim validate --input dp.jsonl
```

| Flag | Type / Default | Description |
|---|---|---|
| `--input` / `-i` | `Path` (required) | Path to datapoints JSONL file to validate. (`validate-dataset` is retained as a compatibility alias.) |

---

## `eq sim validate-dataset` (compatibility alias)

Deprecated alias for `eq sim validate --input PATH`. Retained for compatibility.

```bash
eq sim validate-dataset dp.jsonl
```

| Argument | Type / Default | Description |
|---|---|---|
| `PATH` | `Path` (required) | Path to datapoints JSONL file to validate. |

---

## `eq sim runs`

List recent simulation runs.

```bash
eq sim runs [DIRECTORY] [--limit N]
```

| Flag / Argument | Type / Default | Description |
|---|---|---|
| `DIRECTORY` | `Path \| None` / `None` | Directory to scan. Defaults to `.evaluatorq/sim-runs/`. |
| `--limit` / `-n` | `int` / `20` | Maximum number of runs to show. |

---

## `eq sim ui` (deprecated)

!!! warning "Deprecated — use `eq dashboard`"
    `eq sim ui` is a deprecated legacy Streamlit command. The primary UI for
    browsing simulation runs is the multi-run FastHTML dashboard: `eq dashboard
    .evaluatorq/sim-runs` (scopes to simulation) or `eq dashboard` (both stores).
    Passing a single JSON report file to `eq dashboard` is an optional direct
    deep-link.

Launch the Streamlit dashboard for a saved simulation run.

```bash
eq sim ui [RUN_PATH] [--latest] [--host HOST] [--port PORT]
```

| Flag / Argument | Type / Default | Description |
|---|---|---|
| `RUN_PATH` | `Path \| None` / `None` | Saved run to open. Omit to use the latest auto-saved run. |
| `--latest` / `-l` | `bool` / `False` | Open the most recent run without passing a path. |
| `--host` | `str` / `localhost` | Host to bind the Streamlit server to. |
| `--port` | `int` / `8501` | Port for the Streamlit server. |

Requires `evaluatorq[simulation]`.
