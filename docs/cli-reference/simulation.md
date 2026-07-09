# Simulation (`eq sim`)

Agent simulation subcommand group. Registered only when `evaluatorq[simulation]` is installed. `sim` is shorthand for convenience — the feature is **agent simulation**.

Three main verbs: `generate` (datapoints only), `simulate` (run against pre-built datapoints), `run` (generate then simulate in one shot).

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
| `--save-datapoints` | `Path \| None` / `None` | Write generated datapoints to JSONL for reproducible re-runs. |
| `--output` / `-o` | `Path \| None` / `None` | Path to write results JSONL. |
| `--report-output` | `Path \| None` / `None` | Path to write full SimulationRun report JSON. |
| `--export-md` | `Path \| None` / `None` | Directory for an auto-named Markdown report. |
| `--export-html` | `Path \| None` / `None` | Directory for an auto-named HTML report. |
| `--yes` / `-y` | `bool` / `False` | Skip interactive confirmation prompt. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. `-v` info; `-vv` debug. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |

---

## `eq sim simulate`

Run simulations from a pre-built datapoints JSONL file.

```bash
eq sim simulate --datapoints dp.jsonl --target agent:<key>
```

Targets — same three flags as `eq sim run`. All other flags match `eq sim run` except `--num-personas`, `--num-scenarios`, and `--save-datapoints` are absent (datapoints are already provided).

| Flag | Type / Default | Description |
|---|---|---|
| `--datapoints` / `-d` | `Path` (required) | Path to datapoints JSONL file. |
| `--name` / `-n` | `str` / `sim` | Run name for the run-store entry. |
| `--sim-model` | `str` / `openai/gpt-5.4-mini` | Model for user-simulator and judge. |
| `--max-turns` | `int` / `10` | Maximum conversation turns. |
| `--parallelism` | `int` / `5` | Concurrent simulations. |
| `--evaluator` | `str` (repeatable) / API defaults | Evaluator name(s). Repeatable. |
| `--no-save` | `bool` / `False` | Skip writing to `.evaluatorq/sim-runs/`. |
| `--output` / `-o` | `Path \| None` / `None` | Path to write results JSONL. |
| `--report-output` | `Path \| None` / `None` | Path to write full SimulationRun report JSON. |
| `--export-md` | `Path \| None` / `None` | Directory for an auto-named Markdown report. |
| `--export-html` | `Path \| None` / `None` | Directory for an auto-named HTML report. |
| `--yes` / `-y` | `bool` / `False` | Skip interactive confirmation prompt. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |

---

## `eq sim generate`

Generate simulation datapoints only — no simulation is run.

```bash
eq sim generate --output dp.jsonl --agent-description "..."
```

| Flag | Type / Default | Description |
|---|---|---|
| `--output` / `-o` | `Path` (required) | Path to write generated datapoints JSONL. |
| `--agent-description` | `str \| None` / `None` | Free-text description of the agent. |
| `--target` | `str \| None` / `None` | Agent target used to fetch the description when `--agent-description` is omitted. Accepts `agent:<key>`. |
| `--sim-model` | `str` / `openai/gpt-5.4-mini` | Model for persona/scenario/first-message generation. |
| `--num-personas` | `int` / `5` | Number of personas to generate. |
| `--num-scenarios` | `int` / `5` | Number of scenarios to generate. |
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

## `eq sim validate-dataset`

Validate a simulation datapoints JSONL file.

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

## `eq sim ui`

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
