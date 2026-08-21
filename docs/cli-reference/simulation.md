# Simulation (`eq sim`)

Agent simulation subcommand group. Registered only when `evaluatorq[simulation]` is installed. `sim` is shorthand for convenience — the feature is **agent simulation**.

Three main verbs: `generate` (datapoints only), `simulate` (run against pre-built datapoints), `run` (generate then simulate in one shot).

!!! note "Primary UI — `eq dashboard`"
    The recommended way to browse saved simulation runs is the multi-run FastHTML
    dashboard, `eq dashboard .evaluatorq/sim-runs` (scopes to simulation) or
    `eq dashboard` (both stores). Passing a single JSON report file is an optional
    direct deep-link.

--8<-- "docs/_snippets/openai-direct-model.md"

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
| `--memory-entity` | Memory `entity_id` sent with every `agent:<key>` (or bare `<key>`) target call, for agents with a memory store attached. Omit to mint a fresh id per conversation (parallel conversations never share memory); pass one to reuse a specific (e.g. seeded) entity, shared across the run. |
| `--vercel-url` | Vercel AI SDK HTTP endpoint URL. |
| `--openai-model` | OpenAI-compatible model name. Provider resolved from env: `ORQ_API_KEY` → Orq AI Router; `OPENAI_API_KEY` → OpenAI-compatible. |

| Flag | Type / Default | Description |
|---|---|---|
| `--agent-description` | `str \| None` / `None` | Free-text description of the agent. May be omitted when `--target` is an Orq agent (fetched automatically). |
| `--name` / `-n` | `str` / `sim` | Run name for the run-store entry. |
| `--sim-model` | `str` / `openai/gpt-5.6-luna` | Model for user-simulator, judge, and generation. |
| `--max-turns` | `int` / `10` | Maximum conversation turns. |
| `--datapoint-parallelism` | `int` / `10` | Concurrent simulations. `--parallelism` is a deprecated alias. |
| `--llm-parallelism` | `int` / unset | Ceiling on in-flight LLM requests for the whole run. |
| `--num-personas` | `int` / `5` | Number of personas to generate. |
| `--num-scenarios` | `int` / `5` | Number of scenarios to generate. |
| `--evaluator` | `str` (repeatable) / API defaults | Evaluator name(s). Repeatable. |
| `--no-save` | `bool` / `False` | Skip writing to `.evaluatorq/sim-runs/`. |
| `--recommendations` / `--no-recommendations` | `bool` / `True` | Generate LLM remediation suggestions for failures, tied to their concrete cause. On by default; `--no-recommendations` skips the extra LLM call. Uses `--sim-model`. |
| `--datapoints` / `-d` | `Path \| None` / `None` | Write generated datapoints to JSONL for reproducible re-runs. |
| `--results` / `-r` | `Path \| None` / `None` | Path to write results JSONL (results + scorer averages + metadata). |
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

Targets — same three flags as `eq sim run`. Provide exactly one of four input
sources: `--input` (`-i`), `--dataset-id`, `--experiment-id` (optionally narrowed
by `--experiment-run-id`), or `--from-run`.

| Flag | Type / Default | Description |
|---|---|---|
| `--input` / `-i` | `Path \| None` | Path to datapoints JSONL file. Mutually exclusive with the other input sources. |
| `--dataset-id` | `str \| None` | Fetch datapoints from an Orq dataset instead of a local file. Requires `ORQ_API_KEY`. |
| `--experiment-id` | `str \| None` | Fetch datapoints from an Orq experiment's rows instead of a local file. Requires `ORQ_API_KEY`. |
| `--experiment-run-id` | `str \| None` | Specific run of `--experiment-id` to load. Latest run if omitted. |
| `--from-run` | `str \| None` | Replay a previous run from `.evaluatorq/sim-runs/`: pass its file name, run id, path, or `"latest"`. Re-runs the exact same personas, scenarios, and first messages; only the target/evaluators may differ. |
| `--memory-entity` | `str \| None` / `None` | Memory `entity_id` sent with every `agent:<key>` (or bare `<key>`) target call, for agents with a memory store attached. Omit to mint a fresh id per conversation; pass one to reuse a specific (e.g. seeded) entity, shared across the run. |
| `--name` / `-n` | `str` / `sim` | Run name for the run-store entry. |
| `--sim-model` | `str` / `openai/gpt-5.6-luna` | Model for user-simulator and judge. |
| `--max-turns` | `int` / `10` | Maximum conversation turns. Defaults to the replayed run's cap with `--from-run`. |
| `--datapoint-parallelism` | `int` / `10` | Concurrent simulations. `--parallelism` is a deprecated alias. |
| `--llm-parallelism` | `int` / unset | Ceiling on in-flight LLM requests for the whole run. |
| `--evaluator` | `str` (repeatable) / API defaults | Evaluator name(s). Repeatable. |
| `--no-save` | `bool` / `False` | Skip writing to `.evaluatorq/sim-runs/`. |
| `--recommendations` / `--no-recommendations` | `bool` / `True` | Generate LLM remediation suggestions for failures, tied to their concrete cause. On by default; `--no-recommendations` skips the extra LLM call. Uses `--sim-model`. |
| `--results` / `-r` | `Path \| None` / `None` | Path to write results JSONL. |
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
eq sim generate --datapoints dp.jsonl --agent-description "..."
```

| Flag | Type / Default | Description |
|---|---|---|
| `--datapoints` / `-d` | `Path` (required) | Path to write generated datapoints JSONL. |
| `--agent-description` | `str \| None` / `None` | Free-text description of the agent. |
| `--target` | `str \| None` / `None` | Agent target used to fetch the description when `--agent-description` is omitted. Accepts `agent:<key>`. |
| `--sim-model` | `str` / `openai/gpt-5.6-luna` | Model for persona/scenario/first-message generation. |
| `--num-personas` | `int` / `5` | Number of personas to generate. |
| `--num-scenarios` | `int` / `5` | Number of scenarios to generate. |
| `--persona-seed` | `str` (repeatable) / `None` | Archetype seed for a persona, e.g. `"angry retiree"` (repeatable). Each seed becomes one persona the LLM fleshes out — overrides `--num-personas`. Omit to auto-generate. |
| `--scenario-seed` | `str` (repeatable) / `None` | Situation seed for a scenario, e.g. `"disputes refund denial"` (repeatable). Each seed becomes one scenario — overrides `--num-scenarios`. Omit to auto-generate. |
| `--dataset-format` | `bool` / `False` | Write Orq dataset-row envelopes instead of raw simulation datapoints. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |

---

## `eq sim from-traces`

Build simulation datapoints from Orq production traces.

```bash
eq sim from-traces --output dp.jsonl --limit 50 --lookback-hours 24
eq sim from-traces --output dp.jsonl --extend 20 --agent-description "..."
```

Fetches recent traces from the Orq traces API (requires `ORQ_API_KEY`) and builds
one datapoint per trace conversation: persona and scenario are inferred from the
transcript, and the first message is the real user's opening message verbatim.
Pass `--extend N` to additionally generate `N` new datapoints matching the traffic
distribution of the fetched traces (extra LLM calls). Feed the output file to
`eq sim simulate --input` to run it.

| Flag | Type / Default | Description |
|---|---|---|
| `--output` / `-o` | `Path` (required) | Path to write generated datapoints JSONL. |
| `--limit` | `int` / `20` | Maximum number of traces to fetch. |
| `--lookback-hours` | `float \| None` | Only fetch traces from the last N hours. Default: no time filter. |
| `--search` | `str \| None` | Free-text search applied to the trace list. |
| `--extend` | `int` / `0` | Also generate N distribution-matched datapoints on top of the direct per-trace ones (extra LLM calls). `0` disables extension. |
| `--agent-description` | `str \| None` | Agent description used for `--extend` generation. Optional; inferred from the traffic profile if omitted. |
| `--sim-model` | `str` / `openai/gpt-5.6-luna` | Model for persona/scenario inference and extension generation. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |

---

## `eq sim export`

Export simulation results: OpenResponses payload JSON, or an HTML/Markdown report.

```bash
eq sim export --input results.jsonl --output payload.json
eq sim export --input sim-report.json --output report.html --format html --recommendations
```

Markdown/HTML exports include remediation suggestions if the input run JSON
already carries them (runs generate them by default — see `--no-recommendations`),
or if `--recommendations` is passed here to generate them at export time for a run
that has none stored.

| Flag | Type / Default | Description |
|---|---|---|
| `--input` / `-i` | `Path` (required) | Path to a results JSONL file or a SimulationRun report JSON (`--report` / `--report-output`). |
| `--output` / `-o` | `Path` (required) | Path to write the exported file. |
| `--format` | `str` / `openresponses` | Export format: `openresponses` (payload JSON), `md` (Markdown report), `html` (HTML report). |
| `--recommendations` | `bool` / `False` | For `md`/`html`: generate LLM remediation suggestions at export time if none are stored. Extra LLM cost; uses `--sim-model`. |
| `--sim-model` | `str` / `openai/gpt-5.6-luna` | Model for `--recommendations` generation. |
| `--target-label` | `str` / `agent` | Target name shown in md/html report headers. |

---

## `eq sim validate`

Validate a simulation datapoints JSONL file.

```bash
eq sim validate --input dp.jsonl
eq sim validate dp.jsonl
```

| Flag / Argument | Type / Default | Description |
|---|---|---|
| `PATH` | `Path \| None` | Path to datapoints JSONL file (legacy positional form). |
| `--input` / `-i` | `Path \| None` | Path to datapoints JSONL file to validate. (`validate-dataset` is retained as a compatibility alias.) |

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
| `--full` / `-f` | `bool` / `False` | Render at full content width; do not truncate columns. |
| `--json` | `bool` / `False` | Emit runs as a JSON array on stdout. |

