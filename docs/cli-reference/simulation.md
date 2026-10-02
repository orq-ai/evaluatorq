# Simulation (`eq sim`)

Agent simulation subcommand group. Always registered; the `simulation` extra (`evaluatorq[simulation]`) supplies the optional dependencies some of its commands need at runtime. `sim` is shorthand for convenience — the feature is **agent simulation**.

Three main verbs: `generate` (datapoints only), `simulate` (run against pre-built datapoints), `run` (generate then simulate in one shot).

!!! note "The UI — `eq dashboard`"
    The way to browse saved simulation runs is the multi-run FastHTML dashboard, `eq dashboard .evaluatorq/sim-runs` (scopes to simulation) or `eq dashboard` (both stores). Passing a single JSON report file is an optional direct deep-link.

--8<-- "docs/_snippets/openai-direct-model.md"

## `eq sim run`

Generate personas and scenarios, then run simulations.

```bash
eq sim run --agent-description "..." --openai-model gpt-5.6-luna
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
| `--name` / `-n` | `str \| None` / unset | Run name. Unset unless passed: the SDK then saves the run as `sim` and names an uploaded experiment `simulation-<timestamp>-<id>`. |
| `--sim-model` | `str` / `openai/gpt-5.6-luna` | Model for the user-simulator, the judge, persona/scenario/first-message generation, the recommendations pass and the executive summary. Sets `llm_config.model`. |
| `--max-turns` | `int` / `10` | Maximum conversation turns. Unset unless passed; the SDK default is `10`. |
| `--datapoint-parallelism` | `int` / `10` | Concurrent simulations. Unset unless passed; the SDK default is `10`. `--parallelism` is a deprecated alias. |
| `--llm-parallelism` | `int` / `10` | Ceiling on in-flight LLM requests for the whole run. `-1` disables it. |
| `--num-personas` | `int` / `5` | Number of personas to generate. |
| `--num-scenarios` | `int` / `5` | Number of scenarios to generate. |
| `--persona-seed` | `str` (repeatable) / `None` | Archetype seed for a persona, e.g. `"angry retiree"` (repeatable). Each seed becomes one persona the LLM fleshes out — overrides `--num-personas`. Omit to auto-generate. |
| `--scenario-seed` | `str` (repeatable) / `None` | Situation seed for a scenario, e.g. `"disputes a refund denial"` (repeatable). Each seed becomes one scenario — overrides `--num-scenarios`. Omit to auto-generate. |
| `--generation-instructions` | `str` / `''` | Free-text steer applied to every generated persona AND scenario, e.g. `"enterprise B2B buyers, replying in German"`. Stacks on top of any seeds and `--edge-case-percentage`. Empty leaves the built-in prompts unchanged. |
| `--target-reasoning-effort` | `str \| None` / `None` | Reasoning effort pinned on the target agent under test (`agent:<key>` targets only). Distinct from the user-simulator's and judge's own reasoning effort, which comes from `EVALUATORQ_REASONING_EFFORT` — see [Tuning](../tuning.md). |
| `--evaluator` | `str` (repeatable) / API defaults | Evaluator name(s). Repeatable. |
| `--save` / `--no-save` | `bool` / `True` | Write the run to `.evaluatorq/sim-runs/`, with a run manifest. The SDK saves it after the recommendations and executive summary are attached. `--no-save` skips it. |
| `--recommendations` / `--no-recommendations` | `bool` / `True` | Generate LLM remediation suggestions for failures, tied to their concrete cause. On by default; `--no-recommendations` skips the extra LLM call. Uses `--sim-model`. |
| `--datapoints` / `-d` | `Path \| None` / `None` | Write generated datapoints to JSONL for reproducible re-runs. |
| `--results` / `-r` | `Path \| None` / `None` | Path to write results JSONL, one `SimulationResult` per line. |
| `--report` | `Path \| None` / `None` | Path to write full SimulationRun report JSON. The [output reference](../guides/agent-simulation-output.md) describes this file and the `--results` rows. |
| `--report-md` | `Path \| None` / `None` | Directory for an auto-named Markdown report. |
| `--report-html` | `Path \| None` / `None` | Directory for an auto-named HTML report. |
| `--executive-summary` / `--no-executive-summary` | `bool` / `True` | Generate an LLM narrative executive summary in the report. The SDK generates it before it saves the run. |
| `--yes` / `-y` | `bool` / `False` | Skip interactive confirmation prompt. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. `-v` info; `-vv` debug. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |
| `--config` | `PATH \| -` / `None` | JSON file of `generate_and_simulate()` keyword arguments, or `-` to read it from stdin. See [Driving the CLI from a config file](#driving-the-cli-from-a-config-file); `eq sim schema --command run` prints the accepted shape. |
| `--llm-config` | JSON / `None` | `LLMCallConfig` for every simulation-side LLM call as a JSON object. Merged field by field into `"llm_config"` from `--config`. `--sim-model` wins over its `"model"` when passed. |
| `--json` | `bool` / `False` | Print the final `SimulationRun` as JSON on stdout. Progress and messages go to stderr; exit codes are unchanged. |

---

## `eq sim simulate`

Run simulations from a pre-built datapoints JSONL file.

```bash
eq sim simulate --input dp.jsonl --target agent:<key>
```

Targets — same three flags as `eq sim run`. Provide exactly one input source: `--input` (`-i`), `--dataset-id`, `--experiment-id` (optionally narrowed by `--experiment-run-id`), `--from-run`, or inline `datapoints` (or `personas` with `scenarios`) in `--config`.

There is no `--target-reasoning-effort` flag here — it is a `eq sim run` flag only. To pin the target's reasoning effort on a pre-built datapoint set, set `"target_reasoning_effort"` in `--config`, or call `simulate()` with `target_reasoning_effort=` from Python (see [Tuning](../tuning.md)).

| Flag | Type / Default | Description |
|---|---|---|
| `--input` / `-i` | `Path \| None` | Path to datapoints JSONL file. Mutually exclusive with the other input sources. |
| `--dataset-id` | `str \| None` | Fetch datapoints from an Orq dataset instead of a local file. Requires `ORQ_API_KEY`. |
| `--experiment-id` | `str \| None` | Fetch datapoints from an Orq experiment's rows instead of a local file. Requires `ORQ_API_KEY`. |
| `--experiment-run-id` | `str \| None` | Specific run of `--experiment-id` to load. Latest run if omitted. |
| `--from-run` | `str \| None` | Replay a previous run from `.evaluatorq/sim-runs/`: pass its file name, run id, path, or `"latest"`. Re-runs the exact same personas, scenarios, and first messages; only the target/evaluators may differ. |
| `--memory-entity` | `str \| None` / `None` | Memory `entity_id` sent with every `agent:<key>` (or bare `<key>`) target call, for agents with a memory store attached. Omit to mint a fresh id per conversation; pass one to reuse a specific (e.g. seeded) entity, shared across the run. |
| `--name` / `-n` | `str \| None` / unset | Run name. Unset unless passed: the SDK then saves the run as `sim` and names an uploaded experiment `simulation-<timestamp>-<id>`. |
| `--sim-model` | `str` / `openai/gpt-5.6-luna` | Model for the user-simulator, the judge, the recommendations pass and the executive summary. Sets `llm_config.model`. |
| `--max-turns` | `int` / `10` | Maximum conversation turns. Unset unless passed; the SDK uses the replayed run's cap with `--from-run`, else `10`. |
| `--datapoint-parallelism` | `int` / `10` | Concurrent simulations. Unset unless passed; the SDK default is `10`. `--parallelism` is a deprecated alias. |
| `--llm-parallelism` | `int` / `10` | Ceiling on in-flight LLM requests for the whole run. `-1` disables it. |
| `--evaluator` | `str` (repeatable) / API defaults | Evaluator name(s). Repeatable. |
| `--save` / `--no-save` | `bool` / `True` | Write the run to `.evaluatorq/sim-runs/`, with a run manifest. The SDK saves it after the recommendations and executive summary are attached. `--no-save` skips it. |
| `--recommendations` / `--no-recommendations` | `bool` / `True` | Generate LLM remediation suggestions for failures, tied to their concrete cause. On by default; `--no-recommendations` skips the extra LLM call. Uses `--sim-model`. |
| `--results` / `-r` | `Path \| None` / `None` | Path to write results JSONL. |
| `--report` | `Path \| None` / `None` | Path to write full SimulationRun report JSON. The [output reference](../guides/agent-simulation-output.md) describes this file and the `--results` rows. |
| `--report-md` | `Path \| None` / `None` | Directory for an auto-named Markdown report. |
| `--report-html` | `Path \| None` / `None` | Directory for an auto-named HTML report. |
| `--executive-summary` / `--no-executive-summary` | `bool` / `True` | Generate an LLM narrative executive summary in the report. The SDK generates it before it saves the run. |
| `--yes` / `-y` | `bool` / `False` | Skip interactive confirmation prompt. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |
| `--config` | `PATH \| -` / `None` | JSON file of `simulate()` keyword arguments, or `-` to read it from stdin. See [Driving the CLI from a config file](#driving-the-cli-from-a-config-file); `eq sim schema` prints the accepted shape. |
| `--llm-config` | JSON / `None` | `LLMCallConfig` for every simulation-side LLM call as a JSON object. Merged field by field into `"llm_config"` from `--config`. `--sim-model` wins over its `"model"` when passed. |
| `--json` | `bool` / `False` | Print the final `SimulationRun` as JSON on stdout. Progress and messages go to stderr; exit codes are unchanged. |

### Driving the CLI from a config file

`--config` on `eq sim simulate` and `eq sim run` takes the keyword arguments of the Python `simulate()` and `generate_and_simulate()` functions as one JSON object, under their public names (`run_name`, `experiment_description`, `raise_on_execution_failure`). That reaches every data-shaped parameter, including the ones without a flag: inline `personas` and `scenarios` or `datapoints`, `scoring`, `target_agent_timeout_ms`, `max_tool_result_chars`, `per_simulation_timeout_s`, `upload_results`, the full `llm_config`, and on `sim run` `edge_case_percentage`. Pass a path, or `-` to read it from stdin. YAML is not accepted.

Inline `datapoints`, or `personas` with `scenarios`, count as the input source of `eq sim simulate`, so the file below needs no `--input`:

```json
{
  "target": "agent:my-agent",
  "personas": [
    {
      "name": "Impatient customer",
      "patience": 0.2,
      "assertiveness": 0.8,
      "politeness": 0.4,
      "technical_level": 0.3,
      "communication_style": "terse",
      "background": "Wants a refund today"
    }
  ],
  "scenarios": [
    {
      "name": "Refund",
      "goal": "Get a full refund",
      "criteria": [{"description": "Agent asks for the order number", "type": "must_happen"}]
    }
  ],
  "max_turns": 6,
  "llm_config": {"model": "openai/gpt-5.6-luna", "temperature": 0.2},
  "target_reasoning_effort": "low",
  "per_simulation_timeout_s": 300
}
```

```bash
eq sim simulate --config sim.json --json > run.json
```

`eq sim run` takes the generation keywords the same way:

```json
{
  "target": "agent:my-agent",
  "num_personas": 4,
  "edge_case_percentage": 0.25,
  "generation_instructions": "customers replying in German",
  "llm_config": {"model": "openai/gpt-5.6-luna"}
}
```

```bash
eq sim run --config gen.json --num-scenarios 3 --json
```

A flag passed on the command line beats the file, and the file beats the CLI default. The command checks where a value came from, not what it is, so `--max-turns 10` wins over `"max_turns": 6` even though 10 is the default. A `null` in the file means "not set" and is accepted only on fields that allow `None`, such as `run_name`, `max_turns` or `datapoint_parallelism`; the SDK default then applies. A value nobody sets (no flag, no file entry) is never sent: `max_turns`, `datapoint_parallelism` and `run_name` reach the SDK only when you pass them, so `--from-run` can still restore the replayed cap. `"save": false` in the file has the effect of `--no-save`. `--llm-config` and the file's `"llm_config"` merge field by field, and `--sim-model` wins for `model`.

The target and the input source are each one choice, so a command-line flag replaces the file's whole choice rather than adding a second one. `--target`, `--vercel-url` or `--openai-model` drops the file's `"target"`. On `eq sim simulate`, `--input`, `--dataset-id`, `--experiment-id` or `--from-run` drops the file's `"datapoints"`, `"personas"`, `"scenarios"`, `"dataset_id"`, `"experiment_id"`, `"experiment_run_id"` and `"previous_run"`. `--experiment-run-id` alone narrows the file's `"experiment_id"` instead. Two flags for the same choice on the command line are still rejected.

Unknown keys fail the command before anything runs, at every depth, so `"num_personas"` in a `sim simulate` file or `"patiense"` inside a persona is rejected with the field path. `"target"` takes only the `agent:<key>` and `deployment:<key>` string forms; a callable or `AgentTarget`, `user_simulator`, `judge`, `hooks` and `generation_client` stay Python-only.

With `--json`, stdout carries the `SimulationRun` and nothing else; the run-store save, the report files and every progress line still happen, on stderr.

---

## `eq sim schema`

Print a JSON schema: what `--config` accepts, or what `--json` prints.

```bash
eq sim schema [--input | --output] [--command simulate|run]
```

| Flag | Type / Default | Description |
|---|---|---|
| `--input` / `--output` | `bool` / `--input` | `--input` prints the config file schema. `--output` prints the `SimulationRun` schema, which both commands share. |
| `--command` | `simulate \| run` / `simulate` | Which command's config file `--input` describes: `SimulateRunConfig` for `eq sim simulate`, `GenerateAndSimulateRunConfig` for `eq sim run`. |

---

## `eq sim upload-dataset`

Upload simulation datapoints to an Orq dataset, or append them to an existing dataset.

```bash
eq sim upload-dataset -i cases.jsonl -n "Support simulation set"
eq sim upload-dataset -i more.jsonl --dataset-id <id>
```

The first command prints the new dataset ID. Use that ID, not the display name, with `eq sim simulate --dataset-id` or the Python `extend_from_dataset()` function. Here, `upload-dataset --dataset-id` **adds rows to the stored dataset**; `extend_from_dataset()` **generates new cases without changing the stored dataset**. Persona and scenario objects are JSON-stringified because the Orq dataset API accepts scalar `inputs` values. The simulation reader restores them when the dataset is used with `eq sim simulate --dataset-id`.

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
| `--generation-instructions` | `str` / `''` | Free-text steer applied to every generated persona AND scenario, e.g. `"enterprise B2B buyers, replying in German"`. Stacks on top of any seeds and `--edge-case-percentage`. Empty leaves the built-in prompts unchanged. |
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

Fetches recent traces from the Orq traces API (requires `ORQ_API_KEY`) and builds one datapoint per trace conversation: persona and scenario are inferred from the transcript, and the first message is the real user's opening message verbatim. Pass `--extend N` to additionally generate `N` new datapoints matching the traffic distribution of the fetched traces (extra LLM calls). Feed the output file to `eq sim simulate --input` to run it.

| Flag | Type / Default | Description |
|---|---|---|
| `--output` / `-o` | `Path` (required) | Path to write generated datapoints JSONL. |
| `--limit` | `int` / `20` | Maximum number of traces to fetch. |
| `--lookback-hours` | `float \| None` | Only fetch traces from the last N hours. Default: no time filter. |
| `--search` | `str \| None` | Free-text search applied to the trace list. |
| `--extend` | `int` / `0` | Also generate N distribution-matched datapoints on top of the direct per-trace ones (extra LLM calls). `0` disables extension. |
| `--agent-description` | `str \| None` | Agent description used for `--extend` generation. Optional; inferred from the traffic profile if omitted. |
| `--sim-model` | `str` / `openai/gpt-5.6-luna` | Model for persona/scenario inference and extension generation. |
| `--llm-parallelism` | `int` / `10` | Ceiling on in-flight LLM requests while building datapoints. `-1` disables it. |
| `--verbose` / `-v` | count / `0` | Increase verbosity. |
| `--quiet` / `-q` | `bool` / `False` | Suppress non-error output. |

---

## `eq sim export`

Export simulation results: OpenResponses payload JSON, or an HTML/Markdown report.

```bash
eq sim export --input results.jsonl --output payload.json
eq sim export --input sim-report.json --output report.html --format html --recommendations
```

Markdown/HTML exports include remediation suggestions if the input run JSON already carries them (runs generate them by default — see `--no-recommendations`), or if `--recommendations` is passed here to generate them at export time for a run that has none stored.

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

