# Models (`eq models`)

`eq models pareto` compares the models in the Orq catalogue on a score against a cost and prints the ones nothing beats, or the ones that are beaten and by what. It reads the live catalogue, so it needs `ORQ_API_KEY` or an Orq CLI sign-in.

For what the frontier means, how to read the output and where list price misleads, see [Model Pareto frontier](../model-pareto.md). This page covers the command only. To list catalogue models, use `orq models list`; evaluatorq does not duplicate it.

```bash
eq models pareto [OPTIONS]
```

| Flag | Type / Default | Description |
|---|---|---|
| `--dominated` | `bool` / `false` | List dominated models and the frontier model that beats each, instead of the frontier. |
| `--provider` | `str` (repeatable) | Compare only models from this provider or developer, such as `openai`. |
| `--model` | `str` (repeatable) | Show only this model, still compared against every other model. A `provider/model` id, or a bare id held by exactly one model. Shown whether it is on the frontier or not. |
| `--effort` | `str` / none | Score the intelligence index at this reasoning effort instead of each model's default. Only with `--index intelligence`. |
| `--index` | `str` / `intelligence` | The y axis: `intelligence`, `economics`, `engineering`, `finance_and_accounting`, `healthcare_and_medical`, `legal` or `strategy_and_ops`. |
| `--x` | `str` / `price` | The x axis: `price` (3:1 blended list price per million tokens), `output_speed` or `time_to_first_token`. |
| `--include-disabled` | `bool` / `false` | Include models switched off in the workspace. |
| `--measured-only` | `bool` / `false` | Leave out models whose score Orq estimated rather than measured. |
| `--json` | `bool` / `false` | Print `{index, effort, x, points, excluded}` as JSON on stdout. |

`--provider` narrows the comparison itself: a model is then only checked against models from the same providers, so a model dominated in the full catalogue can be on the frontier of a filtered one. `--model` does the opposite: it narrows only what is printed.

## Examples

```bash
# the frontier on intelligence against price
eq models pareto

# every dominated model and what beats it
eq models pareto --dominated

# OpenAI against Anthropic on the legal index
eq models pareto --index legal --provider openai --provider anthropic

# one model, as JSON, for a script
eq models pareto --model gpt-6.1-sol --json

# high reasoning effort against output speed, measured scores only
eq models pareto --effort high --x output_speed --measured-only
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | The analysis ran. An empty result is still `0`: a `--model` match left out of the comparison prints `<id> is not plotted: <reason>.` on stderr and an empty table. |
| `1` | The catalogue could not be read: no Orq credentials, or the request failed. |
| `2` | A usage error: an unknown `--index` or `--x`, `--effort` with a domain index, or a `--model` that matches no model or more than one. |
