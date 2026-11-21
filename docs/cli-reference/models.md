# Models (`eq models`)

`eq models pareto` compares the models in the Orq catalogue on a score against a cost and prints the ones nothing beats, or the ones that are beaten and by what. It reads the live catalogue, so it needs `ORQ_API_KEY` or an Orq CLI sign-in.

For what the frontier means, how to read the output and where a blended price misleads, see [Model Pareto frontier](../model-pareto.md). This page covers the command only. To list catalogue models, use `orq models list`; evaluatorq does not duplicate it.

```bash
eq models pareto [OPTIONS]
```

| Flag | Type / Default | Description |
|---|---|---|
| `--dominated` | `bool` / `false` | List dominated models and the frontier model that beats each, instead of the frontier. |
| `--provider` | `str` (repeatable) | Compare only models from this provider or developer, such as `openai`. Case and an `-ai` suffix are ignored, so `google` also matches `google-ai`. |
| `--model` | `str` (repeatable) | Show only this model, compared against every other model in the comparison. A `provider/model` id, or a bare id held by exactly one model, looked up in the whole catalogue. Shown whether it is on the frontier or not. |
| `--effort` | `str` / none | Score the intelligence index at this reasoning effort instead of each model's default. Only with `--index intelligence`, and only an effort at least one catalogue model has a score at. |
| `--index` | `str` / `intelligence` | The y axis: `intelligence`, `economics`, `engineering`, `finance_and_accounting`, `healthcare_and_medical`, `legal` or `strategy_and_ops`. An index no catalogue model has a score on is refused. |
| `--x` | `str` / `price` | The x axis: `price` (blended price per million tokens, see `--mix`), `output_speed` or `time_to_first_token`. |
| `--mix` | `str` / `agentic` | How `--x price` blends a card's rates. `agentic`: 7 cache read : 2 fresh input : 1 output, the Artificial Analysis blend. `list`: 3 input : 1 output. Or a `READ:WRITE:FRESH:OUTPUT` ratio such as `7:1:1:1`. A share with a positive read or write weight adds a `Cache read $/1M` or `Cache write $/1M` column, where `(AA)` marks the Artificial Analysis price used because the host lists none and `(input)` marks the input rate used because the card has no cache price. Refused with any other `--x`. |
| `--include-disabled` | `bool` / `false` | Include models switched off in the workspace. |
| `--no-estimate` | `bool` / `false` | Leave out models whose score Orq estimated rather than measured. |
| `--min-score` | `float` / `15` | Hide models scoring below this on the chosen index; they are listed under `excluded` with the reason `score <n> is below min_score <floor>`. `0` shows every model. Must be `0` or more. |
| `--json` | `bool` / `false` | Print `{index, effort, x, x_label, mix, min_score, points, excluded, left_out_by_provider}` as JSON on stdout. `mix` is `null` off the price axis; each point carries `cache_read_source` and `cache_write_source` (`host`, `artificial_analysis`, `input`, or `null` when the mix has no share of that kind). |

`--provider` narrows the comparison itself: a model is then only checked against models from the same providers, so a model dominated in the full catalogue can be on the frontier of a filtered one. `--model` does the opposite: it narrows only what is printed. With both, the named model joins the comparison even when `--provider` would leave it out, and the output counts the models `--provider` left out.

## Examples

```bash
# the frontier on intelligence against price
eq models pareto

# every dominated model and what beats it
eq models pareto --dominated

# the same frontier on the 3:1 input/output list blend
eq models pareto --mix list

# your own token mix, with a cache-write share
eq models pareto --mix 7:1:1:1

# turn the quality floor off and show every model
eq models pareto --min-score 0

# OpenAI against Anthropic on the legal index
eq models pareto --index legal --provider openai --provider anthropic

# one model, as JSON, for a script
eq models pareto --model gpt-6.1-sol --json

# high reasoning effort against output speed, without estimated scores
eq models pareto --effort high --x output_speed --no-estimate
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | The analysis ran. An empty result is still `0`: a `--model` match left out of the comparison prints `<id> is not plotted: <reason>.` on stderr and an empty table. |
| `1` | The catalogue could not be read: no Orq credentials, or the request failed. |
| `2` | A usage error: an unknown `--index`, `--x` or `--mix`, `--mix` with an `--x` other than `price`, an `--index` no catalogue model has a score on, an `--effort` no catalogue model has a score at, `--effort` with a domain index, or a `--model` that matches no model or more than one, or a negative or non-finite `--min-score`. |
