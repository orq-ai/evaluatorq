# Model Pareto frontier

The **Pareto frontier** of the Orq model catalogue is the set of models that no other model dominates. A model is **dominated** when another model is at least as good on both score and price and strictly better on one of them.

Reach for it when you are picking a model for a role and want to rule out the ones that are never the right choice, or when you want to know whether the model you run today has a cheaper or smarter replacement. It does not tell you which frontier model to pick.

The scores are Artificial Analysis (AA) benchmark indices that Orq attaches to each catalogue entry, not results from your own evaluation runs. `eq models` and `load_models()` read the live catalogue, so they need `ORQ_API_KEY` or an Orq CLI sign-in. Without either, the command exits `1` with `Error: No Orq API key or CLI OAuth session is available for the model catalogue.`

## The frontier in one command

=== "CLI"

    ```bash
    eq models pareto
    ```

=== "Python"

    ```python
    import asyncio

    from evaluatorq.models import load_models, pareto

    models = asyncio.run(load_models())
    result = pareto(models)

    for point in result.frontier:
        print(f"{point.model.router_id:45} {point.score:5.1f}  ${point.x:.2f}/1M")

    for point in result.dominated[:3]:
        print(f"{point.model.router_id} is beaten by {point.dominated_by}")
    ```

The CLI run on 2026-11-20 printed:

```console
$ eq models pareto
                 Pareto frontier: intelligence index (default effort) vs price
╭───────────────────────────────────┬───────┬──────────────────┬─────────────┬──────┬──────────╮
│ Model                             │ Score │ Price $/1M (3:1) │ Score per $ │ Est. │ Frontier │
├───────────────────────────────────┼───────┼──────────────────┼─────────────┼──────┼──────────┤
│ anthropic/claude-opus-5-5         │  48.2 │            8.000 │         6.0 │      │ yes      │
│ wafer/Kimi-K3.1-Pro               │  46.5 │            4.000 │        11.6 │      │ yes      │
│ particle/deepseek-v4.1-flash-1017 │  32.6 │            0.438 │        74.5 │      │ yes      │
│ deepseek/deepseek-flash           │  28.6 │            0.175 │       163.4 │      │ yes      │
│ zai/glm-5.3-flash                 │  27.8 │            0.119 │       234.1 │      │ yes      │
╰───────────────────────────────────┴───────┴──────────────────┴─────────────┴──────┴──────────╯
34 model(s) hidden for scoring below --min-score 15. Pass --min-score 0 to show every model.
4 model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.
```

Read the table from the top down: each row is cheaper than the one above it and scores lower. `Score per $` is the score divided by the price. `Est.` marks a score that Orq estimated from an older benchmark version instead of measuring.

## The quality floor

Models scoring below 15 are left out before the frontier is drawn. Without the floor, the cheap end of the frontier holds models that are there only because nothing is cheaper: with the floor off, the run above adds `tensorix/qwen/qwen3-coder-30b-a3b-instruct` at a score of 2.7 and $0.107 per 1M tokens. 15 is the same floor evaluatorq uses when it picks default models for its own roles, `MIN_SEAT_INDEX`.

Set the floor with `--min-score` or `min_score=`. `--min-score 0` or `min_score=None` shows every model:

=== "CLI"

    ```bash
    eq models pareto --min-score 0
    ```

=== "Python"

    ```python
    import asyncio

    from evaluatorq.models import load_models, pareto

    result = pareto(asyncio.run(load_models()), min_score=None)
    print(len(result.frontier), "on the frontier")
    ```

```console
$ eq models pareto --min-score 0
                      Pareto frontier: intelligence index (default effort) vs price
╭────────────────────────────────────────────┬───────┬──────────────────┬─────────────┬──────┬──────────╮
│ Model                                      │ Score │ Price $/1M (3:1) │ Score per $ │ Est. │ Frontier │
├────────────────────────────────────────────┼───────┼──────────────────┼─────────────┼──────┼──────────┤
│ anthropic/claude-opus-5-5                  │  48.2 │            8.000 │         6.0 │      │ yes      │
│ wafer/Kimi-K3.1-Pro                        │  46.5 │            4.000 │        11.6 │      │ yes      │
│ particle/deepseek-v4.1-flash-1017          │  32.6 │            0.438 │        74.5 │      │ yes      │
│ deepseek/deepseek-flash                    │  28.6 │            0.175 │       163.4 │      │ yes      │
│ zai/glm-5.3-flash                          │  27.8 │            0.119 │       234.1 │      │ yes      │
│ tensorix/qwen/qwen3-coder-30b-a3b-instruct │   2.7 │            0.107 │        25.1 │ yes  │ yes      │
╰────────────────────────────────────────────┴───────┴──────────────────┴─────────────┴──────┴──────────╯
4 model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.
```

The floor applies to whichever score you put on the y axis, so with `--index legal` it compares legal scores against 15. The domain indices use the same 0 to 100 scale, but AA scores fewer and mostly stronger models on them, so the floor removes fewer: on 2026-11-20 it left out 2 of the 22 models compared on legal, against 34 of the 69 compared on intelligence. The floor does not change which of the remaining models are on the frontier, because a model that beats another one scores at least as high and so clears the floor too.

## Is my model dominated?

`--model` shows one model, compared against every other model in the comparison, whichever side of the frontier it falls on. With `--provider`, the comparison is the models that provider keeps plus the one you named, even when it comes from another provider:

=== "CLI"

    ```bash
    eq models pareto --model gpt-6.1-sol
    ```

=== "Python"

    ```python
    import asyncio

    from evaluatorq.models import dominated_by, find_model, load_models

    models = asyncio.run(load_models())
    mine = find_model(models, model_id="gpt-6.1-sol")
    if mine is None:
        raise SystemExit("unknown id, or held by more than one host")
    winner = dominated_by(mine, models)
    print(winner.router_id if winner else "on the frontier")
    ```

```console
$ eq models pareto --model gpt-6.1-sol
               Dominated models: intelligence index (default effort) vs price
╭────────────────────┬───────┬──────────────────┬─────────────┬──────┬─────────────────────╮
│ Model              │ Score │ Price $/1M (3:1) │ Score per $ │ Est. │ Dominated by        │
├────────────────────┼───────┼──────────────────┼─────────────┼──────┼─────────────────────┤
│ openai/gpt-6.1-sol │  46.3 │            4.000 │        11.6 │      │ wafer/Kimi-K3.1-Pro │
╰────────────────────┴───────┴──────────────────┴─────────────┴──────┴─────────────────────╯
34 model(s) hidden for scoring below --min-score 15. Pass --min-score 0 to show every model.
4 model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.
```

The model named under `Dominated by` is the frontier model that beats yours at the lowest price. Here it wins by 0.2 points at the same price. Dominance has no margin: any difference counts, so a model can be dominated by a score gap smaller than the benchmark's own noise. Check the gap before you switch.

The flags combine. To ask the same question on the legal index and get JSON for a script:

```bash
eq models pareto --model gpt-6.1-sol --index legal --json
```

```console
$ eq models pareto --model gpt-6.1-sol --index legal --json 2>/dev/null
{
  "index": "legal",
  "effort": null,
  "x": "price",
  "min_score": 15.0,
  "points": [
    {
      "router_id": "openai/gpt-6.1-sol",
      "score": 49.0,
      "x": 4.0,
      "pareto_score": 12.25,
      "on_frontier": false,
      "dominated_by": "wafer/Kimi-K3.1-Pro",
      "dominators": ["wafer/Kimi-K3.1-Pro"]
    }
  ],
  "excluded": [
    {"router_id": "alibaba/qwen3.5-27b", "reason": "no legal score", "below_min_score": false},
    {"router_id": "cerebras/gpt-oss-120b", "reason": "score 8.1 is below min_score 15", "below_min_score": true}
  ],
  "left_out_by_provider": 0
}
```

The output above is trimmed: each point also carries every `ScoredModel` field (prices, developer, effort ladder, domain scores, speed and latency) and the blended `price`, and `excluded` lists every model left out, with `below_min_score` set on the ones the quality floor removed. The JSON lists only the models you asked for; `dominators` names the winners but does not carry their scores. Run without `--model` to see the winner's row. Log lines go to stderr, so `2>/dev/null` leaves clean JSON on stdout.

A model below the floor is not compared, so `--model` with it prints `<id> is not plotted: score <n> is below min_score 15.` on stderr; add `--min-score 0` to compare it. `dominated_by()` in Python applies no floor.

A bare id such as `gpt-6.1-sol` must match exactly one model; otherwise the CLI exits `2` and `find_model` returns `None`. `--model` only sees the one listing kept per model (see [One dot per model](#one-dot-per-model)), so the full id of a host that was not kept also exits `2`. Comparing a specific host is Python-only, through `load_models(all_hosts=True)`.

## Choosing the axes

The y axis is a score and the x axis is a cost. Both have a default and both can be changed.

| Axis | Flag / argument | Values | Better is |
|---|---|---|---|
| Score (y) | `--index` / `index=` | `intelligence` (default), `economics`, `engineering`, `finance_and_accounting`, `healthcare_and_medical`, `legal`, `strategy_and_ops` | higher |
| Reasoning effort | `--effort` / `effort=` | an effort at least one catalogue model has a score at, such as `high`; only with `intelligence` | — |
| Cost (x) | `--x` / `x=` | `price` (default), `output_speed`, `time_to_first_token` | lower price, higher speed, lower latency |

```bash
eq models pareto --index legal --provider openai --provider anthropic
eq models pareto --effort high --x output_speed --no-estimate
```

`price` is Orq's 3:1 blend of input and output list price per million tokens. `output_speed` is AA's median output tokens per second, and `time_to_first_token` is AA's median seconds to the first token.

A domain index covers only the models AA has scored on it (25 of 73 for `legal` on 2026-11-20), and only at each model's default effort, so `--index legal --effort high` is refused. Fewer scored models also means fewer competitors: on 2026-11-20 at the default floor, a model on `legal` was compared against 19 others, against 34 on `intelligence`. Under `--effort`, models that list no score at that effort drop out of the comparison.

The CLI refuses an index or an effort that no loaded model has a score on and exits `2`, and the dashboard lists only the indices and efforts some model has. `pareto()` in Python accepts them and lists every model under `excluded`.

## One dot per model

The catalogue lists the same model several times, once per host and region. `load_models()` keeps one listing per model: the developer's own listing when there is one, otherwise the cheapest host, preferring non-regional ones. `load_models(all_hosts=True)` keeps every listing, which is how you compare hosts of one model against each other.

Models switched off in your workspace are left out. Pass `--include-disabled` or `include_disabled=True` to compare them as well.

## What gets left out

A model is left out of the comparison, and listed under `excluded` in `--json` and in `ParetoResult.excluded`, when it has no value on one of the two axes or scores below the quality floor. The reason strings are:

| Reason | Why |
|---|---|
| `no legal score` (any index name) | AA has not published that index for the model |
| `no intelligence score at effort high` | the model has no score at the `--effort` you asked for |
| `free (no list price)` | a free model would beat everything on price |
| `no output speed data`, `no time to first token data` | AA has no speed or latency measurement for the model |
| `estimated score` | only with `--no-estimate` / `exclude_estimated=True`; estimated scores are compared by default |
| `score 2.7 is below min_score 15` | the score is under `--min-score` / `min_score=` (see [The quality floor](#the-quality-floor)) |

A left-out model is not dominated. It simply was not compared.

## The limits of list price

List price is not what a task costs you. A reasoning model that writes four times as many tokens per answer costs four times its list price per task, and the blend assumes three input tokens for every output token. The frontier on this page compares price tags. To compare what a run costs, run both models on your own task and read the spend from the report.

## In the dashboard

`eq dashboard` has a **Models** page that draws this frontier as a chart. See [Dashboard](dashboard.md#models).

## In code

Every name used above imports from `evaluatorq.models`. `pareto()` returns a `ParetoResult` whose `points` carry the score, the x value, `pareto_score` (score per dollar, `None` off the price axis), `on_frontier`, the nearest frontier model in `dominated_by`, and every model that beats it in `dominators`. `dominates(a, b)` answers the question for a single pair. The full list is in the [API reference](reference/evaluatorq/models.md).

For the CLI flags in one table, see [`eq models`](cli-reference/models.md).
