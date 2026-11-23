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
╭───────────────────────────────────┬───────┬───────────────────────────┬─────────────┬─────────────────┬──────┬──────────╮
│ Model                             │ Score │ Price $/1M (7:2:1 cached) │ Score per $ │ Cache read $/1M │ Est. │ Frontier │
├───────────────────────────────────┼───────┼───────────────────────────┼─────────────┼─────────────────┼──────┼──────────┤
│ anthropic/claude-opus-5-5         │  48.2 │                     2.940 │        16.4 │           0.200 │      │ yes      │
│ wafer/Kimi-K3.1-Pro               │  46.5 │                     1.540 │        30.2 │           0.200 │      │ yes      │
│ openai/gpt-6.1-sol                │  46.3 │                     1.470 │        31.5 │           0.100 │      │ yes      │
│ particle/deepseek-v4.1-flash-1017 │  32.6 │                     0.171 │       190.6 │           0.030 │      │ yes      │
│ deepseek/deepseek-flash           │  28.6 │                     0.058 │       493.4 │           0.003 │      │ yes      │
│ zai/glm-5.3-flash                 │  27.8 │                     0.051 │       550.5 │           0.015 │      │ yes      │
╰───────────────────────────────────┴───────┴───────────────────────────┴─────────────┴─────────────────┴──────┴──────────╯
Price blends the rates per 1M tokens as 7:0:2:1 (cache read : cache write : fresh input : output). Pass --mix list for the 3:1 input/output list blend.
34 model(s) hidden for scoring below --min-score 15. Pass --min-score 0 to show every model.
4 model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.
```

Read the table from the top down: each row is cheaper than the one above it and scores lower. Price is a blend that assumes most input tokens are cache reads, which is what an agent loop looks like; [The price axis](#the-price-axis) explains it and how to switch to the plain 3:1 list blend. `Cache read $/1M` is the cache-read rate the blend used. `Score per $` is the score divided by the price. `Est.` marks a score that Orq estimated from an older benchmark version instead of measuring.

## The quality floor

Models scoring below 15 are left out before the frontier is drawn. Without the floor, the cheap end of the frontier holds models that are there only because nothing is cheaper: with the floor off and the 3:1 list blend, the frontier adds `tensorix/qwen/qwen3-coder-30b-a3b-instruct` at a score of 2.7 and $0.107 per 1M tokens. 15 is the same floor evaluatorq uses when it picks default models for its own roles, `MIN_SEAT_INDEX`.

Set the floor with `--min-score` or `min_score=`. `--min-score 0` or `min_score=None` shows every model:

=== "CLI"

    ```bash
    eq models pareto --min-score 0 --mix list
    ```

=== "Python"

    ```python
    import asyncio

    from evaluatorq.models import load_models, pareto

    result = pareto(asyncio.run(load_models()), min_score=None, mix="list")
    print(len(result.frontier), "on the frontier")
    ```

```console
$ eq models pareto --min-score 0 --mix list
                        Pareto frontier: intelligence index (default effort) vs price
╭────────────────────────────────────────────┬───────┬───────────────────────┬─────────────┬──────┬──────────╮
│ Model                                      │ Score │ Price $/1M (3:1 list) │ Score per $ │ Est. │ Frontier │
├────────────────────────────────────────────┼───────┼───────────────────────┼─────────────┼──────┼──────────┤
│ anthropic/claude-opus-5-5                  │  48.2 │                 8.000 │         6.0 │      │ yes      │
│ wafer/Kimi-K3.1-Pro                        │  46.5 │                 4.000 │        11.6 │      │ yes      │
│ particle/deepseek-v4.1-flash-1017          │  32.6 │                 0.438 │        74.5 │      │ yes      │
│ deepseek/deepseek-flash                    │  28.6 │                 0.175 │       163.4 │      │ yes      │
│ zai/glm-5.3-flash                          │  27.8 │                 0.119 │       234.1 │      │ yes      │
│ tensorix/qwen/qwen3-coder-30b-a3b-instruct │   2.7 │                 0.107 │        25.1 │ yes  │ yes      │
╰────────────────────────────────────────────┴───────┴───────────────────────┴─────────────┴──────┴──────────╯
4 model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.
```

The floor applies to whichever score you put on the y axis, so with `--index legal` it compares legal scores against 15. The domain indices use the same 0 to 100 scale, but AA scores fewer and mostly stronger models on them, so the floor removes fewer: on 2026-11-20 it left out 2 of the 22 models compared on legal, against 34 of the 69 compared on intelligence. The floor does not change which of the remaining models are on the frontier, because a model that beats another one scores at least as high and so clears the floor too.

## Is my model dominated?

`--model` shows one model, compared against every other model in the comparison, whichever side of the frontier it falls on. With `--provider`, the comparison is the models that provider keeps plus the one you named, even when it comes from another provider:

=== "CLI"

    ```bash
    eq models pareto --model gpt-6.1-sol --mix list
    ```

=== "Python"

    ```python
    import asyncio

    from evaluatorq.models import dominated_by, find_model, load_models

    models = asyncio.run(load_models())
    mine = find_model(models, model_id="gpt-6.1-sol")
    if mine is None:
        raise SystemExit("unknown id, or held by more than one host")
    winner = dominated_by(mine, models, mix="list")
    print(winner.router_id if winner else "on the frontier")
    ```

```console
$ eq models pareto --model gpt-6.1-sol --mix list
                 Dominated models: intelligence index (default effort) vs price
╭────────────────────┬───────┬───────────────────────┬─────────────┬──────┬─────────────────────╮
│ Model              │ Score │ Price $/1M (3:1 list) │ Score per $ │ Est. │ Dominated by        │
├────────────────────┼───────┼───────────────────────┼─────────────┼──────┼─────────────────────┤
│ openai/gpt-6.1-sol │  46.3 │                 4.000 │        11.6 │      │ wafer/Kimi-K3.1-Pro │
╰────────────────────┴───────┴───────────────────────┴─────────────┴──────┴─────────────────────╯
34 model(s) hidden for scoring below --min-score 15. Pass --min-score 0 to show every model.
4 model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.
```

The model named under `Dominated by` is the frontier model that beats yours at the lowest price. Here it wins by 0.2 points at the same list price. On the default cache-aware blend the answer flips: GPT-6.1 Sol's cache reads cost 5% of its input rate against 10% for Kimi K3.1 Pro on `wafer`, so it is $1.470 against $1.540 and back on the frontier. Which blend you read decides the verdict, so pick the one that matches how you call the model. Dominance has no margin: any difference counts, so a model can be dominated by a score gap smaller than the benchmark's own noise. Check the gap before you switch.

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
  "x_label": "Price, USD per 1M tokens (7:2:1 cached)",
  "mix": {"read": 7.0, "write": 0.0, "fresh": 2.0, "output": 1.0, "name": "agentic", "tag": "7:2:1 cached"},
  "min_score": 15.0,
  "points": [
    {
      "router_id": "openai/gpt-6.1-sol",
      "score": 49.0,
      "x": 1.47,
      "pareto_score": 33.333333333333336,
      "on_frontier": true,
      "dominated_by": null,
      "dominators": [],
      "cache_read_source": "host",
      "cache_write_source": null
    }
  ],
  "excluded": [
    {"router_id": "alibaba/qwen3.5-27b", "reason": "no legal score", "below_min_score": false},
    {"router_id": "cerebras/gpt-oss-120b", "reason": "score 8.1 is below min_score 15", "below_min_score": true}
  ],
  "left_out_by_provider": 0
}
```

The output above is trimmed: each point also carries every `ScoredModel` field (prices including `cache_read_price` and `aa_cache_read_price`, developer, effort ladder, domain scores, speed and latency) and `price`, which is always the 3:1 list blend whatever `mix` says, and `excluded` lists every model left out, with `below_min_score` set on the ones the quality floor removed. The JSON lists only the models you asked for; `dominators` names the winners but does not carry their scores. Run without `--model` to see the winner's row. Log lines go to stderr, so `2>/dev/null` leaves clean JSON on stdout.

A model below the floor is not compared, so `--model` with it prints `<id> is not plotted: score <n> is below min_score 15.` on stderr; add `--min-score 0` to compare it. `dominated_by()` in Python applies no floor.

A bare id such as `gpt-6.1-sol` must match exactly one model; otherwise the CLI exits `2` and `find_model` returns `None`. `--model` only sees the one listing kept per model (see [One dot per model](#one-dot-per-model)), so the full id of a host that was not kept also exits `2`. Comparing a specific host is Python-only, through `load_models(all_hosts=True)`.

## Choosing the axes

The y axis is a score and the x axis is a cost. Both have a default and both can be changed.

| Axis | Flag / argument | Values | Better is |
|---|---|---|---|
| Score (y) | `--index` / `index=` | `intelligence` (default), `economics`, `engineering`, `finance_and_accounting`, `healthcare_and_medical`, `legal`, `strategy_and_ops` | higher |
| Reasoning effort | `--effort` / `effort=` | an effort at least one catalogue model has a score at, such as `high`; only with `intelligence` | — |
| Cost (x) | `--x` / `x=` | `price` (default), `output_speed`, `time_to_first_token` | lower price, higher speed, lower latency |
| Price blend | `--mix` / `mix=` | `agentic` (default), `list`, or a `READ:WRITE:FRESH:OUTPUT` ratio; only with `price` | — |

```bash
eq models pareto --index legal --provider openai --provider anthropic
eq models pareto --effort high --x output_speed --no-estimate
```

`price` blends the card's rates per million tokens under `--mix` (see [The price axis](#the-price-axis)). `output_speed` is AA's median output tokens per second, and `time_to_first_token` is AA's median seconds to the first token.

A domain index covers only the models AA has scored on it (25 of 73 for `legal` on 2026-11-20), and only at each model's default effort, so `--index legal --effort high` is refused. Fewer scored models also means fewer competitors: on 2026-11-20 at the default floor, a model on `legal` was compared against 19 others, against 34 on `intelligence`. Under `--effort`, models that list no score at that effort drop out of the comparison.

The CLI refuses an index or an effort that no loaded model has a score on and exits `2`, and the dashboard lists only the indices and efforts some model has. `pareto()` in Python accepts them and lists every model under `excluded`.

## The price axis

The price axis is a weighted average of a card's rates per million tokens. `--mix` (`mix=` in Python) sets the weights:

| Mix | Formula, USD per 1M | Label | Use it when |
|---|---|---|---|
| `agentic` (default) | (7 × cache read + 2 × input + 1 × output) / 10 | `Price $/1M (7:2:1 cached)` | The model runs in a loop that re-sends its history: agents, tool use, multi-turn simulation |
| `list` | (3 × input + 1 × output) / 4 | `Price $/1M (3:1 list)` | Single-shot calls with short prompts, or to match what the Orq smart router and evaluatorq's own preset choice rank on |
| `READ:WRITE:FRESH:OUTPUT`, such as `7:1:1:1` | (R × cache read + W × cache write + F × input + O × output) / (R + W + F + O) | `Price $/1M (7:1:1:1)` | You know your own token mix, or your provider charges for cache writes |

`agentic` is the blend [Artificial Analysis](https://artificialanalysis.ai/methodology) publishes: "we calculate a blended price assuming a 7:2:1 ratio of cache hit, input, and output tokens". It is the default because an agent loop sends its whole history again on every turn, and the provider serves most of that history from its prompt cache at a fraction of the input rate. Measured agent workloads cache more than 70% of input: 72% to 82% of all input on one gateway across twelve weeks ([Requesty](https://www.requesty.ai/data/weekly/2026-w46/prompt-cache)), 92% for Claude Code on the same gateway ([Requesty, April 2026](https://www.requesty.ai/data/coding-agent-cache-hit-rate-apr-2026)), 92% in [Bun's Rust rewrite](https://bun.com/blog/bun-in-rust), and 96% in a study of Claude Code repairing multi-hunk bugs ([arXiv 2511.11012](https://arxiv.org/html/2511.11012)). The 3:1 list blend prices every one of those re-sent tokens at the full input rate, so it overstates what an agent pays and hides which models have cheap cache reads.

=== "CLI"

    ```bash
    eq models pareto --mix list
    eq models pareto --mix 7:1:1:1
    ```

=== "Python"

    ```python
    import asyncio

    from evaluatorq.models import load_models, pareto

    models = asyncio.run(load_models())
    for mix in ("agentic", "list", "7:1:1:1"):
        result = pareto(models, mix=mix)
        print(result.x_spec.short_label, [p.model.router_id for p in result.frontier])
    ```

A mix with a write share adds a `Cache write $/1M` column. On 2026-11-20 it printed:

```console
$ eq models pareto --mix 7:1:1:1
                                  Pareto frontier: intelligence index (default effort) vs price
╭────────────────────────────┬───────┬──────────────────────┬─────────────┬─────────────────┬──────────────────┬──────┬──────────╮
│ Model                      │ Score │ Price $/1M (7:1:1:1) │ Score per $ │ Cache read $/1M │ Cache write $/1M │ Est. │ Frontier │
├────────────────────────────┼───────┼──────────────────────┼─────────────┼─────────────────┼──────────────────┼──────┼──────────┤
│ anthropic/claude-opus-5-5  │  48.2 │                3.040 │        15.9 │           0.200 │            5.000 │      │ yes      │
│ wafer/Kimi-K3.1-Pro        │  46.5 │                1.540 │        30.2 │           0.200 │    2.000 (input) │      │ yes      │
│ openai/gpt-6.1-sol         │  46.3 │                1.520 │        30.5 │           0.100 │            2.500 │      │ yes      │
│ particle/deepseek-v4.1-fl… │  32.6 │                0.171 │       190.6 │           0.030 │    0.250 (input) │      │ yes      │
│ deepseek/deepseek-flash    │  28.6 │                0.058 │       493.4 │           0.003 │    0.140 (input) │      │ yes      │
│ zai/glm-5.3-flash          │  27.8 │                0.051 │       550.5 │           0.015 │    0.075 (input) │      │ yes      │
╰────────────────────────────┴───────┴──────────────────────┴─────────────┴─────────────────┴──────────────────┴──────┴──────────╯
Price blends the rates per 1M tokens as 7:1:1:1 (cache read : cache write : fresh input : output). Pass --mix list for the 3:1 input/output list blend.
4 shown model(s) price a cache rate their host does not list: (AA) is the Artificial Analysis price on the same card, (input) bills those tokens at the input rate.
34 model(s) hidden for scoring below --min-score 15. Pass --min-score 0 to show every model.
4 model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.
```

The ratio `7:0:2:1` is the `agentic` blend and `0:0:3:1` is the `list` blend, so either spelling gets the named label. Every price blend is relabelled, in the table header, the dashboard axis and `x_label` in `--json`, so a number never appears without the blend it was computed under.

### When the card has no cache price

A host that lists no cache-read rate is priced from the next source on the same card, and the row is marked:

| Source | Marker in the table | `cache_read_source` | Rate used |
|---|---|---|---|
| The host's own cache-read rate | none | `host` | The host's rate |
| The Artificial Analysis cache price on the same card | `(AA)` | `artificial_analysis` | The developer's own or the median price across hosts, capped at this host's input rate |
| Neither | `(input)` | `input` | The input rate, as if the host gave no discount; this is also what a run's reported cost bills |

A cache-write share uses the host's cache-write rate, else the input rate, marked the same way in `cache_write_source`. The table footer counts the marked rows, and `ParetoResult.price_fallbacks` lists them in Python:

```console
$ eq models pareto --provider azure --min-score 0
                      Pareto frontier: intelligence index (default effort) vs price
╭──────────────────┬───────┬───────────────────────────┬─────────────┬─────────────────┬──────┬──────────╮
│ Model            │ Score │ Price $/1M (7:2:1 cached) │ Score per $ │ Cache read $/1M │ Est. │ Frontier │
├──────────────────┼───────┼───────────────────────────┼─────────────┼─────────────────┼──────┼──────────┤
│ azure/gpt-5-mini │   9.3 │                     0.267 │        34.8 │      0.025 (AA) │ yes  │ yes      │
╰──────────────────┴───────┴───────────────────────────┴─────────────┴─────────────────┴──────┴──────────╯
Price blends the rates per 1M tokens as 7:0:2:1 (cache read : cache write : fresh input : output). Pass --mix list for the 3:1 input/output list blend.
1 shown model(s) price a cache rate their host does not list: (AA) is the Artificial Analysis price on the same card, (input) bills those tokens at the input rate.
70 model(s) from other providers left out by --provider.
```

On 2026-11-20, 106 of the 119 priced models with a score list their own cache-read rate, 7 fall back to the Artificial Analysis price and 6 to the input rate. With one listing per model and the default floor, none of the plotted models falls back. A missing rate is often a catalogue gap rather than a real price: Azure OpenAI does discount cached input, but its cards list no cache rate.

### What the blend cannot see

Each of these makes the `agentic` price wrong for some workloads, and none of them shows in the output:

- **Short prompts never cache.** Providers only cache a prefix above a minimum length: 512 to 4,096 tokens on Anthropic depending on the model, 1,024 on OpenAI, 2,048 to 4,096 on Gemini. A shorter prompt pays full input every time, silently, and `list` is the better guide.
- **Caches expire.** Anthropic keeps a cache entry for 5 minutes by default, OpenAI for 30 minutes on its newest models. In a public trace of GitHub Copilot's coding agent, only 19% of input was cached on calls made after an idle gap of 5 minutes or more ([analysis](https://siddhantkhare.com/writing/coding-agent-cache-five-minutes)). An agent that waits on a human between turns pays more than the blend shows.
- **Anthropic caches only what you mark.** Without `cache_control` breakpoints (or Anthropic's automatic top-level one), a Claude call pays full input on every turn. OpenAI, Gemini 2.5 and newer, and DeepSeek cache automatically.
- **Cache writes cost extra on some providers.** Anthropic bills a 5-minute cache write at 1.25 times the input rate and a 1-hour write at 2 times; the default blend has no write share. Use a ratio with one, such as `7:1:1:1`, for a write-heavy loop.
- **Gemini's cache-write rate is a storage price.** On about 13 Gemini cards the catalogue's cache-write field holds Google's hourly storage price per million tokens, not a per-token write charge, so any mix with a write share overprices those models. The default mix has no write share and is not affected.
- **Hosts of the same model price caching differently.** Kimi K3.1 Pro costs $0.20 per million cache reads on `wafer`, and `berget` lists no cache rate at all, so the frontier can pick a host and not only a model.
- **Off-peak and long-context rates are ignored.** DeepSeek halves its rates off-peak, and some providers charge more above 200k tokens; the blend reads only the standard rate.

## One dot per model

The catalogue lists the same model several times, once per host and region. `load_models()` keeps one listing per model: the developer's own listing when there is one, otherwise the cheapest host, preferring non-regional ones. `load_models(all_hosts=True)` keeps every listing, which is how you compare hosts of one model against each other.

Contributor listings, whose id ends in `-contributor` (Meta's Muse Spark 1.2 and 1.3 Contributor), are never compared. They cost a fraction of the standard listing of the same model because the vendor may train on your prompts and responses, so their price would put them on the frontier for a reason that has nothing to do with the model. `load_models()` keeps them beside the standard listing, `pareto()` lists them in `excluded` with `contributor: true` (`ParetoResult.contributors`), and the CLI counts them in a footer line. Preset and jury seating does not check for them. The catalogue marks them only by the id suffix and the display name; no field says that the vendor trains on the traffic.

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

## The limits of a blended price

A blended price is not what a task costs you. A reasoning model that writes four times as many tokens per answer costs four times as much per task, and every blend assumes one fixed share of cache reads, fresh input and output for every model. The frontier on this page compares price tags. To compare what a run costs, run both models on your own task and read the spend from the report.

## In the dashboard

`eq dashboard` has a **Pareto Frontier** page that draws this frontier as a chart. See [Dashboard](dashboard.md#models).

## In code

Every name used above imports from `evaluatorq.models`. `pareto()` returns a `ParetoResult` with the blend in `mix` and the axis labels in `x_spec`, whose `points` carry the score, the x value, `pareto_score` (score per dollar, `None` off the price axis), `cache_read_source` and `cache_write_source`, `on_frontier`, the nearest frontier model in `dominated_by`, and every model that beats it in `dominators`. `dominates(a, b)` answers the question for a single pair. `dominates`, `pareto_frontier` and `x_of` default to the `list` blend, because that is what evaluatorq's preset choice ranks on; pass `mix="agentic"` to match `pareto()`. The full list is in the [API reference](reference/evaluatorq/models.md).

For the CLI flags in one table, see [`eq models`](cli-reference/models.md).
