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

The CLI run on 2026-11-23 printed:

```console
$ eq models pareto
                           Pareto frontier: intelligence index (default effort) vs price
╭───────────────────────────┬───────┬───────────────────────────┬─────────────┬─────────────────┬──────┬──────────╮
│ Model                     │ Score │ Price $/1M (7:2:1 cached) │ Score per $ │ Cache read $/1M │ Est. │ Frontier │
├───────────────────────────┼───────┼───────────────────────────┼─────────────┼─────────────────┼──────┼──────────┤
│ anthropic/claude-opus-5-5 │  51.2 │                     2.940 │        17.4 │           0.200 │      │ yes      │
│ openai/gpt-6.1-sol        │  47.8 │                     1.470 │        32.5 │           0.100 │      │ yes      │
│ zai/glm-5.3-flash         │  41.8 │                     0.051 │       827.7 │           0.015 │      │ yes      │
╰───────────────────────────┴───────┴───────────────────────────┴─────────────┴─────────────────┴──────┴──────────╯
Price blends the rates per 1M tokens as 7:0:2:1 (cache read : cache write : fresh input : output). Pass --mix list for the 3:1 input/output list blend.
7 model(s) hidden for scoring below --min-score 10. Pass --min-score 0 to show every model.
5 model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.
```

Read the table from the top down: each row is cheaper than the one above it and scores lower. Price is the `agentic` blend, which assumes most input tokens are cache reads, as in an agent loop. As a `--mix` ratio it is `7:0:2:1`, because the ratio has a cache-write slot second; [The price axis](#the-price-axis) explains the blend and how to switch to the 3:1 list blend. `Cache read $/1M` is the cache-read rate the blend used. Pick from these rows: a dominated model is never the better choice on these two axes, and `--dominated` lists each one with the frontier model that beats it. `Est.` marks a score that Orq estimated from an older benchmark version instead of measuring.

## The quality floor

Models scoring below 10 are left out before the frontier is drawn. Without the floor, the cheap end of the frontier can hold models that are there only because nothing is cheaper: on 2026-11-23, with the floor off and the 3:1 list blend, the frontier adds `tensorix/qwen/qwen3-coder-30b-a3b-instruct` at a score of 9.6 and $0.107 per 1M tokens. On the default blend the same day, the frontier was the same three models with or without the floor, so there the floor only removed dominated models from `--dominated`, `--json` and the dashboard chart. 10 is the same floor evaluatorq uses when it picks default models for its own roles, `MIN_SEAT_INDEX`.

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
│ anthropic/claude-opus-5-5                  │  51.2 │                 8.000 │         6.4 │      │ yes      │
│ openai/gpt-6.1-sol                         │  47.8 │                 4.000 │        11.9 │      │ yes      │
│ zai/glm-5.3-flash                          │  41.8 │                 0.119 │       352.0 │      │ yes      │
│ tensorix/qwen/qwen3-coder-30b-a3b-instruct │   9.6 │                 0.107 │        89.3 │      │ yes      │
╰────────────────────────────────────────────┴───────┴───────────────────────┴─────────────┴──────┴──────────╯
5 model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.
```

The floor applies to whichever score you put on the y axis, so with `--index legal` it compares legal scores against 10. The domain indices use the same 0 to 100 scale, but AA scores fewer and mostly stronger models on them, so the floor removes fewer: on 2026-11-23, after one listing per model, it left out 1 of the 38 models with a legal score and a price, against 7 of the 67 with an intelligence score and a price. The floor does not change which of the remaining models are on the frontier, because a model that beats another one scores at least as high and so clears the floor too.

## Is my model dominated?

`--model` shows one model, compared against every other model in the comparison, whichever side of the frontier it falls on. With `--provider`, the comparison is the models that provider keeps plus the one you named, even when it comes from another provider:

=== "CLI"

    ```bash
    eq models pareto --model claude-sonnet-5-5
    ```

=== "Python"

    ```python
    import asyncio

    from evaluatorq.models import dominated_by, find_model, load_models

    models = asyncio.run(load_models())
    mine = find_model(models, model_id="claude-sonnet-5-5")
    if mine is None:
        raise SystemExit("unknown id, or held by more than one host")
    winner = dominated_by(mine, models)
    print(winner.router_id if winner else "on the frontier")
    ```

```console
$ eq models pareto --model claude-sonnet-5-5
                                Dominated models: intelligence index (default effort) vs price
╭─────────────────────────────┬───────┬───────────────────────────┬─────────────┬─────────────────┬──────┬────────────────────╮
│ Model                       │ Score │ Price $/1M (7:2:1 cached) │ Score per $ │ Cache read $/1M │ Est. │ Dominated by       │
├─────────────────────────────┼───────┼───────────────────────────┼─────────────┼─────────────────┼──────┼────────────────────┤
│ anthropic/claude-sonnet-5-5 │  46.8 │                     1.470 │        31.8 │           0.100 │      │ openai/gpt-6.1-sol │
╰─────────────────────────────┴───────┴───────────────────────────┴─────────────┴─────────────────┴──────┴────────────────────╯
Price blends the rates per 1M tokens as 7:0:2:1 (cache read : cache write : fresh input : output). Pass --mix list for the 3:1 input/output list blend.
7 model(s) hidden for scoring below --min-score 10. Pass --min-score 0 to show every model.
5 model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.
```

The model named under `Dominated by` is the frontier model that beats yours at the lowest price. Here GPT-6.1 Sol scores 1.0 point higher at the same price, $1.470 on the default blend and $4.000 on the list blend. The verdict can change with the blend, because two models with the same list price can have different cache-read rates, so run `--model` under the mix that matches how you call the model. Dominance has no margin: any difference counts, so a model can be dominated by a score gap smaller than the benchmark's own noise. A 1.0-point gap, as here, may be that case. Check the gap before you switch.

The flags combine. To ask the same question on the legal index and get JSON for a script:

```bash
eq models pareto --model claude-sonnet-5-5 --index legal --json
```

```console
$ eq models pareto --model claude-sonnet-5-5 --index legal --json 2>/dev/null
{
  "index": "legal",
  "effort": null,
  "x": "price",
  "x_label": "Price, USD per 1M tokens (7:2:1 cached)",
  "mix": {"read": 7.0, "write": 0.0, "fresh": 2.0, "output": 1.0, "name": "agentic", "tag": "7:2:1 cached"},
  "min_score": 10.0,
  "points": [
    {
      "router_id": "anthropic/claude-sonnet-5-5",
      "score": 48.7,
      "x": 1.47,
      "pareto_score": 33.12925170068027,
      "on_frontier": false,
      "dominated_by": "google-ai/gemini-3.8-flash",
      "dominators": ["google-ai/gemini-3.8-flash", "openai/gpt-6.1-sol"],
      "cache_read_source": "host",
      "cache_write_source": null
    }
  ],
  "excluded": [
    {"router_id": "alibaba/qwen3.5-27b", "reason": "no legal score", "below_min_score": false},
    {"router_id": "groq/openai/gpt-oss-20b", "reason": "score 7.3 is below min_score 10", "below_min_score": true}
  ],
  "left_out_by_provider": 0
}
```

Read `x` for the blended price. Each point also carries a `price` field, and `price` is always the 3:1 list blend whatever `mix` says. The output above is trimmed: each point also carries every `ScoredModel` field (prices including `cache_read_price` and `aa_cache_read_price`, developer, effort ladder, domain scores, speed and latency), and `excluded` lists every model left out, with `below_min_score` set on the ones the quality floor removed. The JSON lists only the models you asked for; `dominators` names the winners but does not carry their scores. Run without `--model` to see the winner's row. Log lines go to stderr, so `2>/dev/null` leaves clean JSON on stdout.

A model below the floor is not compared, so `--model` with it prints `<id> is not plotted: score <n> is below min_score 10.` on stderr and `No models on the frontier for these filters.` on stdout; add `--min-score 0` to compare it. `dominated_by()` in Python applies no floor.

`--model` takes the full router id or a shorter one. For `tensorix/qwen/qwen3-coder-30b-a3b-instruct`, `qwen/qwen3-coder-30b-a3b-instruct` (the part after the provider) and `qwen3-coder-30b-a3b-instruct` (the last part) both work. The full id is tried first, then the part after the provider, then the last part. A shorter id must match exactly one model; otherwise the CLI exits `2` and `find_model` returns `None`. `--model` only sees the one listing kept per model (see [One dot per model](#one-dot-per-model)), so the full id of a host that was not kept also exits `2`. Comparing a specific host is Python-only, through `load_models(all_hosts=True)`.

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

A domain index covers only the models AA has scored on it (38 of the 72 models kept for `legal` on 2026-11-23, against 67 for `intelligence`), and only at one effort per model, the one Artificial Analysis benchmarked, which for some reasoning models is higher than the effort a call uses by default. `--index legal --effort high` is therefore refused. Under `--effort`, models that list no score at that effort drop out of the comparison.

The CLI refuses an index or an effort that no loaded model has a score on and exits `2`, and the dashboard lists only the indices and efforts some model has. `pareto()` in Python accepts them and lists every model under `excluded`.

## The price axis

The price axis is a weighted average of a card's rates per million tokens. `--mix` (`mix=` in Python) sets the weights:

| Mix | Formula, USD per 1M | Label | Use it when |
|---|---|---|---|
| `agentic` (default) | (7 × cache read + 2 × input + 1 × output) / 10 | `Price $/1M (7:2:1 cached)` | The model runs in a loop that re-sends its history: agents, tool use, multi-turn simulation |
| `list` | (3 × input + 1 × output) / 4 | `Price $/1M (3:1 list)` | Single-shot calls with short prompts, or to match what the Orq smart router and evaluatorq's own preset choice rank on |
| `READ:WRITE:FRESH:OUTPUT`, such as `7:1:1:1` | (R × cache read + W × cache write + F × input + O × output) / (R + W + F + O) | `Price $/1M (7:1:1:1)` | You know your own token mix, or your provider charges for cache writes |

`agentic` is the blend [Artificial Analysis](https://artificialanalysis.ai/methodology) publishes: "we calculate a blended price assuming a 7:2:1 ratio of cache hit, input, and output tokens". It is the default because an agent loop sends its whole history again on every turn, and the provider serves most of that history from its prompt cache at a fraction of the input rate. Measured agent workloads cache more than 70% of input: 72% to 82% of all input on one gateway across twelve weeks ([Requesty](https://www.requesty.ai/data/weekly/2026-w46/prompt-cache)), 92% for Claude Code on the same gateway ([Requesty, April 2026](https://www.requesty.ai/data/coding-agent-cache-hit-rate-apr-2026)), 92% in [Bun's Rust rewrite](https://bun.com/blog/bun-in-rust), and 96% in a study of Claude Code repairing multi-hunk bugs ([arXiv 2511.11012](https://arxiv.org/html/2511.11012)). The 3:1 list blend prices every one of those re-sent tokens at the full input rate, so it overstates what an agent pays and hides which models have cheap cache reads.

In those terms, 7:2:1 means 78% of input tokens are cache reads and output is 10% of all tokens. For your own cache-hit rate `h`, the matching ratio is `9h:0:9(1-h):1`; at 90% that is `--mix 8.1:0:0.9:1`.

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

A mix with a write share adds a `Cache write $/1M` column. On 2026-11-23 it printed:

```console
$ eq models pareto --mix 7:1:1:1
                                  Pareto frontier: intelligence index (default effort) vs price
╭───────────────────────────┬───────┬──────────────────────┬─────────────┬─────────────────┬──────────────────┬──────┬──────────╮
│ Model                     │ Score │ Price $/1M (7:1:1:1) │ Score per $ │ Cache read $/1M │ Cache write $/1M │ Est. │ Frontier │
├───────────────────────────┼───────┼──────────────────────┼─────────────┼─────────────────┼──────────────────┼──────┼──────────┤
│ anthropic/claude-opus-5-5 │  51.2 │                3.040 │        16.8 │           0.200 │            5.000 │      │ yes      │
│ openai/gpt-6.1-sol        │  47.8 │                1.520 │        31.4 │           0.100 │            2.500 │      │ yes      │
│ zai/glm-5.3-flash         │  41.8 │                0.051 │       827.7 │           0.015 │    0.075 (input) │      │ yes      │
╰───────────────────────────┴───────┴──────────────────────┴─────────────┴─────────────────┴──────────────────┴──────┴──────────╯
Price blends the rates per 1M tokens as 7:1:1:1 (cache read : cache write : fresh input : output). Pass --mix list for the 3:1 input/output list blend.
1 shown model(s) price a cache rate their host does not list: (AA) is the Artificial Analysis price on the same card, (input) bills those tokens at the input rate.
7 model(s) hidden for scoring below --min-score 10. Pass --min-score 0 to show every model.
5 model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.
```

The ratio `7:0:2:1` is the `agentic` blend and `0:0:3:1` is the `list` blend, so either spelling gets the named label. Every price blend is relabelled, in the table header, the dashboard axis and `x_label` in `--json`, so a number never appears without the blend it was computed under.

### When the card has no cache price

A host that lists no cache-read rate is priced from the next source on the same card, and the row is marked:

| Source | Marker in the table | `cache_read_source` | Rate used |
|---|---|---|---|
| The host's own cache-read rate | none | `host` | The host's rate |
| The Artificial Analysis cache price on the same card | `(AA)` | `artificial_analysis` | The developer's own or the median price across hosts, capped at this host's input rate |
| Neither | `(input)` | `input` | The input rate, as if the host gave no discount; this is also what a run's reported cost bills |

A cache-write share uses the host's cache-write rate, else the input rate, marked the same way in `cache_write_source`. The marker sits on the cell it applies to: in the `7:1:1:1` output above, `0.075 (input)` is in the `Cache write $/1M` column, so `zai/glm-5.3-flash`'s cache-read rate is its host's own. The table footer counts the marked rows, and `ParetoResult.price_fallbacks` lists them in Python:

```console
$ eq models pareto --provider azure --min-score 0
                     Pareto frontier: intelligence index (default effort) vs price
╭───────────────┬───────┬───────────────────────────┬─────────────┬─────────────────┬──────┬──────────╮
│ Model         │ Score │ Price $/1M (7:2:1 cached) │ Score per $ │ Cache read $/1M │ Est. │ Frontier │
├───────────────┼───────┼───────────────────────────┼─────────────┼─────────────────┼──────┼──────────┤
│ azure/gpt-4.1 │  12.7 │                     1.550 │         8.2 │      0.500 (AA) │      │ yes      │
╰───────────────┴───────┴───────────────────────────┴─────────────┴─────────────────┴──────┴──────────╯
Price blends the rates per 1M tokens as 7:0:2:1 (cache read : cache write : fresh input : output). Pass --mix list for the 3:1 input/output list blend.
1 shown model(s) price a cache rate their host does not list: (AA) is the Artificial Analysis price on the same card, (input) bills those tokens at the input rate.
70 model(s) from other providers left out by --provider.
```

On 2026-11-23, across every host, 108 of the 119 listings with a price and an intelligence score list their own cache-read rate, 6 fall back to the Artificial Analysis price and 5 to the input rate. With one listing per model and the default floor, 5 of the 60 compared models fall back on the cache-read rate: four Qwen models to the input rate and `azure/gpt-4.1` to the Artificial Analysis price; the CLI footer counts only the rows it prints, while the dashboard summary counts every plotted model. A missing rate is often a catalogue gap rather than a real price: Azure OpenAI does discount cached input, but its cards list no cache rate.

### What the blend cannot see

Each of these makes the `agentic` price wrong for some workloads, and none of them shows in the output:

- **Short prompts never cache.** Providers only cache a prefix above a minimum length: 512 to 4,096 tokens on Anthropic depending on the model ([Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)), 1,024 on OpenAI's GPT-5.6 and later ([OpenAI](https://developers.openai.com/api/docs/guides/prompt-caching)), 2,048 to 4,096 on Gemini ([Google](https://aistudio.google.com/docs/caching)). A shorter prompt pays full input every time, silently, and `list` is the better guide.
- **Caches expire.** Anthropic keeps a cache entry for 5 minutes by default ([Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)), OpenAI for 30 minutes by default on GPT-5.6 and later ([OpenAI](https://developers.openai.com/api/docs/guides/prompt-caching)). In a public trace of GitHub Copilot's coding agent, only 19% of input was cached on calls made after an idle gap of 5 minutes or more ([analysis](https://siddhantkhare.com/writing/coding-agent-cache-five-minutes)). An agent that waits on a human between turns pays more than the blend shows.
- **Anthropic caches only what you mark.** Without `cache_control` breakpoints, or a single top-level `cache_control` that places one automatically, a Claude call pays full input on every turn ([Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#automatic-caching)). OpenAI ([OpenAI](https://developers.openai.com/api/docs/guides/prompt-caching)), Gemini 2.5 and newer ([Google](https://aistudio.google.com/docs/caching)) and DeepSeek ([DeepSeek](https://api-docs.deepseek.com/guides/kv_cache)) cache by default.
- **Cache writes cost extra on some providers.** Anthropic bills a 5-minute cache write at 1.25 times the input rate and a 1-hour write at 2 times ([Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#pricing)); the default blend has no write share. Use a ratio with one, such as `7:1:1:1`, for a write-heavy loop.
- **Gemini's cache-write rate is a storage price.** Google bills explicit cache storage per million tokens per hour ([Google](https://aistudio.google.com/docs/pricing)), and the catalogue's cache-write field on Gemini cards holds that storage price, not a per-token write charge: on 2026-11-23 all 10 Gemini models `load_models()` keeps carry $0.50, $1.00 or $4.50 there. Any mix with a write share overprices those models until the catalogue is fixed (RES-1926). The default mix has no write share and is not affected.
- **Hosts of the same model price caching differently, and the kept listing ignores the cache rate.** On 2026-11-23, `gpt-oss-120b` cost $0.35 per million cache reads on `cerebras` against $0.075 on `groq`, so $0.390 against $0.143 on the default blend, yet the CLI and the dashboard keep the `cerebras` listing because the catalogue names `cerebras` as that card's developer (see [One dot per model](#one-dot-per-model)). Compare hosts in Python with `load_models(all_hosts=True)`.
- **Time-of-day and long-context rates are averaged or ignored.** DeepSeek V4 Pro and DeepSeek Flash bill a peak rate on weekday UTC hours and half that otherwise ([DeepSeek](https://api-docs.deepseek.com/quick_start/pricing)); the blend uses the mean of the two, so a workload that runs entirely off-peak pays less than it shows. Some providers charge more above 200k tokens, and the blend reads only the standard rate.

## One dot per model

The catalogue lists the same model several times, once per host and region. `load_models()` keeps one listing per model: the developer's own listing when there is one; otherwise a listing without a region before a regional one, and then the cheapest 3:1 list price, whatever `--mix` says. The cache rate plays no part in the choice. `load_models(all_hosts=True)` keeps every listing, which is how you compare hosts of one model against each other.

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
| `score 7.3 is below min_score 10` | the score is under `--min-score` / `min_score=` (see [The quality floor](#the-quality-floor)) |

A left-out model is not dominated. It simply was not compared.

## The limits of a blended price

A blended price is not what a task costs you. A reasoning model that writes four times as many tokens per answer costs four times as much per task, and every blend assumes one fixed share of cache reads, fresh input and output for every model. The frontier on this page compares price tags. To compare what a run costs, run both models on your own task and read the spend from the report.

## In the dashboard

`eq dashboard` has a **Models** page that draws this frontier as a chart. See [Dashboard](dashboard.md#models).

## In code

Every name used above imports from `evaluatorq.models`. `pareto()` returns a `ParetoResult` with the blend in `mix` and the axis labels in `x_spec`, whose `points` carry the score, the x value, `pareto_score` (score per dollar, `None` off the price axis), `cache_read_source` and `cache_write_source`, `on_frontier`, the nearest frontier model in `dominated_by`, and every model that beats it in `dominators`. `dominates(a, b)` answers the question for a single pair. `dominates`, `pareto_frontier` and `x_of` default to the `list` blend, because that is what evaluatorq's preset choice ranks on; pass `mix="agentic"` to match `pareto()`. The full list is in the [API reference](reference/evaluatorq/models.md).

For the CLI flags in one table, see [`eq models`](cli-reference/models.md).
