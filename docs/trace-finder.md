# Trace explorer

The dashboard has two trace pages: **Trace search** (`/find`) finds traces from a natural-language question, and **Traces** (`/traces`) browses recent Orq traces before you ask a question. Both pages reuse the same facet menu, filter chips, and trace drawer; each page keeps its own classification run and table for its result type.

Use Traces to inspect real traffic — browse recent traces, sort and page through them, look at token and cost breakdowns, and drill into a trajectory or a message — with or without ever asking a question. Use red teaming or simulation when you need to generate new conversations against a target instead.

## Start the explorer

The dashboard explorer is included in the `dashboard` extra. It needs Orq credentials because it reads live Orq traces and, when you ask a question, routes model calls through Orq. Export `ORQ_API_KEY` or select an `orq` CLI profile in Settings.

```bash
uv add "evaluatorq[dashboard]"
export ORQ_API_KEY=...
eq dashboard --compiler-model openai/gpt-5.6-luna --classifier-model typesafe/jev-latest
```

Open [http://127.0.0.1:8080/traces](http://127.0.0.1:8080/traces) for the explorer or [http://127.0.0.1:8080/find](http://127.0.0.1:8080/find) for the question-first search. The dashboard sidebar has separate **Traces** and **Trace search** items. Without `ORQ_API_KEY` (or a working saved profile), both pages stay available and explain that trace loading is unavailable.

## Loading traces

When the dashboard starts, the backend begins loading up to 200 trace summaries from the last seven days, with no user-entered facet or numeric filters and no AI calls. A project selected in Settings still limits this initial load. It warms the facet catalogue in parallel. Both requests run asynchronously, so startup and page rendering do not wait for them; opening `/traces` also starts them if they have not begun yet. This initial load is the default view, so you can browse without setting filters or pressing a button. The controls row above the table sets **From**, **To** (both shown and edited in your browser's local time), and **Rows** — up to 5000, defaulting to 200. Preset buttons (`15m`, `1h`, `24h`, `7d`, `30d`) fill the range from now. Press **Load** to fetch the newest traces using the selected time range, row limit, and filters; loading does not call a model. The toolbar shows `loading N / limit` while a load is in flight and the trace count once it finishes.

The initial trace request returns summaries for the table (status, timing, tokens, cost, and models). Full message content and trajectory data are more expensive, so they remain lazy: message content loads for the row you open in the drawer, and trajectory data loads for the page of rows shown in the Trajectories view.

## The table and the Columns menu

The default columns are Status, Started, Agent, Model, Tokens in, Tokens out, Cache read %, Cost, Duration, and AI match. The **AI match** column stays in the table when there are no judgments and shows `—` until a result is available. The **Columns ▾** menu in the toolbar lists every available column — also Name, Provider, Product, Operation, Reasoning tokens, Cache writes, Session, Thread, and Trace ID — and your choice is saved immediately to the dashboard settings file as `explorer_columns`, so it persists across reloads and processes.

Results page at 100 rows per page; the pager below the table shows `Page X of Y`.

## Sorting

Click a column header to sort by it; clicking again flips the direction. Sorting only reorders the rows already loaded — it never triggers a new fetch. Rows missing a value for the sorted column sink to the bottom regardless of sort direction, so switching from ascending to descending never surfaces an unset value at the top.

## How tokens in and cache % are computed

Token counts come from the Orq router's usage rollup. The router counts cache reads and writes **inside** `prompt_tokens` for both OpenAI and Anthropic, so Tokens in is normally the router's `prompt_tokens` as-is. A span instrumented outside the router (native Anthropic usage) reports cache tokens **excluded** from `prompt_tokens` instead; the explorer detects that case — `cached + cache_write > prompt_tokens` — and falls back to `prompt_tokens + cached + cache_write` as the input count for that row. Cache % is cached tokens divided by tokens in, shown as a percentage with a fill bar; it reads `—` when tokens in is zero or unknown.

## Trajectories view

The **Trajectories** toggle in the toolbar switches the table to a per-trace bar: each conversation renders as one horizontal bar made of coloured segments, one per message part. Segments are coloured by kind — `system`, `user`, `assistant`, `reasoning`, `call` (tool call), `result` (tool result) — with an unrecognized part type drawn as `other` (a hatched pattern); the legend above the bars shows each kind's share of the total tokens across the visible page. A segment's size is estimated as its text length divided by 4 — a size proxy, not a real tokenizer count. Hovering a segment shows a dark tooltip with its message index and position (`n / N`), its kind, its estimated token count, and a preview of its content; hovering does not dim other rows. Clicking a segment (or a table row) opens the message drawer at that exact message.

## The message drawer, minimap and message stripes

Clicking a trajectory segment or a table row opens the drawer for that trace, scrolled and expanded to the message you clicked — the thread panel scrolls to center it without moving the page itself. Each message in **Full thread** is stripe-coloured on its left edge by the same kind used in Trajectories, and the selected message is highlighted. A minimap strip mirrors the trace's segments; clicking a minimap segment moves the selection to that message locally, without a new request to the server. A table row click (with no segment) opens the drawer at message 1.

If a trace's full messages can't be loaded — for example a loaded-but-not-classified row whose hydration failed — the drawer explains that the messages could not be loaded and suggests opening the trace in Orq instead, rather than rendering an empty thread.

## Ask AI: within results or as a new search

Above the table, **Ask AI** plans a natural-language question before it spends a classifier call on each trace, the same way the earlier trace finder did:

1. You enter a question, such as `Conversations over 20k tokens where the customer was frustrated.`
2. The compiler creates one semantic classifier task and extracts numeric constraints for total tokens or duration. At the same time, one classify request selects categorical metadata filters from the live facet catalogue.
3. The finder merges those selections with any filters you chose explicitly and builds an OQL query. The base filter excludes `generate_content` operations, so the compiler and classifier traces do not crowd the population being searched.
4. The classifier classifies each projected trace through evaluatorq. Results stream into the matrix and the table's **Match** column as each trace finishes, including the final poll after the run ends.

Each trace is projected into a bounded classifier state before judgment: the projection keeps the newest conversation suffix, preserves tool-call arguments and completion status, removes reasoning fields and tool-result bodies, and truncates text from the front when necessary. Ask AI does not upload evaluation result rows. Trace retrieval and model inference call Orq, and OpenTelemetry tracing may export spans when configured through environment variables. Selecting a CLI profile alone does not enable tracing. Set `ORQ_DISABLE_TRACING=1` before starting the command or dashboard to disable that tracing.

A second segmented control next to **Immediate** / **Review first** chooses the **scope**:

- **Within results** classifies every trace already loaded in the table, keeping the loaded population as-is — it reuses the rows you already fetched rather than searching again, and it carries over the same time range, facets and numeric filters the table is currently showing. After traces load, it becomes enabled and selected by default; it is disabled when the table is empty (asking without loaded rows explains that you must load traces first). Because it always classifies every loaded row, the Limit box is hidden while this scope is selected. A run against more than 500 loaded traces is always routed through **Review first**, regardless of which mode you picked, so a large classification run is never billed without a chance to inspect the plan first.
- **New search** is the original flow: it searches independently of whatever is loaded in the table, using the current From/To range, limit and parallelism controls. AI proposes metadata filters but does not suggest or change the selected time range; the lookback window supplies the range when no explicit From/To values are present.

The review panel (shown in **Review first**) gains an **Apply filters only** button below the plan, so you can load the reviewed population into the table without spending a classifier call — the classify button itself is labelled with the trace count it is about to judge.

Filter chips generated by the classifier (rather than chosen explicitly by you) are marked with a small **AI** badge, so it's clear which filters you set and which the model picked.

### Reading results

The status line shows whether the page is loading traces, how many are loaded, and the AI match or judgment-error counts. The table keeps every loaded trace visible; use the **AI match** column or **Matches only** to find judgments. Click a table row or trajectory segment to open its drawer. **Full thread** shows the source conversation; click a message heading to fold or unfold it. **Classifier input** shows the exact bounded projection sent for judgment, and **Raw result** shows the stored evaluator result. The drawer also shows the trace and span IDs, metadata, verdict, and an **Open in Orq** link when the dashboard has a saved workspace slug, an environment override, or an authenticated Orq CLI that can find the credential's workspace.

When a run completes, choose **Download JSON** to export the query, compiled task, filters, selected trace metadata, verdicts, and errors. The export does not include source messages or the classifier projection.

The common failure mode is stopping in **Review first**: the plan is visible, but no trace is judged until you press the classify button. If no traces match the compiled filters, the run ends with an explicit error instead of pretending that zero judgments are a successful result.

## Filter chips

After you submit a question, chips above the field show the categorical and numeric filters in play, whether the classifier picked them or you did — a chip the classifier generated carries the **AI** badge described above. Whenever no run is in progress, click a chip to reopen its category and change the value, or its ✕ to drop it; the next submit uses the edited set. **+ Filter** opens a category list for project, agent, model, provider, status, product, trace type, tool, tokens, and duration; hovering a category opens its live values beside the list. Scroll or search within a category's returned values. Orq supplies each value's frequency; the menu orders the returned values from most to least frequent and says when more values exist beyond the fetched limit. The facet catalogue starts warming in the background as the initial trace summaries load. The filter menu uses the warmed values and refreshes them when you change the window. For the same facet or numeric bound, an explicit value takes precedence over the generated value; generated values fill only facets and bounds you leave empty before OQL runs. The classifier's picks apply to that run only: the next question starts from the filters you set yourself, while a reviewed start keeps the whole population.

## Facets and numeric ranges

The classifier selects categorical metadata from the live catalogue. The eight categorical facets and their OQL fields are:

| Finder facet | OQL field | Meaning |
|---|---|---|
| `project` | `project_id` | Project names are resolved to project IDs through `projects.list`; equal names include their project ID in the menu. |
| `model` | `model` | Model recorded on the trace. |
| `provider` | `provider` | Provider recorded on the trace. |
| `status` | `status` | Trace status. |
| `product` | `product` | Product recorded on the trace. |
| `trace_type` | `attributes.orq.leading_span.span_type` | Leading span type. |
| `agent_name` | `agent_name` | Agent name recorded on the trace. |
| `tool_name` | `tool_name` | Tool name recorded on the trace. |

The compiler handles the two numeric dimensions because trace-finder metadata thresholds are not classify outputs. Classify tasks return a label (`choice`), a yes/no probability (`noul`), or a 0–1 score (`score`); they do not extract numeric metadata thresholds. The compiler extracts inclusive integer ranges for `total_tokens` and `duration_ms` and applies them in OQL; for example, “over 20k tokens” becomes `total_tokens >= 20001`, “under 20k tokens” becomes `total_tokens <= 19999`, and “slower than 30 seconds” becomes `duration_ms >= 30001`. The CLI and dashboard also let you enter minimum and maximum bounds explicitly.

## Settings and precedence

The dashboard **Settings** page at `/settings` has editable fields for the compiler model, classifier model, and apply-recommendations model. Save the form to persist them in `.evaluatorq/dashboard-settings.json`, or point `EVALUATORQ_DASHBOARD_SETTINGS` at another JSON file. Set the lookback window, trace limit, and parallelism per run on Trace search; Traces uses From and To date and time controls for its range and offers the limit and parallelism controls for a new AI search. Their defaults come from the environment variables below or the saved file. **Advanced** also lets you save the Orq workspace slug for trace links and choose a project for the dashboard's trace loading. The project menu comes from `orq projects list` under the active credential; a project key normally exposes one project, and a broader key can expose more. The saved project ID limits the dashboard's Orq trace query, even when another project has the same name, and the active project appears beside the Traces filters. Leave **All accessible projects** selected to search across the key's scope. If the CLI cannot resolve the workspace slug, enter the slug from your Orq URL. The `eq find` CLI uses the saved project ID when its current key and API host match the credentials used to save that project; pass `--project` to choose a project facet for that run instead.

When the `orq` CLI exposes API-key profiles (`orq auth profile list`), **Advanced** lets you choose one for the dashboard's trace loading, Ask AI, and apply flow in place of `ORQ_API_KEY` and `ORQ_BASE_URL`. Settings stores one active profile, workspace, and project together. Changing the profile clears the previous workspace and project in the preview and reloads choices from the new credential; the saved bundle remains active until you press **Save**. The CLI masks keys in its JSON output, so evaluatorq reads the real key from the CLI's private local credential file without writing it to dashboard settings. If that file is unavailable or has permissions that expose it to other users, the masked profile remains disabled. Choosing **Environment** uses the process environment without changing it; an exported `ORQ_API_KEY` takes precedence over `.env`. If a saved profile is unavailable, the dashboard blocks Orq requests and shows the missing profile in Settings. An apply preview must be made again if its credentials change before confirmation. `eq find` uses the saved profile by default, including its API key and host for both trace retrieval and model calls. An explicit `--profile NAME` overrides that choice. Choose **Environment** in Settings to make `eq find` use `ORQ_API_KEY` and `ORQ_BASE_URL`; none of these choices changes the process environment.

Settings are resolved in this order, from strongest to weakest: explicit CLI or dashboard overrides, environment variables, the saved JSON file, and built-in defaults. Invalid environment integers are ignored with a warning; invalid saved settings fall back to built-in defaults.

| Setting | Default | Environment variable |
|---|---|---|
| Compiler model | `openai/gpt-5.6-luna` | `EVALUATORQ_COMPILER_MODEL` |
| Classifier model | `typesafe/jev-latest` | `EVALUATORQ_CLASSIFIER_MODEL` |
| Apply-recommendations model | `openai/gpt-5.6-luna` | `EVALUATORQ_APPLY_MODEL` |
| Search window | 7 days | `EVALUATORQ_FINDER_WINDOW_DAYS` |
| Trace limit | 500 (max 5000) | `EVALUATORQ_FINDER_LIMIT` |
| Classifier parallelism | 100 | `EVALUATORQ_FINDER_PARALLELISM` |

The dashboard command accepts finder overrides for `--compiler-model`, `--classifier-model`, `--window-days`, `--limit`, and `--parallelism`. Those options apply to finder runs started by that dashboard process. The Settings page lets you save the compiler, classifier and apply models, plus the default AI trace limit and parallelism; environment variables still take precedence over saved values.

## CLI reference

`eq find` runs an immediate finder query — the same compile-and-classify pipeline as **New search** — with a terminal activity indicator, then prints a newest-first table of matched traces and a summary of the full run. The CLI has no explorer table, trajectories view, or drawer; it is the classification pipeline only. Add `--json PATH` to write the completed run export. Pass `--positive-only` to keep only matched trace records in that JSON export; its counts still describe the full run. Pass `--debug` to print progress when it changes, the compiler request and structured output, and the filter and per-trace classifier requests and responses. For a small diagnostic run, use `eq find "mentions a refund" --limit 10 --debug`. Debug output includes projected conversation content for every classified trace, even with `--positive-only`, so treat saved logs as trace data. `EVALUATORQ_LOG_LEVEL=DEBUG` enables the same finder diagnostics in CLI and dashboard runs.

The command cancels a run that has not finished after two hours. If any trace classification fails, the command exits with status 1 and does not write the JSON file.

Pass `--profile NAME` to use that `orq` CLI profile for both trace retrieval and model calls. It overrides the saved profile and environment credentials. Without the flag, the saved profile applies; choose **Environment** in Settings to use `ORQ_API_KEY` and `ORQ_BASE_URL`. A saved project ID limits the CLI trace population only while the profile, key, and API host still match the saved selection. If you rotate a key or change hosts, save the project again in Settings; use `--project` to choose a project facet for one run.

```bash
export ORQ_API_KEY=...
eq find "customers asking for a refund" --compiler-model openai/gpt-5.6-luna --classifier-model typesafe/jev-latest --window-days 7 --limit 500 --parallelism 100 --json finder.json
```

The command accepts these options:

| Option | Meaning |
|---|---|
| `--debug` | Print changed progress and compiler and classifier request and response data, including trace content. |
| `--window-days INTEGER` (`1`–`90`) | How many recent days to search. |
| `--limit INTEGER` (`1`–`5000`) | Maximum traces to classify. |
| `--parallelism INTEGER` (`1`–`200`) | Concurrent classify calls. |
| `--compiler-model TEXT` | Model that compiles the search question through the Orq router. Default: `openai/gpt-5.6-luna`. |
| `--classifier-model TEXT` | Model that classifies each trace through the Orq router. Default: `typesafe/jev-latest`. |
| `--json PATH` | Write the completed run export to `PATH`. |
| `--positive-only` | Keep only matched trace records in `--json` exports; the terminal table already shows matches and keeps its full-run summary. |
| `--project TEXT` | Project facet; repeatable. Overrides the saved project ID for this run. |
| `--profile TEXT` | Orq CLI credential profile; overrides the saved profile and environment credentials. |
| `--model TEXT` | Model facet; repeatable. |
| `--provider TEXT` | Provider facet; repeatable. |
| `--status TEXT` | Status facet; repeatable. |
| `--product TEXT` | Product facet; repeatable. |
| `--trace-type TEXT` | Trace type facet; repeatable. |
| `--agent TEXT` | Agent name facet; repeatable. |
| `--tool TEXT` | Tool name facet; repeatable. |
| `--tokens-min INTEGER` (`>= 0`) | Minimum total tokens. |
| `--tokens-max INTEGER` (`>= 0`) | Maximum total tokens. |
| `--duration-ms-min INTEGER` (`>= 0`) | Minimum trace duration in milliseconds. |
| `--duration-ms-max INTEGER` (`>= 0`) | Maximum trace duration in milliseconds. |
| `--help`, `-h` | Show the command help. |

The facet and numeric options are explicit OQL constraints. The natural-language question still supplies the semantic classifier task and can add generated facet or numeric constraints.

## Limits and cost

Loading the table fetches at most 5000 traces per load, with no AI cost. Asking AI **within results** classifies every loaded trace — as many as you loaded, up to that same 5000. Asking AI as a **new search** searches at most 5000 usable traces per run, even if a larger limit is supplied elsewhere; the default is 500. The default lookback is seven days, the default classifier parallelism is 100, and parallelism is capped at 200. Each projected trace has a 25,000-token budget based on the serialized UTF-8 projection estimate; older conversation units are omitted first when the budget is reached.

One completed classification run makes one compiler call, at most one facet-selection classify call, and one classification call per selected trace. If a facet lookup fails, the run warns and skips classifier-generated categorical filters; filters you chose explicitly and semantic classification still run. When Orq reports more facet values than the fetched limit, the finder warns and uses the returned values ranked by frequency. A 500-trace run therefore has up to 502 model calls before retries, so use the limit and window controls when you are exploring a large workspace.
