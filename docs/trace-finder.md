# Trace explorer

The dashboard has two trace pages: **Trace search** (`/find`) finds traces from a natural-language question, and **Traces** (`/traces`) browses recent Orq traces before you ask a question. Both pages reuse the same facet menu, filter chips, and trace drawer; each page keeps its own classification run and table for its result type. Traces keeps loaded rows, filters, hydrated conversations, and Ask AI results in server memory keyed to your browser session, which expires after 30 minutes idle, may be evicted when more than 32 sessions are active, and clears when the dashboard restarts; Trace search continues to use one dashboard-wide result set.

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

When you open `/traces`, the backend begins loading up to 200 trace summaries from the last seven days for that browser session, with no user-entered facet or numeric filters and no AI calls. A project selected in Settings still limits this initial load. The dashboard warms the facet catalogue in parallel. Both requests run asynchronously, so page rendering does not wait for them. This initial load is the default view, so you can browse without setting filters or pressing a button. The compact toolbar shows the active relative range (for example, **Last 7 days**) and a **Rows** control, up to 5000 and defaulting to 200. Open the range menu to choose a preset (`15m`, `1h`, `24h`, `7d`, `30d`) or enter exact **From** and **To** dates and times in your browser's local time. Press **Load** to fetch traces using the selected range, row limit, and filters; loading does not call a model. The status line shows loading progress and then the visible count. If the requested row limit is reached, it says the result may contain more traces and names the cap.

The initial trace request returns summaries for the table (status, timing, tokens, cost, and models). While traces load, the dashboard warms trajectory data for the first page of up to 100 rows, so the Trajectories view is ready when loading finishes. Later pages load trajectory data when you open them, and message content for the drawer loads when you open a row.

On **Traces**, the totals strip and the compact **By model** summary use the rows currently shown, so errors, AI matches, facets, and within-results narrowing update both. A trace with more than one model counts under each model, but its trace-level cost is left unknown because the total cannot be attributed to one model. Costs also show as unknown when a trace has no cost or when currencies are missing or mixed.

## The table and the Columns menu

The default columns are Time, Trace / agent, Status, Model, Tokens in, Tokens out, Cache read %, Cost, Duration, and AI match. The **AI match** column stays in the table when there are no judgments and shows `—` until a result is available. When an Ask AI run judges the loaded rows, it splits into one column per classifier dimension, each headed by the dimension's name. The **Columns ▾** menu in the toolbar lists every available column — also Name, Provider, Product, Operation, Reasoning tokens, Cache writes, Session, Thread, and Trace ID — and your choice is saved immediately to the dashboard settings file as `explorer_columns`, so it persists across reloads and processes. On Traces, choose **Download CSV** to export every row in the current filtered and sorted set across all pages, using the selected columns.

Results page at 100 rows per page; the pager below the table shows `Page X of Y`.

On **Traces**, each known duration keeps its numeric value and adds a small bar scaled to the longest trace in the current filtered and sorted set. Durations at or above that set's nearest-rank p95 receive a warm tint, including ties; both the scale and threshold stay the same as you move between pages. Unknown durations remain `—`, and no p95 tint appears when the set has no known durations.

## Sorting

Click a column header to sort by it; clicking again flips the direction. Sorting only reorders the rows already loaded — it never triggers a new fetch. Rows missing a value for the sorted column sink to the bottom regardless of sort direction, so switching from ascending to descending never surfaces an unset value at the top.

## How tokens in and cache read % are computed

Token counts come from the Orq router's usage rollup. The router counts cache reads and writes **inside** `prompt_tokens` for both OpenAI and Anthropic, so Tokens in is normally the router's `prompt_tokens` as-is. A span instrumented outside the router (native Anthropic usage) reports cache tokens **excluded** from `prompt_tokens` instead; the explorer detects that case — `cached + cache_write > prompt_tokens` — and falls back to `prompt_tokens + cached + cache_write` as the input count for that row. Cache read % is cache-read tokens divided by all input tokens. Cache writes count toward the input total but never toward the cache-read numerator. The percentage reads `—` when tokens in is zero or unknown.

## Trajectories view

The **Trajectories** toggle in the toolbar switches the table to a per-trace bar: each conversation renders as one horizontal bar made of coloured segments, one per captured message part. Segments are coloured by kind — `system`, `user`, `assistant`, `reasoning`, `call` (tool call), `result` (tool result) — with an unrecognized part type drawn as `other` (a hatched pattern). A captured system prompt appears as a `system` segment; tool definitions are intentionally omitted. Each segment estimates tokens as text length divided by 4, so the coloured portions show estimated captured-message tokens rather than exact provider tokenization. When provider input usage is known, a muted hatched `unattributed` segment fills the difference between that trace input total and the captured-message estimate. This difference can include hidden system or tool configuration, provider formatting and tokenizer differences, missing content, or usage from other spans; it does not identify any one source. If the estimate exceeds reported input, the bar keeps the full estimate and shows the overage in its description instead of a negative remainder. Unknown input usage does not create a remainder. The axis uses the p95 of provider-reported input across all loaded rows, so its scale stays fixed while you page; bars above that scale are capped and marked. If none of the loaded rows reports input usage, the axis uses a fixed 1,000-token estimate scale, shown in its tooltip. The legend's kind percentages describe only estimated captured-message text on the visible page. Each row also shows its duration and a text status pill. Traces without conversation text get a labelled `No conversation` bar. Hovering a segment shows a dark tooltip with its message index and position (`n / N`), its kind, its estimated token count, and a preview of its content; hovering does not dim other rows. Clicking a segment (or a table row) opens the message drawer at that exact message.

## The message drawer, minimap and message stripes

Clicking a trajectory segment or a table row opens the drawer for that trace, scrolled and expanded to the message you clicked — the thread panel scrolls to center it without moving the page itself. Each message in **Full thread** is stripe-coloured on its left edge by the same kind used in Trajectories, and the selected message is highlighted. A minimap strip mirrors the trace's segments; clicking a minimap segment moves the selection to that message locally, without a new request to the server. A table row click (with no segment) opens the drawer at message 1.

On `/traces`, if a trace has no messages, the drawer shows a labelled empty state. If a trace's full messages can't be loaded — for example a loaded-but-not-classified row whose hydration failed — the drawer explains that the messages could not be loaded and suggests opening the trace in Orq instead. Its **Spans** tab remains available and loads the trace's span tree when opened, including status, duration, token totals and Orq links for spans with safe IDs. The optional **Error details** column directs failed rows to **Spans**, and the drawer header shows the trace duration. For the first errored span, the tree also shows the raw Orq status message when that detail is available. Trace IDs, metadata, classifier input and raw result sit under the collapsed **Technical details** section. The `/find` drawer keeps its tabbed layout.

On `/traces`, press `/` to focus and select Ask AI, `?` to open the keyboard shortcut guide, and, while a trace drawer is open, `o` to activate its **Open in Orq** link or `c` to copy its trace ID. **Escape** closes the guide or drawer. Letter shortcuts are ignored while a text field or native details menu has focus, when Ctrl, Alt or Command is held, and while another modal is open.

## Ask AI: within results or as a new search

Above the table, **Ask AI** plans a natural-language question before it spends a classifier call on each trace, the same way the earlier trace finder did:

1. You enter a question, such as `Conversations over 20k tokens where the customer was frustrated.`
2. The compiler extracts numeric constraints for total tokens or duration and creates between zero and three classifier dimensions: short-named questions to ask of each trace, such as **Frustrated** or **Refund**. It creates no dimension only when every part of the question is a numeric bound or an exact metadata value; descriptive phrases such as `coding agents` become dimensions because they require reading the trace. A question that filters alone can answer, such as `Any traces above 50k tokens`, gets no dimension, and then no trace is sent to the classifier at all. At the same time, one classify request selects categorical metadata filters from the live facet catalogue. Boolean selections accept either `yes`/`no` or `true`/`false` labels.
3. The finder merges those selections with any filters you chose explicitly and builds an OQL query. The base filter excludes `generate_content` operations, so the compiler and classifier traces do not crowd the population being searched. A model or provider filter drops the base filter, because a bare model call is itself a `generate_content` trace and would otherwise never match.
4. The classifier judges each projected trace through evaluatorq, asking every dimension in one classify call per trace. A trace is included only when it matches every dimension. Results stream into one table column per dimension as each trace finishes, including the final poll after the run ends. With zero dimensions, every trace the filters keep is included without a classifier call.

Each trace is projected into a bounded classifier state before judgment: the projection keeps the newest conversation suffix, preserves tool-call arguments and completion status, removes reasoning fields and tool-result bodies, and truncates text from the front when necessary. Ask AI does not upload evaluation result rows. Trace retrieval and model inference call Orq, and OpenTelemetry tracing may export spans when configured through environment variables. Selecting a CLI profile alone does not enable tracing. Set `ORQ_DISABLE_TRACING=1` before starting the command or dashboard to disable that tracing.

On **Traces**, the Ask AI strip offers two scopes:

- **Within results** classifies the traces already loaded in the table, keeping the loaded population as-is — it reuses the rows you already fetched rather than searching again, and it carries over the same time range, facets and numeric filters the table is currently showing. After traces load, it becomes enabled and selected by default; it is disabled when the table is empty (asking without loaded rows explains that you must load traces first). During loading and classification, the progress line shows live counts and a thin progress bar; the corner status badge is hidden while the run is working. Before classifying, it applies the filters your question implies to the loaded rows: token and duration bounds, so "above 50k tokens" drops smaller rows, and the metadata values the filter model picks, such as a model, status or agent. A row with no token or duration value is dropped when that bound is set. The table narrows to the same rows, and the question's filters appear as chips, in the **Filters** count and in the **Filters** menu, so removing a chip brings the other loaded rows back. If those filters drop every loaded row, the table shows zero rows and the run completes with a notice that names the filter and nearest loaded value, such as `None of the 190 loaded traces have at least 50,001 tokens (the largest has 18,411). Try New search to look beyond the loaded rows.` At most the AI trace limit from Settings is judged; rows past it show as not judged. AI answers stay on their rows when you reload the table, change filters or change the time range, until you press **Clear AI results**. Runs above 500 traces require review before classification.
- **New search** searches independently of the rows already loaded, then reloads the table with the searched traces so the AI match column, quick views, drawer and trajectories show the run’s results. It uses a relative lookback window based on the loaded table’s time span, or the configured default if no table range is available, and anchors that window at submission time. Exact **From** and **To** dates apply to table loads, not new searches. The progress line shows how many traces have loaded out of the requested maximum while the search runs. The search uses the saved AI trace limit and parallelism defaults. The separate **Trace search** page (`/find`) has per-run window, limit, and parallelism controls.

On **Trace search**, the review panel (shown in **Review first**) gains an **Apply filters only** button below the plan, so you can load the reviewed population into the table without spending a classifier call — the classify button itself is labelled with the trace count it is about to judge.

Filter chips generated by the classifier (rather than chosen explicitly by you) are marked with a small **AI** badge, so it's clear which filters you set and which the model picked.

### Reading results

The status line shows trace loading progress or the number of traces loaded. During Ask AI loading and classification, it also shows live counts and a thin progress bar, while the corner status badge is hidden. A zero-dimension plan with no filters warns when words beyond numeric phrasing are not covered, for example `Only the numeric bounds were applied. No AI question or filter covers "coding agents", so rephrase the question or use Review to add one.` When a run finishes, the progress line leads with the answer, for example `6 of 196 traces match “angry customer”`, followed by a **Show only these** button that switches the table to the **AI matches** quick view, and the classifier model the run used (the dashboard does not track a per-run cost). An AI match is a trace the classifier judged a match; rows that only passed filters read **Kept by filters** and never count as AI matches. When the classifier returns a prose explanation, each matched row shows it under the verdict and the drawer shows it in full. When the classifier returns no prose, the row says `No explanation returned` and includes its score when available; the score is not an explanation, and Ask AI makes no extra call to generate one. Asking a question Ask AI cannot answer, such as a total, average or ranking, ends the run early with a neutral **Can't answer** state and the reason, and no counts. Running Ask AI keeps the view you are in, Table or Trajectories. The table keeps the loaded population visible; use the **AI match** column or the **AI matches** quick view to find matching judgments. **All**, **Errors**, and **AI matches** switch the table population shown. Click a table row or trajectory segment to open its drawer. **Full thread** shows the source conversation; click a message heading to fold or unfold it. **Classifier input** shows the exact bounded projection sent for judgment, and **Raw result** shows the stored evaluator result. The drawer also shows the trace and span IDs, metadata, verdict, and an **Open in Orq** link when the dashboard has a saved workspace slug, an environment override, or an authenticated Orq CLI that can find the credential's workspace.

The **Top 10%** menu picks the top tenth, at least one, of the loaded traces, largest first: **Slowest traces** use duration, **Costliest traces** use cost, **Token-heavy traces** use input plus output tokens over every turn, and **Largest context** uses the prompt tokens of the trace's leading model call. Conversation views group loaded traces by thread id, or session id when there is no thread id: **Costliest conversations** sum cost over the conversation, **Token-heavy conversations** sum input plus output tokens over the conversation, and **Longest conversations** rank by the largest message count among their traces because a trace's final call carries the whole history. Longest conversations fetches every loaded trace's messages first, so the first use after a load can take a while. Traces that do not report a metric are left out rather than counted as zero. Choosing a view clears any column sort.

When a run completes, choose **Download JSON** to export the query, the classifier dimensions (name, task and match rule for each), filters, selected trace metadata, each trace's per-dimension answers, the combined match, and errors. The export's `schema_version` is `2`; version 1 had a single top-level `task` and `selection` and one `value` per trace. The export does not include source messages or the classifier projection.

On **Trace search**, the common failure mode is stopping in **Review first**: the plan is visible, but no trace is judged until you press the classify button. If no traces match the compiled filters, the run ends with an explicit error instead of pretending that zero judgments are a successful result.

## Filter chips

Filter chips in the toolbar show the categorical and numeric filters in play, whether the classifier picked them or you did — a chip the classifier generated carries the **AI** badge described above. Whenever no run is in progress, click a chip to reopen its category and change the value, or its ✕ to drop it; the next submit uses the edited set. On **Traces**, the toolbar's **Filters** button (it shows the active count, as in **Filters · 2**) opens the category list directly; the table reloads once when you close the menu after picking values, and at once when you drop a chip with ✕. Reloading makes no AI calls. On **Trace search**, **+ Filter** opens the same category list for project, agent, model, provider, status, product, trace type, tool, tokens, and duration; hovering a category opens its live values beside the list. Scroll or search within a category's returned values. Orq supplies each value's frequency; the menu orders the returned values from most to least frequent and says when more values exist beyond the fetched limit. The facet catalogue starts warming in the background as the initial trace summaries load. The filter menu uses the warmed values and refreshes them when you change the window. For the same facet or numeric bound, an explicit value takes precedence over the generated value; generated values fill only facets and bounds you leave empty before OQL runs. The classifier's picks apply to that run only: the next question starts from the filters you set yourself, while a reviewed start keeps the whole population.

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

The dashboard **Settings** page at `/settings` has editable fields for the compiler model, classifier model, apply-recommendations model, default AI trace limit, classifier parallelism, and **Ask AI on traces**. That last setting controls what Ask AI on **Traces** does after it compiles your question: **Just proceed** (the default) classifies straight away, and **Review first** stops on the compiled plan until you press the classify button. Within results over more than 500 loaded traces always stops for review. Save the form to persist them in `.evaluatorq/dashboard-settings.json`, or point `EVALUATORQ_DASHBOARD_SETTINGS` at another JSON file. Environment variables take precedence over saved defaults. On **Traces**, choose the time range and loaded row count in the toolbar; a new AI search uses the saved limit and parallelism defaults. The separate **Trace search** page has per-run window, limit, and parallelism controls. **Advanced** also lets you save the Orq workspace slug for trace links and choose a project for the dashboard's trace loading. The project menu comes from `orq projects list` under the active credential; a project key normally exposes one project, and a broader key can expose more. The saved project ID limits the dashboard's Orq trace query, even when another project has the same name, and the active project appears beside the Traces filters. Leave **All accessible projects** selected to search across the key's scope. If the CLI cannot resolve the workspace slug, enter the slug from your Orq URL. The `eq find` CLI uses the saved project ID when its current key and API host match the credentials used to save that project; pass `--project` to choose a project facet for that run instead.

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

Loading the table fetches at most 5000 traces per load, with no AI cost. Asking AI **within results** classifies the loaded traces that meet the question's filters, up to the AI trace limit from Settings (default 500). Asking AI as a **new search** searches at most 5000 usable traces per run, even if a larger limit is supplied elsewhere; the default is 500. The default lookback is seven days, the default classifier parallelism is 100, and parallelism is capped at 200. Each projected trace has a 25,000-token budget based on the serialized UTF-8 projection estimate; older conversation units are omitted first when the budget is reached.

One completed classification run makes one compiler call, at most one facet-selection classify call, and one classification call per selected trace when the question has at least one dimension; all of a trace's dimensions share that call. A question with no dimensions makes no per-trace calls. If a facet lookup fails, the run warns and skips classifier-generated categorical filters; filters you chose explicitly and semantic classification still run. When Orq reports more facet values than the fetched limit, the finder warns and uses the returned values ranked by frequency. A 500-trace run therefore has up to 502 model calls before retries, so use the limit and window controls when you are exploring a large workspace.
