# Trace finder

A **trace finder** turns a natural-language question into a JEV classification task, searches recent Orq traces with OQL, and highlights the conversations that match.

Use it when you want to inspect real traffic for a semantic pattern, such as frustrated customers or unsupported claims, without writing a dataset first. Use red teaming or simulation when you need to generate new conversations against a target instead.

## Start the finder

The dashboard finder is included in the `dashboard` extra. The CLI command is available from the regular package installation; both paths require `ORQ_API_KEY` because they read live Orq traces and route their model calls through Orq.

```bash
uv add "evaluatorq[dashboard]"
export ORQ_API_KEY=...
eq dashboard --compiler-model openai/gpt-5.6-luna --jev-model typesafe/jev-latest
```

Open [http://127.0.0.1:8080/find](http://127.0.0.1:8080/find), choose a question, and select **Find traces**. The **Trace search** item in the dashboard sidebar opens the same page.

## How a query becomes matches

The finder plans the query before it spends a JEV call on each trace:

1. You enter a question, such as `Conversations over 20k tokens where the customer was frustrated.`
2. The compiler creates one semantic JEV task and extracts numeric constraints for total tokens or duration. At the same time, one JEV classify request selects categorical metadata filters from the live facet catalogue. The classifier receives one question for each non-empty facet dimension, but this is one classify round trip.
3. The finder merges those selections with any filters you chose explicitly and builds an OQL query. The base filter excludes `generate_content` operations, so the compiler and classifier traces do not crowd the population being searched.
4. Orq returns the newest usable traces in the selected window. The finder hydrates conversations when the trace summary does not contain usable messages, and applies the OQL filters before judging.
5. Each trace is projected into a bounded JEV state. The projection keeps the newest conversation suffix, preserves tool-call names, arguments, and completion status, removes reasoning fields and tool-result bodies, and truncates text from the front when necessary.
6. JEV classifies each projected trace through evaluatorq. Results stream into the matrix and the included-traces table as each trace finishes.

The compiler and facet selector run concurrently, so a slow facet catalogue does not wait for semantic compilation. The facet catalogue follows the current search window, and dashboard windows are bounded to 1–90 days. The population is fixed before per-trace judging begins; changing a finder control does nothing until you submit the form again, which starts a new run.

## Dashboard workflow

The finder has two modes. **Immediate** compiles the task, loads the population, and starts judging. **Review first** stops after compilation and population selection so you can edit the task before any per-trace JEV calls begin.

After you submit a question, the chips above the field show the categorical and numeric filters in play, whether JEV picked them or you did. Whenever no run is in progress, click a chip to reopen its category and change the value, or its ✕ to drop it; the next submit uses the edited set. **+ Filter** opens a category list for project, agent, model, provider, status, product, trace type, tool, tokens, and duration; hovering a category opens its live values beside the list. For the same facet or numeric bound, an explicit value takes precedence over the generated value; generated values fill only facets and bounds you leave empty before OQL runs.

While the compiler and facet selector are still working, the field shows a pulsing **Asking JEV what to look for…** placeholder and the counts read *compiling the question and selecting traces*; the dots appear once the population is loaded. The dot field represents the selected population. Each dot is a trace, including traces that do not match and traces whose judgment failed. Matching dots use the task's legend colors, while failed judgments are marked separately. The included-traces table below the field contains only successful matches and is sorted newest first.

Click a dot or a table row to open its drawer. **Full thread** shows the source conversation, **JEV input** shows the exact bounded projection sent for judgment, and **Raw result** shows the stored evaluator result. The drawer also shows the trace and span IDs, metadata, verdict, and an **Open in Orq** link when the dashboard has a workspace configured.

When a run completes, choose **Download JSON** to export the query, compiled task, filters, selected trace metadata including each trace's agent and tool names, verdicts, and errors. The export does not include source messages or the JEV projection.

The common failure mode is stopping in **Review first**: the plan is visible, but no trace is judged until you press **Start classification**. If no traces match the compiled filters, the run ends with an explicit error instead of pretending that zero judgments are a successful result. Without `ORQ_API_KEY`, the page stays available but shows that trace finding is unavailable.

## Facets and numeric ranges

JEV selects categorical metadata from the live catalogue. The eight categorical facets and their OQL fields are:

| Finder facet | OQL field | Meaning |
|---|---|---|
| `project` | `project_id` | Project names are resolved to project IDs through `projects.list`. |
| `model` | `model` | Model recorded on the trace. |
| `provider` | `provider` | Provider recorded on the trace. |
| `status` | `status` | Trace status. |
| `product` | `product` | Product recorded on the trace. |
| `trace_type` | `attributes.orq.leading_span.span_type` | Leading span type. |
| `agent_name` | `agent_name` | Agent name recorded on the trace. |
| `tool_name` | `tool_name` | Tool name recorded on the trace. |

The compiler handles the two numeric dimensions because trace-finder metadata thresholds are not JEV classify outputs. JEV classify tasks return a label (`choice`), a yes/no probability (`noul`), or a 0–1 score (`score`); they do not extract numeric metadata thresholds. The compiler extracts inclusive ranges for `total_tokens` and `duration_ms` and applies them in OQL; for example, “over 20k tokens” becomes `total_tokens >= 20000`, and “slower than 30 seconds” becomes `duration_ms >= 30000`. The CLI and dashboard also let you enter minimum and maximum bounds explicitly.

## Settings and precedence

The dashboard **Settings** page at `/settings` has editable fields for the compiler model, JEV model, and apply-recommendations model. Save the form to persist them in `.evaluatorq/dashboard-settings.json`, or point `EVALUATORQ_DASHBOARD_SETTINGS` at another JSON file. The window, trace limit, and parallelism are edited per run in the controls row of the Trace search page; their defaults come from the environment variables below or the saved file.

Settings are resolved in this order, from strongest to weakest: explicit CLI or dashboard overrides, environment variables, the saved JSON file, and built-in defaults. Invalid environment integers are ignored with a warning; invalid saved settings fall back to built-in defaults.

| Setting | Default | Environment variable |
|---|---|---|
| Compiler model | `openai/gpt-5.6-luna` | `EVALUATORQ_COMPILER_MODEL` |
| JEV model | `typesafe/jev-latest` | `EVALUATORQ_JEV_MODEL` |
| Apply-recommendations model | `openai/gpt-5.6-luna` | `EVALUATORQ_APPLY_MODEL` |
| Search window | 7 days | `EVALUATORQ_FINDER_WINDOW_DAYS` |
| Trace limit | 500 | `EVALUATORQ_FINDER_LIMIT` |
| JEV parallelism | 100 | `EVALUATORQ_FINDER_PARALLELISM` |

The dashboard command accepts finder overrides for `--compiler-model`, `--jev-model`, `--window-days`, `--limit`, and `--parallelism`. Those options apply to finder runs started by that dashboard process.

## CLI reference

`eq find` runs an immediate finder query and prints progress followed by a newest-first table of matched traces. Add `--json PATH` to write the completed run export.

```bash
export ORQ_API_KEY=...
eq find "customers asking for a refund" --compiler-model openai/gpt-5.6-luna --jev-model typesafe/jev-latest --window-days 7 --limit 500 --parallelism 100 --json finder.json
```

The command accepts these options:

| Option | Meaning |
|---|---|
| `--window-days INTEGER` (`1`–`90`) | How many recent days to search. |
| `--limit INTEGER` (`1`–`500`) | Maximum traces to classify. |
| `--parallelism INTEGER` (`1`–`200`) | Concurrent JEV classifications. |
| `--compiler-model TEXT` | Model that compiles the search question through the Orq router. Default: `openai/gpt-5.6-luna`. |
| `--jev-model TEXT` | JEV model that classifies each trace through the Orq router. Default: `typesafe/jev-latest`. |
| `--json PATH` | Write the completed run export to `PATH`. |
| `--project TEXT` | Project facet; repeatable. |
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

The facet and numeric options are explicit OQL constraints. The natural-language question still supplies the semantic JEV task and can add generated facet or numeric constraints.

## Limits and cost

The finder searches a maximum of 500 usable traces per run, even if a larger limit is supplied elsewhere. The default lookback is seven days, the default JEV parallelism is 100, and parallelism is capped at 200. Each projected trace has a 25,000-token budget based on the serialized UTF-8 projection estimate; older conversation units are omitted first when the budget is reached.

One completed run makes one compiler call, at most one facet-selection classify call, and one JEV classification call per selected trace. The facet-selection call is skipped when the facet catalogue is empty, including when every facet lookup failed. A 500-trace run therefore has up to 502 model calls before retries, so use the limit and window controls when you are exploring a large workspace.
