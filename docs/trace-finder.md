# Trace finder

A **trace finder** turns a natural-language question into a classification task, searches recent Orq traces with OQL, and highlights the conversations that match.

Use it when you want to inspect real traffic for a semantic pattern, such as frustrated customers or unsupported claims, without writing a dataset first. Use red teaming or simulation when you need to generate new conversations against a target instead.

## Start the finder

The dashboard finder is included in the `dashboard` extra. The CLI command is available from the regular package installation; both paths need Orq credentials because they read live Orq traces and route model calls through Orq. Export `ORQ_API_KEY` or select an `orq` CLI profile with `eq find --profile NAME`.

```bash
uv add "evaluatorq[dashboard]"
export ORQ_API_KEY=...
eq dashboard --compiler-model openai/gpt-5.6-luna --classifier-model typesafe/jev-latest
```

Open [http://127.0.0.1:8080/find](http://127.0.0.1:8080/find), choose a question, and select **Find traces**. The **Trace search** item in the dashboard sidebar opens the same page.

## How a query becomes matches

The finder plans the query before it spends a classifier call on each trace:

1. You enter a question, such as `Conversations over 20k tokens where the customer was frustrated.`
2. The compiler creates one semantic classifier task and extracts numeric constraints for total tokens or duration. At the same time, one classify request selects categorical metadata filters from the live facet catalogue. It normally asks one question per non-empty facet dimension; when your query names multiple available values in one dimension, it asks a yes/no question for each named value in the same round trip.
3. The finder merges those selections with any filters you chose explicitly and builds an OQL query. The base filter excludes `generate_content` operations, so the compiler and classifier traces do not crowd the population being searched.
4. Orq returns the newest usable traces in the selected window. The finder hydrates conversations when the trace summary does not contain usable messages, and applies the OQL filters before judging. Hydration stops with an error if a trace exceeds ten span pages or 2,000 spans.
5. Each trace is projected into a bounded classifier state. The projection keeps the newest conversation suffix, preserves tool-call names, arguments, and completion status, removes reasoning fields and tool-result bodies, and truncates text from the front when necessary. Its `omitted_bytes` count measures content dropped to fit the token budget; it excludes fields removed by the projection schema.
6. The classifier classifies each projected trace through evaluatorq. Results stream into the matrix and the included-traces table as each trace finishes.

The compiler and facet selector run concurrently, so a slow facet catalogue does not wait for semantic compilation. The facet catalogue follows the current search window, and dashboard windows are bounded to 1–90 days. The population is fixed before per-trace judging begins; changing a finder control does nothing until you submit the form again, which starts a new run.

## Dashboard workflow

The finder has two modes. **Immediate** compiles the task, loads the population, and starts judging. **Review first** stops after compilation and population selection so you can edit the question, verdict rule, and filters before any per-trace classifier calls begin. The review shows the filter model's selected metadata values separately from the classifier task; open **View structured LLM output** to inspect the filter model's complete structured response, including dimensions it left unfiltered. If filter selection fails, the review shows the error and keeps your explicit filters.

After you submit a question, the chips above the field show the categorical and numeric filters in play, whether the classifier picked them or you did. Whenever no run is in progress, click a chip to reopen its category and change the value, or its ✕ to drop it; the next submit uses the edited set. **+ Filter** opens a category list for project, agent, model, provider, status, product, trace type, tool, tokens, and duration; hovering a category opens its live values beside the list. The values load in the background right after the page renders, and reload when you change the window. For the same facet or numeric bound, an explicit value takes precedence over the generated value; generated values fill only facets and bounds you leave empty before OQL runs. The classifier's picks apply to that run only: the next question starts from the filters you set yourself, while a reviewed start keeps the whole population.

The field shows a pulsing progress message while it plans the search, loads traces, or starts classification after review. Submitting a question, starting a reviewed task, changing the filter window, cancelling, and resetting also show feedback while their requests are pending. The dots appear once the population is loaded. The dot field represents the selected population. Each dot is a trace, including traces that do not match and traces whose judgment failed. Hollow dots are waiting, dots being classified pulse, and finished dots stay still. Matching dots use the task's legend colors, while failed judgments are marked separately. The included-traces table below the field contains only successful matches and is sorted newest first.

Click a dot or a table row to open its drawer. A loading badge appears while the trace details are fetched. **Full thread** shows the source conversation; click a message heading to fold or unfold it. **Classifier input** shows the exact bounded projection sent for judgment, and **Raw result** shows the stored evaluator result. The drawer also shows the trace and span IDs, metadata, verdict, and an **Open in Orq** link when the dashboard has a workspace configured.

When a run completes, choose **Download JSON** to export the query, compiled task, filters, selected trace metadata including each trace's agent and tool names, verdicts, and errors. The export does not include source messages or the classifier projection.

The common failure mode is stopping in **Review first**: the plan is visible, but no trace is judged until you press **Start classification**. If no traces match the compiled filters, the run ends with an explicit error instead of pretending that zero judgments are a successful result. Without `ORQ_API_KEY`, the page stays available but shows that trace finding is unavailable.

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

The dashboard **Settings** page at `/settings` has editable fields for the compiler model, classifier model, and apply-recommendations model. Save the form to persist them in `.evaluatorq/dashboard-settings.json`, or point `EVALUATORQ_DASHBOARD_SETTINGS` at another JSON file. The window, trace limit, and parallelism are edited per run in the controls row of the Trace search page; their defaults come from the environment variables below or the saved file. **Advanced** also lets you save the Orq workspace slug for trace links and choose a project for dashboard Trace search. The project menu comes from `orq projects list` under the active credential; a project key normally exposes one project, and a broader key can expose more. The saved project ID limits the dashboard's Orq trace query, even when another project has the same name, and the active project appears beside the Trace search filters. Leave **All accessible projects** selected to search across the key's scope. If the CLI cannot resolve the workspace slug, enter the slug from your Orq URL. The `eq find` CLI uses the saved project ID by default; pass `--project` to choose a project facet for that run instead.

When the `orq` CLI exposes API-key profiles (`orq auth profile list`), **Advanced** lets you choose one for the dashboard's trace finder and apply flow in place of `ORQ_API_KEY` and `ORQ_BASE_URL`. Settings stores one active profile, workspace, and project together. Changing the profile clears the previous workspace and project in the preview and reloads choices from the new credential; the saved bundle remains active until you press **Save**. The CLI masks keys in its JSON output, so evaluatorq reads the real key from the CLI's private local credential file without writing it to dashboard settings. If that file is unavailable or has permissions that expose it to other users, the masked profile remains disabled. Choosing **Environment** uses the process environment without changing it; an exported `ORQ_API_KEY` takes precedence over `.env`. If a saved profile is unavailable, the dashboard blocks Orq requests and shows the missing profile in Settings. An apply preview must be made again if its credentials change before confirmation. `eq find` uses the saved profile by default, including its API key and host for both trace retrieval and model calls. An explicit `--profile NAME` overrides that choice. Choose **Environment** in Settings to make `eq find` use `ORQ_API_KEY` and `ORQ_BASE_URL`; none of these choices changes the process environment.

Settings are resolved in this order, from strongest to weakest: explicit CLI or dashboard overrides, environment variables, the saved JSON file, and built-in defaults. Invalid environment integers are ignored with a warning; invalid saved settings fall back to built-in defaults.

| Setting | Default | Environment variable |
|---|---|---|
| Compiler model | `openai/gpt-5.6-luna` | `EVALUATORQ_COMPILER_MODEL` |
| Classifier model | `typesafe/jev-latest` | `EVALUATORQ_CLASSIFIER_MODEL` |
| Apply-recommendations model | `openai/gpt-5.6-luna` | `EVALUATORQ_APPLY_MODEL` |
| Search window | 7 days | `EVALUATORQ_FINDER_WINDOW_DAYS` |
| Trace limit | 500 (max 5000) | `EVALUATORQ_FINDER_LIMIT` |
| Classifier parallelism | 100 | `EVALUATORQ_FINDER_PARALLELISM` |

The dashboard command accepts finder overrides for `--compiler-model`, `--classifier-model`, `--window-days`, `--limit`, and `--parallelism`. Those options apply to finder runs started by that dashboard process.

## CLI reference

`eq find` runs an immediate finder query and prints progress followed by a newest-first table of matched traces. Add `--json PATH` to write the completed run export. The command cancels a run that has not finished after two hours. If any trace classification fails, the command exits with status 1 and does not write the JSON file.

Pass `--profile NAME` to use that `orq` CLI profile for both trace retrieval and model calls. It overrides the saved profile and environment credentials. Without the flag, the saved profile applies; choose **Environment** in Settings to use `ORQ_API_KEY` and `ORQ_BASE_URL`. A saved project ID also limits the CLI trace population unless you pass `--project` for a different project facet or choose a different profile with `--profile`.

```bash
export ORQ_API_KEY=...
eq find "customers asking for a refund" --compiler-model openai/gpt-5.6-luna --classifier-model typesafe/jev-latest --window-days 7 --limit 500 --parallelism 100 --json finder.json
```

The command accepts these options:

| Option | Meaning |
|---|---|
| `--window-days INTEGER` (`1`–`90`) | How many recent days to search. |
| `--limit INTEGER` (`1`–`5000`) | Maximum traces to classify. |
| `--parallelism INTEGER` (`1`–`200`) | Concurrent classify calls. |
| `--compiler-model TEXT` | Model that compiles the search question through the Orq router. Default: `openai/gpt-5.6-luna`. |
| `--classifier-model TEXT` | Model that classifies each trace through the Orq router. Default: `typesafe/jev-latest`. |
| `--json PATH` | Write the completed run export to `PATH`. |
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

The finder searches at most 5000 usable traces per run, even if a larger limit is supplied elsewhere; the default is 500. The default lookback is seven days, the default classifier parallelism is 100, and parallelism is capped at 200. Each projected trace has a 25,000-token budget based on the serialized UTF-8 projection estimate; older conversation units are omitted first when the budget is reached.

One completed run makes one compiler call, at most one facet-selection classify call, and one classification call per selected trace. If a facet lookup fails or the catalogue reports more values than the SDK can retrieve, the run warns and skips classifier-generated categorical filters; filters you chose explicitly and semantic classification still run. A 500-trace run therefore has up to 502 model calls before retries, so use the limit and window controls when you are exploring a large workspace.
