# Trace Insights

**Trace Insights** selects a population of Orq traces, answers fixed label questions, and discovers groups in the traces by clustering summary text. Use it when you want to review patterns in filtered trace traffic; use the trace finder alone when you only need to locate individual matches.

## What a run produces

An Insights run keeps three different things separate. A **classifier** is the model that answers fixed-form questions about each trace and, when you provide a semantic query, decides whether the trace matches it.

| Result | How it is chosen | What it contains |
|---|---|---|
| Population | Query, metadata facets, numeric bounds, time window, and limit | The traces included in the run. Labels never change population membership. |
| Labels | Fixed classifier questions | One answer per trace for each yes/no (`noul`), choice, or score question, with confidence and probabilities. |
| Discovered dimensions | Text fields from a per-trace summary | Two-level clusters and saved 3D UMAP coordinates for traces in groups large enough to project. |

The built-in discovered dimensions are `intent` (summary field `request`), `failure` (`assistant_errors`), and `sentiment` (`sentiment_explanation`). The summary model writes these text fields from the trace's conversation, including assistant tool calls and an excerpt from the start of each tool result. The classifier answers labels such as sentiment, customer satisfaction, or your own questions. When you request the discovered `sentiment` dimension without a sentiment label, Insights adds that label so it can group the explanations by sentiment.

The run writes its JSON result to `.evaluatorq/insights-runs/` and its manifest under `.evaluatorq/insights-runs/.manifests/`; `EVALUATORQ_DIR` changes the base directory. It does not store message bodies or embedding vectors in the run JSON.

## Run from Python

Call `insights()` with an `InsightsPopulation` and optional labels and discovered dimensions. This example restricts the population to one agent and adds a custom label alongside two presets. It reads live traces and makes model requests through Orq, so set `ORQ_API_KEY` before running it.

Install the `insights` extra before importing this package path with `uv add "evaluatorq[insights]"`.

```python
import asyncio

from evaluatorq.insights import InsightsPopulation, LabelSpec, insights, presets
from evaluatorq.trace_finder.models import FacetSelection, NumericFilters


async def main() -> None:
    run = await insights(
        population=InsightsPopulation(
            query="customers asking about refunds",
            facets=FacetSelection(agent_name=frozenset({"support-bot"})),
            numeric=NumericFilters(),
            window_days=7,
            limit=500,
        ),
        labels=[
            presets.SENTIMENT,
            presets.CUSTOMER_SATISFACTION,
            LabelSpec(
                name="needs_human_follow_up",
                kind="noul",
                instructions="Decide whether a human should follow up on this conversation.",
                criteria={
                    "true": "The customer needs a human to resolve the request.",
                    "false": "The assistant resolved the request without human help.",
                },
            ),
        ],
        dimensions=["intent", "failure", "sentiment"],
        summary_model="openai/gpt-6-luna",
        classifier_model="typesafe/jev-latest",
        embedding_model="openai/text-embedding-3-small",
    )
    print(run.status, run.counts)


asyncio.run(main())
```

`InsightsPopulation` also accepts just facets and numeric bounds, or `InsightsPopulation.from_finder_export(path)` to use the matches from an `eq find --json` export. The export path already fixes the population, so do not combine it with a query, facets, numeric bounds, time window, or limit.

## Run from the CLI

The `eq insights` command accepts trace population filters, label presets or JSON files, and discovered dimensions. Install `evaluatorq[insights]` and use `ORQ_API_KEY`, the profile selected in dashboard Settings, or an explicit `--profile NAME`. An explicit profile wins over Settings, and Settings wins over the environment. The selected profile supplies its API key and host for both traces and model calls; if it is unavailable, the command exits with an error instead of using another key. With `--query`, Insights asks the classifier to match traces and removes non-matches. Requested labels are sent with the same per-trace request, and traces whose match judgment fails remain in the run with an error. Filter-only runs and `--from-finder` runs do not ask a new match question.

```bash
eq insights --query "customers asking about refunds" --agent support-bot --label sentiment --label customer_satisfaction --dimension intent --dimension failure --limit 500 --json refunds-insights.json
```

The command prints cluster and label summaries, failed-trace counts, and the stored run path. Use `--from-finder PATH` to use matched trace IDs from a finder export instead of `--query`; the command rejects other population filters, `--window-days`, and `--limit` alongside it. `--no-cache` disables the local summary and embedding cache. The summary model defaults to `openai/gpt-6-luna`. If you omit `--classifier-model`, Insights uses `EVALUATORQ_CLASSIFIER_MODEL`, then the classifier model saved in dashboard Settings, then `typesafe/jev-latest`; an explicit option overrides those defaults.

After a finder run completes, its **Analyze matches** section lets you download the finder export. Pass the downloaded filename to `--from-finder`; Insights selects only the matched trace IDs from that export. For a file named `trace-finder-42.json`, this command reviews those matches:

```bash
eq insights --from-finder trace-finder-42.json --label sentiment --dimension intent --json insights.json
```

## Add a fixed label

Save one label object or a list of label objects as JSON, then pass its path with repeatable `--label`. A choice label must provide its answer names and descriptions in `criteria`.

```json
{
  "name": "resolution_status",
  "kind": "choice",
  "instructions": "Classify how the assistant handled the customer's request.",
  "criteria": {
    "resolved": "The assistant completed the requested task.",
    "partially_resolved": "The assistant made progress but left part of the request open.",
    "unresolved": "The assistant did not resolve the request."
  }
}
```

```bash
eq insights --query "customers asking about refunds" --label ./resolution-label.json --dimension intent
```

## Review a run

Start the dashboard with the `dashboard` and `insights` extras, then open **Insights** in the dashboard navigation. Its overview lists saved runs, their status, trace counts, and dimensions. The Stages column beside each run shows its saved progress; the current stage appears beneath the name of an active run. Hover over a stage dot to read its label and status. Select a run to open a dedicated results page without a second run list; **All Insights runs** returns to the overview. In a run, inspect discovered dimensions as a cluster tree or 3D map, review label distributions and confidence, compare dimensions or labels in a crosstab, filter the trace table, or open a trace in Orq. The priority matrix is available when the run includes the customer-satisfaction label; otherwise the page explains why it is empty.

The priority matrix uses error share from summaries when you did not request the `made_errors` label. It divides traces whose `assistant_errors` summary field contains a non-placeholder entry by traces in that cluster that have a summary; traces without a summary are excluded, and a cluster without summaries is skipped.

When you request `made_errors`, its successful answers determine error share instead. The share divides true answers by answers that did not fail; missing or failed answers are excluded, and a cluster without any successful answers is skipped.

The **3D Map** tab shows each discovered dimension in its own UMAP projection. Choose **Projection** to change the coordinates, then **Colour by** to compare that shape with another cluster dimension, a fixed label, agent, or project. The colour selection does not move points. Click **Full screen** to explore the map across the window, and select a point to inspect its trace summary and cluster. UMAP needs at least five traces in a clustering group; smaller groups have no coordinates, and a dimension with no coordinates shows an empty state with the reason.

To start a run from the dashboard, click **+ New Run**. Choose recent traces, a semantic query, or a Finder JSON export stored on the dashboard machine. Then select built-in labels and discovered dimensions, review the expected stages, and click **Start run**. The dashboard opens the new run immediately. The overview shows an active run's current stage; its page shows every stage as pending, running, completed, error, or skipped and refreshes while the run is active. The stages reflect your source, labels, and dimensions; each dimension stage embeds, clusters, and maps its traces. The worker continues if the dashboard is restarted. For more filters and custom labels, use `eq insights` or the Python API.

The dashboard uses the Orq profile selected in **Settings** for both facet values and the run worker, including the profile's API key and host. With **Environment** selected, it uses `ORQ_API_KEY` and `ORQ_BASE_URL`. If a saved profile is unavailable, the wizard cannot start a run until you select another profile or Environment. `eq insights` also reads the saved profile and accepts `--profile` to override it. The Python API uses environment credentials unless you pass clients explicitly.

For recent traces or a semantic query, set the window and choose any project, agent, model, provider, status, product, trace type, or tool values shown under **Filter by facets**. The values come from Orq for that window and reload when you change it. Multiple values within one facet include any matching value; different facets must all match. Your choices are shown on the review step and applied before the trace limit. A Finder export already fixes its population, so the wizard does not add facet filters to that source. If Orq cannot provide the facet list, the wizard says so; check the connection before starting a run that needs filters.

Each run has a state file under `.evaluatorq/insights-runs/.manifests/<run-id>.json` and a report in `.evaluatorq/insights-runs/`. The state file records the stage plan, current stage, outcomes, errors, and report path. If a run fails before writing its report, it still appears in the dashboard with the failed stage and error. The worker log is under `.evaluatorq/insights-runs/.logs/<run-id>.log`.

The JSON report stores provider usage and cost by stage in `cost_by_stage`. A missing stage key means no provider calls ran; a `null` value means calls ran but the provider returned no usage. If only some calls have a model catalogue price, the run header says “priced for N of M calls” and the displayed dollar amount is a lower bound. “Cost unknown” means calls ran but none had a usable price.

On the run overview, stage dots show each stage's status and a count shows how many are complete. The run page shows the current stage and completed stages out of the plan. Label and Summary stages also show completed traces out of the total, such as `running 27/100`; those counts update while the run is active and remain in the saved manifest.

For the finder’s **Analyze matches** handoff, see [Trace Finder](trace-finder.md).

## Failures and cache

A per-trace label, summary, or dimension failure is recorded on that trace and counted as failed; it does not discard the rest of the run. When a dimension has at least five signal traces overall, a sentiment group with one trace is marked unclassified for that dimension. Sentiment groups with two to four traces form one cluster, and Insights records a warning because UMAP coordinates are unavailable below five traces. A dimension with fewer than five signal traces is skipped with a warning. An embedding batch failure marks that discovered-dimension stage as failed. Any whole-stage failure marks the run `error` and preserves partial results. The CLI exits with status 1 for an `error` run and status 0 for a completed run, including one with failed traces. Check the failed count and stage message before treating a result as complete. An empty population completes with a warning and empty results.

If every summary request fails, Insights marks the summary stage failed and sets the run status to `error`.

Placeholder answers such as “no errors” mean there is no failure text to cluster, so the trace has no signal for the failure dimension.

By default, summaries and embeddings are cached per item in `.evaluatorq/cache/insights.sqlite`. Summary cache keys include the trace, span, summary model, and prompt hash; embedding keys include the model and text hash. This cache is local to the working directory. Pass `cache=False` to Python `insights()` or `--no-cache` to the CLI to bypass it.

The `insights` extra installs numpy, scipy, and umap-learn for vector processing, clustering, and the 3D map; the embedding requests go through the Orq router. Add the `dashboard` extra to the Insights extra to review saved runs locally with `eq dashboard`.
