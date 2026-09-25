# Trace Insights

**Trace Insights** selects a population of Orq traces, answers fixed label questions, and discovers groups in the traces by clustering summary text. Use it when you want to review patterns in filtered trace traffic; use the trace finder alone when you only need to locate individual matches.

## What a run produces

An Insights run keeps three different things separate. A **classifier** is the model that answers fixed-form questions about each trace and, when you provide a semantic query, decides whether the trace matches it.

| Result | How it is chosen | What it contains |
|---|---|---|
| Population | Query, metadata facets, numeric bounds, time window, and limit | The traces included in the run. Labels never change population membership. |
| Labels | Fixed classifier questions | One answer per trace for each yes/no (`noul`), choice, or score question, with confidence and probabilities. |
| Discovered dimensions | Text fields from a per-trace summary | Two-level clusters whose categories are named after the traces have been embedded and grouped. |

The built-in discovered dimensions are `intent` (summary field `request`), `failure` (`assistant_errors`), and `sentiment` (`sentiment_explanation`). The summary model writes these text fields. The classifier answers labels such as sentiment, customer satisfaction, or your own questions. When you request the discovered `sentiment` dimension without a sentiment label, Insights adds that label so it can group the explanations by sentiment.

The run writes its JSON result to `.evaluatorq/insights-runs/` and its manifest under `.evaluatorq/insights-runs/.manifests/`; `EVALUATORQ_DIR` changes the base directory. It does not store message bodies or embedding vectors in the run JSON.

## Run from Python

Call `insights()` with an `InsightsPopulation` and optional labels and discovered dimensions. This example restricts the population to one agent and adds a custom label alongside two presets. It reads live traces and makes model requests through Orq, so set `ORQ_API_KEY` or use an active Orq CLI profile before running it.

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

The `eq insights` command accepts trace population filters, label presets or JSON files, and discovered dimensions. Install `evaluatorq[insights]` and set `ORQ_API_KEY` or use an active Orq CLI profile before starting a run. With `--query`, Insights asks the classifier to match traces and removes non-matches. Requested labels are sent with the same per-trace request, and traces whose match judgment fails remain in the run with an error. Filter-only runs and `--from-finder` runs do not ask a new match question.

```bash
eq insights --query "customers asking about refunds" --agent support-bot --label sentiment --label customer_satisfaction --dimension intent --dimension failure --limit 500 --json refunds-insights.json
```

The command prints cluster and label summaries, failed-trace counts, and the stored run path. Use `--from-finder PATH` to use matched trace IDs from a finder export instead of `--query`; the command rejects other population filters, `--window-days`, and `--limit` alongside it. `--no-cache` disables the local summary and embedding cache. The classifier and summary model options default to `typesafe/jev-latest` and `openai/gpt-6-luna` respectively.

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

Start the dashboard with the `dashboard` and `insights` extras, then open **Insights** in the sidebar. The page reads saved runs and does not start new ones. Select a run to inspect discovered dimensions as a cluster tree or 3D map, review label distributions and confidence, compare dimensions or labels in a crosstab, filter the trace table, or open a trace in Orq. The priority matrix is available when the run includes the customer-satisfaction label; otherwise the page explains why it is empty.

For the finder’s **Analyze matches** handoff, see [Trace Finder](trace-finder.md).

## Failures and cache

A per-trace label, summary, or dimension failure is recorded on that trace and counted as failed; it does not discard the rest of the run. When a dimension has at least five signal traces overall, a sentiment group with one trace is marked unclassified for that dimension. Sentiment groups with two to four traces form one cluster, and Insights records a warning because UMAP coordinates are unavailable below five traces. A dimension with fewer than five signal traces is skipped with a warning. An embedding batch failure marks that discovered-dimension stage as failed. Any whole-stage failure marks the run `error` and preserves partial results. The CLI exits with status 1 for an `error` run and status 0 for a completed run, including one with failed traces. Check the failed count and stage message before treating a result as complete. An empty population completes with a warning and empty results.

By default, summaries and embeddings are cached per item in `.evaluatorq/cache/insights.sqlite`. Summary cache keys include the trace, span, summary model, and prompt hash; embedding keys include the model and text hash. This cache is local to the working directory. Pass `cache=False` to Python `insights()` or `--no-cache` to the CLI to bypass it.

The `insights` extra installs numpy, scipy, and umap-learn for vector processing, clustering, and the 3D map; the embedding requests go through the Orq router. Add the `dashboard` extra to the Insights extra to review saved runs locally with `eq dashboard`.
