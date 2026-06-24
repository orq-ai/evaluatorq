## What

Adds `llm_jury()` — an LLM-as-a-judge / LLM-as-a-jury evaluator that drops into the `evaluators=[...]` list of `evaluatorq()`, exposing the full judge panel surface that previously only `red_team()` could reach.

```python
from evaluatorq import evaluatorq, llm_jury

correctness = llm_jury(
    name="correctness",
    criteria="Rate the answer's factual correctness vs. the expected output.",
    judges=["openai/gpt-5.5", "anthropic/claude-opus-4-8", "google/gemini-3.5-flash"],
    min_successful_judges=2,
    labels=["correct", "partially_correct", "incorrect"],
    passing_labels=["correct", "partially_correct"],
    aggregator="majority",  # default is "mode" (plurality)
)
await evaluatorq("run", data=[...], jobs=[...], evaluators=[correctness])
```

## How

- **`execute_chat_parse`** (`common/llm_call.py`) — sibling of `execute_chat_completion` using `client.chat.completions.parse` (real structured outputs). The shared `.create` path is untouched.
- **`run_judge`** (`common/judge.py`) gains additive `response_model` / `structured_output` / sentinel `temperature` params. **Tier 1:** `.parse` with a dynamic Pydantic verdict model. **Tier 2 fallback:** `json_object` + schema injection on `BadRequestError` (gated on the error body). `response_model=None` (the default) is byte-identical — red-team and simulation callers are unchanged.
- **`llm_jury.py`** — factory + validation + per-datapoint scorer; reuses the existing `run_jury` aggregation engine. No `redteam` import.

## Surface

`judges`/`model`, `repetitions`, `replacement_judges`, `min_successful_judges`, `strict_panel` (reserved no-op), `verdict_kind` (categorical + numeric), `labels`/`passing_labels`, `threshold`/`score_range`, `aggregator` (keyword or custom callable — see below), `tie_break`, `structured_output`, `temperature` (None-omit, warns at 0.0), `max_tokens=8000`, `extra_kwargs`, `client` (auto-resolved).

## Aggregation

`aggregator=` selects the panel consensus rule, validated against `verdict_kind` (a mismatch raises `ValueError`):

- **categorical** — `"mode"` (default: most common; plurality ties go to `tie_break`) or `"majority"` (strict >50%, else inconclusive)
- **numeric** — `"mean_std"` (default: mean verdict; std reported in `stats` on a conclusive verdict), `"median"`, `"min"`, `"max"`
- **custom** — `Callable[[list[JuryVote]], bool | float | str | None]` for either kind. Receives **all** votes (including abstained/failed) so it can quorum or weight; returns `None` for "no consensus" (inconclusive). The numeric keyword also collapses a single judge's `repetitions`.

Built-in keywords live in a single keyword→fn registry conforming to the same `Aggregator` schema custom callables use; one `_AGG_KIND` map is the source of truth for the keyword↔kind partition.

## Verdict mapping

Categorical → boolean/label per the chosen aggregator; numeric → aggregated per `aggregator`, clamped to `score_range`, `pass = score >= threshold`. Inconclusive (below `min_successful_judges`, or a custom aggregator returning `None`) → `pass=None`. Refusal → abstain.

## Notes

- Probe confirmed `.parse` works through the Orq router for all three default models.
- `openai` floor bumped to `>=1.92.0` (where `.parse` left beta); it is now a **core** dependency (no `[jury]` extra).
- Responses-API support for judges deferred to **RES-972**; per-judge OTEL span attributes to **RES-985**.
- New unit tests cover every aggregator strategy, the custom-callable contract (sees all votes; `None`→inconclusive), the strict-majority 50% boundary, repetition collapse, and registry/partition/literal parity. Full suite green (9 pre-existing `huggingface_hub` failures unrelated).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
