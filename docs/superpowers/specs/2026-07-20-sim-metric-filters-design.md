# Agent Simulation Metric Filters

## Goal

Extend the Agent Simulation report rail with low-dimensional, actionable filters without adding semantically incomparable criterion IDs as dashboard dimensions.

## Scope

The rail keeps the existing Persona, Scenario, Goal Outcome, and Terminated By controls. It adds:

- an **Any rule broken** boolean chip;
- a maximum **Goal score** slider (0.00–1.00), for finding incomplete outcomes;
- a minimum **Turns** slider (1 through the run maximum);
- a minimum **Total tokens** slider (0 through the observed run maximum, in raw token counts);
- four per-turn threshold sliders in a **More filters** expander:
  - minimum hallucination risk, applied to the maximum risk seen in a run;
  - maximum response-quality score, applied to the minimum quality seen in a run;
  - maximum tone-appropriateness score, applied to the minimum tone score seen in a run;
  - maximum factual-accuracy score, applied to the minimum accuracy seen in a run.

Named rules and failed criteria are deliberately not filters. The judge emits low-cardinality positional IDs (for example, ``criteria_0``), but those IDs refer to different criterion text in different scenarios. A filter for ``criteria_0`` would therefore group unrelated failures. The boolean rule-violation filter captures the useful question without creating that false comparison.

## Behaviour

All controls use the complete run to determine their range and remain visible after filtering. Score bounds are fixed at 0.00–1.00; turns and tokens use their observed integer maximums without normalisation. A default threshold is non-restrictive: goal-score ceiling 1.0, turns 1, tokens 0, hallucination risk 0, and the three quality ceilings 1.0. Turn and token sliders use their raw units and a step of 1.

Per-turn scores are aggregated conservatively for discovery: a run matches a risk threshold when any scored turn reaches it, and matches a quality threshold when any scored turn falls at or below it. Existing turn-quality reporting already represents unmeasured values as gaps, so filtering preserves that policy: a run with no usable score for a selected metric remains visible rather than being misclassified as a pass or failure. A partially scored run aggregates only its numeric turn values. Token usage has no separate unknown state because the persisted model represents absent usage as zero.

The new controls participate in the existing HTMX form POST and recompute all report tabs from the filtered result set. The More filters expander reuses the existing ``filter-dd-more`` state persistence. Numeric values are submitted as one-element form lists, parsed defensively, and clamped to their fixed or observed-run bounds. The unchecked ``Any rule broken`` chip submits no value and means no rule filter; its only submitted value is ``yes``.

## Presentation

The primary rail shows Rule adherence, Goal Outcome, Terminated By, Goal score, Turns, and Persona/Scenario. Total tokens and all four per-turn metric controls live beneath the small, text-style More filters expander so normal runs stay scannable. Slider readouts express their direction, for example `≥ 0.70`, `≤ 0.40`, and `all` for a non-restrictive value. A metric control is omitted when the run has no scores for that metric, matching the existing Turn quality tab's no-data behavior.

## Testing

Tests cover each filter’s predicate, conservative per-turn aggregation, missing score handling, raw-unit bounds, bounds clamping, checked/default states, and the HTMX rendering path. A shared turn-metric descriptor (key, label, and whether high values are risky) supplies the four controls and the existing report renderers, avoiding another hard-coded metric list. Existing filter tests continue to verify fixed full-run option lists.
