# Agent Simulation Metric Filters

## Goal

Extend the Agent Simulation report rail with low-dimensional, actionable filters without introducing arbitrary rule or criterion names as dashboard dimensions.

## Scope

The rail keeps the existing Persona, Scenario, Goal Outcome, and Terminated By controls. It adds:

- an **Any rule broken** boolean chip;
- a minimum **Goal score** slider (0.00–1.00);
- a minimum **Turns** slider (1 through the run maximum);
- a minimum **Total tokens** slider (0 through the run maximum);
- four per-turn threshold sliders in a **More filters** expander:
  - minimum hallucination risk, applied to the maximum risk seen in a run;
  - maximum response-quality score, applied to the minimum quality seen in a run;
  - maximum tone-appropriateness score, applied to the minimum tone score seen in a run;
  - maximum factual-accuracy score, applied to the minimum accuracy seen in a run.

Named rules and failed criteria are deliberately not filters. Rule names are defined by simulation configuration and have unbounded cardinality; criterion labels have the same problem. The boolean rule-violation filter captures the useful question without making the rail dependent on a long, unstable list of labels.

## Behaviour

All controls use the complete run to determine their range and remain visible after filtering. A default threshold is non-restrictive: goal score 0, turns 1, tokens 0, hallucination risk 0, and the three quality ceilings 1.0.

Per-turn scores are aggregated conservatively for discovery: a run matches a risk threshold when any scored turn reaches it, and matches a quality threshold when any scored turn falls at or below it. A run without a score for a selected per-turn metric remains visible; missing scoring must not silently remove older or partially scored runs.

The new controls participate in the existing HTMX form POST and recompute all report tabs from the filtered result set. The More filters expander stays open across swaps and follows the existing one-dropdown-at-a-time behavior for nested multi-selects.

## Presentation

The primary rail shows Rule adherence, Goal Outcome, Terminated By, Goal score, Turns, and Persona/Scenario. Total tokens and the four per-turn metric controls live beneath the small, text-style More filters expander so normal runs stay scannable. Slider readouts express their direction, for example `≥ 0.70`, `≤ 0.40`, and `all` for a non-restrictive value.

## Testing

Tests cover each filter’s predicate, the conservative per-turn aggregation, missing score handling, dynamic slider bounds, checked/default states, and the HTMX rendering path. Existing filter tests continue to verify fixed full-run option lists.
