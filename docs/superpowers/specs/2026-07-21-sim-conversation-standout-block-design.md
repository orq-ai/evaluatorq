# Simulation conversation standout block

## Goal

Give the conversation portion of a simulation transcript drawer the same calm,
contained visual treatment as the dashboard's established standout/callout
surfaces.

## Scope

Only `.sim-transcript-bubbles` changes. The conversation messages remain in
their current order, alignment, role-specific tinting, and avatar treatment.
The summary header, criteria block, judge rationale, table rows, and error
state are unchanged.

## Design

The conversation message list becomes one rounded container with:

- `var(--surface-sunken)` as its light background;
- a `var(--border-subtle)` hairline border;
- the existing shared medium/large radius token;
- interior padding that keeps the first and last messages clear of the edge;
- no extra bottom margin on the final message, so spacing is balanced within
  the container.

The surrounding criteria-to-conversation divider remains the structural break
between outcome and evidence. Individual user and agent bubbles retain their
current surface treatment, ensuring roles remain easy to scan inside the new
transcript boundary.

## Implementation and verification

Add the scoped CSS rules in `_SIM_TRANSCRIPT_OVERRIDES_CSS` after the current
sim message rules. Extend the dashboard transcript rendering tests only if
they assert the exact class structure; the HTML structure itself need not
change. Run the targeted dashboard transcript tests and the dashboard test
suite relevant to styles/rendering.

## Non-goals

This does not introduce a new reusable component, alter data flow, change
transcript copy, or restyle Red Team transcripts.
