# Simulation agent-card icon design

## Goal

Make the agent-under-test card on the Agent Sim report overview easier to identify at a glance.

## Design

Reuse the existing outline bot SVG from the runs overview's target-kind icon set. Render it immediately before the agent name in the simulation agent-card header.

The icon is decorative: the adjacent agent name remains the accessible label. It uses the existing teal accent and a compact size aligned with the 18px agent-name text. The card's model, description, capability groups, links, and responsive behavior remain unchanged.

## Implementation boundary

Keep the SVG owned by the dashboard presentation layer, avoid adding a new external icon dependency, and add a focused renderer assertion that the agent card contains the icon hook.

## Validation

Run the focused dashboard report-tab tests and inspect the rendered HTML for the icon hook and decorative accessibility attribute.
