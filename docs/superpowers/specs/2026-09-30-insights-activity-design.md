# Insights Activity view design

## Purpose and scope

The Activity view shows how tools, skills, and shell commands appear across the traces in one Insights run. It replaces the Priority scatter view in the redesign mock. It helps a reviewer discover common, rare, repeated, and co-occurring activity, then open the traces behind a count. It does not combine saved runs or claim that an activity caused an outcome.

## Data and counting

Each trace may have a `tool_stats` record with maps of tool names to call counts, skill names to load counts, and normalized first shell commands to call counts. Tools exclude the `Skill` tool; shell command calls are already included in shell tool calls, so category totals must never be added together. A skill load means the `Skill` tool recorded that skill, not that the skill ran successfully. A command name is the first normalized program or program/subcommand from a shell call, not the complete command line.

For each item, show `traces using` as the number of eligible traces with a positive count and `total calls` or `total loads` as the sum of counts. An eligible trace has a `tool_stats` record, including an empty one. Show the eligible trace denominator and the number of traces missing tool stats. Missing stats are unknown and never count as zero use. The view starts with all eligible traces in the current run and recomputes every number for the current run filters.

## Page structure

Replace the Priority navigation item with Activity and retain the current run's page shell, filters, URL state, and trace detail drawer. The Activity view opens on Skills, with Tools and Shell commands as adjacent sections. Each section shows summary counts for distinct items, recorded loads or calls, and traces with any recorded use. A searchable ranking shows item name, traces using, total loads or calls, and a coverage bar. Sort by trace coverage by default, with a total-use sort. Selecting an item opens its detail panel and the matching trace list; selecting a trace opens the existing trace detail.

The ranking shows a horizontal bar under both numeric columns. The `Traces using` bar shows the share of eligible traces, and the `Total loads/calls` bar is relative to the largest visible item in that category. Keep the exact counts visible and explain both scales beside the ranking. Shell command program rows use the same bars, with their call bars compared to the largest visible program; expanded command rows compare with the largest visible command. Search and helper visibility update the relative call scale. At narrow widths, stack the detail panel beneath the ranking and put each item's two bars side by side below its name.

The Shell commands section groups normalized commands by program, with expandable individual commands. Hide routine helpers by default behind a visible toggle; search still finds them. Mark commands that might change state with a cautious label such as `Potentially state-changing`. That marker is a heuristic on the normalized command name, not a verdict about what happened in the trace.

## Selected-item detail

Show a weekly chart for the selected item's trace coverage. Each week displays both traces using the item and eligible traces observed that week, so a week with little data cannot resemble a strong trend. Use the trace timestamp and a consistent UTC week boundary. Do not draw a trend when the run has only one populated week; show the week's count instead.

Show `Often together` lists for items in the same category and in the other categories. Rank by the number of traces containing both items, show that count alongside the selected item's trace count and the rounded percentage of selected-item traces, and allow each partner to be selected. Keep the count and percentage together when long item names wrap. Exclude a shell tool's mechanically nested relationship with its own shell commands so `Bash` does not dominate every command's cross-category list. These are co-occurrences within a trace; the stored maps cannot reveal order, arguments, or success of individual calls.

## Empty and degraded states

If no eligible traces have tool stats, explain that activity data was not recorded. If a section has no recorded items, say so; if active filters remove all its items, say that no items match the filters. If a selected item has no co-occurrences, show an empty message. Show the count of missing tool-stat records rather than treating them as unused traces.

## Verification

Check aggregation with mixed positive, empty, and missing `tool_stats` records, including the denominator after filtering. Check weekly bins across a week boundary and sparse weeks, co-occurrence within and across categories, and exclusion of trivial shell-tool pairings. Check search, sorting, URL restoration, row selection, helper toggling, and empty states in the rendered mock. Keep public documentation updates for the later production implementation, when the Activity view becomes a shipped surface.
