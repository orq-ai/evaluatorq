# Insights Activity view design

## Purpose and scope

The Activity view shows how tools, skills, and shell commands appear across the traces in one Insights run. It replaces the Priority scatter view in the redesign mock. It helps a reviewer discover common, rare, repeated, and co-occurring activity, then open the traces behind a count. It does not combine saved runs or claim that an activity caused an outcome.

## Data and counting

Each trace may have a `tool_stats` record with maps of tool names to call counts, skill names to load counts, and normalized first shell commands to call counts. Tools exclude the `Skill` tool; shell command calls are already included in shell tool calls, so category totals must never be added together. A skill load means the `Skill` tool recorded that skill, not that the skill ran successfully. A command name is the first normalized program or program/subcommand from a shell call, not the complete command line.

For each item, show `traces using` as the number and rounded percentage of eligible traces with a positive count. Show total calls for tools and shell commands because repeat calls add information; omit total loads for skills because their loads largely repeat trace coverage. An eligible trace has a `tool_stats` record, including an empty one. Show the eligible trace denominator and the number of traces missing tool stats. Missing stats are unknown and never count as zero use. The view starts with all eligible traces in the current run and recomputes every number for the current run filters.

## Page structure

Replace the Priority navigation item with Activity and retain the current run's page shell, filters, URL state, and trace detail drawer. The Activity view opens on Skills, with Tools and Shell commands as adjacent sections. Skills show summary counts for distinct items and traces with any recorded use; Tools and Shell commands also show recorded calls. A searchable ranking shows item name, traces using, its percentage, and a coverage bar; Tools and Shell commands also show total calls. Skills sort by trace coverage, while Tools and Shell commands can also sort by total calls. Selecting an item opens its detail panel and the matching trace list; selecting a trace opens the existing trace detail.

The ranking shows a horizontal bar under each numeric column. The `Traces using` bar and adjacent percentage show the share of eligible traces. For Tools and Shell commands, the `Total calls` bar is relative to the largest visible item in that category. Keep the exact counts visible and explain the scales beside the ranking. Shell command program rows use the same bars, with their call bars compared to the largest visible program; expanded command rows compare with the largest visible command. Search and helper visibility update the relative call scale. Cap the desktop results scroll region at 460px and keep its header sticky; the summary, search, and selected-item detail remain outside it. Preserve scroll position and keyboard focus when selecting a row. At widths of 600px or less, let the page scroll normally and initially show ten Skills or Tools rows, with a Show all/Show fewer control; a selected row remains visible even beyond the first ten. Label each metric on mobile, where the column header is hidden. Shell commands remain grouped and collapsed unless opened or matched by search. Stack the detail panel beneath the ranking, place the Skills coverage bar below its name, and put the two bars for Tools and Shell commands side by side below each name.

The Shell commands section groups normalized commands by program, with expandable individual commands. Hide routine helpers by default behind a visible toggle; search still finds them. Mark commands that might change state with a cautious label such as `Potentially state-changing`. That marker is a heuristic on the normalized command name, not a verdict about what happened in the trace.

## Selected-item detail

Show a weekly chart for the selected item's trace coverage. Each week displays both traces using the item and eligible traces observed that week, so a week with little data cannot resemble a strong trend. Use the trace timestamp and a consistent UTC week boundary. Do not draw a trend when the run has only one populated week; show the week's count instead.

Show `Often together` lists for items in the same category and in the other categories. Rank by the number of traces containing both items and show the top five in each category. Show each overlap count alongside the selected item's trace count and the rounded percentage of selected-item traces, and allow each partner to be selected. Keep the count and percentage together when long item names wrap. Exclude a shell tool's mechanically nested relationship with its own shell commands so `Bash` does not dominate every command's cross-category list. These are co-occurrences within a trace; the stored maps cannot reveal order, arguments, or success of individual calls.

## Empty and degraded states

If no eligible traces have tool stats, explain that activity data was not recorded. If a section has no recorded items, say so; if active filters remove all its items, say that no items match the filters. If a selected item has no co-occurrences, show an empty message. Show the count of missing tool-stat records when positive; omit the zero badge rather than treating missing records as unused traces.

## Verification

Check aggregation with mixed positive, empty, and missing `tool_stats` records, including the denominator after filtering. Check weekly bins across a week boundary and sparse weeks, co-occurrence within and across categories, the five-item cap, and exclusion of trivial shell-tool pairings. Check desktop scrolling, mobile Show all/Show fewer behavior, search, sorting, URL restoration, row selection, helper toggling, and empty states in the rendered mock. Keep public documentation updates for the later production implementation, when the Activity view becomes a shipped surface.
