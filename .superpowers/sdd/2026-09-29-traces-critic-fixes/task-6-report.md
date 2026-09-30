# Task 6 report: first-use copy and jargon

Final copy strings:
- Lede: "Recent traces from your agents, newest first: ask a question in plain words, or filter the list below."
- Placeholder: "Ask a question, e.g. Did any customers get frustrated?"
- Example chips: "Did any customers get frustrated?" / "Which conversations mention refunds?" / "Where did a failed tool call leave the user stuck?"
- Scope: label "Search in", options "Loaded traces" / "All traces".
- Helper (changes with scope): "Loaded traces: Ask AI reads only the traces already loaded below, up to the row limit." / "All traces: Ask AI turns your question into filters and searches the whole time range again, then loads the matches." followed by "It uses a model, and each question reads the traces it covers."
- Settings link: "AI settings" (gear plus text), title "Choose the models Ask AI uses".
- Trajectories title: "Shows each trace as a bar of its messages, sized by estimated tokens"; legend note "Bar sizes are estimates (text length ÷ 4), not exact token counts."
- Cache read %: "Share of input tokens served from the provider's prompt cache". Tokens in: "Tokens sent to the model, including any served from the cache". Tokens out: "Tokens the model wrote back". Totals strip cells carry the same titles plus p50/p95 explanations. Visible "?" on the titled headers.
- Status line under a narrowing chip: "0 of 200 loaded traces" (was "200 traces loaded").
- Duplicate card h2 "Traces" removed. Result-line separators removed (parts wrap via gap; .sep hidden on /traces) so no dot dangles at a wrap.

Validation: screenshots .context/critics/after-t6-{1-first,2-chip,5-errors,6-900}.png; chip fills box without submitting; titles read via eval; helper switches; Errors chip shows "0 of 200 loaded traces" and strip "traces 0".
Note: scope labels do not include N or the day window (strip only knows has_rows); the helper text explains scope instead.

## Fix round (t6fix)

- Trajectories "?" wrap: `.xr-switch button` now inline-flex, align-items center, nowrap (/traces only). Verified at 1280px in browser.
- /find separators: `_progress_parts` again emits `<span class="sep">·</span>` before every part, inside a `.part` wrapper (display:contents, so layout on /find equals the old flat siblings). The `.sep` is hidden only under `.finder:has(> .finder-command)` (/traces). Unit test added. /find idle in browser has no progress line, so verified by test on the HTML.
- Copy: the four strings applied verbatim (cost note, chip, All-traces helper, cache-read title).
- 900px: lede/helper fonts shrink and the chips row scrolls horizontally (nowrap) under 1000px.
- Screenshots: .context/critics/after-t6fix-*.png (port 8171; 8161 was occupied by another session).
