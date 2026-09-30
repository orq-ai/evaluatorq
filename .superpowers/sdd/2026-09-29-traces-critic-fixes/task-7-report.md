# Task 7 report: toolbar hierarchy and Filters menu

- Search (Ask AI row) and Load (toolbar) are the only solid buttons. Every other control is a 32px ghost with a 6px radius. Table/Trajectories and Search-in are a quiet grey track with a white active chip.
- Filters popover is an opaque panel (white, 1px border, layered shadow, z-index 60).
- Facet counts: `facet_menu(..., loaded_rows=)` tallies model, provider, status, agent and product from the already-loaded rows (no new API calls), sorts most common first, and says "Counts are of the N loaded traces." When Orq's catalogue is unavailable it lists the values seen in the loaded rows. `/find/facets?counts=loaded` keeps counts after the self-load. /find is unchanged (no rows passed). Project, trace type and tool have no counts (rows do not carry them).
- Toolbar stability: the Ask AI result band, criteria and progress now render under the toolbar (display:contents plus order), and the right control group is narrower so the "AI matches" tab no longer wraps it. Measured toolbar y before and after one Ask AI run: 244 -> 244 (Loaded traces) and 260 -> 260 (All traces). Active filter chips sit on their own row under the controls.
- 900px: question box on a full row; Search in / AI settings / Search share row 2; Filters + tabs on toolbar row 1; time, Rows, Columns, Load, switch on row 2. No lone control.
- Screenshots: .context/critics/after-t7-{1280,filters,narrowed,ask,900,find}.png.
- Note: Orq returned 503 "no healthy upstream" on several loads mid-run; retrying Load worked.
