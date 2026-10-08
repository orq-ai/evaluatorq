"""Dashboard chrome CSS.

``load_css()`` (``common.reports``) styles the *report body* and defines the
brand ``:root`` tokens.  ``theme.EDITORIAL_CSS`` supplies the v1 editorial-skin
tokens.  This module supplies the chrome that consumes them: the sidebar shell,
the topbar, the combined landing, the per-kind run lists, and the report-view
filter/body split.

Placed last in ``shell.dashboard_css()`` so its rules win on
equal specificity and all ``var(--…)`` references resolve.
"""

from __future__ import annotations

_TRACES_DENSITY_CSS = """
/* /traces density pass. The command strip exists only on /traces; /find keeps its search hero. */
.finder:not(:has(> .finder-command)) { max-width:1040px; }
.finder:has(> .finder-command) { gap:0; max-width:none; margin:0 auto; overflow:hidden; background:#fff; color:#25232e; }
.finder:has(> .finder-command) .finder-command { align-self:stretch; max-width:none; padding:13px 16px 0; border:0; border-radius:0; background:#fff; box-shadow:none; }
.finder:has(> .finder-command) .finder-command-query { gap:8px; min-height:53px; padding:5px 8px; border:0; border-top:1px solid #e5e6e9; border-bottom:1px solid #e5e6e9; border-radius:0; box-shadow:none; }
.finder:has(> .finder-command) .finder-command-query:focus-within { box-shadow:none; outline:none; border-top-color:#025558; border-bottom-color:#025558; }
.finder:has(> .finder-command) .finder-ai-label { display:inline-flex; align-items:center; white-space:nowrap; padding:8px 10px; border:1px solid #f0cb9f; border-radius:6px; background:#fff4e7; color:#9a5f20; font-size:12px; font-weight:750; }
.finder:has(> .finder-command) .finder-command-query .col { flex:1; min-width:180px; }
.finder:has(> .finder-command) .finder-command-textarea { max-height:4em; padding:8px 4px; font-family:var(--font-sans); font-size:13px; line-height:1.35; }
.finder:has(> .finder-command) .finder-command-query > a { display:inline-flex; align-items:center; gap:5px; flex:0 0 auto; height:42px; min-height:42px!important; padding:0 10px; white-space:nowrap; box-sizing:border-box; border:1px solid #d9dbe0; border-radius:6px; color:#494753; font-size:12px; font-weight:500; line-height:1; text-decoration:none; }
.finder:has(> .finder-command) .finder-command-query > button { flex:0 0 94px; min-height:38px; border-radius:6px; font-size:12px; }
.finder:has(> .finder-command) #finder-body { gap:0; }
.finder:has(> .finder-command) #explorer-results-slot { min-width:0; }
.finder:has(> .finder-command) .xr { gap:0; }
.finder:has(> .finder-command) .xr-toolbar { min-height:49px; flex-wrap:wrap; gap:8px; padding:0 12px; border-bottom:1px solid #dfe1e5; }
.finder:has(> .finder-command) .xr-toolbar-right { grid-template-columns:128px 86px 94px 57px 139px; gap:8px; margin-left:auto; }
.finder:has(> .finder-command) .xr-toolbar-right > * { min-width:0; }
.finder:has(> .finder-command) .xr-toolbar { order:-1; }
.finder:has(> .finder-command) .xr-status { order:0; min-height:32px; padding:0 16px; border-bottom:1px solid #ebebed; font-size:11px; }
.finder:has(> .finder-command) .xr-totals { order:0; padding:6px 16px; border-bottom-color:#ebebed; }
.finder:has(> .finder-command) .xr-totals { align-items:center; gap:4px 17px; min-height:34px; padding:4px 16px; }
.finder:has(> .finder-command) .xr-total { display:inline-flex; flex:0 0 auto; align-items:center; gap:5px; min-width:0; padding:2px 0; border:0; background:transparent; }
.finder:has(> .finder-command) .xr-total-mark { display:grid; flex:none; place-items:center; width:17px; height:17px; border-radius:4px; background:#edf0f5; color:#586783; font-size:11px; font-weight:700; }
.finder:has(> .finder-command) .xr-total-copy { display:flex; min-width:0; flex-direction:row; align-items:baseline; gap:4px; }
.finder:has(> .finder-command) .xr-total-label { color:#696d78; font-size:10px; font-weight:550; line-height:1.1; }
.finder:has(> .finder-command) .xr-total-value { color:#25232e; font-size:12px; font-weight:700; line-height:1.2; font-variant-numeric:tabular-nums; white-space:nowrap; }
.finder:has(> .finder-command) .xr-total-errors .xr-total-mark { background:#fcefed; color:#ad4039; }
.finder:has(> .finder-command) .xr-total-errors.has-errors .xr-total-value { color:#ad302a; }
.finder:has(> .finder-command) .xr-total-cost .xr-total-mark { background:#eaf3ec; color:#286746; }
.finder:has(> .finder-command) .xr-total-in .xr-total-mark { background:#edf2fb; color:#3a5e9a; }
.finder:has(> .finder-command) .xr-total-out .xr-total-mark { background:#f0ecf8; color:#6a4fa2; }
.finder:has(> .finder-command) .xr-total-cache-read .xr-total-mark { background:#e8f3f0; color:#287665; }
.finder:has(> .finder-command) .xr-total-p50 .xr-total-mark,
.finder:has(> .finder-command) .xr-total-p95 .xr-total-mark { background:#fff2e4; color:#995a20; }
.finder:has(> .finder-command) .xr-total.unavailable .xr-total-value { color:#858994; font-size:10px; font-weight:500; }
.finder:has(> .finder-command) .xr-time-menu > summary,
.finder:has(> .finder-command) .xr-toolbar-right > .quiet,
.finder:has(> .finder-command) .xr-cols > summary,
.finder:has(> .finder-command) .xr-sort > summary { display:flex; align-items:center; justify-content:center; box-sizing:border-box; min-height:31px; padding:5px 8px; border:1px solid #d5d8df; border-radius:6px; background:#fff; color:#4d4b56; font-size:12px; white-space:nowrap; }
.finder:has(> .finder-command) .xr-toolbar-right > .quiet { gap:5px; }
.finder:has(> .finder-command) .xr-toolbar-right > .quiet b { font-weight:500; }
.finder:has(> .finder-command) #explorer-rows { width:38px!important; border:0; background:transparent; color:#4d4b56; font:inherit; text-align:center; }
.finder:has(> .finder-command) #explorer-rows:focus:not(:focus-visible) { outline:none; box-shadow:none; }
.finder:has(> .finder-command) .xr-toolbar-right .btn-secondary { min-height:31px; padding:5px 9px; border-radius:6px; background:#25232e; border-color:#25232e; color:#fff; font-size:12px; }
.finder:has(> .finder-command) .xr-toolbar-right .finder-seg { justify-self:stretch; }
.finder:has(> .finder-command) .xr-table-wrap { position:relative; display:block; overflow:auto; max-height:calc(100vh - 160px); margin:0; padding:0; border:0; border-radius:6px; line-height:normal; -webkit-overflow-scrolling:touch; }
/* Many columns scroll sideways inside the card while the header stays visible. */
.finder:has(> .finder-command) .xr-table thead th { position:sticky; top:0; z-index:2; }
.finder:has(> .finder-command) .xr-table th:first-child, .finder:has(> .finder-command) .xr-table td:first-child { z-index:1; background:#fff; box-shadow:inset -1px 0 0 #e8e9ec; }
.finder:has(> .finder-command) .xr-table thead th:first-child { z-index:3; background:#f8f9fa; }
.finder:has(> .finder-command) .xr-table tbody tr:hover td:first-child { background:#f8faf8; }
.finder:has(> .finder-command) .xr-table tr.sel td:first-child { background:#f3f8f7; }
.finder:has(> .finder-command) .xr-table th:last-child, .finder:has(> .finder-command) .xr-table td:last-child { padding-right:18px; }
.finder:has(> .finder-command) .xr-table { width:max-content; table-layout:fixed; border-collapse:collapse; font-size:12px; overflow:visible; border:0; }
.finder:has(> .finder-command) .xr-table thead { display:table-header-group!important; }
.finder:has(> .finder-command) .xr-table tbody tr { display:table-row!important; height:auto!important; }
.finder:has(> .finder-command) .xr-table td { display:table-cell!important; }
.finder:has(> .finder-command) .xr-table th { padding:9px 11px; border-bottom:1px solid #dfe1e5; background:#f8f9fa; color:#686a74; font-size:10px; letter-spacing:.055em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.finder:has(> .finder-command) .xr-ai-sparkle { margin-right:4px; color:#a66124; font-size:12px; }
.finder:has(> .finder-command) .xr-table td { padding:7px 11px; border-bottom:1px solid #e8e9ec; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.finder:has(> .finder-command) .xr-table td.xr-duration { min-width:82px; }
.finder:has(> .finder-command) .xr-duration-bar { display:inline-block; width:36px; height:5px; margin-left:6px; overflow:hidden; vertical-align:middle; border-radius:3px; background:#e8ecec; }
.finder:has(> .finder-command) .xr-duration-bar > span { display:block; height:100%; border-radius:inherit; background:#559b91; }
.finder:has(> .finder-command) .xr-table td.xr-duration-p95 { background:#fff1e9; color:#8a3d20; }
.finder:has(> .finder-command) .xr-table tbody tr:hover td.xr-duration-p95 { background:#ffe8dc; }
.finder:has(> .finder-command) .xr-duration-accessible { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
.finder:has(> .finder-command) .xr-table tbody tr:hover { background:#f8faf8; }
.finder:has(> .finder-command) .xr-status ~ .xr-empty { margin:0 12px 12px; }
/* Square pass: every header control shares one height, square corners, centred labels and the card's 16px gutter. */
.finder:has(> .finder-command) { --ctl:34px; border-radius:0; }
.finder:has(> .finder-command) .finder-command-query { padding:8px 0; min-height:0; }
.finder:has(> .finder-command) .finder-ai-icon { display:none; }
.finder:has(> .finder-command) .finder-ai-label::before { content:'✦'; margin-right:6px; }
.finder:has(> .finder-command) .finder-command-query > *,
.finder:has(> .finder-command) .xr-toolbar > *,
.finder:has(> .finder-command) .xr-toolbar-right > * { align-self:center; }
.finder:has(> .finder-command) .finder-ai-label,
.finder:has(> .finder-command) .finder-command-query > a,
.finder:has(> .finder-command) .finder-command-query > button,
.finder:has(> .finder-command) .finder-seg,
.finder:has(> .finder-command) .xr-filter,
.finder:has(> .finder-command) .xr-time-menu > summary,
.finder:has(> .finder-command) .xr-toolbar-right > .quiet,
.finder:has(> .finder-command) .xr-cols > summary,
.finder:has(> .finder-command) .xr-sort > summary,
.finder:has(> .finder-command) .xr-toolbar-right .btn-secondary { display:flex; align-items:center; justify-content:center; box-sizing:border-box; height:var(--ctl); min-height:var(--ctl)!important; margin:0; padding-top:0; padding-bottom:0; border-radius:0; line-height:1; text-align:center; }
.finder:has(> .finder-command) .finder-command-query > a { flex:0 0 auto; width:auto; height:var(--ctl); font-size:12px; }
.finder:has(> .finder-command) .finder-command-query > a.finder-command-gear { flex:0 0 34px; width:34px; padding:0; }
.finder:has(> .finder-command) .finder-command-query > a > span { font-size:16px; }
.finder:has(> .finder-command) .finder-command-textarea { padding:9px 4px; }
.finder:has(> .finder-command) .finder-seg { display:inline-flex; padding:2px; align-items:stretch; }
.finder:has(> .finder-command) .finder-seg label { display:flex; flex:1 1 0; }
.finder:has(> .finder-command) .finder-seg label > span { flex:1; justify-content:center; height:auto; border-radius:0; }
.finder:has(> .finder-command) .xr-toolbar { padding:0 16px; }
.finder:has(> .finder-command) .xr-toolbar-right { grid-template-columns:172px 86px 94px 57px 180px; align-items:center; }
.finder:has(> .finder-command) .xr-quickviews > button { height:var(--ctl); padding:0 7px; white-space:nowrap; }
.finder:has(> .finder-command) .xr-chips .chip,
.finder:has(> .finder-command) .tv { border-radius:0; }
.finder:has(> .finder-command) .xr-switch button { min-width:0; height:auto; padding:0 8px; border-radius:0; justify-content:center; text-align:center; }
@media (max-width:850px) { .finder:has(> .finder-command) .finder-command-query > button { flex:1 1 100%; } }
/* Quick-view tabs carry the table's status colours: errors red, AI matches amber. */
.finder:has(> .finder-command) .xr-quickviews > .xr-qv-errors { color:#686a74; }
.finder:has(> .finder-command) .xr-quickviews > .xr-qv-errors.has-errors { color:#b3261e; font-weight:700; }
.finder:has(> .finder-command) .xr-quickviews > .xr-qv-errors.has-errors .xr-count { padding:0 5px; border-radius:8px; background:#b3261e; color:#fff; }
.finder:has(> .finder-command) .xr-quickviews > .xr-qv-errors.on { border-color:#b3261e; background:#fdf3f1; }
.finder:has(> .finder-command) .xr-quickviews > button:nth-child(3) { color:#94591f; }
.finder:has(> .finder-command) .xr-quickviews > button:nth-child(3).on { color:#a66124; border-color:#ed8844; background:#fff6ec; }
.finder:has(> .finder-command) .xr-quickviews > button:first-child.on { background:#f4f3f1; }
.finder:has(> .finder-command) .xr-qv-sep { flex:none; width:1px; height:18px; margin:0 4px; background:#d5d8df; }
.finder:has(> .finder-command) .xr-top-menu { position:relative; }
.finder:has(> .finder-command) .xr-top-menu > summary { display:flex; align-items:center; height:var(--ctl); padding:0 7px; list-style:none; cursor:pointer; white-space:nowrap; color:var(--text-muted); border-bottom:2px solid transparent; }
.finder:has(> .finder-command) .xr-top-menu > summary::-webkit-details-marker { display:none; }
.finder:has(> .finder-command) .xr-top-menu > summary:hover { background:#f7f7f8; }
.finder:has(> .finder-command) .xr-top-menu > summary.on { color:#25232e; font-weight:700; border-bottom-color:#25232e; background:#f4f3f1; }
.finder:has(> .finder-command) .xr-top-menu > summary:focus-visible { outline:2px solid #025558; outline-offset:1px; }
.finder:has(> .finder-command) .xr-top-list { position:absolute; top:calc(100% + 4px); left:0; z-index:30; display:flex; flex-direction:column; min-width:230px; padding:4px 0; background:#fff; border:1px solid #d5d8df; border-radius:0; box-shadow:0 6px 18px rgba(20,20,30,.08); }
.finder:has(> .finder-command) .xr-top-head { padding:8px 12px 4px; font-size:10.5px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:#686a74; }
.finder:has(> .finder-command) .xr-top-head:not(:first-child) { margin-top:4px; border-top:1px solid #eceef2; padding-top:10px; }
.finder:has(> .finder-command) .xr-top-list > button { height:34px; padding:0 12px; border:0; border-radius:0; background:#fff; color:#25232e; text-align:left; font:inherit; font-size:13px; cursor:pointer; white-space:nowrap; }
.finder:has(> .finder-command) .xr-top-list > button:hover { background:#f7f7f8; }
.finder:has(> .finder-command) .xr-top-list > button.on { font-weight:700; background:#f4f3f1; }
.finder:has(> .finder-command) .xr-top-list > button:focus-visible { outline:2px solid #025558; outline-offset:1px; }
.finder:has(> .finder-command) .xr-top-list > button.htmx-request::after { content:' · counting messages…'; color:#686a74; font-weight:400; }
.finder:has(> .finder-command) .xr-table tr[data-conv] td { border-top:2px solid #d5d8df; }
/* One weight for secondary controls, one for primary actions; white on bright orange failed AA (2.3:1), so Search matches Load as in the mockup. */
.finder:has(> .finder-command) .xr-filter,
.finder:has(> .finder-command) .xr-time-menu > summary,
.finder:has(> .finder-command) .xr-toolbar-right > .quiet,
.finder:has(> .finder-command) .xr-cols > summary,
.finder:has(> .finder-command) .xr-sort > summary { font-weight:500; cursor:pointer; transition:background .15s ease-out; }
.finder:has(> .finder-command) .finder-command-query > button,
.finder:has(> .finder-command) .xr-toolbar-right .btn-secondary { background:#25232e; border:1px solid #25232e; color:#fff; font-weight:600; transition:background .15s ease-out; }
.finder:has(> .finder-command) .finder-command-query > button:hover,
.finder:has(> .finder-command) .xr-toolbar-right .btn-secondary:hover { background:#3a3844; }
.finder:has(> .finder-command) .xr-filter:hover,
.finder:has(> .finder-command) .xr-time-menu > summary:hover,
.finder:has(> .finder-command) .xr-cols > summary:hover,
.finder:has(> .finder-command) .xr-sort > summary:hover,
.finder:has(> .finder-command) .finder-command-query > a:hover { background:#f5f5f6; }
.finder:has(> .finder-command) .xr-quickviews > button:not(.on):hover { background:#f7f7f8; }
.finder:has(> .finder-command) :is(.finder-command-query > a, .finder-command-query > button, .xr-filter, .xr-quickviews button, .xr-toolbar-right summary, .xr-toolbar-right .btn-secondary, .xr-switch button):focus-visible { outline:2px solid #025558; outline-offset:1px; }
/* Time-range popover: presets as one square segmented row, then a Custom range disclosure with aligned From/To rows. */
.finder:has(> .finder-command) .xr-time-options { left:0; right:auto; width:320px; min-width:0; box-sizing:border-box; gap:0; padding:0; border:1px solid #d5d8df; border-radius:0; box-shadow:0 6px 16px #25232e1f; }
.finder:has(> .finder-command) .xr-time-options .xr-presets { display:grid; grid-template-columns:repeat(5,1fr); gap:0; margin:12px; border:1px solid #d5d8df; }
.finder:has(> .finder-command) .xr-preset { height:32px; border:0; border-left:1px solid #d5d8df; background:#fff; color:#4d4b56; font:inherit; font-size:12px; font-weight:500; font-variant-numeric:tabular-nums; cursor:pointer; transition:background .15s ease-out; }
.finder:has(> .finder-command) .xr-preset:first-child { border-left:0; }
.finder:has(> .finder-command) .xr-preset:hover { background:#f5f5f6; }
.finder:has(> .finder-command) .xr-preset[aria-pressed="true"] { background:#25232e; color:#fff; font-weight:600; }
.finder:has(> .finder-command) .xr-exact { border-top:1px solid #e5e6e9; }
.finder:has(> .finder-command) .xr-exact > summary { display:flex; align-items:center; justify-content:flex-start; height:40px; margin:0 12px; padding:0; color:#25232e; font-size:12px; font-weight:600; }
.finder:has(> .finder-command) .xr-exact > summary::-webkit-details-marker { display:none; }
.finder:has(> .finder-command) .xr-time-menu > summary::after { content:'▾'; flex:none; margin-left:6px; color:#77747e; font-size:10px; }
.finder:has(> .finder-command) .xr-exact > summary::after { content:'▾'; margin-left:6px; color:#77747e; font-size:10px; transition:transform .15s ease-out; }
.finder:has(> .finder-command) .xr-exact[open] > summary::after { transform:rotate(180deg); }
.finder:has(> .finder-command) .xr-exact > summary { cursor:pointer; }
.finder:has(> .finder-command) .xr-exact > summary:hover { color:#000; }
.finder:has(> .finder-command) .xr-exact > summary::before { content:''; flex:none; width:6px; height:6px; margin-right:8px; background:#25232e; visibility:hidden; }
.finder:has(> .finder-command) .xr-exact > summary[data-active]::before { visibility:visible; }
.finder:has(> .finder-command) .xr-tz { margin:0 12px 0 54px; color:#77747e; font-size:11px; }
.finder:has(> .finder-command) .xr-time-menu > summary { justify-content:safe center; min-width:0; }
.finder:has(> .finder-command) .xr-time-label { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.finder:has(> .finder-command) .xr-time-menu > summary::-webkit-details-marker { display:none; }
.finder:has(> .finder-command) .xr-exact[open] { display:grid; gap:8px; padding-bottom:12px; }
.finder:has(> .finder-command) .xr-time-options .xr-range { display:grid; grid-template-columns:36px minmax(0,1fr) 104px; align-items:center; gap:6px; margin:0 12px; }
.finder:has(> .finder-command) .xr-range b { color:#77747e; font-size:11px; font-weight:600; }
.finder:has(> .finder-command) .xr-range input { box-sizing:border-box; width:100%; height:32px; padding:0 8px; border:1px solid #d5d8df; border-radius:0; background:#fff; color:#25232e; font:inherit; font-size:12px; font-variant-numeric:tabular-nums; }
.finder:has(> .finder-command) .xr-range input::-webkit-calendar-picker-indicator { opacity:.35; margin-left:2px; cursor:pointer; }
.finder:has(> .finder-command) .xr-range input:hover::-webkit-calendar-picker-indicator { opacity:.7; }
.finder:has(> .finder-command) .xr-apply { justify-self:stretch; width:calc(100% - 66px); height:32px; margin:4px 12px 0 54px; border:1px solid #25232e; border-radius:0; background:#25232e; color:#fff; font:inherit; font-size:12px; font-weight:600; cursor:pointer; transition:background .15s ease-out; }
.finder:has(> .finder-command) .xr-apply:hover { background:#3a3844; }
.finder:has(> .finder-command) .xr-apply:focus-visible { outline:2px solid #025558; outline-offset:1px; }
@media (max-width:850px) { .finder:has(> .finder-command) .xr-time-options { width:calc(100vw - 50px); } }
@media (max-width:1100px) { .finder:has(> .finder-command) .xr-quickviews { flex:0 1 auto; flex-wrap:wrap; } }
.finder:has(> .finder-command) .xr-range input:focus-visible,
.finder:has(> .finder-command) .xr-preset:focus-visible,
.finder:has(> .finder-command) .xr-exact > summary:focus-visible { outline:2px solid #025558; outline-offset:-2px; }
/* Ask AI results: the answer leads, actions look like buttons, yes/no stays neutral (a match is not an error). */
.finder:has(> .finder-command) .finder-progress { gap:4px 10px; padding:6px 16px; min-height:44px; flex-wrap:wrap; font-family:var(--font-sans); font-size:12px; }
.finder:has(> .finder-command) .finder-progress .state { font-size:10.5px; }
.finder:has(> .finder-command) .finder-progress-answer { flex:1 1 auto; min-width:0; color:var(--text-strong); font-size:13px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.finder:has(> .finder-command) .finder-progress-q { display:inline-block; max-width:min(60ch,100%); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; vertical-align:bottom; }
.finder:has(> .finder-command) .finder-progress .nw { white-space:nowrap; }
.finder:has(> .finder-command) .finder-progress-answer b { font-weight:700; }
.finder:has(> .finder-command) .finder-show-only { padding:3px 10px; font-size:12px; }
.finder:has(> .finder-command) .xr-reason { max-width:34ch; margin-top:2px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--text-muted, #6b6b78); font-size:11px; }
.finder:has(> .finder-command) .fd-reason { margin:2px 0 0; color:var(--text-muted, #6b6b78); font-size:12.5px; }
.finder:has(> .finder-command) .finder-progress-failed b { color:#bd5548; }
.finder:has(> .finder-command) .finder-progress .btn-secondary,
.finder:has(> .finder-command) .finder-progress-action .btn-secondary { display:inline-flex; align-items:center; height:30px; margin-left:0; padding:0 10px; border:1px solid #d5d8df; border-radius:0; background:#fff; color:#25232e; font-family:var(--font-sans); font-size:12px; font-weight:500; text-decoration:none; white-space:nowrap; flex-shrink:0; }
.finder:has(> .finder-command) .finder-progress > a.btn-secondary { margin-left:auto; }
.finder:has(> .finder-command) .finder-progress .btn-secondary:hover { background:#f5f5f6; }
.finder:has(> .finder-command) .finder-progress-action { margin-left:0; }
.finder:has(> .finder-command) .finder-progress-bar { height:3px; }
.finder:has(> .finder-command) .finder-tasks { padding:0; }
.finder:has(> .finder-command) .finder-task { margin:0; border:0; border-bottom:1px solid #e5e6e9; border-radius:0; box-shadow:none; background:#fcfcfd; }
.finder:has(> .finder-command) .finder-task > summary { display:flex; align-items:center; gap:10px; min-height:40px; padding:0 16px; }
.finder:has(> .finder-command) .finder-task .kind { border-radius:0; }
.finder:has(> .finder-command) .finder-task > summary > :not(.finder-task-q) { flex-shrink:0; white-space:nowrap; }
.finder:has(> .finder-command) .finder-task-q { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--text-body); font-size:12px; }
.finder:has(> .finder-command) .finder-task[open] .finder-task-q { display:none; }
.finder:has(> .finder-command) .xr-count { margin-left:3px; color:inherit; opacity:.7; font-variant-numeric:tabular-nums; font-weight:500; }
.finder:has(> .finder-command) .xr-yn { display:inline-flex; align-items:center; gap:6px; font-weight:600; }
.finder:has(> .finder-command) .xr-yn::before { content:''; width:8px; height:8px; box-sizing:border-box; }
.finder:has(> .finder-command) .xr-yn.yes { color:#25232e; }
.finder:has(> .finder-command) .xr-yn.yes::before { background:#25232e; }
.finder:has(> .finder-command) .xr-yn.no { color:#77747e; font-weight:400; }
.finder:has(> .finder-command) .xr-yn.no::before { border:1px solid #a9a7ae; }
.finder:has(> .finder-command) .xr-yn.failed { color:#bd5548; }
.finder:has(> .finder-command) .xr-yn.failed::before { background:#bd5548; }
.finder:has(> .finder-command) .xr-pending { letter-spacing:.1em; }
.finder:has(> .finder-command) .finder-progress[data-state="completed"] .state { color:#686a74; }
body:has(.finder-command) .finder-status.done, body:has(.finder-command) .finder-status.idle { display:none; }
/* Table fills its card; the flex column absorbs the spare width and the inline min-width keeps the rest legible. */
.finder:has(> .finder-command) .xr-table { width:100%; }
/* Errors stay loud: a red bar on the row, status text at 4.5:1 or better, and a shape that survives without colour. */
.finder:has(> .finder-command) .xr-table .status-label.ok { color:#1a6a3c; }
.finder:has(> .finder-command) .xr-table .status-label.err { color:#b3261e; }
.finder:has(> .finder-command) .xr-table .dot.err { border-radius:1px; transform:rotate(45deg) scale(.85); background:#b3261e; }
.finder:has(> .finder-command) .xr-table tr.xr-err td:first-child { box-shadow:inset 3px 0 0 #b3261e, inset -1px 0 0 #e8e9ec; }
.finder:has(> .finder-command) .tv-r.row-err { box-shadow:inset 3px 0 0 #b3261e; }
/* AI matches get an accent bar; rows the question passed over stay fully readable. */
.finder:has(> .finder-command) .xr-table tr.xr-hit td:first-child, .finder:has(> .finder-command) .tv-r.hit { box-shadow:inset 3px 0 0 #ed8844, inset -1px 0 0 #e8e9ec; }
.finder:has(> .finder-command) .tv-r.hit { background:#fffaf3; }
.finder:has(> .finder-command) .tv-r.nomatch { opacity:1; }
.finder:has(> .finder-command) .xr-table th[aria-sort] { color:#25232e; }
.finder:has(> .finder-command) .xr-sort-arrow { margin-left:4px; font-weight:700; }
.finder:has(> .finder-command) .xr-table tbody tr:focus-visible, .finder:has(> .finder-command) .tv-r:focus-visible { outline:2px solid #025558; outline-offset:-2px; }
.finder:has(> .finder-command) .xr-empty { border-radius:0; }
.finder:has(> .finder-command) .xr-empty h4 { text-transform:none; letter-spacing:0; }
@media (prefers-reduced-motion:reduce) { .finder:has(> .finder-command) * { transition:none!important; } }
@media (max-width:850px) {
  body:has(.finder-command) .app-sidebar { display:none; }
  body:has(.finder-command) .app-content { padding:12px; }
  .finder:has(> .finder-command) .finder-command { padding:10px 12px 0; }
  .finder:has(> .finder-command) .finder-command-query { flex-wrap:wrap; }
  .finder:has(> .finder-command) .finder-command-query .col { flex:1 1 calc(100% - 120px); }
  .finder:has(> .finder-command) .xr-toolbar { padding:6px 10px; }
  .finder:has(> .finder-command) .xr-toolbar .spacer { display:none; }
}
/* Toolbar hierarchy: Search and Load are the only solid buttons, every other control is a 32px ghost with a 6px radius, and the
   Table/Trajectories switch is a quiet track whose active segment is a white chip, so it never competes with Load. */
.finder:has(> .finder-command) { --ctl:32px; }
.finder:has(> .finder-command) :is(.finder-ai-label, .finder-command-query > a, .finder-command-query > button, .finder-seg, .xr-filter, .xr-time-menu > summary, .xr-toolbar-right > .quiet, .xr-cols > summary, .xr-sort > summary, .xr-toolbar-right .btn-secondary) { border-radius:6px; }
.finder:has(> .finder-command) .finder-command-query > button { flex:0 0 auto; min-width:88px; padding:0 16px; }
.finder:has(> .finder-command) .finder-seg { padding:2px; border:1px solid #d5d8df; background:#f1f0ee; }
.finder:has(> .finder-command) .finder-seg label > span,
.finder:has(> .finder-command) .xr-switch button { border-radius:4px; background:transparent; color:#5b5964; font-weight:500; }
.finder:has(> .finder-command) .finder-seg input:checked + span,
.finder:has(> .finder-command) .xr-switch button.on { background:#fff; color:#25232e; font-weight:600; box-shadow:0 0 0 1px #d5d8df, 0 1px 2px rgba(20,20,30,.12); }
/* Reserve the Export column in Trajectories so the remaining controls keep their positions. */
.finder:has(> .finder-command) .xr-toolbar-right { grid-template-columns:64px minmax(150px,1fr) 100px 90px 64px 200px; }
.finder:has(> .finder-command) .xr-toolbar-right > .xr-time-menu { grid-column:2; }
.finder:has(> .finder-command) .xr-toolbar-right > .quiet { margin-left:12px; }
.finder:has(> .finder-command) .xr-toolbar-right > .xr-load { grid-column:5; }
.finder:has(> .finder-command) .xr-toolbar-right > .xr-switch { grid-column:6; display:grid; grid-template-columns:72px minmax(0,1fr); width:200px; }
/* The toolbar sits directly under the question box and never moves. The page puts it ahead of the Ask AI band in the DOM, so the
   band, its criteria and the run progress render under it without CSS reordering, and Tab follows what the eye sees. */
.finder:has(> .finder-command) :is(#explorer-toolbar, #finder-body, .finder-body-fragment, #explorer-results-slot, .xr) { display:contents; }
.finder:has(> .finder-command) #explorer-toolbar > .xr-toolbar { order:0; }
/* An empty chip slot would still cost one flex gap. */
.finder:has(> .finder-command) .xr-chips:empty { display:none; }
/* Active filter chips get their own row under the controls, so adding a filter never pushes the controls onto another line. */
.finder:has(> .finder-command) .xr-chips:not(:empty) { order:5; flex:1 0 100%; display:flex; flex-wrap:wrap; gap:6px; padding:0 0 8px; }
/* Filters menu: an opaque panel above the sticky table header, so nothing behind it shows through. */
.finder:has(> .finder-command) .finder-facets { z-index:60; }
.finder:has(> .finder-command) .finder-facets .facet-list,
.finder:has(> .finder-command) .finder-facets .facet-sub { background:#fff; opacity:1; border:1px solid #cfd2d9; border-radius:8px; box-shadow:0 12px 32px rgba(20,20,30,.18), 0 2px 6px rgba(20,20,30,.08); }
.finder:has(> .finder-command) .finder-facets .facet-sub label { border-radius:6px; }
.finder:has(> .finder-command) .finder-facets .facet-n { margin-left:auto; padding-left:12px; color:#686a74; font-size:11.5px; font-variant-numeric:tabular-nums; }
.finder:has(> .finder-command) .finder-facets .facet-scope { margin:2px 10px 6px; color:#686a74; font-size:11px; line-height:1.35; }
.finder:has(> .finder-command) .finder-command-query .col { position:relative; padding:8px 12px; border:1px solid #c4c8ce; border-radius:7px; background:#fff; box-shadow:0 1px 2px #28263005; }
.finder:has(> .finder-command) .finder-command-query .col:focus-within { border-color:#025558; box-shadow:0 0 0 2px #0255580e; }
.finder:has(> .finder-command) .finder-command-examples { position:absolute; left:16px; right:16px; top:50%; transform:translateY(-50%); margin:0; }
.finder:has(> .finder-command) .col:has(.finder-command-textarea:focus) .finder-command-examples,
.finder:has(> .finder-command) .col:has(.finder-command-textarea:not(:placeholder-shown)) .finder-command-examples { display:none; }
.finder:has(> .finder-command) .finder-command-textarea:placeholder-shown:not(:focus)::placeholder { color:transparent; }
.finder:has(> .finder-command) .finder-command-textarea::placeholder { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
/* Below 1100px the question takes a full row, and no control is left alone on a row. In the command strip the second row holds
   Search in, AI settings and Search. In the toolbar the first row holds Filters and the quick views, and the
   second holds Export, the time range, Rows, Columns, Load and the view switch. */
@media (max-width:1100px) {
  .finder:has(> .finder-command) .finder-command-query { flex-wrap:wrap; row-gap:8px; }
  .finder:has(> .finder-command) .finder-command-query .col { flex:1 1 calc(100% - 92px); min-width:0; }
  .finder:has(> .finder-command) .finder-command-query > .finder-command-gear { margin-left:auto; }
  .finder:has(> .finder-command) .xr-toolbar { row-gap:6px; padding-top:8px; padding-bottom:8px; }
  .finder:has(> .finder-command) .xr-toolbar-right { display:contents; }
  .finder:has(> .finder-command) .xr-toolbar > .xr-filter,
  .finder:has(> .finder-command) .xr-toolbar > .xr-quickviews { order:1; }
  .finder:has(> .finder-command) .xr-toolbar .spacer { display:none; }
  .finder:has(> .finder-command) .xr-toolbar-right .xr-switch { flex:0 0 200px; width:200px; }
  .finder:has(> .finder-command) .xr-toolbar::after { content:''; order:2; flex:1 0 100%; height:0; margin-top:-6px; }
  .finder:has(> .finder-command) .xr-toolbar-right > * { order:3; }
  .finder:has(> .finder-command) .xr-time-menu { flex:1 1 110px; min-width:0; }
  .finder:has(> .finder-command) .xr-toolbar-right > .quiet { flex:0 0 86px; }
  .finder:has(> .finder-command) .xr-cols, .finder:has(> .finder-command) .xr-sort { flex:0 0 94px; }
  .finder:has(> .finder-command) .xr-toolbar-right .btn-secondary { flex:0 0 57px; }
  .finder:has(> .finder-command) .xr-toolbar-right .xr-export { flex-basis:64px; }
}
"""


_DASHBOARD_CSS_HEAD = """
/* ==== shell: sidebar + main ========================================= */
body.eq-dashboard { margin: 0; background: var(--surface-app); }
.app-shell {
    display: flex;
    min-height: 100vh;
    background: var(--surface-app);
    color: var(--text-body);
    font-family: var(--font-sans);
}

.app-sidebar {
    width: 232px;
    flex-shrink: 0;
    background: var(--app-gray-100);
    border-right: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    padding: 16px 12px;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow: hidden;
    transition: width 0.15s ease, padding 0.15s ease;
}

/* Collapse toggle: sits at the bottom of the sidebar (margin-top:auto pushes
   it down). Class lives on <html> (set in shell.py before paint) so it
   persists across the full-page navigations between reports. */
.sidebar-toggle {
    margin-top: auto;
    display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; border: none; width: 100%;
    background: transparent; color: var(--text-muted);
    font-family: var(--font-sans); font-size: 13px; text-align: left;
    border-radius: var(--radius-md); cursor: pointer;
}
.sidebar-toggle:hover { background: rgba(10,10,11,0.04); color: var(--text-strong); }
.sidebar-toggle .nav-icon { transition: transform 0.15s ease; }
.sidebar-toggle-kbd {
    margin-left: auto; font-family: var(--font-sans); font-size: 11px;
    font-weight: 500; letter-spacing: 0.3px;
    color: var(--text-muted); background: var(--surface-sunken);
    border: 1px solid var(--border-subtle);
    border-radius: 5px; padding: 2px 7px; line-height: 1.5;
}
html.sidebar-collapsed .app-sidebar { width: 60px; padding: 16px 8px; }
html.sidebar-collapsed .app-brand { justify-content: center; padding: 4px 0 18px; }
html.sidebar-collapsed .app-brand .brand-name { display: none; }
html.sidebar-collapsed .nav-item { justify-content: center; padding: 8px; }
html.sidebar-collapsed .nav-item span,
html.sidebar-collapsed .sidebar-toggle span,
html.sidebar-collapsed .sidebar-toggle-kbd { display: none; }
html.sidebar-collapsed .sidebar-toggle { justify-content: center; padding: 8px; }
html.sidebar-collapsed .sidebar-toggle .nav-icon { transform: rotate(180deg); }
.app-brand {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 4px 8px 18px;
    text-decoration: none;
}
.app-brand .nav-mark { display: inline-flex; flex-shrink: 0; }
.app-brand .nav-mark svg { width: 24px; height: 24px; display: block; }
.app-brand .brand-name {
    font-family: var(--font-display);
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-strong);
}
.app-brand .brand-q { color: var(--orange-500); }

.app-nav { display: flex; flex-direction: column; gap: 2px; }
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border-radius: var(--radius-md);
    text-decoration: none;
    color: var(--text-muted);
    font-size: 14px;
    font-weight: 500;
}
.nav-item:hover { background: rgba(10,10,11,0.04); }
.nav-item.active {
    background: var(--surface-card);
    color: var(--text-strong);
    font-weight: 600;
    box-shadow: 0 1px 2px rgba(37,35,46,0.06);
}
.nav-item .nav-icon { color: currentColor; flex-shrink: 0; }
.nav-item.active .nav-icon { color: var(--orange-500); }

.app-main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.app-topbar {
    height: 56px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 24px;
    background: var(--surface-card);
    border-bottom: 1px solid var(--border-subtle);
    position: sticky;
    top: 0;
    z-index: 20;
}
.app-title {
    font-family: var(--font-display);
    font-size: 17px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text-strong);
    margin: 0;
}
.app-topbar > .report-back { margin-bottom: 0; font-size: 14px; }
.app-actions { margin-left: auto; display: flex; align-items: center; gap: 8px; }
.app-content { flex: 1; padding: 24px; }

/* ==== shared chrome primitives ====================================== */
.panel {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 18px 20px;
}
.panel-title {
    font-family: var(--font-display);
    font-size: 16px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text-strong);
    margin: 0;
}
.panel-sub {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-faint);
    margin: 2px 0 14px;
}

.btn-secondary {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 32px;
    padding: 0 12px;
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    background: var(--surface-card);
    color: var(--text-body);
    font-family: var(--font-sans);
    font-size: 13px;
    font-weight: 500;
    text-decoration: none;
    cursor: pointer;
}
.btn-secondary:hover { background: var(--app-gray-50); color: var(--text-strong); }

.kind-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 20px;
    padding: 0 7px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 500;
    white-space: nowrap;
}
.kind-badge.redteam { background: var(--red-100); color: var(--red-600); }
.kind-badge.sim { background: var(--teal-100); color: var(--teal-600); }

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 500;
    line-height: 1;
    white-space: nowrap;
}
.status-badge .dot { width: 5px; height: 5px; border-radius: 50%; }
.status-badge.passed { background: var(--green-100); color: var(--teal-600); }
.status-badge.passed .dot { background: var(--green-600); }
.status-badge.failed { background: var(--red-100); color: var(--red-600); }
.status-badge.failed .dot { background: var(--red-600); }
.status-badge.warning { background: var(--amber-100); color: var(--red-600); }
.status-badge.warning .dot { background: var(--orange-500); }
.status-badge.cancelled { background: var(--app-gray-100, #ececec); color: var(--text-muted, #8a8a8a); }
.status-badge.cancelled .dot { background: var(--text-muted, #8a8a8a); }

/* sim run-level table: two-line job cell + outline target pill */
.sim-job { display: inline-flex; flex-direction: column; gap: 2px; text-decoration: none; }
.sim-job-name { font-weight: 600; color: var(--text-strong, #1a1a1a); }
.sim-job-sub { font-size: 12px; color: var(--text-muted, #8a8a8a); }
.target-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 24px;
    padding: 0 10px;
    border: 1px solid var(--border-subtle);
    border-radius: 999px;
    font-size: 12px;
    font-weight: 500;
    white-space: nowrap;
}
.target-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green-600); }
.target-pill svg { width: 13px; height: 13px; color: var(--green-600); flex: 0 0 auto; }
.row-chevron { width: 16px; height: 16px; color: var(--text-faint); display: block; margin-left: auto; }

/* Run-level surface tables: clickable anchor-grid, borderless (no table chrome) */
.runs-grid { display: flex; flex-direction: column; }
/* Fixed/fr columns only — NO content-based (auto/max-content) widths, so every
   row (each its own grid) resolves identical column edges and the bars align. */
.runs-grid-head, .runs-grid-row {
    display: grid;
    grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr) 92px 56px 56px 88px 18px;
    align-items: center;
    gap: 16px;
}
.runs-grid-head > span, .runs-grid-row > span { min-width: 0; }
.runs-grid-row .target-pill { max-width: 100%; overflow: hidden; text-overflow: ellipsis; }
.rg-targets { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; min-width: 0; }
.runs-grid-head {
    padding: 0 4px 10px;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-faint);
}
.runs-grid-row {
    padding: 12px 4px;
    text-decoration: none;
    color: inherit;
    border-top: 1px solid var(--border-subtle);
}
.runs-grid-row:hover { background: var(--app-gray-50); }
.rg-job { display: inline-flex; flex-direction: column; gap: 2px; min-width: 0; }
.rg-name { font-weight: 600; color: var(--text-strong, #1a1a1a); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rg-sub { font-size: 12px; color: var(--text-muted, #8a8a8a); }
.rg-num { font-family: var(--font-mono); font-size: 13px; text-align: right; }
.runs-grid-row .run-score { font-family: var(--font-mono); font-size: 13px; font-weight: 500; text-align: right; }
.runs-grid-head span:nth-child(4) { text-align: right; }

/* Run-overview pager: size picker on the left; count + prev/next on the right */
.runs-pager {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 4px 4px;
    font-size: 13px;
    color: var(--text-muted, #8a8a8a);
}
.runs-pager-right { margin-left: auto; display: inline-flex; align-items: center; gap: 16px; }
.runs-pager-nav { display: inline-flex; gap: 8px; }
.runs-pager-count { color: var(--text-strong, #1a1a1a); }
.runs-pager-sizes { display: inline-flex; align-items: center; gap: 6px; }
.runs-pager-link {
    text-decoration: none;
    color: var(--text-muted, #8a8a8a);
    padding: 2px 6px;
    border-radius: 4px;
}
.runs-pager-link:hover { background: var(--app-gray-50); color: var(--text-strong, #1a1a1a); }
.runs-pager-link.is-active { color: var(--text-strong, #1a1a1a); font-weight: 600; }
.runs-pager-link.is-disabled { opacity: 0.4; pointer-events: none; }

/* Solid status pill mirroring the main platform (success green / error red) */
.run-status {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    color: #fff;
    text-transform: lowercase;
    white-space: nowrap;
}
.run-status.finished { background: var(--green-600); }
.run-status.error { background: var(--red-600); }
.run-status.running { background: var(--orange-500); }
.run-status.cancelled { background: var(--text-muted, #8a8a8a); }

/* Landing 'Recent runs' Type column: surface glyph + label (no colored bubble) */
.type-cell { display: inline-flex; align-items: center; gap: 6px; font-weight: 500; white-space: nowrap; }
.type-cell svg { width: 15px; height: 15px; flex: 0 0 auto; }
.type-cell.redteam svg { color: var(--red-600); }
.type-cell.sim svg { color: var(--teal-600); }
.type-cell.pairwise svg { color: var(--orange-500); }

/* Landing 'Recent runs' — airy aligned columns, no table chrome */
.recent-runs { display: flex; flex-direction: column; }
/* Fixed/fr columns only — NO content-based (auto/max-content) widths, so every
   row (each its own grid) resolves identical column edges and the values align
   with the header. Mirrors the runs-grid fix in .runs-grid-head/.runs-grid-row. */
.rr-head, .rr-row {
    display: grid;
    grid-template-columns: minmax(130px, 1.2fr) minmax(0, 2fr) 90px 130px 56px minmax(0, 0.8fr);
    align-items: center;
    gap: 16px;
}
.rr-head {
    padding: 0 4px 10px;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-faint);
}
.rr-row {
    padding: 12px 4px;
    text-decoration: none;
    color: inherit;
    border-top: 1px solid var(--border-subtle);
}
.rr-row:hover { background: var(--app-gray-50); }
.rr-job {
    font-size: 13px; font-weight: 500; color: var(--text-strong);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.rr-meta { font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rr-row .run-score { font-family: var(--font-mono); font-size: 13px; font-weight: 500; }
/* Align the numeric Score column head + cell. */
.rr-head span:nth-child(5), .rr-row .run-score { text-align: right; }

/* ==== combined landing ============================================== */
.dash-wrap { display: flex; flex-direction: column; gap: 16px; max-width: 1100px; }
.stat-band { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.stat-tile {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 16px 18px;
}
.stat-tile .stat-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-faint);
}
.stat-tile .stat-value {
    font-family: var(--font-display);
    font-size: 26px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text-strong);
    margin-top: 6px;
}
.stat-tile .stat-value .stat-unit {
    font-family: var(--font-mono);
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted);
    margin-left: 3px;
}
.stat-tile .stat-sub {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-faint);
    margin-top: 4px;
}
.dash-row2 { display: grid; grid-template-columns: 1.5fr 1fr; gap: 16px; }
.dash-row-eq { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

/* horizontal proportion bars (severity / by-kind) */
.bars { display: flex; flex-direction: column; gap: 14px; padding-top: 4px; }
.bar-row .bar-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 5px;
}
.bar-row .bar-name { font-size: 13px; color: var(--text-body); }
.bar-row .bar-val { font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); }
.bar-row .bar-val .bar-pct { color: var(--text-faint); }
.bar-track { height: 9px; border-radius: 5px; background: var(--chart-track); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 5px; }
.bars-total {
    display: flex;
    justify-content: space-between;
    border-top: 1px solid var(--border-subtle);
    padding-top: 10px;
    margin-top: 2px;
}
.bars-total .t-label {
    font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.05em;
    text-transform: uppercase; color: var(--text-faint);
}
.bars-total .t-val {
    font-family: var(--font-mono); font-size: 13px; font-weight: 600; color: var(--text-strong);
}

/* donut (pass rate) */
.donut-wrap { display: flex; justify-content: center; padding-top: 6px; }
.donut { position: relative; width: 150px; height: 150px; }
.donut svg { transform: rotate(-90deg); }
.donut .donut-center {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.donut .donut-value {
    font-family: var(--font-display); font-size: 28px; font-weight: 600; color: var(--text-strong);
}
.donut .donut-label {
    font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.05em;
    text-transform: uppercase; color: var(--text-faint);
}
.donut-wrap { gap: 20px; align-items: center; }
.donut-legend {
    list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column;
    gap: 6px; font-family: var(--font-sans); font-size: 12px; color: var(--text-muted);
}
.donut-legend li { display: flex; align-items: center; gap: 8px; }
.donut-key { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }

/* ==== search results dropdown (styling retained for the /search route) === */
.search-results {
    position: absolute; top: calc(100% + 6px); right: 0; width: 320px; max-width: 60vw;
    background: var(--surface-card, #fff); border: 1px solid var(--border, #e2e0da);
    border-radius: 10px; box-shadow: 0 8px 28px rgba(0,0,0,0.12); overflow: hidden; z-index: 40;
}
.search-results:empty { display: none; }
.search-hit { display: flex; flex-direction: column; gap: 2px; padding: 8px 12px; text-decoration: none; }
.search-hit:hover { background: var(--surface-app, #faf9f5); }
.search-hit-kind { font-family: var(--font-mono); font-size: 10px; text-transform: uppercase; color: var(--text-faint); }
.search-hit-name { font-size: 13px; color: var(--text-strong); }
.search-empty { padding: 10px 12px; font-size: 13px; color: var(--text-muted); }

/* ==== settings ====================================================== */
.config-list { display: flex; flex-direction: column; }
.config-row { display: flex; align-items: baseline; gap: 24px; padding: 8px 0; border-bottom: 1px solid var(--border, #eee); }
.config-key { flex: 0 0 160px; font-family: var(--font-sans); font-size: 13px; color: var(--text-muted); }
/* Value column takes the remaining width; list items stack one per line. */
.config-val { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; font-family: var(--font-mono); font-size: 12px; color: var(--text-strong); }
.config-val-item { word-break: break-all; }
.config-note { font-size: 13px; color: var(--text-muted); line-height: 1.5; }
.settings-form { display: flex; flex-direction: column; gap: 14px; }
.settings-field .config-key { cursor: pointer; }
.settings-field input { width: 100%; box-sizing: border-box; border: 1px solid var(--border-default, #d8d5cf); border-radius: 6px; padding: 7px 9px; color: var(--text-strong); background: var(--surface-card, #fff); font: inherit; }
.settings-field input:focus { outline: none; border-color: var(--accent); box-shadow: var(--ring); }
.settings-error { color: var(--red-600, #dc2626); font-family: var(--font-sans); font-size: 12px; }
.settings-saved { margin: 0 0 12px; color: var(--green-600, #16a34a); font-size: 13px; }
.settings-advanced { margin: 12px 0; }
.settings-advanced > summary { cursor: pointer; color: var(--text-muted); font-size: 13px; }
.model-pick { position: relative; display: block; }
.model-pick-btn { width: 100%; text-align: left; border: 1px solid var(--border-default, #d8d5cf); border-radius: 6px; padding: 7px 9px; color: var(--text-strong); background: var(--surface-card, #fff); font: inherit; font-family: var(--font-mono); cursor: pointer; }
.model-pick-btn::after { content: "▾"; float: right; color: var(--text-faint); }
.model-pick:has(.finder-facets.open) .model-pick-btn { border-color: var(--accent); box-shadow: var(--ring); }
.model-pick .finder-facets { top: calc(100% + 4px); }
.model-pick .finder-facets .facet-item > span:first-child { text-transform: none; }
.model-pick .finder-facets .facet-list { max-height: 420px; overflow-y: auto; }
.model-pick .model-option { display: block; width: 100%; padding: 6px 10px; border: 0; border-radius: 8px; background: transparent; color: var(--text-body); font: inherit; font-size: 12px; text-align: left; word-break: break-all; cursor: pointer; }
.model-pick .model-option:hover, .model-pick .model-option:focus-visible { background: var(--surface-sunken); color: var(--text-strong); outline: none; }
.model-pick .model-option[hidden] { display: none; }
.model-pick .model-option.is-selected { background: var(--surface-sunken); color: var(--text-strong); font-weight: 600; }
.model-pick .model-custom { box-sizing: border-box; width: calc(100% - 12px); margin: 4px 6px 8px; }
.rich-pick .model-pick-btn, .rich-pick .model-option { display: flex; align-items: center; gap: 12px; font-family: var(--font-sans); word-break: normal; }
.rich-pick .model-pick-btn::after { float: none; order: 3; margin-left: auto; }
.rich-pick .finder-facets { left: 0; right: 0; }
.rich-pick .finder-facets .facet-list { width: auto; }
.rich-pick .model-option, .rich-pick-disabled { padding: 8px 10px; }
.rich-pick-disabled { display: flex; align-items: center; gap: 12px; cursor: not-allowed; opacity: .7; }
.rich-pick-placeholder { flex: 1; color: var(--text-muted); font-size: 13px; }
.rich-pick-id { display: grid; flex: 1; gap: 1px; min-width: 0; }
.rich-pick-id strong { color: var(--text-strong); font-size: 13px; font-weight: 600; }
.rich-pick-id .config-note { min-width: 0; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
/* Compact: one line per row, the note right after the title. */
.rich-pick--compact .model-option, .rich-pick--compact .rich-pick-disabled { padding: 5px 10px; }
.rich-pick--compact .rich-pick-id { display: flex; align-items: baseline; gap: 8px; }
.rich-pick--compact .rich-pick-id strong { flex: none; }
.settings-field select { width: 100%; box-sizing: border-box; appearance: none; border: 1px solid var(--border-default, #d8d5cf); border-radius: 6px; padding: 7px 28px 7px 9px; color: var(--text-strong); background: var(--surface-card, #fff); font: inherit; font-family: var(--font-mono); cursor: pointer; }
/* Same ▾ glyph and inset as the model pickers above it, in place of the native arrow. */
.settings-select { position: relative; display: block; }
.settings-select::after { content: "▾"; position: absolute; right: 10px; top: 50%; transform: translateY(-50%); color: var(--text-faint); font-family: var(--font-mono); pointer-events: none; }
.settings-auth-step { margin: 10px 0 0; color: var(--text-muted); font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.settings-auth-choices { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); grid-auto-rows: 1fr; gap: 10px; padding: 12px 0 2px; }
.settings-auth-choice { min-width: 0; height: 100%; border: 1px solid var(--border-default); border-radius: 12px; background: var(--surface-card); overflow: hidden; }
.settings-auth-choice:has(input[type="radio"]:checked) { border-color: var(--teal-600); box-shadow: 0 0 0 1px var(--teal-600); }
.settings-auth-card { display: flex; align-items: flex-start; gap: 11px; min-height: 82px; height: 100%; padding: 15px; cursor: pointer; }
.settings-auth-card input { margin: 3px 0 0; flex: 0 0 auto; accent-color: var(--teal-600); }
.settings-auth-card-copy { display: grid; gap: 4px; min-width: 0; }
.settings-auth-card-copy strong { color: var(--text-strong); font-size: 13px; font-weight: 650; }
.settings-auth-card-copy small { color: var(--text-muted); font-size: 12px; line-height: 1.4; }
.settings-auth-card-check { display: none; }
.settings-auth-config { min-height: 188px; margin-top: 16px; padding: 16px 18px; border: 1px solid var(--border-default); border-radius: 12px; background: var(--surface-sunken); }
.settings-auth-panel:has(input[name="orq_auth_method"][value="environment"]:checked) .settings-auth-config { display: none; }
.settings-auth-config > .settings-auth-step { margin: 0 0 14px; }
.settings-auth-detail { display: none; max-width: 640px; gap: 7px; }
.settings-oauth-field { display: grid; gap: 7px; }
.settings-auth-panel:has(input[value="environment"]:checked) .settings-auth-detail[data-auth-method="environment"],
.settings-auth-panel:has(input[value="cli_profile"]:checked) .settings-auth-detail[data-auth-method="cli_profile"],
.settings-auth-panel:has(input[value="cli_oauth"]:checked) .settings-auth-detail[data-auth-method="cli_oauth"],
.settings-auth-panel:has(input[value="stored_api_key"]:checked) .settings-auth-detail[data-auth-method="stored_api_key"] { display: grid; }
.settings-auth-detail-label { color: var(--text-muted); font-size: 11px; font-weight: 650; }
.settings-auth-detail input, .settings-auth-detail select {
  width: 100%; min-width: 0; min-height: 36px; box-sizing: border-box;
  padding: 7px 10px; border: 1px solid var(--border-default); border-radius: 6px;
  background: var(--surface-card); color: var(--text-strong);
  font-family: var(--font-mono); font-size: 12px; line-height: 1.4;
}
.settings-auth-detail input::placeholder { color: var(--text-muted); opacity: 1; }
.settings-auth-detail :is(input, select):hover { border-color: var(--teal-600); }
.settings-auth-detail :is(input, select):focus-visible { outline: 2px solid var(--teal-600); outline-offset: 2px; }
.settings-auth-hint { color: var(--text-muted); font-size: 11px; line-height: 1.4; }
.eq-auth-toast { position: fixed; z-index: 1000; right: 20px; bottom: 20px; display: flex; align-items: center; gap: 12px; max-width: min(440px, calc(100vw - 32px)); padding: 12px 14px; border: 1px solid var(--border-default); border-left: 3px solid var(--red-700); border-radius: 10px; background: var(--surface-card); box-shadow: 0 10px 30px rgba(30, 28, 35, .16); color: var(--text-body); font-size: 12px; line-height: 1.4; }
.eq-auth-toast[hidden] { display: none; }
.eq-auth-toast.is-warning { border-left-color: var(--teal-600); }
.eq-auth-toast a { color: var(--teal-600); font-weight: 650; white-space: nowrap; }
.eq-auth-toast button { border: 0; background: transparent; color: var(--text-muted); cursor: pointer; font-size: 18px; line-height: 1; }
.settings-auth-note { margin: 2px 0 0; color: var(--text-muted); font-size: 12px; line-height: 1.5; }
.settings-save { align-self: flex-start; }
@media (max-width: 600px) {
  body.eq-dashboard:has(.settings-auth-panel) .config-row { flex-direction: column; align-items: stretch; gap: 6px; }
  body.eq-dashboard:has(.settings-auth-panel) .config-key { flex-basis: auto; }
  body.eq-dashboard:has(.settings-auth-panel) .config-val { width: 100%; }
  .settings-auth-choices { grid-template-columns: 1fr; }
}

/* ==== run rows (recent + per-kind list) ============================= */
.run-list { display: flex; flex-direction: column; }
.run-row {
    display: grid;
    grid-template-columns: 1fr auto auto;
    align-items: center;
    gap: 12px;
    padding: 12px 4px;
    text-decoration: none;
    border-top: 1px solid var(--border-subtle);
    color: inherit;
}
.run-row:first-child { border-top: none; }
.run-row:hover { background: var(--app-gray-50); }
.run-row .run-id { min-width: 0; }
.run-row .run-name-line { display: flex; align-items: center; gap: 8px; }
.run-row .run-name {
    font-size: 13px; font-weight: 500; color: var(--text-strong);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.run-row .run-meta { font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); }
.run-row .run-score { font-family: var(--font-mono); font-size: 13px; font-weight: 500; }
.run-score.good { color: var(--teal-600); }
.run-score.warn { color: var(--orange-700); }
.run-score.none { color: var(--text-faint); }
/* Marks a score the dashboard re-derived; the Score tooltip names the recorded value. */
.score-recalc { color: var(--text-faint); font-size: 11px; vertical-align: super; margin-left: 1px; }

/* per-kind run-list screen: card-wrapped table with a header strip */
.runs-screen { max-width: 1100px; }
.runs-card {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    overflow: hidden;
}
.runs-head {
    display: grid;
    grid-template-columns: 2fr auto auto auto;
    gap: 12px;
    padding: 11px 20px;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--app-gray-50);
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-faint);
}
.runs-card .run-row { grid-template-columns: 2fr auto auto auto; padding: 13px 20px; }
.runs-empty {
    padding: 48px 20px; text-align: center; color: var(--text-faint);
    font-family: var(--font-sans); font-size: 14px;
}

/* settings stub */
.settings-stub {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 10px; height: 50vh; color: var(--text-faint); font-size: 14px;
}

/* ==== report view: filter sidebar + body =========================== */
.report-back {
    display: inline-flex; align-items: center; gap: 6px;
    color: var(--text-muted); font-size: 13px; text-decoration: none; margin-bottom: 12px;
}
.report-back:hover { color: var(--text-strong); }
.report-title {
    font-family: var(--font-display); font-size: 22px; font-weight: 600;
    letter-spacing: -0.02em; color: var(--text-strong); margin: 0;
}

.filter-swap-container { display: flex; align-items: flex-start; gap: 28px; }
.filter-form { flex: 0 0 230px; position: sticky; top: 80px; }
.report-body-area { flex: 1 1 auto; min-width: 0; }

/* ==== sim + redteam filter rail (right side) — scoped to .filter-form--sim
   / .filter-form--redteam so the generic .filter-sidebar form (other
   surfaces) is untouched. */
.filter-form--sim,
.filter-form--redteam {
    flex: 0 0 208px;
    position: static;
    display: flex;
    flex-direction: column;
    gap: 16px;
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 16px;
}
.filter-form--redteam .filter-group { margin-bottom: 0; }
.filter-form--redteam .filter-label {
    font-size: 10.5px; margin-bottom: 8px;
}
.filter-rail-header {
    display: flex; align-items: center; gap: 6px;
    color: var(--text-faint);
}
.filter-rail-title {
    font-family: var(--font-mono);
    font-size: 11px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-faint);
}
.filter-form--sim .filter-group { margin-bottom: 0; }
.filter-form--sim .filter-label {
    font-size: 10.5px; margin-bottom: 8px;
}
.filter-chip-row { display: flex; flex-wrap: wrap; gap: 6px; }
.filter-chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 6px;
    border-radius: 999px;
    border: 1px solid var(--border-subtle);
    background: transparent;
    color: var(--text-faint);
    font-size: 10.5px; font-weight: 500;
    cursor: pointer;
    user-select: none;
    transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease;
}
/* Selected = teal-tinted chip (teal carries selection state; semantic hue lives
   on the check mark only, never the chip fill/border — outcome polarity must
   never rely on color alone, so the dot->check shape swap below is the real
   accessible signal). */
.filter-chip.is-active {
    background: color-mix(in srgb, var(--teal-600) 10%, transparent);
    border-color: color-mix(in srgb, var(--teal-600) 55%, var(--border-default));
    color: var(--text-body);
    font-weight: 600;
}
.filter-chip-input {
    /* visually-hidden — checked state drives .is-active via server re-render */
    position: absolute; width: 1px; height: 1px;
    padding: 0; margin: -1px; overflow: hidden;
    clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
}
/* Keyboard focus ring only — mouse clicks must not leave a lingering outline. */
.filter-chip:has(:focus-visible) { outline: 2px solid var(--teal-600); outline-offset: 2px; }
/* Unselected: hollow ring dot — reads as "empty" without relying on color. */
.filter-chip-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: transparent;
    box-shadow: inset 0 0 0 1px var(--border-default);
    flex-shrink: 0;
}
/* Selected: dot swaps out for a checkmark — a shape change, not just a color
   change, so on/off is legible without color (WCAG AA polarity requirement).
   Semantic hue (chip-dot-*) is reused here so category meaning is preserved. */
.filter-chip-check {
    display: none;
    width: 5px; height: 5px; line-height: 5px;
    font-size: 8px; font-weight: 700;
    flex-shrink: 0;
}
.filter-chip.is-active .filter-chip-dot { display: none; }
.filter-chip.is-active .filter-chip-check { display: inline-block; }
.filter-chip.is-active .filter-chip-check.chip-dot-green { color: var(--green-600); }
.filter-chip.is-active .filter-chip-check.chip-dot-red { color: var(--red-600); }
.filter-chip.is-active .filter-chip-check.chip-dot-jade { color: var(--green-600); }
.filter-chip.is-active .filter-chip-check.chip-dot-amber { color: var(--amber-600); }
.filter-chip.is-active .filter-chip-check.chip-dot-gray { color: var(--text-faint); }
@media (prefers-reduced-motion: reduce) {
    .filter-chip { transition: none; }
}

/* min-turns slider (redteam rail, multi-turn runs only) */
.filter-slider-row {
    display: flex; align-items: center; gap: 8px;
}
.filter-slider {
    flex: 1 1 auto;
    accent-color: var(--green-600);
}
.filter-slider-readout {
    font-family: var(--font-mono);
    font-size: 10.5px; font-weight: 500;
    color: var(--text-faint);
    min-width: 2.5em; text-align: right;
    flex-shrink: 0;
}
/* A slider moved off its no-op bound is actively filtering — promote the
   readout (teal + bold) so it reads as engaged, matching the chip/dropdown
   selection vocabulary. */
.filter-slider-readout.is-engaged { color: var(--teal-600); font-weight: 700; }
.filter-slider-max {
    font-family: var(--font-mono);
    font-size: 10.5px; font-weight: 500;
    color: var(--text-faint);
    flex-shrink: 0;
}

/* "More filters" expander (redteam rail: technique/delivery/vulnerability).
   Styled as a subtle text toggle — not a boxed card — so it reads as a
   section reveal, not another filter dropdown. */
.filter-dd-more .filter-dd-trigger {
    justify-content: flex-start; gap: 4px;
    padding: 4px 2px;
    background: none; border: none; border-radius: 0;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase;
    color: var(--text-faint);
}
.filter-dd-more .filter-dd-trigger:hover { color: var(--text-body); }
/* Count of active filters hidden inside the collapsed expander (teal pill) so
   engaged controls are never silent. */
.filter-dd-more-badge {
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 15px; height: 15px; padding: 0 4px;
    border-radius: 999px;
    background: var(--teal-600); color: #fff;
    font-size: 9.5px; font-weight: 700; line-height: 1;
    letter-spacing: 0;
    flex-shrink: 0;
}
.filter-dd-more-body {
    display: flex; flex-direction: column; gap: 10px;
    margin-top: 10px;
}

/* <details> Persona/Scenario dropdowns */
.filter-dd { position: relative; }
.filter-dd-trigger {
    display: flex; align-items: center; gap: 6px;
    padding: 7px 9px;
    background: var(--surface-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    font-size: 12.5px; color: var(--text-body);
    cursor: pointer; list-style: none;
}
.filter-dd-trigger::-webkit-details-marker { display: none; }
.filter-dd-status {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--border-default);
    flex-shrink: 0;
}
/* Status dots use only tokens that clear WCAG 1.4.11 (≥3:1 non-text) against
   the white trigger; a narrowed filter (partial) reads orange = "engaged", an
   exclude-all filter (none) reads red — never a decorative chart hue. */
.filter-dd-status.is-all { background: var(--green-600); }
.filter-dd-status.is-partial { background: var(--orange-600); }
.filter-dd-status.is-none { background: var(--red-700); }
.filter-dd-name { color: var(--text-faint); }
.filter-dd-value { flex: 1 1 auto; color: var(--text-body); font-weight: 500; }
/* Engaged states also promote the status text so the trigger differs at a
   glance from the "All" default without parsing the 6px dot. */
.filter-dd-value.is-engaged { color: var(--teal-600); font-weight: 700; }
.filter-dd-value.is-none { color: var(--red-700); font-weight: 700; }
.filter-dd-chevron { flex-shrink: 0; color: var(--text-faint); }
.filter-dd[open] .filter-dd-chevron { transform: rotate(180deg); }
.filter-dd-menu {
    position: absolute; z-index: 20;
    top: calc(100% + 4px); right: 0; left: auto;
    min-width: 100%; width: max-content; max-width: 340px;
    max-height: 230px; overflow-y: auto;
    background: var(--surface-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-lg);
    padding: 6px;
}
.filter-dd-row {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 6px;
    font-size: 13px; color: var(--text-body);
    cursor: pointer;
}
.filter-dd-row input {
    width: 14px; height: 14px; border-radius: 4px;
    accent-color: var(--green-600);
}
.filter-rail-footer {
    margin-top: auto;
    padding-top: 10px;
    border-top: 1px solid var(--border-subtle);
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-faint);
}
/* "More filters" toggle rides at the very bottom, just under the counter. */
.filter-group--more { margin-top: -6px; }

.filter-sidebar,
.rt-panel {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 18px;
}
.filter-sidebar { padding: 24px 22px; }
.filter-title {
    margin: 0 0 20px;
    font-family: var(--font-display);
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text-strong);
}
.filter-group { margin-bottom: 22px; }
.filter-group:last-child { margin-bottom: 0; }
.filter-label {
    display: block;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-faint);
    margin-bottom: 10px;
}
.filter-checkbox, .filter-radio {
    display: flex; align-items: flex-start; gap: 10px;
    font-size: 14px; line-height: 1.35; color: var(--text-body);
    padding: 5px 0; cursor: pointer;
}
.filter-checkbox input, .filter-radio input {
    flex: 0 0 auto;
    width: 17px; height: 17px;
    margin: 1px 0 0;
    accent-color: var(--teal-600);
    cursor: pointer;
}


/* ==== interactive panels ============================================ */
.rt-interactive-panels, .sim-interactive-panels { margin-top: 32px; }
.rt-panel { margin-bottom: 22px; }
.rt-panel-title { margin: 0 0 14px; font-size: 17px; font-family: var(--font-display); }
.rt-panel-loading { color: var(--text-faint); font-style: italic; }

/* ==== report hero (above tabs) ===================================== */
.report-hero { margin: 4px 0 20px; }
.report-hero-kicker {
    margin: 0 0 4px;
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-faint);
}
.report-hero-title {
    margin: 0;
    font-family: var(--font-display);
    font-size: 26px; font-weight: 700; letter-spacing: -0.02em;
    color: var(--text-strong);
}
.report-hero-sub {
    margin: 4px 0 16px;
    font-size: 13px; color: var(--text-muted);
    font-family: var(--font-mono);
}

/* ==== CSS-only tabs (report bodies) ================================= */
/* Radios carry the state; labels are the tab bar; panels show on :checked.
   Radios must be direct children of .tabs and precede .tab-bar/.tab-panels so
   the sibling combinators resolve. nth-of-type pairs each radio to its panel. */
.tabs { margin-top: 8px; }
.tabs > .tab-radio { position: absolute; opacity: 0; pointer-events: none; }
.tab-bar {
    display: flex; flex-wrap: wrap; gap: 2px;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 22px;
}
.tab-label {
    padding: 9px 14px;
    font-size: 13px; font-weight: 500;
    color: var(--text-muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    white-space: nowrap;
    user-select: none;
}
.tab-label:hover { color: var(--text-strong); }
.tab-panel { display: none; }
.tab-panel.active-fallback { display: block; }
"""

# CSS-only tab switching: pair the Nth radio (by document order) to the Nth
# label and Nth panel. Generated for up to 9 tabs so the rules stay declarative
# in the inlined stylesheet (no per-instance <style> blocks).
_TAB_RULES = ''.join(
    f'.tabs > .tab-radio:nth-of-type({i}):checked ~ .tab-bar > .tab-label:nth-child({i}) '
    '{ color: var(--text-strong); border-bottom-color: var(--teal-600); }\n'
    f'.tabs > .tab-radio:nth-of-type({i}):checked ~ .tab-panels > .tab-panel:nth-child({i}) '
    '{ display: block; }\n'
    f'.tabs > .tab-radio:nth-of-type({i}):focus-visible ~ .tab-bar > .tab-label:nth-child({i}) '
    '{ outline: 2px solid var(--teal-600); outline-offset: 2px; border-radius: 4px; }\n'
    for i in range(1, 10)
)

_DASHBOARD_CSS_TAIL = """
@media (max-width: 760px) {
    .filter-swap-container { flex-direction: column; }
    .filter-form { position: static; flex-basis: auto; width: 100%; }
    .stat-band { grid-template-columns: repeat(2, 1fr); }
    .dash-row2, .dash-row-eq { grid-template-columns: 1fr; }
}
"""

# Active-tab underline: orange accent, scoped to `.report-aligned .tabs`.
# Originally scoped to `.sim-report .tabs` only (the RT tab bar was out of
# scope then); aligning RT is precisely this task, so the restriction is
# superseded — deliberately, not silently (spec §Shared infrastructure #1).
# The `.report-aligned` class gives this higher specificity than the
# surface-neutral `_TAB_RULES` above, so it wins without `!important`.
_SIM_TAB_ACCENT = ''.join(
    f'.report-aligned .tabs > .tab-radio:nth-of-type({i}):checked ~ .tab-bar > .tab-label:nth-child({i}) '
    '{ border-bottom-color: var(--orange-500); }\n'
    for i in range(1, 10)
)

# ==== .sim-report — Agent Sim report design-mockup alignment ============
# Rules scoped under `.sim-report` (report_tabs.sim_report_tabs' wrapper) per
# docs/superpowers/specs/2026-07-10-agent-sim-report-alignment-design.md.
# Surface-identical rules (exec-summary/callout, KPI band, `.rk-*`
# primitives, panel, design tables) have been promoted to `.report-aligned`
# (both `sim_report_tabs` and `redteam_report_tabs` carry that class — spec
# 2026-07-10-redteam-report-alignment-design.md §Shared infrastructure #1) so
# they apply to both surfaces; only sim-only composition (personas/scenarios
# overview, breakdown grids, config rows) stays scoped to `.sim-report` here.
# Consumes report_kit.py primitives (exec_summary/panel/bar_rows/tag) and the
# shared `.kpi-band`/`.kpi-card` markup from common/reports/report.css — the
# overrides here only apply inside `.report-aligned`, so the landing-page KPI
# tiles and the flat HTML export (which never carry this class) are untouched.
_SIM_REPORT_CSS = """
.report-aligned .tab-count {
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    background: var(--surface-sunken); color: var(--text-muted);
    border-radius: 999px; padding: 1px 7px; margin-left: 5px;
}

/* ---- Executive summary callout (spec Overview.1) ---- */
.report-aligned .exec-summary {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
}
.report-aligned .es-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.report-aligned .es-label {
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-faint);
}
.report-aligned .es-confidence {
    font-family: var(--font-mono); font-size: 10px; font-weight: 600;
    border: 1px solid; border-radius: 999px; padding: 2px 8px;
}
.report-aligned .es-body {
    margin: 8px 0 0; font-size: 14px; line-height: 1.6; color: var(--text-body);
    max-width: 760px;
}
.report-aligned .es-body strong { color: var(--text-strong); }

/* ---- KPI band (spec Overview.2). One equal column per card, whatever the
   count (red team = 5, sim = 6), so neither leaves a blank slot. ---- */
.report-aligned .kpi-band {
    display: grid; grid-auto-flow: column; grid-auto-columns: 1fr; gap: 12px;
    margin: 16px 0;
}
.report-aligned .kpi-card {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 15px 16px;
}
/* State lives on the number, not a bar. Neutral keeps --text-strong. */
.report-aligned .kpi-card--pass .kpi-value { color: var(--green-600); }
.report-aligned .kpi-card--fail .kpi-value { color: var(--red-600); }
.report-aligned .kpi-card--warn .kpi-value { color: var(--amber-600); }
.report-aligned .kpi-value {
    font-family: var(--font-mono); font-size: 28px; font-weight: 600;
    color: var(--text-strong); line-height: 1.1;
}
.report-aligned .kpi-label { font-size: 12px; color: var(--text-muted); margin-top: 7px; }

/* ---- 2-col grids (donut+tokens, personas+scenarios) ---- */
.sim-report .sim-overview-grid-2 {
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;
}
@media (max-width: 760px) {
    .sim-report .sim-overview-grid-2 { grid-template-columns: 1fr; }
}

/* ---- Panel wrapper (report_kit.panel) ---- */
.report-aligned .rk-panel {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px 20px;
}
.report-aligned .rk-panel-title {
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-faint);
}
.report-aligned .rk-panel-sub { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
.report-aligned .rk-panel-body { margin-top: 12px; }

/* ---- Tag (report_kit.tag) ---- */
.report-aligned .rk-tag {
    display: inline-block; font-size: 11px; font-weight: 500;
    border: 1px solid var(--border-default); border-radius: 999px;
    padding: 1px 8px; margin-left: 8px; color: var(--text-muted);
}

.report-aligned .rk-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 2px 9px; border-radius: 999px;
    font-size: 11px; font-weight: 600; line-height: 1.5; white-space: nowrap;
}
.report-aligned .rk-pill-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }

/* ---- Personas panel (Overview) ---- */
.sim-report .sim-persona-item + .sim-persona-item { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-subtle); }
.sim-report .sim-persona-row { display: flex; align-items: center; gap: 4px; }
.sim-report .sim-persona-name { font-family: var(--font-mono); font-size: 13px; font-weight: 600; color: var(--text-strong); }
.sim-report .sim-persona-count { margin-left: auto; font-size: 11px; color: var(--text-faint); }
.sim-report .sim-trait-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px;
}
.sim-report .sim-trait-bar { display: flex; align-items: center; gap: 6px; }
.sim-report .sim-trait-label { width: 86px; flex-shrink: 0; font-size: 11px; color: var(--text-faint); }
.sim-report .sim-trait-track {
    flex: 1; height: 5px; border-radius: 3px; background: var(--surface-sunken); overflow: hidden;
}
.sim-report .sim-trait-fill { display: block; height: 100%; background: var(--teal-600); border-radius: 3px; }

/* ---- Scenarios panel (Overview) ---- */
.sim-report .sim-scenario-item + .sim-scenario-item { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-subtle); }
.sim-report .sim-scenario-name { font-size: 13px; font-weight: 600; color: var(--text-strong); }
.sim-report .sim-scenario-goal { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
.sim-report .sim-criterion {
    display: flex; align-items: baseline; gap: 8px; margin-top: 6px; font-size: 12px; color: var(--text-muted);
}
.sim-report .sim-criterion-type {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase; flex-shrink: 0;
}

/* ---- Breakdown tab: stacked panels + heatmap/histogram/tables ---- */
.report-aligned .rk-panel + .rk-panel,
.report-aligned .rk-panel + .report-card,
.report-aligned .report-card + .rk-panel {
    margin-top: 20px;
}

/* HTML-table heatmap (report_kit.heatmap) */
.report-aligned .rk-heatmap { border-collapse: separate; border-spacing: 4px; }
.report-aligned .rk-heat-col {
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    color: var(--text-muted); text-align: center; padding: 0 4px 6px;
}
.report-aligned .rk-heat-row {
    font-size: 12px; font-weight: 600; color: var(--text-strong);
    text-align: right; padding-right: 10px; white-space: nowrap;
}
.report-aligned .rk-heat-cell {
    min-width: 50px; height: 34px; border-radius: 5px;
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    text-align: center; vertical-align: middle;
}
.report-aligned .rk-heat-empty { background: var(--surface-sunken); color: var(--text-faint); }

/* Per-persona / per-scenario tables (html_table output) + failures table */
.report-aligned .rk-panel-body { display: grid; }
.report-aligned table {
    width: 100%; border-collapse: collapse;
}
.report-aligned table thead th {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase;
    font-weight: 600; color: var(--text-faint); background: var(--surface-sunken);
    padding: 11px 16px; text-align: left;
}
.report-aligned table tbody td {
    font-size: 13px; padding: 12px 16px; border-bottom: 1px solid var(--border-subtle);
}
.report-aligned table tbody tr:last-child td { border-bottom: none; }
.report-aligned table thead th:not(:first-child),
.report-aligned table tbody td:not(:first-child) {
    text-align: right; font-variant-numeric: tabular-nums;
}

/* ---- Turn quality tab (spec §Turn) ---- */
/* Line chart legend (report_kit.line_chart) */
.report-aligned .rk-legend {
    display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px;
}
.report-aligned .rk-legend-item {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; color: var(--text-muted);
}
.report-aligned .rk-legend-swatch {
    display: inline-block; width: 8px; height: 8px; border-radius: 2px;
}
/* Average quality metric stat tiles (spec §Turn.3) */
.sim-report .sim-stat-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
}
.sim-report .sim-stat-tile {
    background: var(--surface-sunken); border-radius: 8px; padding: 12px 14px;
}
.sim-report .sim-stat-value {
    font-family: var(--font-mono); font-size: 24px; font-weight: 600;
    color: var(--text-strong);
}
.sim-report .sim-stat-label {
    font-size: 11px; color: var(--text-muted); margin-top: 4px;
    text-transform: capitalize;
}

/* ---- Config tab (spec §Config) ---- */
/* Run-scale stat tiles (persona/scenario/conversation counts) — big number on
   a sunken card so the run's scale reads at a glance above the textual config. */
.report-aligned .rk-stat-row {
    display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 22px;
}
.report-aligned .rk-stat {
    flex: 1 1 120px; min-width: 120px;
    display: flex; flex-direction: column; gap: 4px;
    padding: 14px 16px; border-radius: 10px;
    background: var(--surface-sunken); border: 1px solid var(--border-subtle);
}
.report-aligned .rk-stat-num {
    font-family: var(--font-display); font-size: 28px; font-weight: 700;
    line-height: 1; letter-spacing: -0.02em; color: var(--text-strong);
}
.report-aligned .rk-stat-cap {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--text-faint);
}
/* Grouped config metadata — a mono sub-label + divider per semantic cluster. */
.report-aligned .rk-meta-group + .rk-meta-group { margin-top: 20px; }
.report-aligned .rk-meta-group-title {
    font-family: var(--font-mono); font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted);
    padding-bottom: 8px; margin-bottom: 12px;
    border-bottom: 1px solid var(--border-subtle);
}
/* Run-configuration meta grid (report_kit.meta_grid) */
.report-aligned .rk-meta-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 16px 24px;
}
.report-aligned .rk-meta-key {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--text-faint);
}
.report-aligned .rk-meta-value {
    font-size: 13.5px; color: var(--text-body); margin-top: 4px;
}
/* Personas panel — grid so name · tone · trait bars align under column headers */
.sim-report .sim-config-persona-head,
.sim-report .sim-config-persona-row {
    display: grid;
    /* fixed widths (not auto/1fr per-content) so columns line up across the separate per-row grids;
       trait columns are wider than the longest header label so the labels don't collide */
    grid-template-columns: minmax(0, 1fr) 110px repeat(4, 104px);
    align-items: center; gap: 16px;
}
.sim-report .sim-config-persona-head {
    padding-bottom: 8px; margin-bottom: 12px; border-bottom: 1px solid var(--border-subtle);
}
.sim-report .sim-config-persona-head > span {
    font-family: var(--font-mono); font-size: 10px; color: var(--text-faint);
    white-space: nowrap;
}
.sim-report .sim-config-persona-row {
    margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border-subtle);
}
.sim-report .sim-config-persona-row:first-of-type { margin-top: 0; padding-top: 0; border-top: none; }
.sim-report .sim-config-persona-name {
    font-size: 13px; font-weight: 400; color: var(--text-strong);
}
.sim-report .sim-config-persona-style {
    font-family: var(--font-mono); font-size: 11px; color: var(--text-faint);
}
.sim-report .sim-config-persona-bg { font-size: 12.5px; color: var(--text-muted); }
/* Scenarios panel (name + goal + criteria chips) */
.sim-report .sim-config-scenario-head,
.sim-report .sim-config-scenario-row {
    display: grid;
    /* fixed right columns so the header labels line up over the values across per-row grids */
    grid-template-columns: minmax(0, 1fr) 200px 76px;
    align-items: start; gap: 16px;
}
.sim-report .sim-config-scenario-head {
    padding-bottom: 8px; margin-bottom: 12px; border-bottom: 1px solid var(--border-subtle);
}
.sim-report .sim-config-scenario-head > span {
    font-family: var(--font-mono); font-size: 10px; color: var(--text-faint); white-space: nowrap;
}
.sim-report .sim-config-scenario-head > span:nth-child(n+2) { justify-self: end; }
.sim-report .sim-config-scenario-checks,
.sim-report .sim-config-scenario-rate-cell { justify-self: end; align-self: center; }
.sim-report .sim-config-scenario-row {
    margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border-subtle);
}
.sim-report .sim-config-scenario-row:first-of-type { margin-top: 0; padding-top: 0; border-top: none; }
.sim-report .sim-config-scenario-name { font-size: 13px; font-weight: 400; color: var(--text-strong); }
.sim-report .sim-config-scenario-goal { font-size: 12.5px; color: var(--text-muted); margin-top: 3px; }
.sim-report .sim-config-criteria {
    display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;
}
.sim-report .sim-config-criterion {
    font-family: var(--font-mono); font-size: 11.5px; background: var(--surface-sunken);
    border-radius: 6px; padding: 3px 9px;
}
"""

# ==== .sim-report — Transcripts redesign (Tasks 11+12) ===================
# Appended as its own block (rather than edited into _SIM_REPORT_CSS above)
# so a concurrently-landing `.sim-report` CSS addition on another branch
# merges cleanly. Covers: tinted conversation drawer rows (Task 11) +
# judge callout / chat bubbles / two-state criteria (Task 12), both scoped
# under `.sim-report` per docs/superpowers/specs/2026-07-10-agent-sim-report-
# alignment-design.md §Transcripts.
_SIM_TRANSCRIPT_CSS = """
/* ---- Conversation table (sortable + paginated) ---- */
.sim-report .sim-conv-table {
    width: 100%; border-collapse: collapse; font-size: 13px;
    border: 1px solid var(--border-subtle); border-radius: 8px; overflow: hidden;
}
.sim-report .sim-conv-table thead th {
    text-align: left; padding: 0; background: var(--surface-sunken);
    border-bottom: 1px solid var(--border-subtle); white-space: nowrap;
}
.sim-report .sim-th-sort {
    display: inline-flex; align-items: center; gap: 4px; width: 100%;
    padding: 10px 12px; background: none; border: 0; cursor: pointer;
    font: inherit; font-weight: 600; color: var(--text-strong); text-align: left;
}
.sim-report .sim-th-sort:hover { color: var(--teal-600); }
/* Inactive sortable columns show a faint ⇅; it strengthens on hover so the
   header reads as clickable. The active column's ▲/▼ is always full-strength. */
.sim-report .sim-th-caret { font-size: 9px; color: var(--text-faint); }
.sim-report .sim-th-sort:hover .sim-th-caret { color: var(--teal-600); }
.sim-report .sim-th-caret-active { color: var(--teal-600); }
.sim-report .sim-conv-table tbody td { padding: 10px 12px; }
.sim-report .sim-conv-row {
    border-top: 1px solid var(--border-subtle);
    cursor: pointer;
}
.sim-report .sim-conv-row:focus-visible {
    outline: 2px solid var(--teal-600); outline-offset: -2px;
}
.sim-report .sim-conv-idx { font-family: var(--font-mono); font-size: 12px; color: var(--text-faint); }
.sim-report .sim-conv-persona { font-weight: 600; color: var(--text-strong); }
.sim-report .sim-conv-scenario { color: var(--text-body); }
.sim-report .sim-conv-turns, .sim-report .sim-conv-score { font-variant-numeric: tabular-nums; }
.sim-report .sim-conv-trace .trace-link { white-space: nowrap; }

/* ---- Pager ---- */
.sim-report .sim-pager {
    display: grid; grid-template-columns: 1fr auto 1fr; align-items: center;
    margin-top: 10px; font-size: 12px; color: var(--text-faint);
}
/* Pager group centred (col 2); size selector pinned right (col 3). */
.sim-report .sim-pager-nav { grid-column: 2; display: flex; align-items: center; gap: 12px; }
.sim-report .sim-size { grid-column: 3; justify-self: end; }
.sim-report .sim-pager-btn {
    padding: 5px 12px; border: 1px solid var(--border-subtle); border-radius: 6px;
    background: var(--surface-app); color: var(--text-body); cursor: pointer; font: inherit;
}
.sim-report .sim-pager-btn:hover:not([disabled]) { border-color: var(--teal-600); color: var(--teal-600); }
.sim-report .sim-pager-btn[disabled] { opacity: 0.4; cursor: default; }
.sim-report .sim-size { display: flex; align-items: center; gap: 4px; }
.sim-report .sim-size-label { margin-right: 2px; }
.sim-report .sim-size-btn {
    padding: 4px 9px; border: 1px solid var(--border-subtle); border-radius: 6px;
    background: var(--surface-app); color: var(--text-body); cursor: pointer; font: inherit;
}
.sim-report .sim-size-btn:hover:not([disabled]) { border-color: var(--teal-600); color: var(--teal-600); }
/* Active size is disabled (current selection), so give it the teal fill instead of the faded look. */
.sim-report .sim-size-btn[aria-current="true"] {
    opacity: 1; cursor: default; border-color: var(--teal-600);
    background: var(--teal-600); color: var(--surface-app);
}

/* ---- Transcript fragment: judge callout / bubbles / criteria (Task 12) ---- */
.sim-report .sim-judge {
    background: var(--surface-sunken);
    border-radius: 4px; padding: 10px 14px; margin-bottom: 14px;
}
.sim-report .sim-judge-label {
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    text-transform: uppercase; color: var(--teal-600); display: block;
}
.sim-report .sim-judge-reason { font-size: 13px; line-height: 1.55; margin: 4px 0 0; color: var(--text-body); }
.sim-report .sim-transcript-error {
    background: var(--red-100); color: var(--red-600); border-radius: 4px;
    padding: 10px 14px; margin-bottom: 14px; font-size: 13px;
}
.sim-report .sim-transcript-grid { display: flex; flex-direction: column; gap: 0; }
/* Hairline between the criteria block and the conversation below it, matching
   the summary's bottom rule so the drawer reads as three stacked sections. */
.sim-report .sim-transcript-grid > * + * {
    margin-top: 24px; padding-top: 24px; border-top: 1px solid var(--border-subtle);
}

/* Chat bubbles (render_message_list avatar + side extension). Shared with
   the Red Team transcript fragment, which calls the same
   `render_message_list` with `class_prefix='rt'` (spec 2026-07-10-redteam-
   report-alignment-design.md §Shared infrastructure #4) — one declaration
   set, two selectors per rule, so the two reports cannot drift. */
.report-aligned .sim-msg, .report-aligned .rt-msg {
    display: flex; gap: 8px; margin-bottom: 10px; max-width: 88%;
}
.report-aligned .sim-msg-user, .report-aligned .sim-msg-system,
.report-aligned .rt-msg-user, .report-aligned .rt-msg-system {
    margin-right: auto;
}
.report-aligned .sim-msg-assistant, .report-aligned .sim-msg-tool,
.report-aligned .rt-msg-assistant, .report-aligned .rt-msg-tool {
    margin-left: auto; flex-direction: row-reverse;
}
.report-aligned .sim-msg-avatar, .report-aligned .rt-msg-avatar {
    flex-shrink: 0; width: 30px; height: 30px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-mono); font-size: 9px; font-weight: 600;
    background: var(--ink-900); color: #fff;
}
.report-aligned .sim-msg-assistant .sim-msg-avatar, .report-aligned .sim-msg-tool .sim-msg-avatar,
.report-aligned .rt-msg-assistant .rt-msg-avatar, .report-aligned .rt-msg-tool .rt-msg-avatar {
    background: var(--teal-50); color: var(--teal-600);
}
.report-aligned .sim-msg-bubble, .report-aligned .rt-msg-bubble {
    background: var(--surface-sunken); border-radius: 12px; padding: 9px 13px;
}
.report-aligned .sim-msg-assistant .sim-msg-bubble, .report-aligned .sim-msg-tool .sim-msg-bubble,
.report-aligned .rt-msg-assistant .rt-msg-bubble, .report-aligned .rt-msg-tool .rt-msg-bubble {
    background: var(--teal-50);
}
.report-aligned .sim-msg-role, .report-aligned .rt-msg-role {
    display: block; font-family: var(--font-mono); font-size: 10px;
    text-transform: uppercase; color: var(--text-faint); margin-bottom: 3px;
}
.report-aligned .sim-msg-content, .report-aligned .rt-msg-content {
    font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-word;
    margin: 0; font-family: inherit;
}

/* Criteria column (two-state per deviation #4) */
.sim-report .sim-criteria-header {
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    text-transform: uppercase; color: var(--text-faint); margin-bottom: 8px;
}
.sim-report .sim-criteria-list { list-style: none; margin: 0; padding: 0; }
.sim-report .sim-criterion {
    display: flex; align-items: flex-start; gap: 8px; padding: 8px 0;
    border-top: 1px solid var(--border-subtle);
}
.sim-report .sim-criterion:first-child { border-top: none; }
.sim-report .sim-criterion-icon {
    flex-shrink: 0; width: 18px; height: 18px; border-radius: 999px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; color: #fff;
}
.sim-report .sim-criterion-pass .sim-criterion-icon { background: var(--green-600); }
.sim-report .sim-criterion-fail .sim-criterion-icon { background: var(--red-600); }
.sim-report .sim-criterion-desc { font-size: 13px; color: var(--text-body); flex: 1; }
.sim-report .sim-ctype {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase;
    color: var(--text-faint); white-space: nowrap;
}
.sim-report .sim-ctype-unsafe { color: var(--red-600); }
"""

# ==== .sim-report — Agent Sim overrides layered on `.report-aligned` ======
# `.sim-report` and `.report-aligned` have equal specificity (one class
# each), so these rules must stay *after* _SIM_REPORT_CSS / _SIM_TRANSCRIPT_CSS
# in source order for the sim-specific values to win over the shared
# `.report-aligned` defaults on the sim report only (Option 3: mostly split,
# reuse identicals — spec docs/superpowers/specs/2026-07-10-agent-sim-report-
# alignment-design.md). Rules whose declarations were identical to their
# `.report-aligned` twin were dropped as redundant; only the ones that
# diverge (font, spacing, frameless tables/heatmap, etc.) or are sim-only are
# kept here.
_SIM_REPORT_OVERRIDES_CSS = """
/* Outcomes donut lives in a .chart-card (not a .rk-panel). Give it the same
   white-card chrome as its grid sibling so it doesn't sit bare on the page
   background, and make it a flex column that stretches to the sibling's height
   with the donut + legend centered on both axes. */
.sim-report .chart-card {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 16px 20px;
    margin: 0;
}
.sim-report .sim-overview-grid-2 > .chart-card { display: flex; flex-direction: column; }
.sim-report .chart-card .donut-wrap { flex-direction: column; justify-content: center; align-items: center; gap: 20px; flex: 1; }
.sim-report .chart-card .donut-legend { flex-direction: row; flex-wrap: wrap; justify-content: center; gap: 16px; }
/* Sim report is all-sans to match the mockup (no monospace anywhere). */
.sim-report .report-hero-sub { font-family: var(--font-sans); }

.sim-report .tab-count {
    font-family: var(--font-sans); font-size: 11px; font-weight: 600;
    background: var(--surface-sunken); color: var(--text-muted);
    border-radius: 999px; padding: 1px 7px; margin-left: 5px;
}
.sim-report .es-label {
    font-family: var(--font-sans); font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-faint);
}
.sim-report .es-confidence {
    font-family: var(--font-sans); font-size: 10px; font-weight: 600;
    border: 1px solid; border-radius: 999px; padding: 2px 8px;
}
.sim-report .kpi-value {
    font-family: var(--font-sans); font-size: 28px; font-weight: 600;
    color: var(--text-strong); line-height: 1.1;
}

/* ---- 2-col grids (donut+tokens, personas+scenarios): stretch-height fixes ---- */
.sim-report .sim-overview-grid-2 {
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;
    align-items: stretch;
}
/* Panels in this grid stretch to equal height; let each panel's body fill that
   height so the donut can center vertically and the stat grid fills its half. */
.sim-report .sim-overview-grid-2 > .rk-panel { display: flex; flex-direction: column; }
/* The 2nd grid child matches the stacked-panel `.rk-panel + .rk-panel` rule and
   inherits its 20px top margin, which (in a stretch grid) shrinks its border-box
   by 20px so the pair no longer matches. Same-specificity + later source = it wins,
   so override with an extra-class selector. */
.sim-report .sim-overview-grid-2 > .rk-panel + .rk-panel { margin-top: 0; }
.sim-report .sim-overview-grid-2 > .rk-panel > .rk-panel-body { flex: 1; }
.sim-report .sim-overview-grid-2 .sim-aq-grid { height: 100%; grid-auto-rows: 1fr; }
.sim-empty-note { color: var(--text-faint); font-size: 12px; padding: 12px 2px; margin: 0; }
/* Personas + scenarios row: unlike the donut row above it, these list-cards have
   no reason to match heights — stretching the shorter one leaves a dead void.
   Let each hug its content. */
.sim-report .sim-overview-grid-2--top { align-items: start; }
@media (max-width: 760px) {
    .sim-report .sim-overview-grid-2 { grid-template-columns: 1fr; }
}

/* Body rhythm: the report stylesheet sets line-height 1.65 globally; the mockup
   is 1.5. Set it once at the region root so every tab inherits the tighter
   rhythm (elements that need their own leading still override locally). */
.sim-report { line-height: var(--leading-body); }

.sim-report .rk-panel {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 16px 20px;
}
/* Panel titles match the mockup: serif (display), sentence-case, ~18px bold.
   Mono-uppercase is the mockup's *eyebrow* style (exec-summary label, filter
   labels, table column headers) — not panel titles. */
.sim-report .rk-panel-title {
    font-family: var(--font-display); font-size: 16px; font-weight: 600;
    letter-spacing: -0.01em; color: var(--text-strong);
}
.sim-report .rk-panel-sub { font-size: 12px; color: var(--text-muted); margin-top: 3px; }
.rk-panel-info { margin-left: 6px; color: var(--text-muted); cursor: help; font-size: 0.85em; vertical-align: middle; }
.rk-panel-info:hover { color: var(--text-strong); }

.sim-report .sim-persona-name { font-family: var(--font-sans); font-size: 13px; font-weight: 600; color: var(--text-strong); }
.sim-report .sim-criterion-type {
    font-family: var(--font-sans); font-size: 10px; text-transform: uppercase; flex-shrink: 0;
}

/* HTML-table heatmap (report_kit.heatmap) */
/* Cell metrics measured off the mockup DOM: 92x34, radius 5, 11px mono 600,
   compact (NOT full-width blow-up). border-spacing matches the mockup gap. */
/* No table chrome: mockup floats the cells directly on the panel — no enclosing
   frame, no fill, no radius (the global report-table border/bg/radius must be
   stripped here). */
.sim-report .rk-heatmap {
    border-collapse: separate; border-spacing: 6px;
    border: none; background: transparent; border-radius: 0;
}
/* All heatmap header cells (incl. the empty top-left corner) are transparent and
   carry no rules — mockup has no header underline or row-label divider. */
.sim-report .rk-heatmap th { background: transparent; border: none; }
.sim-report .rk-heatmap tbody tr,
.sim-report .rk-heatmap tbody tr:hover,
.sim-report .rk-heatmap tbody tr:nth-child(even) { background: transparent; }
/* Both axes AND the cells use the same grotesque sans (--font-sans) in the mockup
   — not mono for the x-axis and serif for the y-axis (that split read as two
   mismatched fonts). Measured: col 11/600 muted, row 12/600 strong, cell 16/400. */
.sim-report .rk-heat-col {
    font-family: var(--font-sans); font-size: 11px; font-weight: 600;
    color: var(--text-muted); text-align: center; padding: 0 4px 8px;
    text-transform: none;  /* override global report-table th uppercasing */
    background: transparent;
}
.sim-report .rk-heat-row {
    font-family: var(--font-sans); font-size: 12px; font-weight: 600;
    color: var(--text-strong); text-align: right; padding-right: 16px; line-height: 1.3;
    text-transform: none;  /* override global report-table th uppercasing */
    background: transparent;
}
.sim-report .rk-heat-cell {
    min-width: 88px; height: 34px; border-radius: 5px;
    font-family: var(--font-sans); font-size: 16px; font-weight: 400;
    text-align: center; vertical-align: middle;
}

/* Data tables — exclude the heatmap (.rk-heatmap has its own cell styling; the
   generic thead rule was bleeding uppercase + sunken bg onto its headers). */
/* Frameless like the mockup: strip the global report-table chrome (outer border,
   surface fill, radius, margin) and the nth-child zebra. Rows are separated by
   padding only — the mockup has no row dividers, just a header underline. */
.sim-report table:not(.rk-heatmap) {
    width: 100%; border-collapse: collapse;
    border: none; background: transparent; border-radius: 0; margin: 0;
}
.sim-report table:not(.rk-heatmap) thead th {
    font-family: var(--font-sans); font-size: 10px; text-transform: uppercase;
    font-weight: 600; color: var(--text-faint); background: var(--surface-sunken);
    padding: 11px 16px; text-align: left; border-bottom: 1px solid var(--border-subtle);
}
.sim-report table:not(.rk-heatmap) tbody td {
    font-size: 13px; padding: 12px 16px; border-bottom: none;
}
.sim-report table:not(.rk-heatmap) tbody tr,
.sim-report table:not(.rk-heatmap) tbody tr:nth-child(even) { background: transparent; }
.sim-report table:not(.rk-heatmap) thead th:not(:first-child),
.sim-report table:not(.rk-heatmap) tbody td:not(:first-child) {
    text-align: right; font-variant-numeric: tabular-nums;
}

/* Failures table: collapsed pass/fail dots that unfold to full criteria text.
   Height animates both ways via ::details-content + interpolate-size where
   supported (Chrome/Edge); elsewhere it snaps open — both fully usable. */
.sim-report .crit-cell { display: inline-block; interpolate-size: allow-keywords; }
.sim-report .crit-summary {
    list-style: none; cursor: pointer; display: inline-flex; gap: 6px;
    align-items: center; justify-content: flex-end; padding: 3px 6px;
    margin: -3px -6px; border-radius: 999px; border: 1px solid transparent;
    transition: background .15s ease, border-color .15s ease;
}
.sim-report .crit-summary::-webkit-details-marker { display: none; }
.sim-report .crit-summary:hover { background: var(--surface-sunken); border-color: var(--border-subtle); }
.sim-report .crit-summary:focus-visible { outline: 2px solid var(--green-600); outline-offset: 2px; }
.sim-report .crit-dots { display: inline-flex; gap: 4px; align-items: center; }
.sim-report .crit-dot {
    width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none;
    transition: transform .15s ease;
}
.sim-report .crit-summary:hover .crit-dot { transform: scale(1.15); }
.sim-report .crit-dot--pass { background: var(--green-600); }
.sim-report .crit-dot--fail { background: var(--red-600); }
.sim-report .crit-dot--safety { background: var(--orange-500); }
/* Unaudited: hollow, so it reads as "unknown" rather than as a pass. */
.sim-report .crit-dot--unknown { background: var(--surface-sunken); box-shadow: inset 0 0 0 1px var(--border-strong); }
.sim-report .crit-caret {
    width: 13px; height: 13px; flex: none; color: var(--text-faint);
    transition: transform .2s ease, color .15s ease;
}
.sim-report .crit-summary:hover .crit-caret { color: var(--text-muted); }
.sim-report .crit-cell[open] .crit-caret { transform: rotate(180deg); }
.sim-report .crit-empty { color: var(--text-faint); }
.sim-report .crit-cell::details-content {
    height: 0; overflow: hidden; opacity: 0;
    transition: height .22s ease, opacity .18s ease, content-visibility .22s allow-discrete;
    content-visibility: hidden;
}
.sim-report .crit-cell[open]::details-content { height: auto; opacity: 1; content-visibility: visible; }
.sim-report .crit-list {
    margin: 8px 0 0; padding: 10px 12px; list-style: none; text-align: left;
    font-size: 12px; line-height: 1.45; min-width: 240px; max-width: 460px;
    background: var(--surface-sunken); border: 1px solid var(--border-subtle);
    border-radius: 8px;
}
.sim-report .crit-li { padding: 3px 0 3px 16px; position: relative; color: var(--text-body); }
.sim-report .crit-li + .crit-li { border-top: 1px solid var(--border-subtle); margin-top: 3px; padding-top: 6px; }
.sim-report .crit-li::before {
    content: ''; position: absolute; left: 0; top: 8px;
    width: 7px; height: 7px; border-radius: 50%;
}
.sim-report .crit-li + .crit-li::before { top: 11px; }
.sim-report .crit-li--pass::before { background: var(--green-600); }
.sim-report .crit-li--fail::before { background: var(--red-600); }
.sim-report .crit-li--safety::before { background: var(--orange-500); }
.sim-report .crit-li--unknown::before { background: var(--text-faint); }
.sim-report .crit-evidence { margin: 4px 0 0; padding-left: 18px; font-size: 12px; font-style: italic; color: var(--text-muted); }
@media (prefers-reduced-motion: reduce) {
    .sim-report .crit-summary, .sim-report .crit-dot, .sim-report .crit-caret,
    .sim-report .crit-cell::details-content { transition: none; }
}

/* Full-width per-persona / per-scenario tables: name column takes the slack so
   long names sit on one line; goal-rate + avg-score values are tinted by value. */
.sim-report .sim-bd-table th:first-child,
.sim-report .sim-bd-table td:first-child { width: 52%; }
.sim-report .sim-td-tint { font-weight: 600; }
.sim-report .sim-breakdown-grid-2 {
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
    /* Top/bottom margin so the row doesn't hug the panel above it — this div isn't
       a .rk-panel, so the `.rk-panel + .rk-panel` gap rule never fires for it. */
    margin: 20px 0;
    /* Each table panel sizes to its own content — don't stretch the shorter one
       and leave a blank gap under its last row. */
    align-items: start;
}
@media (max-width: 760px) {
    .sim-report .sim-breakdown-grid-2 { grid-template-columns: 1fr; }
}
/* The 2nd child otherwise matches the stacked-panel `.rk-panel + .rk-panel` rule
   and inherits its 20px top margin, pushing it below the first panel so their
   tops no longer line up. Zero it here (same fix as sim-overview-grid-2). */
.sim-report .sim-breakdown-grid-2 > .rk-panel + .rk-panel { margin-top: 0; }
/* Fixed-size SVG line chart: never upscale above native (that magnified the axis
   text); shrink to fit on narrow (<720px) cards instead of overflowing. */
.sim-report .rk-line-chart { max-width: 100%; height: auto; }

/* ---- Turn quality tab: legend + editorial average-quality quadrants ---- */
.sim-report .rk-legend {
    /* padding-left aligns the swatches with the chart's plot origin (line_chart
       pad_left = 36px), so the legend reads as belonging to the axes. */
    display: flex; flex-wrap: wrap; gap: 16px; margin-top: 10px; padding-left: 36px;
}
.sim-report .rk-legend-item {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; color: var(--text-muted);
}
.sim-report .rk-legend-swatch {
    /* A short bar (not a square dot) mirrors the line stroke it labels. */
    display: inline-block; width: 12px; height: 2.5px; border-radius: 2px;
}
/* Average quality metrics — editorial quadrants: big mono values, small-caps
   labels, per-metric accent tick (color from _interp_color). Dividers come from
   a 1px grid gap over a tinted background so they render cleanly for any metric
   count (1/2/3/4+), not just an exact 2x2. */
.sim-report .sim-aq-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 1px; background: var(--border-subtle);
}
.sim-report .sim-aq-cell { background: var(--surface-card); padding: 14px 16px; }
.sim-report .sim-aq-label {
    font-family: var(--font-sans); font-size: 10px; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--text-faint);
    /* Reserve two lines so single- and double-line labels keep their values on
       a common baseline across the 2x2. */
    min-height: 2.2em; line-height: 1.25;
}
.sim-report .sim-aq-value {
    font-family: var(--font-sans); font-size: 30px; font-weight: 600; line-height: 1.1;
    margin-top: 8px; color: var(--text-strong);
}
.sim-report .sim-aq-value::before {
    content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 2px;
    margin-right: 9px; vertical-align: middle; background: var(--aq-accent, var(--orange-500));
}

/* ---- Config tab: sans-serif label overrides ---- */
.sim-report .rk-meta-key {
    font-family: var(--font-sans); font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text-faint);
}
.sim-report .sim-entity-row {
    width: 100%; border: 0; background: transparent; font: inherit; color: inherit;
    appearance: none; padding-right: 0; padding-bottom: 0; padding-left: 0;
    text-align: left; cursor: pointer;
}
.sim-report .sim-entity-row:hover .sim-config-persona-name,
.sim-report .sim-entity-row:hover .sim-config-scenario-name { color: var(--orange-500); }
.sim-report .sim-config-persona-style {
    font-family: var(--font-sans); font-size: 12px; color: var(--text-muted);
}
/* display:contents lets the 4 bars sit as direct grid cells under the trait headers */
.sim-report .sim-config-persona-row .sim-trait-minis { display: contents; }
.sim-report .sim-trait-mini {
    position: relative; width: 34px; height: 8px; border-radius: 999px;
    background: var(--surface-sunken); overflow: visible; vertical-align: middle;
}
.sim-report .sim-trait-mini--empty { background: transparent; }
.sim-report .sim-trait-mini-fill {
    display: block; height: 100%; border-radius: inherit; background: var(--teal-600);
}
/* instant tooltip on hover — native title has a ~1s delay */
.sim-report .sim-trait-mini[data-tip]:hover::after {
    content: attr(data-tip);
    position: absolute; bottom: calc(100% + 6px); left: 0;
    padding: 3px 7px; border-radius: 5px; white-space: nowrap;
    font-family: var(--font-sans); font-size: 11px; line-height: 1.2;
    background: var(--text-strong); color: #fff;
    box-shadow: var(--shadow-lg); pointer-events: none; z-index: 5;
}
.sim-report .sim-config-check-count {
    font-size: 12px; color: var(--text-muted); white-space: nowrap;
}
.sim-report .sim-config-scenario-main { min-width: 0; }
.sim-report .sim-config-criterion {
    font-family: var(--font-sans); font-size: 11.5px; background: var(--surface-sunken);
    border-radius: 6px; padding: 3px 9px;
}

/* Top failure modes panel — min-count slider filters bars client-side. */
.sim-report .sim-fm-head {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
}
.sim-report .sim-fm-filter {
    display: flex; align-items: center; gap: 8px;
    text-transform: none; letter-spacing: normal;
}
.sim-report .sim-fm-filter label {
    font-family: var(--font-mono); font-size: 10px; color: var(--text-faint);
}
.sim-report .sim-fm-slider { width: 120px; accent-color: var(--red-600); }
.sim-report .sim-fm-out {
    font-family: var(--font-mono); font-size: 12px; font-weight: 600;
    color: var(--text-strong); min-width: 1.4em; text-align: right;
}
.sim-report .sim-fm-bars { display: flex; flex-direction: column; gap: 8px; }
.sim-report .sim-fm-row {
    display: grid; grid-template-columns: minmax(0, 1fr) 200px 32px;
    align-items: center; gap: 12px;
}
/* display:grid above beats the UA [hidden]{display:none}; restore it so the
   server-side default filter and the slider actually hide sub-threshold rows. */
.sim-report .sim-fm-row[hidden] { display: none; }
.sim-report .sim-fm-label {
    font-size: 12px; color: var(--text-body); white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}
.sim-report .sim-fm-track {
    height: 14px; background: var(--chart-track); border-radius: 3px; overflow: hidden;
}
.sim-report .sim-fm-fill { display: block; height: 100%; background: var(--red-600); border-radius: 3px; }
.sim-report .sim-fm-count {
    font-family: var(--font-sans); font-size: 11px; font-weight: 600; color: var(--text-strong);
}
.sim-report .sim-fm-empty { font-size: 12px; color: var(--text-muted); margin: 8px 0 0; }

/* Scenario pass-rate pill (Config panel) */
.sim-report .sim-scenario-rate {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 2px 9px; border-radius: 999px;
    font-family: var(--font-mono); font-size: 11px; font-weight: 400; white-space: nowrap;
}
.sim-report .sim-scenario-rate-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.sim-report .sim-scenario-rate--high { background: var(--green-100); color: var(--green-600); }
.sim-report .sim-scenario-rate--high .sim-scenario-rate-dot { background: var(--green-600); }
.sim-report .sim-scenario-rate--mid { background: var(--amber-100); color: var(--amber-600); }
.sim-report .sim-scenario-rate--mid .sim-scenario-rate-dot { background: var(--amber-600); }
.sim-report .sim-scenario-rate--low { background: var(--red-100); color: var(--red-600); }
.sim-report .sim-scenario-rate--low .sim-scenario-rate-dot { background: var(--red-600); }
.sim-report .sim-entity-dialog {
    inset: 0 0 0 auto; margin: 0; width: 50vw; height: 100vh; max-height: 100vh;
    padding: 0; border: 0; border-left: 1px solid var(--border-subtle); border-radius: 12px 0 0 12px;
    background: var(--surface-app); color: var(--text-body); box-shadow: var(--shadow-lg);
}
.sim-report .sim-entity-dialog[open] { animation: sim-drawer-in 160ms ease-out; }
@keyframes sim-drawer-in { from { transform: translateX(100%); } to { transform: translateX(0); } }
.sim-report .sim-entity-dialog--closing { animation: sim-drawer-out 160ms ease-in forwards; }
@keyframes sim-drawer-out { from { transform: translateX(0); } to { transform: translateX(100%); } }
.sim-report .sim-entity-dialog::backdrop { background: rgb(18 17 15 / 0.42); }
.sim-report .sim-entity-modal-shell { display: flex; flex-direction: column; height: 100%; max-height: none; }
.sim-report .sim-entity-modal-content { flex: 1; min-height: 0; padding: 24px; overflow: auto; }
.sim-report .sim-entity-dialog .sim-transcript-grid > * { min-width: 0; }
@media (max-width: 760px) {
    .sim-report .sim-entity-dialog { width: 100vw; border-radius: 0; }
}
.sim-report .sim-drawer-row,
.sim-report .sim-conv-row {
    cursor: pointer; transition: background 120ms ease, border-color 120ms ease;
    box-shadow: inset 3px 0 0 var(--border-subtle);
}
/* Outcome accent on the leading edge, so rows are scannable by result. */
.sim-report .sim-conv-row--pass { box-shadow: inset 3px 0 0 var(--green-600); }
.sim-report .sim-conv-row--fail { box-shadow: inset 3px 0 0 var(--red-600); }
.sim-report .sim-conv-row--error { box-shadow: inset 3px 0 0 var(--orange-500); }
.sim-report .sim-drawer-row:hover,
.sim-report table.sim-conv-table tbody tr.sim-conv-row:hover { background: var(--surface-sunken); }
.sim-report .sim-drawer-row:focus-visible,
.sim-report .sim-conv-row:focus-visible {
    outline: 2px solid var(--teal-600); outline-offset: 2px;
}
.sim-report .sim-cohort-stats {
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px;
    margin: 12px 0 16px;
}
.sim-report .sim-cohort-stats span {
    display: grid; gap: 2px; padding: 8px 10px; border-radius: 6px;
    background: var(--surface-sunken); font-size: 12px; color: var(--text-body);
}
.sim-report .sim-cohort-stats b {
    font-family: var(--font-mono); font-size: 10px; font-weight: 500;
    text-transform: uppercase; color: var(--text-faint);
}
.sim-report .sim-cohort-conversations { margin-top: 18px; }
.sim-report .sim-cohort-conversations h3 { margin-bottom: 8px; }
.sim-report .sim-cohort-list { display: grid; gap: 6px; }
.sim-report .sim-cohort-conversation {
    display: flex; align-items: center; justify-content: space-between; gap: 10px; width: 100%;
    padding: 9px 10px; border: 1px solid var(--border-subtle); border-radius: 6px;
    background: var(--surface-app); color: var(--text-body); font: inherit; text-align: left;
    cursor: pointer; transition: background 120ms ease, border-color 120ms ease;
}
.sim-report .sim-cohort-conversation:hover {
    background: var(--surface-sunken); border-color: var(--teal-600);
}
.sim-report .sim-cohort-conversation:focus-visible {
    outline: 2px solid var(--teal-600); outline-offset: 2px;
}
.sim-report .sim-cohort-empty { margin: 8px 0 0; color: var(--text-muted); font-size: 13px; }
.sim-report .sim-entity-kicker {
    font-size: 11px; color: var(--text-faint); text-transform: uppercase; letter-spacing: .06em;
}
.sim-report .sim-entity-detail h2 {
    margin: 6px 0 10px; font-size: 22px; line-height: 1.2; color: var(--text-strong);
}
.sim-report .sim-entity-detail h3 {
    margin: 18px 0 6px; font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
    color: var(--text-faint);
}
.sim-report .sim-entity-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
.sim-report .sim-entity-pill {
    font-size: 12px; color: var(--text-muted); background: var(--surface-sunken);
    border-radius: 999px; padding: 4px 9px;
}
.sim-report .sim-entity-prose {
    margin: 10px 0 0; font-size: 13.5px; line-height: 1.55; color: var(--text-body);
}
.sim-report .sim-entity-traits { display: grid; gap: 8px; margin: 16px 0; }
.sim-report .sim-entity-trait-row {
    display: grid; grid-template-columns: 130px 1fr 44px; gap: 12px; align-items: center;
    font-size: 13px;
}
.sim-report .sim-entity-trait-track {
    height: 7px; border-radius: 999px; background: var(--surface-sunken); overflow: hidden;
}
.sim-report .sim-entity-trait-fill { display: block; height: 100%; background: var(--teal-600); }
.sim-report .sim-entity-trait-value { font-variant-numeric: tabular-nums; color: var(--text-muted); }
.sim-report .sim-entity-criteria {
    list-style: none; padding: 0; margin: 8px 0 0; display: grid; gap: 8px;
}
.sim-report .sim-entity-criterion {
    display: grid; grid-template-columns: 18px 1fr auto; gap: 8px; align-items: start;
    padding: 10px 12px; border-radius: 8px; background: var(--surface-sunken);
    font-size: 13px; line-height: 1.4;
}
.sim-report .sim-entity-criterion-mark { font-weight: 700; }
.sim-report .sim-entity-criterion--positive .sim-entity-criterion-mark { color: var(--green-600); }
.sim-report .sim-entity-criterion--negative .sim-entity-criterion-mark { color: var(--red-600); }
.sim-report .sim-entity-criterion-type {
    font-size: 11px; color: var(--text-faint); text-transform: uppercase; letter-spacing: .04em;
}
.sim-report .sim-entity-modal-actions {
    display: flex; justify-content: flex-end; gap: 8px; padding: 12px 16px;
    border-top: 1px solid var(--border-subtle); background: var(--surface-sunken);
}
.sim-report .sim-entity-back,
.sim-report .sim-entity-nav,
.sim-report .sim-entity-close {
    border: 1px solid var(--border-subtle); border-radius: 6px; background: var(--surface-app);
    color: var(--text-body); padding: 6px 10px; font: inherit; cursor: pointer;
}
.sim-report .sim-entity-nav {
    font-family: var(--font-mono); min-width: 32px;
    display: inline-flex; align-items: center; gap: 6px;
}
/* j/k hotkey hint on the modal prev/next buttons. */
.sim-report .sim-entity-kbd {
    font-family: var(--font-mono); font-size: 10px; line-height: 1;
    padding: 2px 5px; border-radius: 4px;
    border: 1px solid var(--border-subtle); background: var(--surface-sunken);
    color: var(--text-faint);
}
.sim-report .sim-entity-back:hover,
.sim-report .sim-entity-nav:hover,
.sim-report .sim-entity-close:hover { border-color: var(--orange-500); color: var(--orange-500); }
@media (max-width: 480px) {
    .sim-report .sim-entity-dialog { width: 92vw; border-radius: 10px 0 0 10px; }
    .sim-report .sim-entity-modal-content { padding: 16px; }
    .sim-report .sim-entity-trait-row { grid-template-columns: 88px minmax(0, 1fr) 36px; gap: 8px; }
    .sim-report .sim-entity-modal-actions { padding: 10px 12px; }
}

/* ---- Agent info card ---- */
.sim-report .sim-agent-card { margin-bottom: 16px; }
/* Provenance note + "show captured snapshot" disclosure (agent info enriched
   live from Orq when the run didn't capture it). */
.sim-report .sim-agent-source { margin-top: 14px; font-size: 12px; color: var(--text-muted); }
.sim-report .sim-agent-original { margin-top: 8px; }
.sim-report .sim-agent-original > summary {
    cursor: pointer; font-size: 12px; font-weight: 500; color: var(--orange-500); width: fit-content;
}
.sim-report .sim-agent-original-body {
    margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-subtle);
}
.sim-report .sim-agent-empty { margin: 0; font-size: 13px; color: var(--text-muted); }
.sim-report .sim-agent-head {
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
}
.sim-report .sim-agent-identity {
    display: inline-flex; align-items: center; gap: 8px; min-width: 0;
}
.sim-report .sim-agent-icon {
    width: 18px; height: 18px; color: var(--teal-600); flex: 0 0 auto;
}
.sim-report .sim-agent-name {
    font-family: var(--font-sans); font-size: 18px; font-weight: 600; color: var(--text-strong);
}
.sim-report .sim-agent-role, .sim-report .sim-agent-chip {
    display: inline-block; font-family: var(--font-sans); font-size: 11.5px;
    background: var(--surface-sunken); border-radius: 999px; padding: 2px 8px;
    color: var(--text-muted); margin-left: 8px;
}
.sim-report .sim-agent-model {
    font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); margin-top: 4px;
}
.sim-report .sim-agent-desc {
    font-size: 14px; color: var(--text-body); max-width: 66.667%; margin-top: 8px;
}
@media (max-width: 760px) {
    .sim-report .sim-agent-desc { max-width: 100%; }
}
.sim-report .sim-agent-delegates {
    display: flex; align-items: center; gap: 8px; margin-top: 10px;
    font-size: 12.5px; color: var(--text-muted);
}
.sim-report .sim-agent-delegates .sim-agent-chip { margin-left: 0; }
.sim-report .sim-agent-groups {
    display: flex; flex-wrap: wrap; gap: 32px; margin-top: 12px;
}
.sim-report .sim-agent-group { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.sim-report .sim-agent-group-label {
    font-family: var(--font-sans); font-size: 10.5px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text-faint); margin-right: 2px;
}
.sim-report .sim-agent-group .sim-agent-chip { margin-left: 0; }
.sim-report .sim-agent-open {
    border: 1px solid var(--border-subtle); border-radius: 8px; padding: 6px 12px;
    font-size: 12px; color: var(--text-body); text-decoration: none; white-space: nowrap;
}
.sim-report .sim-agent-open:hover { background: var(--surface-sunken); }
"""

# ==== .sim-report — Transcript overrides layered on `.report-aligned` =====
# Same equal-specificity/source-order rationale as _SIM_REPORT_OVERRIDES_CSS
# above: the sim report gets its own conversation-row chrome and a distinct
# chat-bubble skin (asymmetric tail, no in-bubble role label) instead of the
# shared `.report-aligned .sim-msg/.rt-msg` bubbles used by the Red Team
# report.
_SIM_TRANSCRIPT_OVERRIDES_CSS = """
.sim-report .sim-conv-table-shell {
    border: 1px solid var(--border-subtle); border-radius: var(--radius-lg);
    overflow-x: auto; background: var(--surface-card);
}
.sim-report .sim-conv-table { border: 0; border-radius: 0; }
.sim-report .sim-conv-row {
    border: 0; background: transparent; transition: background 150ms ease;
}
.sim-report .sim-conv-row:hover { background: var(--surface-sunken); }
.sim-report .sim-conv-row + .sim-conv-row { border-top: 1px solid var(--border-subtle); }
@media (prefers-reduced-motion: reduce) {
    .sim-report .sim-conv-row { transition: none; }
}
.sim-report .sim-conv-idx {
    font-family: var(--font-sans); font-size: 12px; color: var(--text-faint);
}
.sim-report .sim-judge {
    background: var(--surface-sunken);
    border-radius: var(--radius-md); padding: 10px 14px; margin-bottom: 14px;
}
.sim-report .sim-judge-label {
    font-family: var(--font-sans); font-size: 11px; font-weight: 600;
    text-transform: uppercase; color: var(--teal-600); display: block;
}
.sim-report .sim-transcript-error {
    background: var(--red-100); color: var(--red-600); border-radius: var(--radius-md);
    padding: 10px 14px; margin-bottom: 14px; font-size: 13px;
}
.sim-report .sim-transcript-bubbles {
    background: var(--surface-sunken); border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg); padding: 16px;
}
.sim-report .sim-transcript-bubbles .sim-msg:last-child { margin-bottom: 0; }

/* Chat bubbles (render_message_list avatar + side extension) — sim-only skin,
   distinct from the shared `.report-aligned .sim-msg/.rt-msg` bubbles. */
.sim-report .sim-msg { display: flex; gap: 10px; margin-bottom: 10px; max-width: 88%; }
/* Sim swaps sides vs the shared `.report-aligned` base (user right, agent left),
   so it must reset BOTH margins + flex-direction — the element matches both the
   base rule and this one (drawer lives inside `.report-aligned sim-report`). */
.sim-report .sim-msg-user, .sim-report .sim-msg-system { margin: 0 0 16px auto; flex-direction: row-reverse; }
.sim-report .sim-msg-assistant, .sim-report .sim-msg-tool { margin: 0 auto 16px 0; flex-direction: row; }
.sim-report .sim-msg-avatar {
    flex-shrink: 0; width: 30px; height: 30px; border-radius: var(--radius-md);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-sans); font-size: 9px; font-weight: 600;
    background: var(--ink-900); color: #fff;
}
.sim-report .sim-msg-assistant .sim-msg-avatar, .sim-report .sim-msg-tool .sim-msg-avatar {
    background: var(--teal-50); color: var(--teal-600);
}
/* Single flat bubble with an asymmetric tail corner, like the mockup — white +
   hairline border for the user, teal tint for the agent (tail mirrored to the
   avatar side). */
.sim-report .sim-msg-bubble {
    background: var(--surface-card); border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg) 3px var(--radius-lg) var(--radius-lg); padding: 9px 13px;
}
.sim-report .sim-msg-assistant .sim-msg-bubble, .sim-report .sim-msg-tool .sim-msg-bubble {
    background: var(--teal-50); border-radius: 3px var(--radius-lg) var(--radius-lg) var(--radius-lg);
}
/* The avatar already carries USR/AGT; the mockup has no second in-bubble label. */
.sim-report .sim-msg-role { display: none; }
/* .sim-msg-content is a <pre> — strip the global code-block chrome (bg, border,
   radius, padding) so text sits flat in the bubble, not in a nested box. */
.sim-report .sim-msg-content {
    font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-word;
    margin: 0; font-family: inherit;
    background: transparent; border: none; border-radius: 0; padding: 0;
}

/* Conversation summary header: persona + scenario recap and turn-count chip,
   shown above the criteria and transcript inside the drawer. */
.sim-report .sim-conv-summary {
    display: flex; flex-direction: column;
    gap: 16px; margin-bottom: 20px;
    padding-bottom: 16px; border-bottom: 1px solid var(--border-subtle);
}
/* Index row: the large # identity anchor left, turn count right. */
.sim-report .sim-conv-head {
    display: flex; align-items: center; justify-content: space-between; gap: 14px;
}
/* Teal index chip — the conversation's # from the row table, sized up as the
   drawer's identity anchor (DESIGN.md: teal leads). */
.sim-report .sim-conv-index {
    flex-shrink: 0; font-family: var(--font-mono);
    font-size: 20px; font-weight: 700; line-height: 1;
    color: #fff; background: var(--teal-600);
    padding: 8px 12px; border-radius: var(--radius-md);
    font-variant-numeric: tabular-nums;
}
.sim-report .sim-conv-meta {
    display: flex; flex-direction: column; gap: 10px; min-width: 0;
}
.sim-report .sim-conv-field {
    display: flex; align-items: flex-start; gap: 10px; min-width: 0;
}
/* Glyph tile anchoring each field (Option A). Orange punctuation accent —
   DESIGN.md: teal carries (index chip + value links), orange punctuates. */
.sim-report .sim-conv-ico {
    flex-shrink: 0; width: 26px; height: 26px; border-radius: 8px;
    display: grid; place-items: center;
    background: var(--orange-50); color: var(--orange-700);
}
.sim-report .sim-conv-ico svg { width: 15px; height: 15px; }
.sim-report .sim-conv-field-text {
    display: flex; flex-direction: column; gap: 2px; min-width: 0;
}
.sim-report .sim-conv-label {
    font-family: var(--font-sans); font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.03em; color: var(--text-faint);
}
.sim-report .sim-conv-value {
    font-size: 14px; color: var(--text-body); line-height: 1.4;
}
/* Click-through persona/scenario value: opens the cohort card. Reset button
   chrome, keep it inline text with a teal underline affordance. */
.sim-report button.sim-conv-value--link {
    display: inline; margin: 0; padding: 0; border: none; background: none;
    font: inherit; text-align: left; cursor: pointer;
    color: var(--teal-600); text-decoration: underline;
    text-decoration-color: var(--teal-100); text-underline-offset: 3px;
    transition: text-decoration-color 150ms ease;
}
.sim-report button.sim-conv-value--link:hover {
    text-decoration-color: var(--teal-600);
}
.sim-report button.sim-conv-value--link:focus-visible {
    outline: 2px solid var(--teal-600); outline-offset: 2px; border-radius: 3px;
}
.sim-report .sim-conv-turns-pill {
    flex-shrink: 0; font-family: var(--font-sans); font-size: 12px; font-weight: 600;
    white-space: nowrap; padding: 3px 10px; border-radius: 999px;
    color: var(--text-muted); background: var(--surface-sunken);
    border: 1px solid var(--border-subtle);
}

/* Judge rationale folded into the criteria block: sits under the list, its own
   margins reset so it reads as the verdict's explanation, not a stray callout. */
.sim-report .sim-criteria .sim-judge { margin: 16px 0 0; }

/* Criteria column (two-state per deviation #4) */
.sim-report .sim-criteria-head {
    display: flex; align-items: baseline; gap: 10px;
    margin-bottom: 8px;
}
.sim-report .sim-criteria-header {
    font-family: var(--font-sans); font-size: 11px; font-weight: 600;
    text-transform: uppercase; color: var(--text-faint); margin-bottom: 0;
    margin-right: auto;
}
.sim-report .sim-criteria-verdict {
    font-family: var(--font-sans); font-size: 12px; font-weight: 700;
    letter-spacing: 0.02em; white-space: nowrap;
}
.sim-report .sim-criteria-verdict--pass { color: var(--green-600); }
.sim-report .sim-criteria-verdict--fail { color: var(--red-600); }
.sim-report .sim-criteria-count {
    font-family: var(--font-sans); font-size: 12px; color: var(--text-faint); white-space: nowrap;
}
.sim-report .sim-criteria-list {
    list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px;
}
/* Scoped to the criteria list so it no longer collides with the Scenarios-panel
   .sim-criterion rule above. Plain gapped column (no dividers). */
.sim-report .sim-criteria-list .sim-criterion {
    /* wraps so the evidence quote drops onto its own full-width line under the row */
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
    /* Reset the row-divider + padding the legacy `.sim-criterion` rule adds; the
       criteria list is a plain gapped column so only the section-level rules
       (general info ↔ criteria ↔ conversation) draw dividers. */
    padding: 0; border-top: none;
}
/* The requirement chip sits right after the result icon so the reader never has
   to reconcile a green check against a far-right "must not happen" label. */
.sim-report .sim-ctype {
    flex-shrink: 0; font-family: var(--font-sans); font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.03em; white-space: nowrap;
    padding: 2px 7px; border-radius: 999px;
    color: var(--text-faint); background: var(--surface-sunken);
    border: 1px solid var(--border-subtle);
}
.sim-report .sim-ctype-unsafe {
    color: var(--red-600); background: var(--red-100); border-color: transparent;
}
/* Unaudited criteria read as neither pass nor fail: a muted "?" tile, a "not
   audited" chip and a run-level note, so an unverified run can never look like a
   clean one (RES-1308). */
.sim-report .sim-criterion-unknown .sim-criterion-icon {
    background: var(--surface-sunken); color: var(--text-faint);
    border: 1px solid var(--border-subtle);
}
.sim-report .sim-criterion-unaudited {
    flex-shrink: 0; font-family: var(--font-sans); font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.03em; white-space: nowrap;
    color: var(--text-faint);
}
.sim-report .sim-criterion-evidence {
    flex: 1 1 100%; font-size: 12px; font-style: italic; color: var(--text-muted);
    padding-left: 26px;
}
.sim-report .sim-criteria-unknown {
    font-family: var(--font-sans); font-size: 12px; color: var(--text-faint); white-space: nowrap;
}
.sim-report .sim-criteria-unverified {
    margin: 0 0 10px; font-size: 12px; line-height: 1.5; color: var(--text-muted);
    padding: 8px 10px; border-radius: 6px;
    background: var(--surface-sunken); border: 1px solid var(--border-subtle);
}
.sim-report .sim-criteria-empty {
    margin: 0; font-size: 13px; color: var(--text-faint);
}
"""

# ==== .rt-report — Red Team report design-mockup alignment ==============
# All rules scoped under `.rt-report` (report_tabs.redteam_report_tabs'
# wrapper) per docs/superpowers/specs/2026-07-10-redteam-report-alignment-
# design.md. Surface-identical widgets (exec-summary, KPI band, `.rk-*`
# primitives, chat bubbles, active-tab underline) are already covered by the
# `.report-aligned` promotion above; this block only carries RT-only
# composition (hero agent pills, Overview 2-col grid, agents-under-test).
_RT_REPORT_CSS = """
/* ---- Run header (spec §Run header) ---- */
/* Shares .report-hero-title's type scale; only adds the flex row for the inline agents pill. */
.rt-hero-title { display: flex; align-items: center; gap: 10px; }
.rt-hero-agents-pill {
    font-family: var(--font-mono); font-size: 11px;
    background: var(--surface-sunken); color: var(--text-muted);
    border-radius: 999px; padding: 2px 8px;
}
/* Occupies the same slot as `.report-hero-sub` on the sim report — the line
   under the run name answering "what was tested?" — so it carries that rule's
   13px size and `4px 0 16px` margins. The mono 12px sub is a nested element
   here (sim has no equivalent), one step down from the pill's own 13px. */
.rt-hero-agent-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 16px; }
.rt-hero-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--surface-card); border: 1px solid var(--border-subtle);
    border-radius: 999px; padding: 3px 10px; font-size: 13px;
}
.rt-hero-dot { display: inline-block; width: 7px; height: 7px; border-radius: 999px; }
.rt-hero-dot--critical { background: var(--red-600); }
.rt-hero-dot--vuln { background: var(--orange-500); }
.rt-hero-dot--clean { background: var(--green-600); }
.rt-hero-pill-name { color: var(--text-strong); }
.rt-hero-pill-sub {
    font-family: var(--font-mono); font-size: 12px; color: var(--text-faint);
}

/* ---- Overview tab (spec §Overview.4) ---- */
.rt-report .rt-overview-grid-2 {
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;
}
@media (max-width: 760px) {
    .rt-report .rt-overview-grid-2 { grid-template-columns: 1fr; }
}
.rt-report .rt-agents-table { display: flex; flex-direction: column; gap: 10px; }
.rt-report .rt-agent-row {
    display: grid; grid-template-columns: 1.4fr 64px 1fr 56px; align-items: center; gap: 12px;
}
.rt-report .rt-agent-row-name {
    display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-strong);
}
.rt-report .rt-agent-row-model {
    font-family: var(--font-mono); font-size: 10.5px; color: var(--text-faint);
}
.rt-report .rt-agent-row-count {
    font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); text-align: right;
}
.rt-report .rt-agent-row-track {
    height: 7px; border-radius: 999px; background: var(--chart-track); overflow: hidden;
}
.rt-report .rt-agent-row-fill { height: 100%; border-radius: 999px; }
.rt-report .rt-agent-row-asr {
    font-family: var(--font-mono); font-size: 12px; font-weight: 600; text-align: right;
}

/* ---- Breakdowns tab (spec §Breakdowns) ---- */
.rt-report .rt-breakdowns-grid-2 {
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;
}
@media (max-width: 760px) {
    .rt-report .rt-breakdowns-grid-2 { grid-template-columns: 1fr; }
}
.rt-report .rt-breakdowns-leadin {
    font-size: 13px; color: var(--text-muted); margin: 0 0 10px;
}
.rt-report .rt-breakdowns-footnote {
    font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); margin-top: 12px;
}

/* ---- Agents tab (spec §Agents) ---- */
.rt-report .rt-agents-intro { font-size: 13px; color: var(--text-muted); margin: 0 0 16px; max-width: 720px; }
.rt-report .rt-agent-card { display: flex; gap: 16px; margin-bottom: 22px; }
.rt-report .rt-agent-card-dial { flex: 0 0 64px; }
.rt-report .rt-agent-card-main { flex: 1 1 auto; min-width: 0; }
.rt-report .rt-agent-card-name-row { display: flex; align-items: center; gap: 8px; }
.rt-report .rt-agent-card-name { font-family: var(--font-display); font-size: 16px; font-weight: 600; }
.rt-report .rt-agent-card-critical {
    font-family: var(--font-mono); font-size: 10px; font-weight: 600; text-transform: uppercase;
    color: var(--red-600); background: var(--red-100); border-radius: 5px; padding: 2px 6px;
}
.rt-report .rt-agent-card-studio {
    margin-left: auto; font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    text-decoration: none; color: var(--teal-600); background: var(--teal-100);
    border-radius: 5px; padding: 3px 9px; white-space: nowrap;
    transition: background .15s ease, color .15s ease;
}
.rt-report .rt-agent-card-studio:hover { background: var(--teal-600); color: #fff; }
.rt-report .rt-agent-card-model { font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); margin-top: 3px; }
.rt-report .rt-agent-card-desc { font-size: 13px; color: var(--text-body); margin-top: 6px; }
.rt-report .rt-agent-card-stats {
    display: flex; gap: 22px; margin-top: 11px; padding-top: 11px; border-top: 1px solid var(--border-subtle);
}
.rt-report .rt-agent-card-stat { display: flex; flex-direction: column; gap: 2px; }
.rt-report .rt-agent-card-stat-key {
    font-family: var(--font-mono); font-size: 9px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--text-faint);
}
.rt-report .rt-agent-card-stat-value { font-family: var(--font-mono); font-size: 14px; font-weight: 600; }
.rt-report .rt-agent-card-chiprow { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
.rt-report .rt-agent-card-chip-label {
    font-size: 11px; color: var(--text-faint); flex: 0 0 70px;
}
.rt-report .rt-agent-card-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.rt-report .rt-agent-card-chip-empty { color: var(--text-faint); font-size: 12px; }

/* ---- Focus areas tab (spec §Focus areas) ---- */
.report-aligned .rt-focus-intro { font-size: 13px; color: var(--text-muted); margin: 0 0 16px; max-width: 720px; }
.report-aligned .rt-focus-intro code { font-family: var(--font-mono); }
.report-aligned .rt-focus-card { display: flex; flex-direction: column; margin-bottom: 22px; padding: 0; overflow: hidden; }
.report-aligned .rt-focus-head {
    display: flex; align-items: center; justify-content: space-between; gap: 20px;
    padding: 14px 18px; border-bottom: 1px solid var(--border-subtle);
}
.report-aligned .rt-focus-head-main { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.report-aligned .rt-focus-head-stats {
    display: flex; align-items: center; flex: 0 0 auto;
    background: var(--surface-sunken); border: 1px solid var(--border-subtle); border-radius: 9px;
    padding: 6px 0;
}
.report-aligned .rt-focus-stat {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 2px; padding: 0 16px; min-width: 64px;
}
.report-aligned .rt-focus-stat + .rt-focus-stat { border-left: 1px solid var(--border-default); }
.report-aligned .rt-focus-stat--dial { padding: 0 10px; }
.report-aligned .rt-focus-body { display: flex; flex-direction: column; gap: 12px; padding: 14px 18px 16px; }
.report-aligned .rt-focus-tier-row { display: flex; align-items: center; gap: 8px; }
.report-aligned .rt-focus-tier-dot { width: 8px; height: 8px; border-radius: 999px; display: inline-block; }
.report-aligned .rt-focus-tier-label {
    font-family: var(--font-mono); font-size: 10.5px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em;
}
.report-aligned .rt-focus-category-name { font-family: var(--font-display); font-size: 17px; font-weight: 600; }
.report-aligned .rt-focus-category-code { font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); font-weight: 400; margin-left: 4px; }
.report-aligned .rt-focus-patterns {
    display: flex; align-items: center; gap: 8px; font-size: 12px; background: var(--surface-sunken);
    border-radius: 6px; padding: 4px 10px; width: fit-content; margin: 0;
}
.report-aligned .rt-focus-pattern-dot { width: 5px; height: 5px; border-radius: 999px; display: inline-block; flex: 0 0 auto; }
.report-aligned .rt-focus-fixbox { background: var(--surface-sunken); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; margin: 0; }
.report-aligned .rt-focus-fixbox-label {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase; color: var(--accent-hover);
    letter-spacing: 0.05em;
}
.report-aligned .rt-focus-fixbox-body { font-size: 13px; line-height: 1.55; margin-top: 6px; }
.report-aligned .rt-focus-mini-key {
    font-family: var(--font-mono); font-size: 9px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--text-faint);
}
.report-aligned .rt-focus-mini-value { font-family: var(--font-mono); font-size: 14px; font-weight: 600; }

/* ---- Apply recommendations: bar, bullets, right drawer (RES-1143) ---- */
.report-aligned .rt-focus-recs-section { display: flex; flex-direction: column; gap: 6px; }
.report-aligned .rt-focus-recs-label { margin: 0; }
/* One grouped list, hairline dividers between rows — not a stack of boxes. */
.report-aligned .rt-focus-recs {
    list-style: none; margin: 6px 0 0; padding: 0;
    border: 1px solid var(--border-default); border-radius: 8px; background: var(--surface-card, #fdfcfb);
    overflow: hidden;
}
.report-aligned .rt-focus-rec {
    font-size: 13px; line-height: 1.5; display: flex; align-items: center; gap: 14px;
    padding: 10px 12px;
}
.report-aligned .rt-focus-rec + .rt-focus-rec { border-top: 1px solid var(--border-subtle); }
.report-aligned .rt-focus-rec:hover { background: var(--surface-sunken); }
.report-aligned .rt-focus-rec-text { flex: 1 1 auto; min-width: 0; }
/* The action slot is a fixed-width right rail so every button and applied
   pill lines up down the column regardless of text length. */
.report-aligned .rt-focus-rec-apply {
    flex: 0 0 auto; margin: 0 0 0 auto; width: 76px; display: flex; justify-content: flex-end;
}
.rt-apply-btn--sm {
    font-size: 11px; font-weight: 600; padding: 4px 12px; border-radius: 6px;
    background: transparent; color: var(--accent); border: 1px solid var(--accent);
}
.rt-apply-btn--sm:hover { background: var(--accent); color: #fff; }
.report-aligned .rt-focus-rec--applied .rt-focus-rec-text { color: var(--text-muted); }
.report-aligned .rt-focus-rec-applied {
    font-family: var(--font-mono); font-size: 10px; color: var(--green-600, #16a34a);
    background: color-mix(in srgb, #16a34a 12%, transparent);
    border-radius: 999px; padding: 2px 9px; margin-left: auto; white-space: nowrap;
    flex: 0 0 auto;
}
.report-aligned .rt-apply-bar {
    display: flex; align-items: center; justify-content: space-between; gap: 20px; flex-wrap: wrap;
    /* Raised like the cards, not sunken: this is the tab's primary action,
       and the recessed background read as a backdrop (review feedback). The
       accent edge marks it as the call to action. */
    background: var(--surface-card, #fdfcfb); border: 1px solid var(--border-default);
    border-left: 3px solid var(--accent);
    border-radius: 10px; padding: 14px 18px; margin: 0 0 16px;
    box-shadow: 0 1px 2px rgba(20, 18, 15, 0.05);
}
/* Cap the text column so the action side stays on the same row on wide
   screens instead of wrapping underneath. */
.report-aligned .rt-apply-bar-text { display: flex; flex-direction: column; gap: 3px; min-width: 0; flex: 1 1 320px; max-width: 640px; }
.report-aligned .rt-apply-count { font-size: 13px; font-weight: 600; }
.report-aligned .rt-apply-count--done { color: var(--green-600, #16a34a); }
.report-aligned .rt-apply-hint { font-size: 12px; color: var(--text-muted); }
.report-aligned .rt-apply-form { display: flex; align-items: center; gap: 10px; flex: 0 0 auto; margin-left: auto; }
.rt-apply-btn {
    font-size: 13px; font-weight: 600; padding: 7px 14px; border-radius: 8px; cursor: pointer;
    border: 1px solid var(--accent); background: var(--accent); color: #fff;
}
.rt-apply-btn:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
.rt-apply-btn--confirm { background: var(--green-600, #16a34a); border-color: var(--green-600, #16a34a); }
.rt-apply-btn--confirm:hover { filter: brightness(0.92); background: var(--green-600, #16a34a); }
.rt-apply-btn--ghost { background: transparent; color: inherit; border-color: var(--border-default); }
.rt-apply-btn--ghost:hover { background: var(--surface-sunken); border-color: var(--border-default); }
.rt-drawer-overlay {
    position: fixed; inset: 0; background: rgba(15, 15, 15, 0.42); z-index: 90; cursor: pointer;
}
.rt-drawer {
    position: fixed; top: 0; right: 0; bottom: 0; width: min(560px, 92vw); z-index: 91;
    background: var(--surface-card, #fdfcfb); border-left: 1px solid var(--border-default);
    box-shadow: -18px 0 48px rgba(0, 0, 0, 0.18);
    display: flex; flex-direction: column;
    animation: rt-drawer-in 0.18s ease-out;
}
@keyframes rt-drawer-in { from { transform: translateX(24px); opacity: 0; } to { transform: none; opacity: 1; } }
.rt-drawer-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 20px; border-bottom: 1px solid var(--border-subtle); flex: 0 0 auto;
}
.rt-drawer-title { margin: 0; font-size: 15px; font-weight: 700; }
.rt-drawer-close {
    border: none; background: transparent; color: var(--text-muted); font-size: 22px; line-height: 1;
    cursor: pointer; padding: 2px 6px; border-radius: 6px;
}
.rt-drawer-close:hover { background: var(--surface-sunken); color: inherit; }
.rt-drawer-body { padding: 16px 20px; overflow-y: auto; flex: 1 1 auto; }
.rt-drawer-section-label {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--accent-hover); margin: 14px 0 6px;
}
.rt-drawer-section-label:first-child { margin-top: 0; }
.rt-drawer-agent { font-family: var(--font-mono); font-size: 13px; }
.rt-drawer-area {
    display: flex; flex-direction: column; gap: 3px;
    background: var(--surface-sunken); border: 1px solid var(--border-default);
    border-radius: 8px; padding: 10px 12px;
}
.rt-drawer-area-tier { font-family: var(--font-mono); font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
.rt-drawer-area-name { font-size: 13px; font-weight: 600; }
.rt-drawer-area-code { font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); font-weight: 400; }
.rt-drawer-area-meta { font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); }
.rt-drawer-patterns { font-size: 12px; color: var(--text-muted); line-height: 1.5; margin-top: 4px; }
.rt-drawer-recs { margin: 0; padding-left: 18px; display: flex; flex-direction: column; gap: 5px; }
.rt-drawer-recs li { font-size: 13px; line-height: 1.5; }
.rt-drawer-error { color: var(--sev-high, #dc2626); font-size: 13px; line-height: 1.55; }
/* Read-the-diff callout above the preview diff: the merge is an LLM rewrite of a
   live prompt, and the diff is the only thing standing between it and the agent. */
.rt-drawer-review {
    display: flex; gap: 9px; align-items: flex-start;
    background: var(--surface-sunken); border: 1px solid var(--border-default);
    border-left: 3px solid var(--sev-medium, #d97706);
    border-radius: 8px; padding: 10px 12px; margin-bottom: 10px;
}
.rt-drawer-review-icon { font-size: 14px; line-height: 1.4; }
.rt-drawer-review-text { font-size: 12px; line-height: 1.55; color: var(--text-default); }
.rt-drawer-review-text b { font-weight: 600; }
.rt-drawer-success { font-size: 14px; line-height: 1.6; }
/* Applied celebration screen: centered, green check drawn in (RES-1143). */
.rt-drawer-body--applied {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; gap: 6px; padding: 56px 24px 32px; min-height: 60%;
}
.rt-applied-check { width: 84px; height: 84px; margin-bottom: 14px; animation: rt-applied-pop 0.45s cubic-bezier(0.22, 1.4, 0.36, 1); }
.rt-applied-check-ring {
    stroke: var(--green-600, #299D8F); stroke-dasharray: 202; stroke-dashoffset: 202;
    animation: rt-draw 0.6s ease-out 0.1s forwards;
}
.rt-applied-check-mark {
    stroke: var(--green-600, #299D8F); stroke-dasharray: 44; stroke-dashoffset: 44;
    animation: rt-draw 0.35s ease-out 0.55s forwards;
}
@keyframes rt-draw { to { stroke-dashoffset: 0; } }
@keyframes rt-applied-pop { from { transform: scale(0.6); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.rt-applied-headline { font-family: var(--font-display); font-size: 19px; font-weight: 700; margin: 0; }
.rt-applied-target { font-size: 14px; color: var(--text-muted); margin: 0 0 10px; display: flex; align-items: center; gap: 8px; }
.rt-applied-version {
    font-family: var(--font-mono); font-size: 11px; color: var(--green-600, #299D8F);
    background: color-mix(in srgb, #299D8F 12%, transparent);
    border-radius: 999px; padding: 2px 9px;
}
.rt-drawer-body--applied .rt-drawer-note { max-width: 360px; }
.rt-drawer-note { font-size: 12px; color: var(--text-muted); line-height: 1.55; }
.rt-diff {
    font-family: var(--font-mono); font-size: 11.5px; line-height: 1.5; margin: 0;
    background: var(--surface-sunken); border: 1px solid var(--border-default); border-radius: 8px;
    padding: 10px 12px; overflow-x: auto; display: flex; flex-direction: column;
}
.rt-diff-line { white-space: pre; }
.rt-diff-add { color: var(--green-600, #16a34a); background: color-mix(in srgb, #16a34a 9%, transparent); }
.rt-diff-del { color: var(--sev-high, #dc2626); background: color-mix(in srgb, #dc2626 8%, transparent); }
.rt-diff-hunk { color: var(--accent-hover); }
.rt-diff-file { color: var(--text-faint); }
.rt-drawer-footer {
    display: flex; align-items: center; gap: 10px; padding: 14px 20px;
    border-top: 1px solid var(--border-subtle); flex: 0 0 auto;
}
.rt-drawer-footnote { font-size: 11px; color: var(--text-faint); margin-left: auto; text-align: right; }
/* Loading-drawer styles live in dashboard.js (injected with the markup) so a
   cached page can never render the spinner unstyled. */
/* htmx tags the in-flight form; freeze its button so a double-click cannot
   queue a second merge. */
.rt-apply-form.htmx-request .rt-apply-btn,
.rt-focus-rec-apply.htmx-request .rt-apply-btn,
.rt-drawer-footer form.htmx-request .rt-apply-btn { opacity: 0.55; pointer-events: none; }

/* ---- Attack evidence fragment (spec §Attacks, Task 13) ---- */
.rt-report .rt-attack-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.rt-report .rt-verdict {
    background: var(--surface-sunken); border-radius: 4px; padding: 10px 14px; margin-bottom: 14px;
}
.rt-report .rt-verdict-label {
    font-family: var(--font-mono); font-size: 10px; font-weight: 600; text-transform: uppercase;
    color: var(--text-faint); display: block;
}
.rt-report .rt-verdict-body { font-size: 13px; line-height: 1.55; margin: 4px 0 0; color: var(--text-body); }
.rt-report .rt-verdict-vuln { background: var(--red-50); }
.rt-report .rt-verdict-safe { background: var(--green-50); }
.rt-report .rt-verdict-error .rt-verdict-body { color: var(--red-600); }

/* ---- Attacks evidence table (spec §Attacks, Task 14) ---- */
.rt-report .rt-attacks-intro { color: var(--text-faint); font-size: 13px; margin-bottom: 12px; }
.rt-report .rt-attack-table { padding: 0; overflow: hidden; }
.rt-report .rt-attack-row-header,
.rt-report .rt-attack-row-summary {
    display: grid; grid-template-columns: 1.9fr 1.1fr 1fr 0.85fr 0.95fr 20px;
    align-items: center; gap: 10px; padding: 10px 14px;
}
.rt-report .rt-attack-row-header {
    font-family: var(--font-mono); font-size: 10px; font-weight: 600; text-transform: uppercase;
    color: var(--text-faint); border-bottom: 1px solid var(--border-subtle);
}
.rt-report .rt-attack-row { border-bottom: 1px solid var(--border-subtle); }
.rt-report .rt-attack-row:last-child { border-bottom: none; }
.rt-report .rt-attack-row-summary {
    cursor: pointer; list-style: none; font-size: 13px;
}
.rt-report .rt-attack-row-summary::-webkit-details-marker { display: none; }
.rt-report .rt-attack-row-summary:hover { background: var(--surface-sunken); }
.rt-report .rt-attack-row-title { display: flex; flex-direction: column; gap: 2px; overflow: hidden; }
.rt-report .rt-attack-row-id {
    font-family: var(--font-mono); font-size: 10px; color: var(--text-faint);
}
.rt-report .rt-attack-row-chevron {
    font-size: 10px; color: var(--text-faint); transition: transform 0.15s ease;
}
.rt-report .rt-attack-row[open] .rt-attack-row-chevron { transform: rotate(180deg); }
.rt-report .rt-attack-row-body { padding: 14px; background: var(--surface-sunken); }

/* Unify the shared report.css palette onto the brand semantic tokens within the
   RT report only (flat export + sim keep report.css defaults). Custom props
   cascade, so this one block repoints every --c-*/--orq-orange consumer —
   KPI cards, badges, risk pills, status badges, verdict lines — to brand. */
.rt-report {
    --c-fail: var(--outcome-vulnerable);
    --c-pass: var(--outcome-resistant);
    --c-warn: var(--outcome-error);
    --orq-orange: var(--orange-500);
    --clay: var(--orange-500);
}

/* Severity-definitions table (and any .severity-* label) uses the semantic
   --sev-* scale (4-step: critical/high/medium/low) so medium stays a neutral
   tint distinct from high, rather than collapsing onto --c-warn. */
.rt-report .severity-critical { color: var(--red-700); }
.rt-report .severity-high     { color: var(--orange-700); }
.rt-report .severity-medium   { color: var(--sev-medium); }
.rt-report .severity-low      { color: var(--sev-low); }
"""

# Side-by-side sim run comparison page. The hero reuses the shared .report-hero
# classes; this block only carries compare-specific rules, on the editorial
# theme tokens. Kept with the other dashboard CSS blocks rather than inlined
# per-request in sim_compare.py.
_SIM_COMPARE_CSS = """
/* Hero action row: trace-link button + the on-report compare control */
.report-hero-actions { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-top: 10px; }
.report-hero-actions .cmp-bar { margin: 0; }

/* Compare picker bar on the sim run overview and in the report hero. Controls
   mirror the filter-rail trigger look (12.5px sans, card surface, default
   hairline, md radius) so the bar reads as part of the same control family. */
.cmp-bar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 2px 0 14px; }
.cmp-bar-label { font-family: var(--font-mono); font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .08em; color: var(--text-faint); }
.cmp-bar select { height: 32px; padding: 0 9px; max-width: 320px;
  border: 1px solid var(--border-default); border-radius: var(--radius-md);
  background: var(--surface-card); color: var(--text-body);
  font-family: var(--font-sans); font-size: 12.5px; font-weight: 500; cursor: pointer; }
.cmp-bar select:hover { background: var(--app-gray-50); }
.cmp-bar select:focus-visible { outline: none; box-shadow: var(--ring); }
.cmp-bar .btn-secondary { font-size: 12.5px; }
.cmp-bar-vs { font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); }

.report-hero-title .cmp-vs { color: var(--orange-500); }
.cmp-note { font-size: 12px; color: var(--text-muted); margin: 0 0 12px; }
.cmp-warn { color: var(--orange-700); font-weight: 600; }
.cmp-body { display: flex; flex-direction: column; gap: 24px; }
.cmp-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 24px; }
.cmp-charts .panel { margin: 0; }
/* Vega SVGs carry a viewBox, so max-width:100% + height:auto scales them DOWN to
   fit their column and keeps aspect ratio — never upscaling past native size
   (which made full-width charts oversized). Centered for the odd/full-width one. */
.cmp-body svg.marks { max-width: 100%; height: auto; display: block; margin: 0 auto; }
/* The full-width charts (direct children of cmp-body, not in the 2-up grid) fill
   their panel instead of sitting capped-and-centered with whitespace either side. */
.cmp-body > .panel svg.marks { width: 100%; }
.cmp-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.cmp-table th { font-family: var(--font-mono); font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .08em; color: var(--text-faint);
  text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border-subtle); }
.cmp-table td { padding: 7px 10px; border-bottom: 1px solid var(--border-subtle); color: var(--text-body); }
.cmp-diff-row:hover { background: var(--app-gray-50); }
.cmp-delta { font-variant-numeric: tabular-nums; font-family: var(--font-mono); }
.cmp-up { color: var(--green-600); } .cmp-down { color: var(--red-700); } .cmp-flat { color: var(--text-faint); }
.cmp-flip { color: var(--red-700); font-weight: 600; font-size: 11px; margin-left: 6px; font-family: var(--font-mono); }
.cmp-unmatched { margin-top: 12px; font-size: 12px; color: var(--text-muted); }
.cmp-unmatched ul { margin: 4px 0 0; padding-left: 18px; }
.cmp-transcript-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }
.cmp-side-title { font-family: var(--font-mono); font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .08em; color: var(--text-muted); margin: 0 0 8px; }
"""

# Keeps the wireframe's class names; its mock state-switcher rules are omitted because HTMX fragments drive state.
_FINDER_CSS = """
.finder { display:flex; flex-direction:column; gap:20px; width:100%; max-width:1600px; margin:0 auto; }
.finder-hero { position:relative; width:100%; max-width:1040px; box-sizing:border-box; align-self:center; padding:34px 36px 26px; border-radius:16px; background:linear-gradient(135deg,#fff 0%,#fbf7f2 55%,#f3f6f6 100%); border:1px solid var(--border-subtle); box-shadow:0 1px 2px rgba(20,18,30,.04); }
.finder-hero-bg { position:absolute; inset:0; border-radius:16px; overflow:hidden; pointer-events:none; }
.finder-hero-bg::before { content:''; position:absolute; inset:-40% -20% auto auto; width:420px; height:420px; border-radius:50%; background:radial-gradient(closest-side,rgba(255,143,52,.16),transparent 70%); }
.finder-hero-bg::after { content:''; position:absolute; inset:auto auto -50% -10%; width:380px; height:380px; border-radius:50%; background:radial-gradient(closest-side,rgba(2,85,88,.10),transparent 70%); }
.finder-hero > * { position:relative; }
.finder-title { font-family:var(--font-display); font-size:30px; font-weight:600; letter-spacing:-.02em; line-height:1.15; color:var(--text-strong); margin:0 0 6px; }
.finder-sub { margin:0 0 20px; font-size:14px; color:var(--text-muted); }
.finder-query { display:flex; align-items:center; gap:14px; padding:10px 10px 10px 22px; background:var(--surface-card); border:1px solid var(--border-default); border-radius:999px; box-shadow:0 1px 0 rgba(255,255,255,.8) inset,0 4px 14px -8px rgba(20,18,30,.25); }
.finder-query:focus-within { border-color:var(--accent); box-shadow:0 0 0 4px rgba(255,143,52,.14),0 4px 14px -8px rgba(20,18,30,.25); }
.finder-query svg { flex:0 0 auto; color:var(--text-faint); }
.finder-query .col { flex:1; min-width:0; }
.finder-query textarea { width:100%; border:0; outline:0; resize:none; background:transparent; margin:0; display:block; field-sizing:content; max-height:7.8em; padding:6px 0; font-family:var(--font-display); font-size:22px; line-height:1.3; color:var(--text-strong); }
.finder-query textarea::placeholder { color:var(--text-faint); }
.finder-go { flex:0 0 auto; display:inline-flex; align-items:center; gap:6px; height:44px; padding:0 22px; border:0; border-radius:999px; background:var(--accent); color:#fff; font:inherit; font-size:14px; font-weight:600; cursor:pointer; transition:background .15s ease-out; }
.finder-go:hover { background:var(--accent-hover); }
.finder-go:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.finder-go[disabled] { opacity:.4; cursor:default; }
.finder-key-hint { color:var(--text-faint); font-size:10.5px; white-space:nowrap; }
.finder-below { display:flex; align-items:center; gap:14px; margin-top:14px; font-size:12px; color:var(--text-muted); }
.finder-below .ex,.finder-controls .addwrap { position:relative; }
.finder-below .link,.xr .link,.xr-presets .link { border:0; background:transparent; padding:0; font:inherit; cursor:pointer; color:var(--text-muted); text-decoration:underline dotted; text-underline-offset:3px; }
.finder-below .spacer,.finder-controls .spacer,.xr .spacer { flex:1; }
.finder-below .hint { color:var(--text-faint); }
.finder-below kbd { font-family:var(--font-mono); font-size:10.5px; border:1px solid var(--border-default); border-radius:4px; padding:0 5px; background:var(--surface-sunken); }
.finder-seg { display:inline-flex; padding:3px; border:1px solid var(--border-default); border-radius:999px; background:var(--surface-card); }
.finder-seg label { position:relative; cursor:pointer; }
.finder-seg input { position:absolute; inset:0; opacity:0; margin:0; cursor:pointer; }
.finder-seg label > span { display:inline-flex; white-space:nowrap; align-items:center; height:24px; padding:0 12px; border-radius:999px; font-size:12px; color:var(--text-muted); transition:background .15s ease-out,color .15s ease-out; }
.finder-seg input:checked + span { background:var(--text-strong); color:#fff; }
.finder-seg input:focus-visible + span { outline:2px solid var(--accent); outline-offset:1px; }
.finder-hint-line { flex-basis:100%; margin:0; color:var(--text-muted); font-size:11px; }
body:has(input[name="scope"][value="within"]:checked) .finder-limit-field { display:none; }
.finder-examples { display:none; opacity:0; transform:translateY(-4px); transition:opacity .15s cubic-bezier(.22,1,.36,1),transform .15s cubic-bezier(.22,1,.36,1),display .15s allow-discrete; position:absolute; top:24px; left:-8px; z-index:30; min-width:460px; padding:6px; background:var(--surface-card); border:1px solid var(--border-default); border-radius:12px; box-shadow:var(--shadow-lg); }
.finder-below .ex:hover .finder-examples,.finder-below .ex:focus-within .finder-examples { display:block; opacity:1; transform:none; }
@starting-style { .finder-below .ex:hover .finder-examples,.finder-below .ex:focus-within .finder-examples { opacity:0; transform:translateY(-4px); } }
/* Invisible bridge over the gap under the trigger, so moving the pointer down keeps :hover. */
.finder-examples::before { content:""; position:absolute; left:0; right:0; top:-12px; height:12px; }
.finder-examples .hd { font-size:10.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--text-faint); padding:6px 10px 4px; }
.finder-examples button { display:block; width:100%; text-align:left; border:0; background:transparent; font:inherit; font-size:13px; color:var(--text-body); padding:7px 10px; border-radius:8px; cursor:pointer; transition:background .15s cubic-bezier(.22,1,.36,1),color .15s cubic-bezier(.22,1,.36,1),padding-left .15s cubic-bezier(.22,1,.36,1); }
.finder-examples button:hover,.finder-examples button:focus-visible { background:var(--surface-sunken); color:var(--text-strong); padding-left:14px; outline:0; }
@media (prefers-reduced-motion:reduce) { .finder-examples,.finder-examples button { transition:none; } .finder-examples { transform:none; } .finder-examples button:hover,.finder-examples button:focus-visible { padding-left:10px; } }
#finder-body,.finder-body-fragment { display:flex; flex-direction:column; gap:12px; }
/* The poll's swap slot for the run's progress; it adds no box, so its children lay out as the body's own. */
#finder-run-status { display:contents; }
.finder-controls { display:flex; flex-wrap:wrap; align-items:center; gap:8px 10px; font-size:12px; color:var(--text-body); padding:0 4px; }
/* /traces keeps the filter menu in #finder-controls (the Ask AI form reads it) but opens it from the toolbar's
   Filters button: the row collapses to an invisible anchor, and active filters show as toolbar chips. */
.finder-command ~ #finder-body .finder-controls { display:block; position:absolute; width:0; height:0; padding:0; }
.finder-command ~ #finder-body .finder-controls > :not(.addwrap),
.finder-command ~ #finder-body .finder-controls .add { display:none; }
.xr-chips { display:inline-flex; flex-wrap:wrap; gap:6px; font-size:12px; }
:is(.finder-controls,.xr-chips) .chip { display:inline-flex; align-items:stretch; height:28px; border-radius:999px; background:var(--surface-card); border:1px solid var(--border-default); color:var(--text-strong); box-shadow:0 1px 2px rgba(20,18,30,.05); overflow:hidden; transition:border-color .15s cubic-bezier(.22,1,.36,1),box-shadow .15s cubic-bezier(.22,1,.36,1),transform .15s cubic-bezier(.22,1,.36,1); }
:is(.finder-controls,.xr-chips) .chip b { display:inline-flex; align-items:center; padding:0 8px 0 10px; font-size:10px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--text-muted); background:var(--surface-sunken); border-right:1px solid var(--border-subtle); }
:is(.finder-controls,.xr-chips) .chip .v { display:inline-flex; align-items:center; padding:0 10px; font-size:12px; font-weight:500; }
:is(.finder-controls,.xr-chips) .chip.ai { background:var(--orange-50); border-color:var(--orange-100); }
:is(.finder-controls,.xr-chips) .ai-badge { font-size:9px; color:var(--orange-700); margin-right:4px; }
:is(.finder-controls,.xr-chips) .chip .chip-open { display:inline-flex; align-items:stretch; border:0; padding:0; background:transparent; font:inherit; color:inherit; cursor:pointer; }
:is(.finder-controls,.xr-chips) .chip.is-editable .v { padding-right:4px; }
:is(.finder-controls,.xr-chips) .chip.is-editable:hover { border-color:var(--border-strong); box-shadow:0 2px 6px -2px rgba(20,18,30,.2); transform:translateY(-1px); }
:is(.finder-controls,.xr-chips) .chip.is-editable:hover b { color:var(--text-strong); }
:is(.finder-controls,.xr-chips) .chip .finder-chip-remove { border:0; background:transparent; padding:0 8px 0 4px; font:inherit; font-size:11px; color:var(--text-faint); cursor:pointer; transition:color .15s ease-out; }
:is(.finder-controls,.xr-chips) .chip .finder-chip-remove:hover { color:var(--danger,#c2410c); }
.finder-controls .add { display:inline-flex; align-items:center; height:28px; padding:0 12px; border-radius:999px; border:1px dashed var(--border-strong); color:var(--text-muted); background:transparent; font:inherit; font-size:12px; cursor:pointer; transition:border-color .15s ease-out,color .15s ease-out,background .15s ease-out; }
.finder-controls .add:hover,.finder-controls .addwrap:has(.finder-facets.open) .add { border-style:solid; border-color:var(--accent); color:var(--accent); background:var(--surface-card); }
.finder-facet-loading { display:none; align-items:center; gap:6px; margin-left:8px; color:var(--text-muted); font-size:11px; }
.finder-facets[data-pending].open + .finder-facet-loading,.finder-facets.open + .finder-facet-loading.htmx-request { display:inline-flex; }
.finder-facets .facet-n { margin-left:auto; padding-left:12px; color:var(--text-muted); font-size:11px; font-variant-numeric:tabular-nums; }
.finder-facets .facet-scope { margin:2px 10px 6px; color:var(--text-muted); font-size:11px; line-height:1.35; }
.finder-facet-loading::before { content:""; width:9px; height:9px; border:1.5px solid var(--border-strong); border-top-color:var(--accent); border-radius:50%; animation:finder-spin .8s linear infinite; }
.finder-go-working,.finder-start-working { display:none; }
.finder-query.htmx-request .finder-go-idle { display:none; }
.finder-query.htmx-request .finder-go-working { display:inline; }
.finder-query.htmx-request .finder-go[disabled],#finder-start-form.htmx-request button[disabled] { opacity:1; }
#finder-mode-working { display:none; color:var(--text-muted); font-size:11px; }
#finder-mode-working.htmx-request { display:inline; }
#finder-drawer-loading { display:none; position:fixed; top:20px; right:20px; z-index:101; padding:8px 12px; border-radius:999px; background:#16151c; color:#f2f1f5; box-shadow:var(--shadow-lg); font-size:12px; }
#finder-drawer-loading.htmx-request { display:block; }
#finder-start-form.htmx-request .finder-start-working { display:inline-flex; align-items:center; margin:0 16px 16px; color:var(--text-muted); font-size:12px; }
.finder-progress-action { display:flex; align-items:center; gap:8px; margin-left:auto; white-space:nowrap; flex-shrink:0; }
.finder-progress-action > span { display:none; }
.finder-progress-action.htmx-request > span { display:inline; }
.finder-progress-action.htmx-request button { display:none; }
.finder-controls .quiet { display:inline-flex; align-items:center; gap:6px; color:var(--text-muted); }
.finder-controls .quiet input { height:28px; border:1px solid var(--border-default); border-radius:8px; background:var(--surface-card); color:var(--text-strong); padding:0 8px; font-family:var(--font-mono); font-size:12px; font-variant-numeric:tabular-nums; box-shadow:0 1px 2px rgba(20,18,30,.05); transition:border-color .15s ease-out,box-shadow .15s ease-out; }
.finder-controls .quiet input:hover { border-color:var(--border-strong); }
.finder-controls .quiet input:focus { outline:0; border-color:var(--accent); box-shadow:0 0 0 3px rgba(255,143,52,.18); }
.finder-controls .xr-range { gap:4px; }
.finder-controls .xr-range .xr-date { width:142px; padding:0 5px; border-radius:6px; }
.finder-controls .xr-range .xr-time { width:104px; padding:0 5px; border-radius:6px; }
.finder-controls .quiet b { font-size:10.5px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; color:var(--text-muted); }
.finder-controls .count { color:var(--text-muted); font-variant-numeric:tabular-nums; }
.finder-facets { display:none; position:absolute; top:34px; left:0; z-index:30; background:transparent; }
.finder-facets.open { display:block; animation:finder-menu-in .15s cubic-bezier(.22,1,.36,1); }
@keyframes finder-menu-in { from { opacity:0; transform:translateY(-4px); } to { opacity:1; transform:none; } }
.finder-facets .facet-list { width:200px; padding:6px; background:var(--surface-card); border:1px solid var(--border-default); border-radius:12px; box-shadow:var(--shadow-lg); }
.finder-facets .facet-item { display:flex; align-items:center; gap:8px; width:100%; padding:7px 8px 7px 10px; border:0; border-radius:8px; background:transparent; font:inherit; font-size:12.5px; color:var(--text-strong); text-align:left; cursor:pointer; transition:background .15s cubic-bezier(.22,1,.36,1),padding-left .15s cubic-bezier(.22,1,.36,1); }
.finder-facets .facet-item > span:first-child { flex:1; text-transform:capitalize; }
.finder-facets .facet-item .count { min-width:18px; padding:0 5px; border-radius:999px; background:var(--teal-50); color:var(--teal-600); font-size:10.5px; font-weight:600; text-align:center; line-height:16px; }
.finder-facets .facet-item .chev { color:var(--text-faint); font-size:14px; line-height:1; transition:transform .15s cubic-bezier(.22,1,.36,1); }
.finder-facets .facet-item:hover,.finder-facets .facet-item.is-active { background:var(--surface-sunken); padding-left:14px; }
.finder-facets .facet-item.is-active .chev { transform:translateX(2px); color:var(--text-strong); }
.finder-facets .facet-sub { position:absolute; top:0; left:206px; min-width:240px; max-width:360px; max-height:420px; overflow:auto; padding:6px; background:var(--surface-card); border:1px solid var(--border-default); border-radius:12px; box-shadow:var(--shadow-lg); animation:finder-sub-in .15s cubic-bezier(.22,1,.36,1); }
/* No room beside the category list on a phone: open the value list under it instead of off-screen. */
@media (max-width:600px) { .finder-facets .facet-sub { top:auto; left:0; max-width:calc(100vw - 48px); } }
@keyframes finder-sub-in { from { opacity:0; transform:translateX(-6px); } to { opacity:1; transform:none; } }
.finder-facets .facet-sub .hd { padding:6px 10px 4px; font-size:10.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--text-faint); }
.finder-facets .facet-search { box-sizing:border-box; width:calc(100% - 12px); margin:4px 6px 8px; padding:7px 9px; border:1px solid var(--border-default); border-radius:7px; background:var(--surface-card); color:var(--text-strong); font:inherit; font-size:12px; }
.finder-facets .facet-search:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
.finder-facets .facet-values { max-height:290px; overflow-y:auto; }
.finder-facets .facet-values label[hidden] { display:none; }
.finder-facets .facet-no-results,.finder-facets .facet-note { margin:5px 10px 8px; color:var(--text-muted); font-size:11px; line-height:1.4; }
.finder-facets .facet-no-results[hidden] { display:none; }
.finder-facets .facet-sub label { display:flex; align-items:center; gap:8px; padding:6px 10px; border-radius:8px; color:var(--text-body); font-size:12px; cursor:pointer; transition:background .15s ease-out; }
.finder-facets .facet-sub label:hover { background:var(--surface-sunken); color:var(--text-strong); }
.finder-facets .facet-sub label input[type=number] { width:96px; height:26px; border:1px solid var(--border-default); border-radius:6px; padding:0 6px; font-family:var(--font-mono); font-size:12px; }
.finder-facets .facet-sub label span { word-break:break-all; }
@media (prefers-reduced-motion:reduce) { .finder-facets.open,.finder-facets .facet-sub { animation:none; } .finder-facets .facet-item,.finder-controls .chip,.finder-controls .add { transition:none; } .finder-facets .facet-item:hover,.finder-facets .facet-item.is-active { padding-left:10px; } .finder-controls .chip.is-editable:hover { transform:none; } }
.finder-field { position:relative; border-radius:18px; overflow:hidden; background:#16151c; background-image:radial-gradient(ellipse at 20% 0%,rgba(255,143,52,.10),transparent 55%),radial-gradient(ellipse at 90% 100%,rgba(2,85,88,.25),transparent 55%); box-shadow:0 1px 2px rgba(20,18,30,.06),0 20px 50px -24px rgba(20,18,30,.45); }
.finder-progress { display:flex; align-items:center; gap:14px; padding:14px 20px; font-size:12.5px; color:#b9b7c2; border-bottom:1px solid rgba(255,255,255,.06); font-family:var(--font-mono); }
.finder-progress b { color:#f2f1f5; font-weight:500; }
.finder-progress { position:relative; }
.finder-progress-bar { position:absolute; left:0; right:0; bottom:-1px; height:2px; background:transparent; }
.finder-progress-bar i { display:block; height:100%; background:var(--accent); transition:width .4s ease; }
/* /traces renders the progress line on a light card, not the dark /find field. */
.finder:has(> .finder-command) .finder-progress { color:var(--text-muted); border-bottom-color:var(--border-subtle); }
.finder:has(> .finder-command) .finder-progress b { color:var(--text-strong); }
.finder:has(> .finder-command) .finder-progress .sep { color:var(--border-strong); }
.finder-progress .sep { color:rgba(255,255,255,.18); }
.finder-progress .state { color:var(--accent); letter-spacing:.08em; text-transform:uppercase; font-size:10.5px; font-weight:600; }
.finder-progress .btn-secondary { height:26px; font-size:12px; margin-left:auto; background:rgba(255,255,255,.06); border-color:rgba(255,255,255,.14); color:#f2f1f5; }
.finder-progress .live { width:7px; height:7px; border-radius:50%; background:var(--accent); box-shadow:0 0 10px var(--accent); animation:finder-live 1.2s ease-in-out infinite; }
@keyframes finder-live { 0%,100% { opacity:1; box-shadow:0 0 10px var(--accent); } 50% { opacity:.45; box-shadow:0 0 2px var(--accent); } }
.finder-pulse { display:inline-flex; gap:6px; margin-bottom:4px; }
.finder-pulse i { width:8px; height:8px; border-radius:50%; background:var(--accent); animation:finder-pulse 1.1s ease-in-out infinite; }
.finder-pulse i:nth-child(2) { animation-delay:.18s; }
.finder-pulse i:nth-child(3) { animation-delay:.36s; }
@keyframes finder-pulse { 0%,100% { transform:scale(.6); opacity:.35; } 50% { transform:scale(1); opacity:1; } }
@media (prefers-reduced-motion:reduce) { .finder-progress .live,.finder-pulse i { animation:none; } .finder-pulse i { opacity:1; transform:none; } }
.finder-matrix { display:grid; grid-template-columns:repeat(auto-fill,18px); grid-auto-rows:18px; gap:2px; justify-content:center; padding:26px 24px 22px; }
.finder-matrix i { display:grid; place-items:center; width:18px; height:18px; cursor:pointer; border-radius:4px; }
.finder-matrix i::before { content:''; width:5px; height:5px; border-radius:50%; background:#4a4957; }
.finder-matrix i.active::before { width:6px; height:6px; background:#d9d8e0; animation:finder-dot-pulse 1.2s ease-in-out infinite; }
@keyframes finder-dot-pulse { 0%,100% { transform:scale(1); opacity:1; } 50% { transform:scale(1.6); opacity:.55; } }
@media (prefers-reduced-motion:reduce) { .finder-matrix i.active::before { animation:none; } }
.finder-matrix i.unmatched::before { background:var(--c); opacity:.55; }
.finder-matrix i.match::before { width:9px; height:9px; background:var(--c); box-shadow:0 0 8px var(--c); }
.finder-matrix i.failed::before { width:8px; height:8px; border-radius:2px; background:#7a3b2e; }
.finder-matrix i.pending::before { width:6px; height:6px; background:transparent; border:1.5px solid #6e6c7c; }
.finder-matrix.idle { display:block; min-height:420px; padding:0; background-image:radial-gradient(circle,#6e6c7c 2.5px,transparent 3px); background-size:20px 20px; background-repeat:space; }
.finder-hint { position:absolute; inset:0; display:grid; place-items:center; text-align:center; padding:24px; background:radial-gradient(ellipse at center,rgba(22,21,28,.85) 25%,rgba(22,21,28,.4) 60%,transparent 100%); }
.finder-hint .inner { max-width:820px; display:flex; flex-direction:column; align-items:center; gap:10px; }
.finder-hint h4 { margin:0; text-transform:none; letter-spacing:-0.01em; font-family:var(--font-display); font-size:20px; font-weight:600; color:#f2f1f5; text-wrap:balance; }
.finder-hint p { margin:0; max-width:460px; font-size:13px; color:#a9a7b4; line-height:1.5; text-wrap:pretty; }
.finder-legend { display:flex; flex-wrap:wrap; align-items:center; gap:6px 18px; font-size:12px; color:#b9b7c2; padding:0 20px 16px; font-family:var(--font-mono); justify-content:center; }
.finder-legend .sw { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; box-shadow:0 0 0 1px rgba(255,255,255,.18); }
.finder-legend .sw.match { width:9px; height:9px; background:var(--chart-5); }
.finder-legend .sw.failed { border-radius:2px; background:#7a3b2e; }
.finder-legend b,.finder-legend .num { color:#f2f1f5; }
.finder-legend .num.hot { color:var(--accent); font-size:15px; }
.finder-legend .muted { color:#7c7a88; }
.finder-section { background:var(--surface-card); border:1px solid var(--border-subtle); border-radius:16px; padding:18px 20px 6px; box-shadow:0 1px 2px rgba(20,18,30,.04),0 10px 30px -20px rgba(20,18,30,.2); }
.finder-section-title { font-family:var(--font-display); font-size:16px; font-weight:600; margin:0 0 8px; color:var(--text-strong); }
.finder-section-sub { font-size:12px; color:var(--text-muted); margin:-4px 0 10px; }
.finder-table { width:100%; border-collapse:collapse; font-size:12.5px; background:var(--surface-card); }
.finder-table th { text-align:left; font-size:10.5px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:var(--text-faint); padding:10px 8px; background:var(--app-gray-50); border-bottom:1px solid var(--border-subtle); }
.finder-table td { padding:9px 8px; border-bottom:1px solid var(--border-subtle); color:var(--text-body); }
.finder-table tr:hover td { background:var(--app-gray-50); cursor:pointer; }
.finder-table td.id,.finder-table .conf { font-family:var(--font-mono); font-size:11.5px; }
.finder-table .verdict { display:inline-flex; align-items:center; gap:6px; }
.finder-table .verdict .sw { width:8px; height:8px; border-radius:2px; }
.finder-empty { color:var(--text-muted); font-size:12.5px; padding:10px 0; }
.finder-task { border:1px solid var(--border-subtle); border-radius:var(--radius-lg); background:var(--surface-card); }
.finder-task summary { list-style:none; display:flex; align-items:center; gap:10px; padding:14px 16px; cursor:pointer; font-size:13px; color:var(--text-body); }
.finder-task summary::-webkit-details-marker { display:none; }
.finder-task summary .chev { color:var(--text-faint); transition:transform .15s ease; }.finder-task[open] > summary .chev { transform:rotate(90deg); }
.finder-task-title { color:var(--text-strong); font-weight:600; }.finder-task-count { color:var(--text-muted); font-size:12px; }
.finder-task summary .kind { font-size:10.5px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:var(--teal-600); background:var(--teal-50); border:1px solid var(--teal-100); border-radius:4px; padding:1px 6px; }
.finder-task-body { padding:0 16px 16px; display:flex; flex-direction:column; gap:14px; font-size:13px; }
.finder-task-body h5 { margin:0 0 7px; font-size:10.5px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:var(--text-faint); }
.finder-task-body p { margin:0; color:var(--text-body); line-height:1.55; }
.finder-task-question { padding:16px; background:var(--surface-sunken); border:1px solid var(--border-subtle); border-radius:10px; }.finder-task-question p { font-size:14px; color:var(--text-strong); max-width:85ch; }
.finder-task-flow { display:flex; align-items:stretch; gap:10px; }.finder-task-stage { flex:1; min-width:0; padding:12px 14px; border:1px solid var(--border-subtle); border-radius:9px; }.finder-task-stage strong { font-size:14px; font-weight:600; color:var(--text-strong); }.finder-task-stage p { font-size:13px; }.finder-task-arrow { align-self:center; color:var(--text-faint); font-size:18px; }
.finder-task-criteria { padding:2px 0; }
.finder-tasks { display:flex; flex-direction:column; gap:10px; }.finder-tasks > .finder-task { margin:0; }
.finder:has(> .finder-command) .finder-tasks { padding:14px 20px; border-bottom:1px solid var(--border-subtle); }.finder-tasks > .rt-apply-btn { align-self:flex-start; }
.finder-task-none { margin:0; color:var(--text-muted); font-size:13px; line-height:1.5; }
.finder-task summary .finder-task-name { width:auto; max-width:24ch; border:1px solid var(--border-default); border-radius:6px; padding:3px 7px; font:inherit; font-weight:600; color:var(--text-strong); background:var(--surface-card); }
.finder-task-body textarea,.finder-task-body input { width:100%; border:1px solid var(--border-default); border-radius:6px; padding:6px 8px; font:inherit; font-size:13px; color:var(--text-strong); background:var(--surface-card); }
.finder-task-body textarea { min-height:96px; resize:vertical; }
.finder-crit { display:grid; grid-template-columns:110px 1fr; gap:4px 10px; font-size:12.5px; }
.finder-crit .lab { display:inline-flex; align-items:center; gap:6px; font-weight:600; color:var(--text-strong); }
.finder-crit .lab .sw { width:9px; height:9px; border-radius:2px; }
.finder-filter-output { border:1px solid var(--border-subtle); border-radius:var(--radius-lg); background:var(--surface-card); padding:16px; }.finder-filter-heading { display:flex; align-items:start; justify-content:space-between; gap:12px; }.finder-filter-heading h3 { margin:0 0 3px; font-size:14px; font-weight:600; color:var(--text-strong); }.finder-filter-heading p { margin:0; color:var(--text-muted); font-size:12px; line-height:1.45; }.finder-filter-count { flex:none; color:var(--teal-600); background:var(--teal-50); border:1px solid var(--teal-100); border-radius:999px; padding:3px 9px; font-size:11px; font-weight:600; }
.finder-filter-chips { display:flex; flex-wrap:wrap; gap:7px; margin:14px 0; }.finder-filter-chip { display:inline-flex; gap:7px; align-items:center; border:1px solid var(--border-default); border-radius:6px; padding:5px 8px; font-size:12px; color:var(--text-strong); }.finder-filter-chip b { font-size:10px; color:var(--text-muted); text-transform:uppercase; letter-spacing:.04em; }.finder-filter-none,.finder-filter-note { color:var(--text-muted); font-size:12px; }.finder-filter-note { margin:0; }
.finder-filter-json { border-top:1px solid var(--border-subtle); padding-top:10px; }.finder-filter-json summary { width:max-content; cursor:pointer; color:var(--teal-600); font-size:12px; font-weight:600; }.finder-filter-json summary:focus-visible,.finder-task > summary:focus-visible { outline:2px solid var(--accent); outline-offset:3px; }.finder-filter-json pre { max-height:380px; overflow:auto; margin:10px 0 0; padding:12px; background:var(--surface-sunken); border-radius:8px; color:var(--text-body); font-size:11.5px; line-height:1.5; }
@media (max-width:640px) { .finder-task-flow { flex-direction:column; }.finder-task-arrow { display:none; } }
@media (prefers-reduced-motion:reduce) { .finder-task summary .chev { transition:none; } }
.finder-review { border:1px solid var(--orange-100); background:var(--orange-50); border-radius:var(--radius-lg); padding:14px 16px; display:flex; align-items:center; gap:14px; font-size:13px; color:var(--text-body); }
.finder-review b { color:var(--text-strong); }
.finder-review .rt-apply-btn { margin-left:auto; }
.finder-task > form > .rt-apply-btn { margin:0 16px 16px; }
.finder-form-error { color:var(--red-700); margin:8px 0 0; font-size:12px; }
.finder-field.unavailable { filter:grayscale(.35); }
.finder-status { position:fixed; right:20px; bottom:20px; z-index:40; display:flex; align-items:center; gap:8px; padding:7px 12px; border-radius:999px; background:#16151c; border:1px solid rgba(255,255,255,.12); color:#d8d6e0; font-size:12px; font-family:var(--font-mono); box-shadow:0 6px 20px rgba(0,0,0,.25); }
.finder-status .icon { display:inline-flex; align-items:center; justify-content:center; width:14px; height:14px; font-size:12px; }
.finder-status.idle .icon::before { content:""; width:7px; height:7px; border-radius:50%; background:#77757f; }
.finder-status .spin { width:12px; height:12px; border-radius:50%; border:2px solid rgba(255,255,255,.18); border-top-color:var(--accent); animation:finder-spin .8s linear infinite; }
.finder-status.done .icon { color:var(--green-600); font-weight:700; }
.finder-status.failed .icon { color:var(--red-700); }
@keyframes finder-spin { to { transform:rotate(360deg); } }
@media (prefers-reduced-motion:reduce) { .finder-status .spin,.finder-facet-loading::before { animation:none; } }
.finder-progress-error { min-width:0; flex:1 1 auto; margin-left:auto; padding:4px 8px; font-size:12px; color:var(--red-700); }
.finder-progress-warning { color:var(--orange-700); }
.finder-task-body select { width:100%; border:1px solid var(--border-default); border-radius:6px; padding:6px 8px; font:inherit; font-size:13px; color:var(--text-strong); background:var(--surface-card); }
.finder-crit-row { display:grid; grid-column:1 / -1; grid-template-columns:110px 1fr; gap:4px 10px; }
.fd-meta { display:grid; grid-template-columns:auto 1fr; gap:4px 14px; font-size:12px; margin-bottom:14px; }
.fd-meta dt { color:var(--text-faint); }.fd-meta dd { margin:0; color:var(--text-strong); font-family:var(--font-mono); font-size:11.5px; }
.fd-row-head { display:flex; align-items:center; gap:8px; flex-wrap:wrap; font-size:14px; margin-bottom:4px; }
.fd-row-meta { font-size:12px; color:var(--text-muted); font-variant-numeric:tabular-nums; margin-bottom:12px; }
.fd-traces .fd-technical { margin-top:16px; border-top:1px solid var(--border-subtle); padding-top:12px; }
.fd-traces .fd-span-tree { display:flex; flex-direction:column; gap:5px; max-height:60vh; overflow:auto; }
.fd-traces .fd-span-node { min-width:0; }
.fd-traces .fd-span-node > summary { cursor:pointer; list-style:none; }
.fd-traces .fd-span-node > summary::-webkit-details-marker { display:none; }
.fd-traces .fd-span-node > summary:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.fd-traces .fd-span-row { display:grid; grid-template-columns:minmax(0,.65fr) minmax(0,1.2fr) minmax(96px,1.1fr) 54px 40px auto; gap:6px; align-items:center; padding:8px; border:1px solid var(--border-subtle); border-radius:6px; font-size:11px; }
.fd-traces .fd-span-row > * { min-width:0; }
.fd-traces .fd-span-row .fd-span-kind,.fd-traces .fd-span-row > b,.fd-traces .fd-span-row > span:not(.fd-span-duration):not(.fd-span-status) { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.fd-traces .fd-span-row .trace-link { padding:5px 6px; white-space:nowrap; }
.fd-traces .fd-span-kind,.fd-traces .fd-span-duration,.fd-traces .fd-span-status { color:var(--text-muted); }
.fd-traces .fd-span-duration { display:flex; align-items:center; gap:6px; white-space:nowrap; font-variant-numeric:tabular-nums; }
.fd-traces .fd-span-duration i { height:5px; min-width:2px; border-radius:3px; background:var(--accent); }
.fd-traces .fd-span-children { margin:5px 0 0 18px; padding-left:10px; border-left:1px solid var(--border-default); display:flex; flex-direction:column; gap:5px; }
.fd-traces .fd-span-error { border-left:3px solid var(--red-700); }
.fd-traces .fd-span-first-error { box-shadow:0 0 0 2px color-mix(in srgb,var(--red-700) 22%,transparent); }
.fd-traces .fd-span-message { margin:4px 8px 6px; color:var(--red-700); font-size:11px; white-space:pre-wrap; overflow-wrap:anywhere; }
.fd-traces .fd-technical > summary { cursor:pointer; color:var(--text-muted); font-size:12px; font-weight:600; }
.fd-traces .fd-technical[open] > summary { margin-bottom:12px; color:var(--text-strong); }
.fd-traces .fd-no-messages { display:flex; flex-direction:column; gap:4px; padding:18px; border:1px dashed var(--border-default); border-radius:8px; color:var(--text-muted); font-size:12px; }
.fd-traces .fd-no-messages b { color:var(--text-strong); font-size:13px; }
.fd-msg summary .fd-msg-meta { white-space:nowrap; color:var(--text-muted); background:var(--surface-sunken); border-radius:4px; padding:1px 4px; }
.fd-verdict { display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:8px; background:var(--surface-sunken); margin-bottom:14px; font-size:13px; }
.fd-verdict .sw { width:12px; height:12px; border-radius:3px; }.fd-tabs { display:flex; border-bottom:1px solid var(--border-subtle); margin-bottom:12px; }.fd-tabs button { padding:8px 12px; font:inherit; font-size:13px; color:var(--text-muted); border:0; border-bottom:2px solid transparent; background:transparent; cursor:pointer; }.fd-tabs button.on { color:var(--text-strong); border-color:var(--accent); }
.fd-msg { border-left:4px solid var(--border-default); padding:6px 10px; margin-bottom:10px; font-size:12.5px; color:var(--text-body); }.fd-msg.k-user { border-left-color:var(--traj-user); }.fd-msg.k-assistant { border-left-color:var(--traj-assistant); }.fd-msg.k-system { border-left-color:var(--traj-system); }.fd-msg.k-reasoning { border-left-color:var(--traj-reasoning); }.fd-msg.k-call { border-left-color:var(--traj-call); }.fd-msg.k-result { border-left-color:var(--traj-result); }.fd-msg.k-other { border-left-color:#aaa; }.fd-msg.on { background:#fff8f1; box-shadow:inset 0 0 0 1px #ffd2ad; }
.fd-msg summary { display:flex; align-items:center; gap:8px; min-width:0; cursor:pointer; list-style:none; }.fd-msg summary::-webkit-details-marker { display:none; }.fd-msg summary::before { content:'>'; display:inline-block; color:var(--text-muted); font-size:16px; line-height:1; transition:transform .15s ease; }.fd-msg[open] summary::before { transform:rotate(90deg); }.fd-msg summary:focus-visible { outline:2px solid var(--accent); outline-offset:3px; border-radius:2px; }
.fd-msg .role { flex:none; font-size:10.5px; font-weight:600; text-transform:uppercase; color:var(--text-faint); }.fd-msg-index { font-weight:400; }.fd-msg-preview { min-width:0; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; color:var(--text-muted); }.fd-msg[open] .fd-msg-preview { display:none; }.fd-msg-content { padding-top:6px; white-space:pre-wrap; overflow-wrap:anywhere; }
@media (prefers-reduced-motion:reduce) { .fd-msg summary::before { transition:none; } }
.fd-mini { display:flex; gap:2px; height:9px; margin:0 0 12px; }.fd-mini i { min-width:2px; border-radius:2px; opacity:.55; }.fd-mini i.on { opacity:1; outline:2px solid var(--ink-900); outline-offset:1.5px; } #fd-thread { overflow:auto; max-height:60vh; }
.fd-panel { margin:12px 0; }.fd-panel-title { font-size:10.5px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:var(--text-faint); margin-bottom:6px; }.fd-panel pre { white-space:pre-wrap; overflow:auto; background:var(--surface-sunken); border-radius:6px; padding:10px; font-family:var(--font-mono); font-size:11px; }

/* Trace explorer */
.xr { display:flex; flex-direction:column; gap:10px; }
.xr-toolbar { display:flex; align-items:center; gap:12px; font-size:12px; color:var(--text-muted); }
.xr-toolbar { min-height:42px; flex-wrap:wrap; }.xr-toolbar .spacer { flex:1; }.xr-toolbar-right { display:grid; grid-template-columns:minmax(185px,1fr) 72px 96px 64px 142px; gap:8px; align-items:center; }.xr-toolbar-right > * { box-sizing:border-box; }
.xr-status { min-height:28px; display:flex; align-items:center; gap:8px; padding:0 2px; border-bottom:1px solid var(--border-subtle); color:var(--text-muted); font-size:11px; font-variant-numeric:tabular-nums; }
.xr-totals { display:flex; flex-wrap:wrap; align-items:center; gap:2px 16px; padding:6px 2px; border-bottom:1px solid var(--border-subtle); color:var(--text-strong); font-size:12px; font-variant-numeric:tabular-nums; }.xr-total-label { color:var(--text-muted); font-size:11px; margin-right:3px; }
.xr-quickviews { display:flex; align-items:center; gap:2px; }.xr-quickviews button { border:0; border-bottom:2px solid transparent; background:transparent; padding:9px 8px 7px; color:var(--text-muted); font:inherit; cursor:pointer; }.xr-quickviews button.on { border-color:var(--accent); color:var(--text-strong); font-weight:700; }
.xr-filter { border:1px solid var(--border-subtle); border-radius:6px; background:var(--surface-card); padding:6px 9px; color:var(--text-body); font:inherit; cursor:pointer; }.xr-progress { white-space:nowrap; }
.xr-time-menu,.xr-sort,.xr-cols { position:relative; }.xr-time-menu > summary,.xr-sort > summary,.xr-exact > summary { cursor:pointer; list-style:none; white-space:nowrap; }.xr-time-options,.xr-cols form { position:absolute; top:calc(100% + 6px); right:0; z-index:6; display:grid; gap:8px; min-width:245px; padding:10px 12px; border:1px solid var(--border-subtle); border-radius:8px; background:#fff; box-shadow:0 8px 24px #0000001a; }.xr-time-options .xr-range { margin:2px 0; }.xr-time-options .xr-presets { display:flex; flex-wrap:wrap; gap:8px; }.xr-exact[open] { display:grid; gap:7px; }
.xr-sort[open] { display:block; }.xr-toolbar-right .xr-switch { justify-self:end; }
.xr-sort-options { position:absolute; top:calc(100% + 6px); right:0; z-index:30; display:flex; flex-direction:column; min-width:152px; padding:4px; border:1px solid var(--border-subtle); border-radius:8px; background:var(--surface-card); box-shadow:0 8px 24px #0000001a; }
.xr-sort-options > button { min-height:32px; padding:0 10px; border:0; border-radius:4px; background:transparent; color:var(--text-body); font:inherit; font-size:12px; text-align:left; white-space:nowrap; cursor:pointer; }
.xr-sort-options > button:hover { background:var(--surface-sunken); }
.xr-sort-options > button:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
.xr-cols { position:relative; }.xr-cols form { position:absolute; right:0; z-index:6; background:#fff; border:1px solid var(--border-subtle); border-radius:10px; padding:8px 12px; display:grid; gap:4px; min-width:200px; box-shadow:0 8px 24px #0000001a; }
.xr-table td.num,.xr-table th.num { text-align:right; font-variant-numeric:tabular-nums; }
.xr-table td small { display:block; margin-top:2px; color:var(--text-muted); font-size:10.5px; }.xr-table .trace-name { font-weight:600; }.xr-table .status-label { display:inline-flex; align-items:center; gap:6px; }.xr-table .status-label.ok { color:var(--green-600); font-weight:600; }.xr-table .status-label.err { color:var(--red-600); font-weight:600; }.xr-table .xr-match { color:#a66124; }
.xr-table tr.sel,.tv-r.sel { box-shadow:inset 3px 0 0 var(--traj-assistant); background:#f3f8f7; }
.finder:has(> .finder-command) .xr-skeleton td { height:30px; }
.finder:has(> .finder-command) .xr-skeleton td i { display:block; height:10px; max-width:85%; border-radius:5px; background:linear-gradient(90deg,#f0f1f3 25%,#e5e7eb 40%,#f0f1f3 65%); background-size:300% 100%; animation:xr-shimmer 1.3s ease infinite; }
.tv-status { display:inline-flex; justify-self:start; align-items:center; border-radius:999px; padding:2px 7px; font-size:9.5px; font-weight:700; line-height:1.3; }
.tv-status.ok { color:#155c36; background:#e5f3e9; }.tv-status.err { color:#9a1c17; background:#fde9e7; }.tv-status.other { color:#4f5663; background:#eceef1; }
.tv-end { display:flex; flex-direction:column; align-items:flex-end; gap:3px; }
.tv-nomsg { display:flex; align-items:center; justify-content:center; border:1px dashed #c9c8c2; background:#faf9f6; }
.tv-nomsg::after { content:"No conversation"; color:var(--text-muted); font:11px ui-sans-serif,system-ui; }
.finder:has(> .finder-command) .tv-skeleton { height:58px; }
.finder:has(> .finder-command) .tv-skeleton i { display:block; height:10px; border-radius:5px; background:linear-gradient(90deg,#f0f1f3 25%,#e5e7eb 40%,#f0f1f3 65%); background-size:300% 100%; animation:xr-shimmer 1.3s ease infinite; }
.finder:has(> .finder-command) .tv-skeleton i:first-child { width:70%; }.finder:has(> .finder-command) .tv-skeleton i:nth-child(2) { width:100%; }.finder:has(> .finder-command) .tv-skeleton i:last-child { width:60%; }
@keyframes xr-shimmer { to { background-position:-150% 0; } }
@media (prefers-reduced-motion:reduce) { .finder:has(> .finder-command) .xr-skeleton td i,.finder:has(> .finder-command) .tv-skeleton i { animation:none; } }
.cache-value { display:inline-grid; grid-template-columns:34px 4ch; align-items:center; gap:6px; vertical-align:middle; }
.cache-percent { text-align:right; }
.cachebar { display:inline-block; width:34px; height:4px; background:#ece9e4; border-radius:2px; }
.cachebar i { display:block; height:4px; border-radius:2px; background:var(--traj-assistant); }
.xr-pager { display:flex; gap:14px; justify-content:center; font-size:12px; color:var(--text-muted); }
.xr-presets { display:inline-flex; gap:7px; }
.xr-empty { padding:24px; text-align:center; color:var(--text-muted); background:var(--surface-card); border:1px solid var(--border-subtle); border-radius:10px; }
.xr-empty h4 { margin:0 0 6px; color:var(--text-strong); font-family:var(--font-display); font-size:16px; }
.xr-empty p { margin:0; font-size:13px; }
.xr-switch button { border:0; background:transparent; padding:0 12px; height:24px; border-radius:999px; font:inherit; font-size:12px; color:var(--text-muted); cursor:pointer; transition:background .15s ease-out,color .15s ease-out; }
.xr-switch button.on { background:var(--text-strong); color:#fff; }
.xr-switch button:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
.tv { font:13px ui-sans-serif,system-ui; color:var(--ink-900); background:#fff; border:1px solid var(--border-subtle); border-radius:12px; overflow-x:auto; position:relative; }
.tv .mono { font-family:var(--font-mono); }
.tv-lg { display:flex; gap:14px; align-items:center; padding:12px 18px; border-bottom:1px solid #ebe9e4; background:#f9f8f6; font-size:12px; color:var(--text-body); }
.tv-lg span { display:flex; gap:6px; align-items:center; }.tv-lg i { width:10px; height:10px; border-radius:3px; display:block; }.tv-lg em { font-style:normal; color:var(--text-muted); font-variant-numeric:tabular-nums; }
.tv-tool-toggle { margin-left:auto; border:1px solid #dcd8d0; border-radius:6px; background:#fff; color:var(--text-body); padding:5px 9px; font:inherit; cursor:pointer; white-space:nowrap; }.tv-tool-toggle[aria-pressed="true"] { background:#eae4f7; border-color:#bca9de; color:#573890; }.tv-tool-toggle:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.tv-hd,.tv-r { display:grid; grid-template-columns:minmax(220px,280px) 90px minmax(320px,1fr) 260px; align-items:center; column-gap:18px; box-sizing:border-box; min-width:1020px; padding:0 18px; }
.tv-hd { height:38px; font-size:10.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--text-muted); border-bottom:1px solid #ebe9e4; background:#fcfbf9; }
.tv-hd .ax,.tv-bar { display:grid; grid-template-columns:minmax(0,1fr) 160px; align-items:center; gap:12px; min-width:0; }
.tv-hd .ax > span { text-align:right; }.tv-scale { position:relative; height:24px; }.tv-scale b { position:absolute; bottom:2px; font-weight:500; transform:translateX(-50%); letter-spacing:0; text-transform:none; font:10px var(--font-mono); }.tv-scale b:first-child { transform:none; }.tv-scale b:last-child { transform:translateX(-100%); }
.tv-mh,.tv-m { display:grid; grid-template-columns:90px 80px minmax(0,1fr); align-items:center; gap:12px; }.tv-mh > span:last-child { text-align:right; }
.tv-r { min-height:58px; border-bottom:1px solid #f3f1ed; cursor:pointer; position:relative; transition:background .12s,opacity .15s; }.tv-r.nomatch { opacity:.45; }.tv-r:hover { background:#faf9f6; }
.tv-id { display:flex; gap:10px; align-items:center; min-width:0; }.tv-id .t { min-width:0; line-height:1.3; }.tv-id .a { font-weight:600; font-size:12.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }.tv-id .tv-model { display:block; overflow:hidden; color:var(--text-body); font-size:10.5px; text-overflow:ellipsis; white-space:nowrap; }.tv-id .s { font-size:10.5px; color:var(--text-muted); }
.tv .dot,.xr-table .dot { width:8px; height:8px; border-radius:50%; flex:none; }.tv .dot.ok,.xr-table .dot.ok { background:var(--traj-assistant); }.tv .dot.err,.xr-table .dot.err { background:var(--red-600); }
.tv-plot { position:relative; min-width:0; height:12px; }.tv-track { position:absolute; inset:4px 0; background:#f4f3ef; border-radius:3px; }.tv-segs { position:relative; display:flex; gap:1.5px; max-width:100%; height:12px; overflow:hidden; border-radius:3px; }.tv-segs i { display:block; height:12px; border-radius:2.5px; min-width:1px; transition:transform .1s,filter .1s; }.tv-segs i:hover { transform:scaleY(1.45); filter:brightness(1.05); }
.tv-end { font:10.5px var(--font-mono); color:var(--text-muted); text-align:right; white-space:nowrap; }.tv-embedding { display:flex; align-items:center; color:var(--text-muted); font-size:11px; }.tv-m { min-width:0; font-size:11.5px; font-variant-numeric:tabular-nums; color:var(--text-body); }.tv-m .io { display:flex; flex-direction:column; line-height:1.3; }.tv-m .io > span { display:flex; align-items:baseline; gap:4px; }.tv-m .io b { font-weight:600; color:var(--ink-900); }.tv-m .io small { font-size:10px; color:var(--text-muted); }.tv-m .cc { display:flex; flex-direction:column; align-items:flex-start; gap:4px; }.tv-m .cc b { font-weight:600; }.tv-m .cache-track { width:48px; height:4px; border-radius:3px; background:#e9e8e4; overflow:hidden; }.tv-m .cache-track i { display:block; height:100%; background:var(--traj-assistant); }.tv-cost { text-align:right; white-space:nowrap; }.tv-tick { width:3px; height:18px; border-radius:2px; flex:none; }
.tv-tip { position:absolute; pointer-events:none; background:#25232e; color:#efeee9; border-radius:10px; padding:11px 13px 10px; font-size:12px; line-height:1.4; width:320px; box-shadow:0 12px 32px #0000003a; display:block; z-index:5; }.tv-tip[hidden] { display:none; }.tv-tip .h { display:flex; align-items:center; gap:8px; white-space:nowrap; }.tv-tip .h i { width:10px; height:10px; border-radius:3px; flex:none; }.tv-tip .h b { font-weight:600; color:#fff; }.tv-tip .h .n { margin-left:auto; font:10.5px var(--font-mono); color:#9d9b96; }.tv-tip .sub { display:flex; align-items:center; gap:6px; margin-top:6px; }.tv-tip .tool { font:11px var(--font-mono); color:#ffc899; background:#ff97471f; border-radius:5px; padding:2px 7px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:200px; }.tv-tip .tk { font-size:10.5px; color:#b9b7b1; background:#ffffff12; border-radius:5px; padding:2px 7px; white-space:nowrap; }.tv-tip pre { margin:8px 0 0; padding:8px 9px; background:#ffffff0d; border-radius:6px; font:10.5px/1.5 var(--font-mono); color:#d9d7d1; white-space:pre-wrap; word-break:break-word; max-height:108px; overflow:hidden; }.tv-tip .c { margin-top:8px; font-size:10.5px; color:#8c8a91; display:flex; justify-content:space-between; }
i.k-system { background:var(--traj-system); }i.k-user { background:var(--traj-user); }i.k-assistant { background:var(--traj-assistant); }i.k-reasoning { background:var(--traj-reasoning); }i.k-call { background:var(--traj-call); }i.k-result { background:var(--traj-result); }i.k-other { background:repeating-linear-gradient(45deg,#ddd 0 3px,#eee 3px 6px); }i.k-tools { background:#9673c5; }
.finder:has(> .finder-command) .tv-overflow { position:absolute; right:2px; top:-2px; color:#5b5964; font:700 16px/1 ui-sans-serif,system-ui; text-shadow:0 0 2px #fff; }
@media (max-width:1020px) { .tv-lg { min-width:1020px; } }
.finder-command-lede { margin:0 0 8px; color:var(--text-body); font-size:13px; }
.finder-scope-label { color:var(--text-muted); font-size:12px; white-space:nowrap; }
.finder-command-help { margin:6px 0 0; color:var(--text-muted); font-size:11.5px; line-height:1.4; }
.finder-command-examples { display:flex; flex-wrap:nowrap; align-items:center; gap:8px; overflow:hidden; margin:0 4px 6px; font-size:12px; color:var(--text-muted); }
.finder-command-examples > * { flex:none; white-space:nowrap; }
.finder-command-examples button[hidden] { display:none; }
.finder-command-examples button { border:1px solid #bfc6cf; border-radius:6px; background:#f3f5f7; color:#494753; font:inherit; padding:6px 12px; box-shadow:0 1px 1px #0000000a; cursor:pointer; }
.finder-command-examples button:hover { background:#e5eeee; border-color:#025558; color:#025558; }
.finder-command-examples button:active { background:#d6e5e5; box-shadow:none; }
.finder-command-examples button:focus-visible { outline:2px solid #025558; outline-offset:1px; }
.finder-command:has(input[name="scope"][value="within"]:checked) .scope-new,
.finder-command:has(input[name="scope"][value="new"]:checked) .scope-within { display:none; }
.finder-progress .part { display:contents; }
.finder:has(> .finder-command) .finder-progress { column-gap:16px; }
.finder:has(> .finder-command) .finder-progress .sep { display:none; }
.finder:has(> .finder-command) .xr-switch button { display:inline-flex; align-items:center; white-space:nowrap; }
.finder:has(> .finder-command) .xr-switch button { gap:6px; }
.finder:has(> .finder-command) .xr-view-icon { display:inline-flex; flex:none; width:14px; height:14px; }
.finder:has(> .finder-command) .xr-view-icon svg { display:block; width:14px; height:14px; fill:none; stroke:currentColor; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round; }
.xr-help { display:inline-grid; place-items:center; width:13px; height:13px; margin-left:4px; border:1px solid currentColor; border-radius:50%; font-size:9px; font-weight:700; line-height:1; opacity:.6; vertical-align:1px; }
@media (max-width:1000px) { .finder-command-lede { font-size:12px; margin-bottom:6px; }.finder-command-help { font-size:11px; } }
"""


_INSIGHTS_CSS = """
.insights-layout { min-width:0; min-height:calc(100vh - 100px); }
.insights-overview { max-width:1180px; }
.insights-overview-head { display:flex; align-items:end; justify-content:space-between; gap:16px; margin:4px 0 22px; }
.insights-overview-head h2 { margin:0 0 4px; color:var(--text-strong); font:600 21px var(--font-display); }
.insights-overview-head p { margin:0; color:var(--text-muted); font-size:12.5px; }
.insights-overview-head > span { color:var(--text-muted); font-size:12px; white-space:nowrap; }
.insights-overview-new { display:inline-flex; align-items:center; justify-content:center; padding:8px 14px; border-radius:6px; background:var(--teal-600); color:white; font-size:12.5px; font-weight:600; text-decoration:none; white-space:nowrap; }
.insights-overview-new:hover { filter:brightness(.86); }
.insights-overview-table { border-top:1px solid var(--border-default); }
.insights-overview-columns,.insights-overview-row { display:grid; grid-template-columns:minmax(180px,1.6fr) minmax(170px,.9fr) 100px 70px minmax(120px,1fr) 20px; align-items:center; gap:12px; }
.insights-overview-columns { padding:10px 15px; color:var(--text-muted); font-size:10.5px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; }
.insights-overview-row { min-height:68px; padding:10px 15px; border-top:1px solid var(--border-subtle); color:var(--text-body); text-decoration:none; font-size:12.5px; }
.insights-overview-row:hover { background:var(--teal-50); }
.insights-overview-name { display:flex; flex-direction:column; min-width:0; gap:4px; }
.insights-overview-name strong { overflow:hidden; color:var(--text-strong); font-weight:600; text-overflow:ellipsis; white-space:nowrap; }
.insights-overview-name small { overflow:hidden; color:var(--text-muted); font-size:11.5px; text-overflow:ellipsis; white-space:nowrap; }
.insights-overview-stage-wrap { display:flex; align-items:center; gap:7px; min-width:0; color:var(--text-muted); font-size:10.5px; white-space:nowrap; }
.insights-overview-stage-wrap > span:first-child { display:none; font-weight:600; }
.insights-overview-stages { display:flex; flex:1; min-width:0; max-width:155px; margin:0; padding:0; list-style:none; }
.insights-overview-stages li { position:relative; flex:1; height:10px; }
.insights-overview-stages li::before { content:''; position:absolute; top:4px; left:0; right:0; height:2px; background:var(--border-default); }
.insights-overview-stages li:last-child::before { display:none; }
.insights-overview-stages li::after { content:''; position:absolute; top:1px; left:0; width:8px; height:8px; border-radius:50%; background:var(--border-default); }
.insights-overview-stages li.completed::before,.insights-overview-stages li.completed::after { background:var(--green-600); }
.insights-overview-stages li:has(+ li.error)::before { background:var(--red-700); }
.insights-overview-stages li.running::after { background:var(--teal-600); }
.insights-overview-stages li.error::before,.insights-overview-stages li.error::after { background:var(--red-700); }
.insights-overview-no-stages { color:var(--text-muted); font-size:10.5px; }
.insights-overview-status { display:inline-flex; justify-self:start; align-items:center; gap:6px; color:var(--text-body); text-transform:capitalize; }
.insights-overview-status::before { content:''; width:7px; height:7px; border-radius:50%; background:var(--green-600); }
.insights-overview-status.running::before { background:var(--orange-600); }
.insights-overview-status.error,.insights-overview-status.unreadable { color:var(--red-700); }
.insights-overview-status.error::before,.insights-overview-status.unreadable::before { background:var(--red-700); }
.insights-overview-status.cancelled::before,.insights-overview-status.other::before { background:var(--text-muted); }
.insights-overview-number { font-family:var(--font-mono); }
.insights-overview-dimensions { overflow:hidden; color:var(--text-muted); text-overflow:ellipsis; white-space:nowrap; }
.insights-overview-arrow { color:var(--teal-600); font-size:18px; text-align:right; }
.insights-overview-empty { max-width:620px; margin:40px auto; padding:32px; border:1px solid var(--border-subtle); border-radius:10px; background:var(--surface-card); text-align:center; }
.insights-overview-empty h2 { margin:0 0 8px; color:var(--text-strong); font-size:19px; }
.insights-overview-empty p { margin:0 0 18px; color:var(--text-muted); font-size:13px; line-height:1.5; }
.insights-muted { color:var(--text-muted); font-size:11.5px; }
.insights-main { min-width:0; padding:4px 0 24px; }
.insights-header { display:flex; justify-content:space-between; gap:12px 20px; }
.insights-header-copy { min-width:0; }
.insights-header h2 { margin:0 0 6px; color:var(--text-strong); font:600 22px var(--font-display); overflow-wrap:anywhere; }
.insights-subtitle { display:flex; align-items:center; flex-wrap:wrap; gap:6px 12px; margin:0 0 14px; color:var(--text-muted); font-size:12px; }
.insights-header-count { color:var(--teal-600); font-weight:700; }
.insights-chip-group { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin:6px 0; }
.insights-group-label,.insights-detail-kicker { min-width:68px; color:var(--text-faint); font-size:10px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }
.insights-chip { padding:3px 8px; border:1px solid var(--border-subtle); border-radius:5px; background:var(--surface-card); color:var(--text-body); font-size:11px; }
.insights-chip b { color:var(--text-muted); font-weight:600; }
.insights-models { margin:9px 0 0 74px; color:var(--text-muted); font-size:11px; }
.insights-models summary { width:max-content; cursor:pointer; color:var(--teal-600); }
.insights-models span { display:block; margin-top:5px; overflow-wrap:anywhere; }
.insights-actions { display:flex; align-items:flex-start; gap:8px; flex:none; }
.insights-action { display:inline-flex; padding:7px 10px; border:1px solid var(--border-default); border-radius:6px; color:var(--text-body); font-size:12px; text-decoration:none; }
.insights-action:hover { background:var(--surface-sunken); }
.insights-error,.insights-warning { margin:12px 0; padding:10px 13px; border:1px solid var(--red-100); border-radius:8px; background:var(--red-50); color:var(--red-700); font-size:12.5px; }
.insights-projection { margin:12px 0; padding:11px 13px; border:1px solid var(--border-default); border-radius:8px; background:var(--surface-card); color:var(--text-body); font-size:12.5px; }
.insights-projection p { margin:5px 0 0; color:inherit; }
.insights-error ul { margin:6px 0 0; padding-left:20px; }
.insights-warning { border-color:var(--orange-100); background:var(--orange-50); color:var(--text-body); }
.insights-warning:has(ul) { display:flex; flex-wrap:wrap; align-items:baseline; gap:4px 12px; }
.insights-warning ul { margin:0; padding-left:0; list-style:none; }
.insights-progress { width:100%; box-sizing:border-box; margin:14px 0; padding:13px 15px 15px; border:1px solid var(--border-subtle); border-radius:8px; background:var(--surface-card); }
.insights-progress-head { display:flex; justify-content:space-between; align-items:center; gap:24px; margin-bottom:10px; }
.insights-progress-head h3 { margin:0; color:var(--text-strong); font-size:13px; }
.insights-progress-head span { color:var(--text-muted); font-size:12px; }
.insights-progress ol { display:flex; margin:0; padding:24px 0 0; list-style:none; overflow-x:auto; }
.insights-stage { position:relative; flex:1 0 88px; display:flex; flex-direction:column; align-items:center; min-width:0; padding:0 3px; color:var(--text-body); font-size:11.5px; text-align:center; }
.insights-stage::before { content:""; position:absolute; top:3px; left:-50%; right:50%; height:1px; background:var(--border-default); }
.insights-stage:first-child::before { display:none; }
.insights-stage.completed::before,.insights-stage.running::before,.insights-stage.error::before { height:2px; background:var(--green-600); }
.insights-stage-mark { position:relative; z-index:1; width:8px; height:8px; flex:none; margin-bottom:7px; border-radius:50%; background:var(--border-default); }
.insights-stage-label { min-height:2.5em; max-width:100%; line-height:1.25; text-wrap:balance; }
.insights-stage small { margin-top:3px; color:var(--text-muted); font-size:10px; font-weight:400; font-variant-numeric:tabular-nums; line-height:1.2; white-space:nowrap; }
.insights-stage.completed { color:var(--text-muted); }
.insights-stage.completed .insights-stage-mark { background:var(--green-600); }
.insights-stage.running { color:var(--teal-600); font-weight:600; }
.insights-stage.running .insights-stage-mark,.insights-stage.error .insights-stage-mark { width:12px; height:12px; margin:-2px 0 5px; background:var(--teal-600); }
.insights-stage.running .insights-stage-mark { box-shadow:0 0 0 0 color-mix(in srgb,var(--teal-600) 40%,transparent); animation:insights-stage-pulse 1.6s ease-in-out infinite; }
.insights-stage.error { color:var(--red-700); font-weight:600; }
.insights-stage.error .insights-stage-mark { background:var(--red-700); }
.insights-stage.error::before { background:var(--red-700); }
.insights-stage.skipped { opacity:.5; }
.insights-stage-flag { position:absolute; top:-24px; left:50%; transform:translateX(-50%); padding:3px 7px; border-radius:4px; background:var(--teal-600); color:#fff; font:600 10px/1.2 system-ui,sans-serif; white-space:nowrap; }
.insights-stage-flag::after { content:""; position:absolute; top:100%; left:50%; margin-left:-4px; border:4px solid transparent; border-bottom:0; border-top-color:var(--teal-600); }
.insights-stage.error .insights-stage-flag { background:var(--red-700); }
.insights-stage.error .insights-stage-flag::after { border-top-color:var(--red-700); }
.insights-progress .sr-only { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }
@keyframes insights-stage-pulse { 50% { box-shadow:0 0 0 6px color-mix(in srgb,var(--teal-600) 0%,transparent); } }
@media (prefers-reduced-motion:reduce) { .insights-stage.running .insights-stage-mark { animation:none; } }
.insights-run-page { max-width:850px; }
.insights-run-head h2 { margin:0 0 4px; color:var(--text-strong); font:600 22px var(--font-display); }
.insights-run-head p { margin:0 0 18px; color:var(--text-muted); font-size:12.5px; }
.insights-run-form { color:var(--text-body); font-size:12.5px; }
.insights-run-form .irf-steps { display:flex; gap:7px; margin:0 0 14px; padding:0; list-style:none; }
.insights-run-form .irf-steps li { flex:1; padding:9px 10px; border-bottom:2px solid var(--border-default); color:var(--text-muted); font-size:12px; }
.insights-run-form .irf-steps li.active { border-color:var(--teal-600); color:var(--teal-600); font-weight:600; }
.insights-run-form .irf-steps li.completed { border-color:var(--green-600); color:var(--text-body); }
.insights-run-form .irf-step { padding:20px; border:1px solid var(--border-subtle); border-radius:10px; background:var(--surface-card); }
.insights-run-form .irf-step h3 { margin:0 0 14px; color:var(--text-strong); font-size:16px; }
.insights-run-form .irf-step h4 { margin:18px 0 8px; color:var(--text-faint); font-size:11px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; }
.insights-run-form .seg { display:flex; gap:2px; padding:3px; border-radius:9px; background:var(--surface-sunken); }
.insights-run-form .seg label { position:relative; flex:1; min-width:0; }
.insights-run-form .seg input { position:absolute; opacity:0; pointer-events:none; }
.insights-run-form .seg span { display:block; padding:6px clamp(3px,1vw,12px); border-radius:7px; color:var(--text-muted); font-size:clamp(10.5px,1.7vw,12px); font-weight:500; text-align:center; white-space:nowrap; cursor:pointer; }
.insights-run-form .seg .irf-tab-icon { display:inline-block; margin:-2px 6px 0 0; vertical-align:middle; }
.insights-run-form .seg label:has(input:checked) span { background:var(--surface-card); color:var(--text-strong); box-shadow:0 1px 2px rgba(0,0,0,.08); }
.insights-run-form .seg label:has(:focus-visible) span { outline:2px solid var(--teal-600); outline-offset:2px; }
.insights-run-form .irf-field { display:block; margin-top:14px; }
.insights-run-form .irf-label { display:block; margin-bottom:5px; color:var(--text-strong); font-size:12.5px; font-weight:600; }
.insights-run-form .irf-label i { color:var(--text-muted); font-style:normal; font-weight:400; }
.insights-run-form input[type="text"],.insights-run-form input[type="number"],.insights-run-form textarea,.insights-run-form select { box-sizing:border-box; width:100%; padding:8px 10px; border:1px solid var(--border-default); border-radius:7px; background:var(--surface-card); color:var(--text-strong); font:inherit; font-weight:400; }
.insights-run-form textarea { resize:vertical; }
.insights-run-form input[readonly] { background:var(--surface-sunken); cursor:default; }
.insights-run-form .irf-hint { display:block; margin:6px 0 0; color:var(--text-muted); font-size:12px; }
.insights-run-form .irf-row { display:flex; flex-wrap:wrap; align-items:center; gap:8px 22px; margin-top:14px; }
.insights-run-form .irf-row label { display:inline-flex; align-items:center; gap:8px; }
.insights-run-form .irf-row input { width:84px; font-variant-numeric:tabular-nums; }
.insights-run-form .irf-file { display:flex; gap:8px; }
.insights-run-form .irf-file input[type="text"] { flex:1; min-width:0; }
.insights-run-form .irf-facets { margin-top:18px; }
.insights-run-form #insights-snapshot-preview { margin-top:10px; }
.insights-run-form #insights-facet-options { position:relative; min-width:0; }
.insights-run-form #insights-facet-options[aria-busy="true"] { opacity:.58; }
.insights-run-form #insights-facet-options .finder-controls { margin:0; min-width:0; max-width:100%; }
.insights-run-form #insights-facet-options .addwrap { position:static; min-width:0; max-width:100%; }
.insights-run-form #insights-facet-options .finder-facets { left:0; width:100%; max-width:100%; display:none; grid-template-columns:minmax(108px,.72fr) minmax(0,1.28fr); gap:0; overflow:hidden; background:var(--surface-card); border:1px solid var(--border-default); border-radius:12px; box-shadow:var(--shadow-lg); }
.insights-run-form #insights-facet-options .finder-facets.open { display:grid; }
.insights-run-form #insights-facet-options .finder-facets .facet-list { width:auto; min-width:0; max-height:320px; overflow-y:auto; padding:6px; background:transparent; border:0; border-radius:0; box-shadow:none; }
.insights-run-form #insights-facet-options .finder-facets .facet-sub { position:static; min-width:0; max-width:none; max-height:320px; overflow:auto; padding:6px; border:0; border-left:1px solid var(--border-subtle); border-radius:0; box-shadow:none; animation:none; }
.insights-run-form #insights-facet-options .finder-facets .facet-sub[hidden] { display:none; }
.insights-run-form #insights-facet-options .finder-facets .facet-item { min-width:0; gap:5px; padding-left:7px; padding-right:6px; font-size:11.5px; }
.insights-run-form #insights-facet-options .finder-facets .facet-item:hover,.insights-run-form #insights-facet-options .finder-facets .facet-item.is-active { padding-left:9px; }
.insights-run-form #insights-facet-options .finder-facets .facet-sub label { min-width:0; padding:6px 7px; }
.insights-run-form #insights-facet-options .finder-facets .facet-sub input[type="checkbox"] { flex:none; width:16px; min-width:16px; height:16px; margin:0; padding:0; appearance:auto; accent-color:var(--teal-600); }
.insights-run-form #insights-facet-options .finder-facets .facet-sub label span { min-width:0; overflow-wrap:anywhere; }
.insights-run-form #insights-facet-options .finder-facets .facet-search { width:calc(100% - 12px); max-width:100%; }
.insights-run-form .insights-selected-facets { display:flex; flex-wrap:wrap; align-items:center; gap:6px; min-width:0; max-width:100%; }
.insights-run-form .insights-selected-facets .chip b { padding:0 6px 0 10px; background:transparent; border-right:0; }
.insights-run-form .insights-selected-facets .chip .v { padding:0 8px 0 0; }
.insights-run-form .presets { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
.insights-run-form .preset { padding:12px; border:1px solid var(--border-subtle); border-radius:12px; background:var(--surface-card); color:var(--text-body); font:inherit; text-align:left; cursor:pointer; }
.insights-run-form .preset:hover { border-color:var(--border-strong); }
.insights-run-form .preset.on { border-color:var(--teal-600); box-shadow:0 0 0 3px var(--teal-100); }
.insights-run-form .preset b { display:block; margin:0 0 2px; color:var(--text-strong); }
.insights-run-form .preset span { color:var(--text-muted); font-size:12px; }
.insights-run-form .toggles { display:flex; flex-wrap:wrap; gap:8px; }
.insights-run-form .tg { position:relative; display:inline-flex; align-items:center; gap:6px; max-width:100%; padding:5px 12px; border:1px solid var(--border-default); border-radius:999px; background:var(--surface-card); color:var(--text-body); font:inherit; cursor:pointer; }
.insights-run-form .tg input { position:absolute; opacity:0; pointer-events:none; }
.insights-run-form .tg:has(input:checked),.insights-run-form .tg.on { border-color:var(--text-strong); background:var(--text-strong); color:#fff; }
.insights-run-form .tg:has(input:checked) .tg-title::before,.insights-run-form .tg.on .tg-title::before { content:"✓ "; }
.insights-run-form .tg:has(:focus-visible) { outline:2px solid var(--teal-600); outline-offset:2px; }
.insights-run-form .tg small { max-width:22ch; overflow:hidden; opacity:.7; text-overflow:ellipsis; white-space:nowrap; }
.insights-run-form .tg button { padding:0 0 0 2px; border:0; background:none; color:inherit; font:inherit; cursor:pointer; }
.insights-run-form .tg-add { border-style:dashed; color:var(--text-muted); }
.insights-run-form .irf-custom-list { display:contents; }
.insights-run-form .card { margin-top:12px; padding:14px; border:1px solid var(--border-subtle); border-radius:12px; background:var(--surface-app); }
.insights-run-form .irf-editor-actions { display:flex; gap:8px; margin-top:12px; }
.insights-run-form .irf-compact { margin:0 0 14px; padding:8px 12px; border:1px solid var(--border-subtle); border-radius:8px; background:var(--surface-sunken); color:var(--text-body); font-size:12px; }
.insights-run-form .irf-review { margin-top:16px; border:1px solid var(--border-subtle); border-radius:8px; background:var(--surface-sunken); }
.insights-run-form .irf-review:not(:has(.irf-summary, .irf-estimate)) { display:none; }
.insights-run-form .irf-review :is(h4,h5) { margin:0 0 8px; color:var(--text-strong); font-size:12.5px; font-weight:600; letter-spacing:0; text-transform:none; }
.insights-run-form .irf-plan:not(:empty) { padding:14px 16px; }
.insights-run-form .irf-plan p { margin:0 0 12px; color:var(--text-strong); font-size:13px; }
.insights-run-form .irf-stages { display:flex; flex-wrap:wrap; gap:6px; margin:0; padding:0; list-style:none; counter-reset:irf-stage; }
.insights-run-form .irf-stages li { display:inline-flex; align-items:baseline; gap:6px; padding:3px 9px 3px 4px; border:1px solid var(--border-default); border-radius:999px; background:var(--surface-card); color:var(--text-body); font-size:11.5px; counter-increment:irf-stage; }
.insights-run-form .irf-stages li::before { content:counter(irf-stage); min-width:16px; padding:0 4px; border-radius:999px; background:var(--app-gray-50); color:var(--text-muted); font-size:10.5px; font-weight:600; font-variant-numeric:tabular-nums; text-align:center; }
.insights-run-form .irf-estimate { padding:14px 16px; }
.insights-run-form .irf-plan:not(:empty) + div .irf-estimate { border-top:1px solid var(--border-subtle); }
.insights-run-form .irf-estimate h5 { margin:14px 0 4px; }
.insights-run-form .irf-estimate-totals { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px 20px; margin:0 0 14px; }
.insights-run-form .irf-estimate-totals dt { color:var(--text-muted); font-size:11.5px; font-weight:500; }
.insights-run-form .irf-estimate-totals dd { margin:0; }
.insights-run-form .irf-total-value { margin-top:2px !important; color:var(--text-strong); font-size:14px; font-weight:600; font-variant-numeric:tabular-nums; }
.insights-run-form .irf-total-basis { margin-top:2px !important; color:var(--text-muted); font-size:11px; line-height:1.4; }
.insights-run-form .irf-estimate-scroll { overflow-x:auto; border:1px solid var(--border-subtle); border-radius:6px; background:var(--surface-card); }
.insights-run-form .irf-estimate-table { width:100%; margin:0; border:0; border-collapse:collapse; background:transparent; font-size:12px; }
.insights-run-form .irf-estimate-table :is(th,td) { padding:7px 10px; border:0; border-bottom:1px solid var(--border-subtle); text-align:left; vertical-align:top; }
.insights-run-form .irf-estimate-table tbody tr:last-child > * { border-bottom:0; }
.insights-run-form .irf-estimate-table thead th { background:var(--app-gray-50); color:var(--text-muted); font-size:10.5px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; white-space:nowrap; }
.insights-run-form .irf-estimate-table tbody th { background:transparent; color:var(--text-strong); font-size:12px; font-weight:500; letter-spacing:0; text-transform:none; white-space:nowrap; }
.insights-run-form .irf-estimate-table .num { color:var(--text-body); font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; }
.insights-run-form .irf-estimate-basis { min-width:220px; color:var(--text-muted); font-size:11.5px; line-height:1.45; }
.insights-run-form .irf-estimate-unknowns { margin:0; padding-left:18px; color:var(--text-body); font-size:12px; line-height:1.6; }
.insights-run-form .irf-estimate .irf-hint { margin:10px 0 0; font-size:11.5px; }
/* report.css turns every table into stacked cards below 640px; this one scrolls instead, so its columns keep their headers. */
@media (max-width:640px) { .insights-run-form .irf-estimate-table thead { display:table-header-group; } .insights-run-form .irf-estimate-table tr { display:table-row; margin:0; border:0; border-radius:0; } .insights-run-form .irf-estimate-table :is(th,td) { display:table-cell; } }
.insights-run-form .irf-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:14px; }
.insights-run-form .irf-btn { padding:8px 15px; border:1px solid var(--border-default); border-radius:7px; background:var(--surface-card); color:var(--text-strong); font:inherit; font-size:12.5px; cursor:pointer; }
.insights-run-form .irf-btn.primary { border-color:var(--teal-600); background:var(--teal-600); color:#fff; }
.insights-run-form .irf-btn.start { border-color:var(--orange-700); background:var(--orange-700); color:#fff; }
.insights-run-form .irf-btn:hover:not(:disabled) { filter:brightness(.93); }
.insights-run-form .irf-btn:disabled { opacity:.55; cursor:wait; }
.insights-run-form [role="alert"] { max-width:100%; line-height:1.45; }
.insights-run-form :is(input,select,textarea,button):focus-visible { outline:2px solid var(--teal-600); outline-offset:2px; }
.insights-run-form[data-mount="dialog"] .irf-step { padding:14px; }
.insights-run-form[data-mount="dialog"] .irf-actions { position:sticky; bottom:0; margin:0; padding:12px 0; background:var(--surface-app); }
@media (max-width:650px) { .insights-run-form .presets { grid-template-columns:1fr; }.insights-run-form .irf-step { padding:14px; }.insights-run-form .seg { display:grid; grid-template-columns:1fr 1fr; }.insights-run-form .seg span { font-size:12px; }.insights-run-form #insights-facet-options .finder-facets { grid-template-columns:minmax(96px,.7fr) minmax(0,1.3fr); }.insights-run-form #insights-facet-options .finder-facets .facet-list,.insights-run-form #insights-facet-options .finder-facets .facet-sub { max-height:min(320px,55vh); }.insights-run-form .irf-file { flex-wrap:wrap; } }
.insights-run-form [hidden] { display:none !important; }
.insights-tool-counts { display:flex; flex-wrap:wrap; gap:6px; list-style:none; padding:0; margin:4px 0 12px; }
.insights-tool-counts li { padding:3px 8px; border:1px solid var(--border-default); border-radius:6px; font-size:12px; }
.insights-tool-counts b { font-variant-numeric:tabular-nums; margin-left:4px; }
.insights-facet-unavailable { grid-column:1/-1; margin:0; padding:10px; border:1px solid var(--border-default); border-radius:7px; color:var(--text-muted); font-size:12px; }
.insights-facet-unavailable button { border:0; padding:0; background:none; color:var(--teal-600); font:inherit; font-weight:600; text-decoration:underline; cursor:pointer; }
.insights-tabs { display:flex; gap:3px; margin:18px 0 16px; border-bottom:1px solid var(--border-subtle); }
.insights-tab { padding:9px 12px; border-bottom:2px solid transparent; color:var(--text-muted); text-decoration:none; font-size:12.5px; }
.insights-tab.active { border-color:var(--teal-600); color:var(--teal-600); font-weight:600; }
.insights-tab:hover { color:var(--teal-600); }
.insights-dimensions { display:grid; grid-template-columns:minmax(0,1.1fr) minmax(300px,1fr); gap:16px; }
.insights-dimension-control { display:flex; align-items:center; gap:9px; margin:0 0 12px; color:var(--text-muted); font-size:12px; }
.insights-dimension-control select { padding:6px 8px; border:1px solid var(--border-default); border-radius:6px; background:var(--surface-card); color:var(--text-strong); font:inherit; }
.insights-dimension h3 { margin:0 0 8px; color:var(--text-strong); font-size:14px; }
.insights-dimension[hidden] { display:none; }
.insights-tree,.insights-detail,.insights-label-card { padding:14px; border:1px solid var(--border-subtle); border-radius:10px; background:var(--surface-card); }
.insights-cluster-row { display:flex; align-items:center; gap:9px; width:100%; min-height:34px; padding:5px 7px; border:0; border-bottom:1px solid var(--border-subtle); background:transparent; color:var(--text-body); text-align:left; font:inherit; font-size:12.5px; }
.insights-cluster-row.child { cursor:pointer; padding-left:22px; }
.insights-cluster-row.child:hover,.insights-cluster-row.child.selected { background:var(--teal-50); }
.insights-tree-marker { color:var(--text-faint); }
.insights-count { margin-left:auto; color:var(--text-muted); font-size:11.5px; white-space:nowrap; }
.insights-bar { display:inline-block; width:64px; height:7px; overflow:hidden; border-radius:5px; background:var(--surface-sunken); }
.insights-bar i { display:block; height:100%; border-radius:5px; background:var(--green-600); }
.insights-cluster-row.failed { color:var(--red-700); }
.insights-cluster-row.muted { color:var(--text-muted); }
.insights-detail { align-self:start; background:var(--surface-sunken); }
.insights-detail h3 { margin:5px 0; color:var(--text-strong); font-size:16px; }
.insights-detail p { color:var(--text-body); font-size:12.5px; line-height:1.5; }
.insights-detail-kicker { margin:16px 0 7px; }
.insights-label-summary,.insights-example { display:flex; align-items:flex-start; gap:10px; padding:7px 0; border-bottom:1px solid var(--border-subtle); color:var(--text-body); font-size:12px; }
.insights-example code { flex:none; color:var(--text-muted); font-size:10.5px; }
.insights-example span { flex:1; min-width:0; }
.insights-trace-id { color:var(--teal-600); text-decoration:none; }
.insights-trace-id:hover { text-decoration:underline; }
.insights-trace-detail { max-width:850px; }
.insights-trace-detail h2 { margin:4px 0 8px; color:var(--text-strong); font-size:19px; }
.insights-trace-detail h2 code { overflow-wrap:anywhere; }
.insights-trace-detail section { margin-top:18px; padding:16px; border:1px solid var(--border-subtle); border-radius:10px; background:var(--surface-card); }
.insights-trace-detail section h3 { margin:0 0 10px; color:var(--text-strong); font-size:13px; }
.insights-trace-detail section p,.insights-trace-detail section li { color:var(--text-body); font-size:12.5px; line-height:1.5; }
.insights-trace-detail dl { display:grid; grid-template-columns:max-content minmax(0,1fr); gap:8px 18px; margin:12px 0 0; font-size:12px; }
.insights-trace-detail dt { color:var(--text-muted); }
.insights-trace-detail dd { min-width:0; margin:0; overflow-wrap:anywhere; color:var(--text-body); }
.insights-label-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; }
.insights-card-head { display:flex; justify-content:space-between; gap:8px; }
.insights-card-head h3 { margin:0; color:var(--text-strong); font-size:14px; }
.insights-card-head span { color:var(--text-muted); font-size:11px; }
.insights-label-instructions { color:var(--text-muted); font-size:12px; }
.insights-label-row { display:flex; align-items:center; gap:9px; padding:7px 0; border-bottom:1px solid var(--border-subtle); color:var(--text-body); text-decoration:none; font-size:12px; }
.insights-label-row b { margin-left:auto; }
.insights-table-wrap { overflow-x:auto; border:1px solid var(--border-subtle); border-radius:9px; background:var(--surface-card); }
.insights-table { width:100%; border-collapse:collapse; font-size:12px; }
.insights-table th { padding:9px 10px; background:var(--surface-sunken); color:var(--text-muted); text-align:left; text-transform:uppercase; font-size:10px; letter-spacing:.04em; }
.insights-table td { padding:9px 10px; border-top:1px solid var(--border-subtle); color:var(--text-body); white-space:nowrap; }
.insights-table .id { font-family:var(--font-mono); font-size:11px; }
.insights-filter-bar { display:flex; flex-wrap:wrap; align-items:center; gap:7px; margin:0 0 12px; }
.insights-filter-chip { display:inline-flex; align-items:center; gap:8px; padding:4px 8px; border:1px solid var(--border-default); border-radius:999px; background:var(--surface-card); color:var(--text-body); font-size:11.5px; text-decoration:none; }
.insights-filter-chip span { color:var(--text-muted); font-size:14px; }
.insights-clear-filters { color:var(--teal-600); font-size:11.5px; text-decoration:none; }
.insights-empty,.insights-empty-state { color:var(--text-muted); font-size:12.5px; }
.insights-empty-state { padding:28px; border:1px dashed var(--border-default); border-radius:10px; background:var(--surface-card); }
.insights-empty-state h2,.insights-empty-state h3 { margin:0 0 8px; color:var(--text-strong); }
.insights-map-view { margin-top:8px; }
.insights-map-toolbar,.insights-chart-controls { display:flex; flex-wrap:wrap; align-items:center; gap:12px; margin:0 0 12px; color:var(--text-muted); font-size:12px; }
.insights-map-toolbar label,.insights-chart-controls label { display:flex; align-items:center; gap:6px; }
.insights-map-toolbar select,.insights-chart-controls select { padding:6px 8px; border:1px solid var(--border-default); border-radius:6px; background:var(--surface-card); color:var(--text-strong); font:inherit; }
.insights-layout .vega-embed { width:100%; }
#insights-crosstab { max-width:100%; overflow-x:auto; }
.insights-map-layout { display:grid; grid-template-columns:minmax(0,1fr) minmax(260px,320px); gap:12px; }
.insights-map-chart { min-width:0; height:470px; border:1px solid var(--border-subtle); border-radius:8px; background:var(--surface-card); }
.insights-map-empty { padding:18px; border:1px dashed var(--border-default); border-radius:8px; background:var(--surface-card); color:var(--text-muted); }
.insights-map-empty h4 { margin:0 0 6px; color:var(--text-strong); }
.insights-map-empty p { margin:0; }
.insights-map-fullscreen-viewer { min-width:0; }
.insights-map-head { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:12px; }
.insights-map-head h3 { margin:0 0 4px; color:var(--text-strong); font:600 17px var(--font-display); }
.insights-map-head p { margin:0; color:var(--text-muted); font-size:12px; line-height:1.45; }
.insights-map-head button { flex:none; background:var(--surface-card); font:inherit; cursor:pointer; }
.insights-map-projection { margin-bottom:12px; color:var(--text-body); font-size:12px; }
.insights-map-projection select { margin-left:6px; padding:6px 8px; border:1px solid var(--border-default); border-radius:6px; background:var(--surface-card); color:var(--text-strong); font:inherit; }
.insights-map-fullscreen-viewer .insights-map-chart { height:min(68vh,720px); }
.insights-map-fullscreen-viewer:fullscreen,.insights-map-fullscreen-viewer.is-expanded { position:fixed; inset:0; z-index:1000; display:flex; flex-direction:column; box-sizing:border-box; width:100vw; height:100vh; padding:20px 24px; overflow:hidden; background:var(--surface-card); }
.insights-map-fullscreen-viewer:fullscreen [data-map-projection-panel]:not([hidden]),.insights-map-fullscreen-viewer.is-expanded [data-map-projection-panel]:not([hidden]) { display:flex; flex:1; flex-direction:column; min-height:0; }
.insights-map-fullscreen-viewer:fullscreen .insights-map-view,.insights-map-fullscreen-viewer.is-expanded .insights-map-view { display:flex; flex:1; flex-direction:column; min-height:0; }
.insights-map-fullscreen-viewer:fullscreen .insights-map-layout,.insights-map-fullscreen-viewer.is-expanded .insights-map-layout { flex:1; min-height:0; }
.insights-map-fullscreen-viewer:fullscreen .insights-map-chart,.insights-map-fullscreen-viewer.is-expanded .insights-map-chart { height:100%; }
.insights-map-fullscreen-viewer:fullscreen [data-map-detail],.insights-map-fullscreen-viewer.is-expanded [data-map-detail] { box-sizing:border-box; align-self:stretch; min-height:0; height:100%; overflow:auto; }
.insights-map-fullscreen-viewer button:focus-visible,.insights-map-fullscreen-viewer select:focus-visible { outline:2px solid var(--teal-600); outline-offset:2px; }
@media (max-width:1100px) { .insights-map-fullscreen-viewer:fullscreen .insights-map-layout,.insights-map-fullscreen-viewer.is-expanded .insights-map-layout { grid-template-columns:1fr; grid-template-rows:minmax(0,2fr) minmax(90px,1fr); }.insights-map-fullscreen-viewer:fullscreen .insights-map-chart,.insights-map-fullscreen-viewer.is-expanded .insights-map-chart { min-height:0; } }
@media (max-width:550px) { .insights-map-head { flex-wrap:wrap; }.insights-map-fullscreen-viewer:fullscreen,.insights-map-fullscreen-viewer.is-expanded { padding:12px; } }
.insights-overview-new:focus-visible,.insights-overview a:focus-visible,.insights-layout a:focus-visible,.insights-layout button:focus-visible,.insights-layout summary:focus-visible,.insights-layout select:focus-visible,.insights-layout input:focus-visible,.insights-layout textarea:focus-visible { outline:2px solid var(--teal-600); outline-offset:2px; }
.finder-analyze-matches { margin:16px 0; padding:16px; border:1px solid var(--border-subtle); border-radius:10px; background:var(--surface-card); }
.finder-analyze-matches h3 { margin:0 0 6px; color:var(--text-strong); }
.finder-analyze-matches p { color:var(--text-muted); font-size:12.5px; }
.finder-analyze-matches label { display:block; margin:12px 0 4px; color:var(--text-muted); font-size:11px; font-weight:700; }
.finder-analyze-matches pre { overflow:auto; margin:0; padding:10px; border-radius:6px; background:var(--surface-sunken); font-size:12px; }
@media (max-width:1000px) { .insights-header { flex-wrap:wrap; }.insights-actions { order:2; }.insights-tabs { overflow-x:auto; white-space:nowrap; }.insights-tab { flex:none; } }
@media (max-width:1100px) { .insights-overview-columns,.insights-overview-row { grid-template-columns:minmax(140px,1fr) minmax(140px,.8fr) 95px 20px; }.insights-overview-columns span:nth-child(4),.insights-overview-columns span:nth-child(5),.insights-overview-number,.insights-overview-dimensions { display:none; } }
@media (max-width:1350px) { .insights-map-fullscreen-viewer:not(:fullscreen):not(.is-expanded) .insights-map-chart { height:min(55vh,600px); } }
@media (max-width:1100px) { .insights-map-fullscreen-viewer:not(:fullscreen):not(.is-expanded) .insights-map-layout { grid-template-columns:1fr; } }
@media (max-width:850px) { .insights-dimensions,.insights-map-layout { grid-template-columns:1fr; }.insights-map-chart,.insights-map-fullscreen-viewer:not(:fullscreen):not(.is-expanded) .insights-map-chart { height:360px; } }
@media (max-width:550px) { .insights-overview-columns,.insights-overview-row { grid-template-columns:minmax(120px,1fr) 90px 20px; gap:8px; padding-left:9px; padding-right:9px; }.insights-overview-columns span:nth-child(2) { display:none; }.insights-overview-row { grid-template-areas:'name status arrow' 'stages stages stages'; }.insights-overview-name { grid-area:name; }.insights-overview-stage-wrap,.insights-overview-no-stages { grid-area:stages; margin:3px 0 2px; }.insights-overview-stage-wrap > span:first-child { display:inline; }.insights-overview-status { grid-area:status; }.insights-overview-arrow { grid-area:arrow; }.insights-overview-head { align-items:start; }.insights-header h2 { font-size:19px; }.insights-group-label { min-width:100%; }.insights-models { margin-left:0; } }
@media (max-width:600px) {
  body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .app-shell { flex-direction:column; }
  body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .app-sidebar,html.sidebar-collapsed body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .app-sidebar { position:static; width:auto; height:auto; padding:8px 12px 0; overflow:visible; }
  body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .app-brand,html.sidebar-collapsed body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .app-brand { justify-content:flex-start; padding:4px 6px 8px; }
  html.sidebar-collapsed body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .app-brand .brand-name { display:inline; }
  body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .app-nav { flex-direction:row; overflow-x:auto; padding-bottom:8px; }
  body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .nav-item,html.sidebar-collapsed body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .nav-item { flex:none; justify-content:flex-start; padding:7px 9px; white-space:nowrap; }
  body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .nav-item.active { order:-1; }
  html.sidebar-collapsed body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .nav-item span { display:inline; }
  body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .sidebar-toggle { display:none; }
  body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .app-content { padding:16px; }
  body.eq-dashboard:has(:is(.insights-layout,.dash-wrap)) .app-topbar { padding:0 16px; }
}
"""

DASHBOARD_CSS = (
    _DASHBOARD_CSS_HEAD
    + _TAB_RULES
    + _SIM_TAB_ACCENT
    + _DASHBOARD_CSS_TAIL
    + _SIM_REPORT_CSS
    + _SIM_TRANSCRIPT_CSS
    + _SIM_REPORT_OVERRIDES_CSS
    + _SIM_TRANSCRIPT_OVERRIDES_CSS
    + _RT_REPORT_CSS
    + _SIM_COMPARE_CSS
    + _FINDER_CSS
    + _TRACES_DENSITY_CSS
    + _INSIGHTS_CSS
)
