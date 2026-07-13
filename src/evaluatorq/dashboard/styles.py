"""Dashboard chrome CSS.

``load_css()`` (``common.reports``) styles the *report body* and defines the
brand ``:root`` tokens.  ``theme.EDITORIAL_CSS`` supplies the v1 editorial-skin
tokens.  This module supplies the chrome that consumes them: the sidebar shell,
the topbar, the combined landing, the per-kind run lists, and the report-view
filter/body split.

Inlined as the last ``<style>`` block by ``shell.page()`` so its rules win on
equal specificity and all ``var(--…)`` references resolve.
"""

from __future__ import annotations

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
}
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
.runs-grid-row .run-score { font-family: var(--font-mono); font-size: 13px; font-weight: 600; text-align: right; }
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

/* Landing 'Recent runs' Type column: surface glyph + label (no colored bubble) */
.type-cell { display: inline-flex; align-items: center; gap: 6px; font-weight: 500; white-space: nowrap; }
.type-cell svg { width: 15px; height: 15px; flex: 0 0 auto; }
.type-cell.redteam svg { color: var(--red-600); }
.type-cell.sim svg { color: var(--teal-600); }

/* Landing 'Recent runs' — airy aligned columns, no table chrome */
.recent-runs { display: flex; flex-direction: column; }
.rr-head, .rr-row {
    display: grid;
    grid-template-columns: minmax(130px, 1.2fr) minmax(0, 2fr) auto auto 56px auto;
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
.rr-meta { font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); }
.rr-row .run-score { font-family: var(--font-mono); font-size: 13px; font-weight: 600; }
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
    font-family: var(--font-sans); font-size: 10px; letter-spacing: 0.05em;
    text-transform: uppercase; color: var(--text-faint);
}
.donut-wrap { gap: 20px; align-items: center; }
/* Outcomes donut sits in a white .rk-panel on the sim report: center the
   donut + legend row both axes so it fills the (stretched) panel height. */
.sim-report .rk-panel .donut-wrap { flex-direction: column; justify-content: center; align-items: center; gap: 20px; height: 100%; }
.sim-report .rk-panel .donut-legend { flex-direction: row; flex-wrap: wrap; justify-content: center; gap: 16px; }
.donut-legend {
    list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column;
    gap: 6px; font-family: var(--font-sans); font-size: 12px; color: var(--text-muted);
}
.donut-legend li { display: flex; align-items: center; gap: 8px; }
.donut-key { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }

/* ==== ⌘K global search (topbar) ==================================== */
.topbar-search { position: relative; margin-left: auto; margin-right: 14px; }
.search-input {
    width: 260px; max-width: 40vw; padding: 7px 30px 7px 12px;
    font-family: var(--font-sans); font-size: 13px; color: var(--text-strong);
    background: var(--surface-card, #fff); border: 1px solid var(--border, #e2e0da);
    border-radius: 8px;
}
.search-input:focus { outline: none; border-color: var(--teal-600); }
.search-kbd {
    position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
    font-family: var(--font-mono); font-size: 10px; color: var(--text-faint);
    pointer-events: none;
}
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

/* ==== settings — read-only config ================================= */
.config-list { display: flex; flex-direction: column; }
.config-row { display: flex; justify-content: space-between; gap: 16px; padding: 8px 0; border-bottom: 1px solid var(--border, #eee); }
.config-key { font-family: var(--font-sans); font-size: 13px; color: var(--text-muted); }
.config-val { font-family: var(--font-mono); font-size: 12px; color: var(--text-strong); word-break: break-all; text-align: right; }
.config-note { font-size: 13px; color: var(--text-muted); line-height: 1.5; }

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
.run-row .run-score { font-family: var(--font-mono); font-size: 13px; font-weight: 600; }
.run-score.good { color: var(--green-600); }
.run-score.warn { color: var(--amber-600); }
.run-score.none { color: var(--text-faint); }

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
.report-head { margin-bottom: 18px; }
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

/* ==== sim filter rail (right side) — scoped to .filter-form--sim so the
   redteam .filter-sidebar form (still generic radio/checkbox) is untouched. */
.filter-form--sim {
    flex: 0 0 208px;
    position: static;
    /* Drop the rail down so its top lines up with the tab bar rather than the
       hero title. ponytail: magic offset = hero (title+sub, no KPI cards) + tab
       bar height; recompute if the hero grows a line or the tab bar resizes. */
    margin-top: 138px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 16px;
}
.filter-rail-header {
    display: flex; align-items: center; gap: 6px;
    color: var(--text-faint);
}
.filter-rail-title {
    font-family: var(--font-sans);
    font-size: 11px; font-weight: 400;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--text-faint);
}
.filter-form--sim .filter-group { margin-bottom: 0; }
.filter-form--sim .filter-label {
    font-family: var(--font-sans);
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
}
.filter-chip.is-active {
    background: var(--surface-card);
    border-color: var(--border-default);
    color: var(--text-body);
}
.filter-chip-input {
    /* visually-hidden — checked state drives .is-active via server re-render */
    position: absolute; width: 1px; height: 1px;
    padding: 0; margin: -1px; overflow: hidden;
    clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
}
.filter-chip-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--border-default);
    flex-shrink: 0;
}
.filter-chip.is-active .chip-dot-green { background: var(--green-600); }
.filter-chip.is-active .chip-dot-red { background: var(--red-600); }
.filter-chip.is-active .chip-dot-jade { background: var(--green-600); }

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
.filter-dd-status.is-all { background: var(--green-600); }
.filter-dd-status.is-partial { background: var(--chart-3); }
.filter-dd-status.is-none { background: var(--border-default); }
.filter-dd-name { color: var(--text-faint); }
.filter-dd-value { flex: 1 1 auto; color: var(--text-body); font-weight: 500; }
.filter-dd-chevron { flex-shrink: 0; color: var(--text-faint); }
.filter-dd[open] .filter-dd-chevron { transform: rotate(180deg); }
.filter-dd-menu {
    position: absolute; z-index: 20;
    top: calc(100% + 4px); left: 0; right: 0;
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
    padding-top: 10px;
    border-top: 1px solid var(--border-subtle);
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-faint);
}

.filter-sidebar,
.download-sidebar,
.rt-panel {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 18px;
}
.download-title { margin: 0 0 14px; font-size: 15px; font-family: var(--font-display); }
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

.download-sidebar { margin-top: 20px; }
.download-link {
    display: inline-block;
    margin-right: 8px;
    margin-top: 4px;
    padding: 6px 14px;
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    text-decoration: none;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-body);
    background: var(--surface-card);
}
.download-link:hover { background: var(--app-gray-50); color: var(--text-strong); }

/* ==== interactive panels ============================================ */
.rt-interactive-panels, .sim-interactive-panels { margin-top: 32px; }
.rt-panel { margin-bottom: 22px; }
.rt-panel-title { margin: 0 0 14px; font-size: 17px; font-family: var(--font-display); }
.rt-panel-loading { color: var(--text-faint); font-style: italic; }

/* ==== report hero (above tabs) ===================================== */
.report-hero { margin: 4px 0 20px; }
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
/* Sim report is all-sans to match the mockup (no monospace anywhere). */
.sim-report .report-hero-sub { font-family: var(--font-sans); }

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
    for i in range(1, 10)
)

_DASHBOARD_CSS_TAIL = """
@media (max-width: 760px) {
    .filter-swap-container { flex-direction: column; }
    .filter-form { position: static; flex-basis: auto; width: 100%; }
    /* Rail stacks full-width here, so the tab-bar-alignment offset no longer
       applies — drop it to avoid a dead gap above the stacked rail. */
    .filter-form--sim { margin-top: 0; }
    .stat-band { grid-template-columns: repeat(2, 1fr); }
    .dash-row2, .dash-row-eq { grid-template-columns: 1fr; }
}
"""

# Active-tab underline scoped to the Agent Sim report only (spec: "Active-tab
# underline: orange accent, scoped to `.sim-report .tabs` only — not
# dashboard-wide"). Extra `.sim-report` class gives this higher specificity
# than the surface-neutral `_TAB_RULES` above, so it wins without `!important`
# and the shared Red Team tab bar is untouched.
_SIM_TAB_ACCENT = ''.join(
    f'.sim-report .tabs > .tab-radio:nth-of-type({i}):checked ~ .tab-bar > .tab-label:nth-child({i}) '
    '{ border-bottom-color: var(--orange-500); }\n'
    for i in range(1, 10)
)

# ==== .sim-report — Agent Sim report design-mockup alignment ============
# All rules scoped under `.sim-report` (report_tabs.sim_report_tabs' wrapper)
# per docs/superpowers/specs/2026-07-10-agent-sim-report-alignment-design.md.
# Consumes report_kit.py primitives (exec_summary/panel/bar_rows/tag) and the
# shared `.kpi-band`/`.kpi-card` markup from common/reports/report.css — the
# overrides here only apply inside `.sim-report`, so the landing-page KPI
# tiles and the flat HTML export (which never carry this class) are untouched.
_SIM_REPORT_CSS = """
.sim-report .tab-count {
    font-family: var(--font-sans); font-size: 11px; font-weight: 600;
    background: var(--surface-sunken); color: var(--text-muted);
    border-radius: 999px; padding: 1px 7px; margin-left: 5px;
}

/* ---- Executive summary callout (spec Overview.1) ---- */
.sim-report .exec-summary {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--orange-500);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
}
.sim-report .es-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.sim-report .es-label {
    font-family: var(--font-sans); font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-faint);
}
.sim-report .es-confidence {
    font-family: var(--font-sans); font-size: 10px; font-weight: 600;
    border: 1px solid; border-radius: 999px; padding: 2px 8px;
}
.sim-report .es-body {
    margin: 8px 0 0; font-size: 14px; line-height: 1.6; color: var(--text-body);
    max-width: 760px;
}
.sim-report .es-body strong { color: var(--text-strong); }

/* ---- 5-card KPI band (spec Overview.2) ---- */
.sim-report .kpi-band {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;
    margin: 16px 0;
}
.sim-report .kpi-card {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-left: none;
    border-top: 3px solid var(--teal-600);
    border-radius: 12px;
    padding: 15px 16px;
}
.sim-report .kpi-card--pass    { border-top-color: var(--green-600); }
.sim-report .kpi-card--fail    { border-top-color: var(--red-600); }
.sim-report .kpi-card--warn    { border-top-color: var(--amber-600); }
.sim-report .kpi-card--neutral { border-top-color: var(--teal-600); }
.sim-report .kpi-value {
    font-family: var(--font-sans); font-size: 28px; font-weight: 600;
    color: var(--text-strong); line-height: 1.1;
}
.sim-report .kpi-label { font-size: 12px; color: var(--text-muted); margin-top: 7px; }

/* ---- 2-col grids (donut+tokens, personas+scenarios) ---- */
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

/* ---- Panel wrapper (report_kit.panel) ---- */
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
.sim-report .rk-panel-body { margin-top: 12px; }

/* ---- Tag (report_kit.tag) ---- */
.sim-report .rk-tag {
    display: inline-block; font-size: 11px; font-weight: 500;
    border: 1px solid var(--border-default); border-radius: 999px;
    padding: 1px 8px; margin-left: 8px; color: var(--text-muted);
}

/* ---- Personas panel (Overview) ---- */
.sim-report .sim-persona-item + .sim-persona-item { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-subtle); }
.sim-report .sim-persona-row { display: flex; align-items: center; gap: 4px; }
.sim-report .sim-persona-name { font-family: var(--font-sans); font-size: 13px; font-weight: 600; color: var(--text-strong); }
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
    font-family: var(--font-sans); font-size: 10px; text-transform: uppercase; flex-shrink: 0;
}

/* ---- Breakdown tab: stacked panels + heatmap/histogram/tables ---- */
.sim-report .rk-panel + .rk-panel,
.sim-report .rk-panel + .report-card,
.sim-report .report-card + .rk-panel {
    margin-top: 20px;
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
.sim-report .rk-heat-empty { background: var(--surface-sunken); color: var(--text-faint); }

/* Per-persona / per-scenario tables (html_table output) + failures table */
.sim-report .rk-panel-body { display: grid; }
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
    /* Each table panel sizes to its own content — don't stretch the shorter one
       and leave a blank gap under its last row. */
    align-items: start;
}
/* The 2nd child otherwise matches the stacked-panel `.rk-panel + .rk-panel` rule
   and inherits its 20px top margin, pushing it below the first panel so their
   tops no longer line up. Zero it here (same fix as sim-overview-grid-2). */
.sim-report .sim-breakdown-grid-2 > .rk-panel + .rk-panel { margin-top: 0; }
@media (max-width: 760px) {
    .sim-report .sim-breakdown-grid-2 { grid-template-columns: 1fr; }
}

/* ---- Turn quality tab (spec §Turn) ---- */
/* Line chart legend (report_kit.line_chart) */
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

/* ---- Config tab (spec §Config) ---- */
/* Run-configuration meta grid (report_kit.meta_grid) */
.sim-report .rk-meta-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 16px 24px;
}
.sim-report .rk-meta-key {
    font-family: var(--font-sans); font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text-faint);
}
.sim-report .rk-meta-value {
    font-size: 13.5px; color: var(--text-body); margin-top: 4px;
}
/* Personas panel (name · communication style · background) */
.sim-report .sim-config-persona-row {
    display: flex; align-items: baseline; gap: 12px;
    margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border-subtle);
}
.sim-report .sim-config-persona-row:first-child { margin-top: 0; padding-top: 0; border-top: none; }
.sim-report .sim-config-persona-name {
    font-size: 13px; font-weight: 600; color: var(--text-strong); min-width: 160px;
}
.sim-report .sim-config-persona-style {
    font-family: var(--font-sans); font-size: 11px; color: var(--text-faint);
}
.sim-report .sim-config-persona-bg { font-size: 12.5px; color: var(--text-muted); }
/* Scenarios panel (name + goal + criteria chips) */
.sim-report .sim-config-scenario-row {
    margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border-subtle);
}
.sim-report .sim-config-scenario-row:first-child { margin-top: 0; padding-top: 0; border-top: none; }
.sim-report .sim-config-scenario-name { font-size: 13px; font-weight: 600; color: var(--text-strong); }
.sim-report .sim-config-scenario-goal { font-size: 12.5px; color: var(--text-muted); margin-top: 3px; }
.sim-report .sim-config-criteria {
    display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;
}
.sim-report .sim-config-criterion {
    font-family: var(--font-sans); font-size: 11.5px; background: var(--surface-sunken);
    border-radius: 6px; padding: 3px 9px;
}
"""

# ==== .sim-report — Transcripts redesign (Tasks 11+12) ===================
# Appended as its own block (rather than edited into _SIM_REPORT_CSS above)
# so a concurrently-landing `.sim-report` CSS addition on another branch
# merges cleanly. Covers: collapsed tinted conversation cards (Task 11) +
# judge callout / chat bubbles / two-state criteria (Task 12), both scoped
# under `.sim-report` per docs/superpowers/specs/2026-07-10-agent-sim-report-
# alignment-design.md §Transcripts.
_SIM_TRANSCRIPT_CSS = """
/* ---- Conversation cards (spec §Transcripts, Task 11) ---- */
.sim-report .sim-row-list { display: flex; flex-direction: column; gap: 10px; }
.sim-report .sim-conv-card {
    border: 1px solid var(--border-subtle); border-radius: var(--radius-lg); overflow: hidden;
    background: var(--surface-card);
}
.sim-report .sim-conv-summary {
    display: flex; align-items: center; gap: 8px; padding: 12px 16px;
    cursor: pointer; list-style: none;
}
.sim-report .sim-conv-summary::-webkit-details-marker { display: none; }
.sim-report .sim-conv-idx {
    font-family: var(--font-sans); font-size: 12px; color: var(--text-faint);
}
.sim-report .sim-conv-persona { font-size: 14px; font-weight: 600; color: var(--text-strong); }
.sim-report .sim-conv-sep { color: var(--text-faint); }
.sim-report .sim-conv-scenario { font-size: 13px; color: var(--text-body); }
.sim-report .sim-conv-right {
    display: flex; align-items: center; gap: 6px; margin-left: auto;
}
.sim-report .sim-tint-achieved { background: var(--green-100); }
.sim-report .sim-tint-missed { background: var(--red-100); }
.sim-report .sim-tint-error { background: var(--amber-100); }
.sim-report .sim-conv-body { padding: 16px; border-top: 1px solid var(--border-subtle); }

/* ---- Transcript fragment: judge callout / bubbles / criteria (Task 12) ---- */
.sim-report .sim-judge {
    background: var(--surface-sunken); border-left: 3px solid var(--teal-600);
    border-radius: var(--radius-md); padding: 10px 14px; margin-bottom: 14px;
}
.sim-report .sim-judge-label {
    font-family: var(--font-sans); font-size: 11px; font-weight: 600;
    text-transform: uppercase; color: var(--teal-600); display: block;
}
.sim-report .sim-judge-reason { font-size: 13px; line-height: 1.55; margin: 4px 0 0; color: var(--text-body); }
.sim-report .sim-transcript-error {
    background: var(--red-100); color: var(--red-600); border-radius: var(--radius-md);
    padding: 10px 14px; margin-bottom: 14px; font-size: 13px;
}
.sim-report .sim-transcript-grid {
    display: grid; grid-template-columns: 1.6fr 1fr; gap: 20px;
}
@media (max-width: 760px) {
    .sim-report .sim-transcript-grid { grid-template-columns: 1fr; }
}

/* Chat bubbles (render_message_list avatar + side extension) */
.sim-report .sim-msg { display: flex; gap: 10px; margin-bottom: 10px; max-width: 88%; }
.sim-report .sim-msg-user, .sim-report .sim-msg-system { margin-right: auto; }
.sim-report .sim-msg-assistant, .sim-report .sim-msg-tool { margin-left: auto; flex-direction: row-reverse; }
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
    border-radius: 3px var(--radius-lg) var(--radius-lg) var(--radius-lg); padding: 9px 13px;
}
.sim-report .sim-msg-assistant .sim-msg-bubble, .sim-report .sim-msg-tool .sim-msg-bubble {
    background: var(--teal-50); border-radius: var(--radius-lg) 3px var(--radius-lg) var(--radius-lg);
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

/* Criteria column (two-state per deviation #4) */
.sim-report .sim-criteria-header {
    font-family: var(--font-sans); font-size: 11px; font-weight: 600;
    text-transform: uppercase; color: var(--text-faint); margin-bottom: 8px;
}
.sim-report .sim-criteria-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
/* Scoped to the criteria list so it no longer collides with the Scenarios-panel
   .sim-criterion rule above. Mockup uses a plain gapped column (no dividers). */
.sim-report .sim-criteria-list .sim-criterion {
    display: flex; align-items: flex-start; gap: 8px;
}
.sim-report .sim-criterion-icon {
    flex-shrink: 0; width: 18px; height: 18px; border-radius: 999px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; color: #fff;
}
.sim-report .sim-criterion-pass .sim-criterion-icon { background: var(--green-600); }
.sim-report .sim-criterion-fail .sim-criterion-icon { background: var(--red-600); }
.sim-report .sim-criterion-desc { font-size: 13px; color: var(--text-body); flex: 1; }
.sim-report .sim-ctype {
    font-family: var(--font-sans); font-size: 10px; text-transform: uppercase;
    color: var(--text-faint); white-space: nowrap;
}
.sim-report .sim-ctype-unsafe { color: var(--red-600); }
"""

DASHBOARD_CSS = (
    _DASHBOARD_CSS_HEAD + _TAB_RULES + _SIM_TAB_ACCENT + _DASHBOARD_CSS_TAIL + _SIM_REPORT_CSS + _SIM_TRANSCRIPT_CSS
)
