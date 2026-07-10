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
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
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
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-faint);
}
.sim-report .es-confidence {
    font-family: var(--font-mono); font-size: 10px; font-weight: 600;
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
    font-family: var(--font-mono); font-size: 28px; font-weight: 600;
    color: var(--text-strong); line-height: 1.1;
}
.sim-report .kpi-label { font-size: 12px; color: var(--text-muted); margin-top: 7px; }

/* ---- 2-col grids (donut+tokens, personas+scenarios) ---- */
.sim-report .sim-overview-grid-2 {
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;
}
@media (max-width: 760px) {
    .sim-report .sim-overview-grid-2 { grid-template-columns: 1fr; }
}

/* ---- Panel wrapper (report_kit.panel) ---- */
.sim-report .rk-panel {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px 20px;
}
.sim-report .rk-panel-title {
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-faint);
}
.sim-report .rk-panel-sub { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
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
.sim-report .rk-panel + .rk-panel,
.sim-report .rk-panel + .report-card,
.sim-report .report-card + .rk-panel {
    margin-top: 20px;
}

/* HTML-table heatmap (report_kit.heatmap) */
.sim-report .rk-heatmap { border-collapse: separate; border-spacing: 4px; }
.sim-report .rk-heat-col {
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    color: var(--text-muted); text-align: center; padding: 0 4px 6px;
}
.sim-report .rk-heat-row {
    font-size: 12px; font-weight: 600; color: var(--text-strong);
    text-align: right; padding-right: 10px; white-space: nowrap;
}
.sim-report .rk-heat-cell {
    min-width: 50px; height: 34px; border-radius: 5px;
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    text-align: center; vertical-align: middle;
}
.sim-report .rk-heat-empty { background: var(--surface-sunken); color: var(--text-faint); }

/* Per-persona / per-scenario tables (html_table output) + failures table */
.sim-report .rk-panel-body { display: grid; }
.sim-report table {
    width: 100%; border-collapse: collapse;
}
.sim-report table thead th {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase;
    font-weight: 600; color: var(--text-faint); background: var(--surface-sunken);
    padding: 11px 16px; text-align: left;
}
.sim-report table tbody td {
    font-size: 13px; padding: 12px 16px; border-bottom: 1px solid var(--border-subtle);
}
.sim-report table tbody tr:last-child td { border-bottom: none; }
.sim-report table thead th:not(:first-child),
.sim-report table tbody td:not(:first-child) {
    text-align: right; font-variant-numeric: tabular-nums;
}
.sim-report .sim-breakdown-grid-2 {
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
}
@media (max-width: 760px) {
    .sim-report .sim-breakdown-grid-2 { grid-template-columns: 1fr; }
}

/* ---- Turn quality tab (spec §Turn) ---- */
/* Line chart legend (report_kit.line_chart) */
.sim-report .rk-legend {
    display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px;
}
.sim-report .rk-legend-item {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; color: var(--text-muted);
}
.sim-report .rk-legend-swatch {
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
/* Run-configuration meta grid (report_kit.meta_grid) */
.sim-report .rk-meta-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 16px 24px;
}
.sim-report .rk-meta-key {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--text-faint);
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
    font-family: var(--font-mono); font-size: 11px; color: var(--text-faint);
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
    font-family: var(--font-mono); font-size: 11.5px; background: var(--surface-sunken);
    border-radius: 6px; padding: 3px 9px;
}
"""

DASHBOARD_CSS = _DASHBOARD_CSS_HEAD + _TAB_RULES + _SIM_TAB_ACCENT + _DASHBOARD_CSS_TAIL + _SIM_REPORT_CSS
