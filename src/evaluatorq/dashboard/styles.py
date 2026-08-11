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

/* ==== settings — read-only config ================================= */
.config-list { display: flex; flex-direction: column; }
.config-row { display: flex; align-items: baseline; gap: 24px; padding: 8px 0; border-bottom: 1px solid var(--border, #eee); }
.config-key { flex: 0 0 160px; font-family: var(--font-sans); font-size: 13px; color: var(--text-muted); }
/* Value column takes the remaining width; list items stack one per line. */
.config-val { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; font-family: var(--font-mono); font-size: 12px; color: var(--text-strong); }
.config-val-item { word-break: break-all; }
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
    display: flex; align-items: center; gap: 8px;
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
.rt-hero-agent-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.rt-hero-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--surface-card); border: 1px solid var(--border-subtle);
    border-radius: 999px; padding: 3px 10px; font-size: 12px;
}
.rt-hero-dot { display: inline-block; width: 7px; height: 7px; border-radius: 999px; }
.rt-hero-dot--critical { background: var(--red-600); }
.rt-hero-dot--vuln { background: var(--orange-500); }
.rt-hero-dot--clean { background: var(--green-600); }
.rt-hero-pill-name { color: var(--text-strong); }
.rt-hero-pill-sub {
    font-family: var(--font-mono); font-size: 10.5px; color: var(--text-faint);
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
.rt-report .rt-focus-intro { font-size: 13px; color: var(--text-muted); margin: 0 0 16px; max-width: 720px; }
.rt-report .rt-focus-intro code { font-family: var(--font-mono); }
.rt-report .rt-focus-card { display: flex; flex-direction: column; margin-bottom: 22px; padding: 0; overflow: hidden; }
.rt-report .rt-focus-head {
    display: flex; align-items: center; justify-content: space-between; gap: 20px;
    padding: 14px 18px; border-bottom: 1px solid var(--border);
}
.rt-report .rt-focus-head-main { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.rt-report .rt-focus-head-stats {
    display: flex; align-items: center; flex: 0 0 auto;
    background: var(--surface-sunken); border: 1px solid var(--border); border-radius: 9px;
    padding: 6px 0;
}
.rt-report .rt-focus-stat {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 2px; padding: 0 16px; min-width: 64px;
}
.rt-report .rt-focus-stat + .rt-focus-stat { border-left: 1px solid var(--border); }
.rt-report .rt-focus-stat--dial { padding: 0 10px; }
.rt-report .rt-focus-body { display: flex; flex-direction: column; gap: 12px; padding: 14px 18px 16px; }
.rt-report .rt-focus-tier-row { display: flex; align-items: center; gap: 8px; }
.rt-report .rt-focus-tier-dot { width: 8px; height: 8px; border-radius: 999px; display: inline-block; }
.rt-report .rt-focus-tier-label {
    font-family: var(--font-mono); font-size: 10.5px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em;
}
.rt-report .rt-focus-category-name { font-family: var(--font-display); font-size: 17px; font-weight: 600; }
.rt-report .rt-focus-category-code { font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); font-weight: 400; margin-left: 4px; }
.rt-report .rt-focus-patterns {
    display: flex; align-items: center; gap: 8px; font-size: 12px; background: var(--surface-sunken);
    border-radius: 6px; padding: 4px 10px; width: fit-content; margin: 0;
}
.rt-report .rt-focus-pattern-dot { width: 5px; height: 5px; border-radius: 999px; display: inline-block; flex: 0 0 auto; }
.rt-report .rt-focus-fixbox { background: var(--surface-sunken); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; margin: 0; }
.rt-report .rt-focus-fixbox-label {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase; color: var(--accent-hover);
    letter-spacing: 0.05em;
}
.rt-report .rt-focus-fixbox-body { font-size: 13px; line-height: 1.55; margin-top: 6px; }
.rt-report .rt-focus-mini-key {
    font-family: var(--font-mono); font-size: 9px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--text-faint);
}
.rt-report .rt-focus-mini-value { font-family: var(--font-mono); font-size: 14px; font-weight: 600; }

/* ---- Apply recommendations: bar, bullets, right drawer (RES-1143) ---- */
.rt-report .rt-focus-recs-section { display: flex; flex-direction: column; gap: 6px; }
.rt-report .rt-focus-recs-label { margin: 0; }
/* One grouped list, hairline dividers between rows — not a stack of boxes. */
.rt-report .rt-focus-recs {
    list-style: none; margin: 6px 0 0; padding: 0;
    border: 1px solid var(--border); border-radius: 8px; background: var(--surface);
    overflow: hidden;
}
.rt-report .rt-focus-rec {
    font-size: 13px; line-height: 1.5; display: flex; align-items: center; gap: 14px;
    padding: 10px 12px;
}
.rt-report .rt-focus-rec + .rt-focus-rec { border-top: 1px solid var(--border); }
.rt-report .rt-focus-rec:hover { background: var(--surface-sunken); }
.rt-report .rt-focus-rec-text { flex: 1 1 auto; min-width: 0; }
/* The action slot is a fixed-width right rail so every button and applied
   pill lines up down the column regardless of text length. */
.rt-report .rt-focus-rec-apply {
    flex: 0 0 auto; margin: 0 0 0 auto; width: 76px; display: flex; justify-content: flex-end;
}
.rt-apply-btn--sm {
    font-size: 11px; font-weight: 600; padding: 4px 12px; border-radius: 6px;
    background: transparent; color: var(--accent); border: 1px solid var(--accent);
}
.rt-apply-btn--sm:hover { background: var(--accent); color: #fff; }
.rt-report .rt-focus-rec--applied .rt-focus-rec-text { color: var(--text-muted); }
.rt-report .rt-focus-rec-applied {
    font-family: var(--font-mono); font-size: 10px; color: var(--green-600, #16a34a);
    background: color-mix(in srgb, #16a34a 12%, transparent);
    border-radius: 999px; padding: 2px 9px; margin-left: auto; white-space: nowrap;
    flex: 0 0 auto;
}
.rt-report .rt-apply-bar {
    display: flex; align-items: center; justify-content: space-between; gap: 20px; flex-wrap: wrap;
    background: var(--surface-sunken); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px; margin: 0 0 16px;
}
/* Cap the text column so the action side stays on the same row on wide
   screens instead of wrapping underneath. */
.rt-report .rt-apply-bar-text { display: flex; flex-direction: column; gap: 3px; min-width: 0; flex: 1 1 320px; max-width: 640px; }
.rt-report .rt-apply-count { font-size: 13px; font-weight: 600; }
.rt-report .rt-apply-count--done { color: var(--green-600, #16a34a); }
.rt-report .rt-apply-hint { font-size: 12px; color: var(--text-muted); }
.rt-report .rt-apply-form { display: flex; align-items: center; gap: 10px; flex: 0 0 auto; margin-left: auto; }
.rt-report .rt-apply-agent {
    font-size: 12px; font-family: var(--font-mono); padding: 6px 8px; border-radius: 7px;
    border: 1px solid var(--border); background: var(--surface); color: var(--text);
}
.rt-report .rt-apply-agent-name {
    font-family: var(--font-mono); font-size: 12px; color: var(--text-muted);
    background: var(--surface); border: 1px solid var(--border); border-radius: 999px;
    padding: 4px 12px; white-space: nowrap;
}
.rt-apply-btn {
    font-size: 13px; font-weight: 600; padding: 7px 14px; border-radius: 8px; cursor: pointer;
    border: 1px solid var(--accent); background: var(--accent); color: #fff;
}
.rt-apply-btn:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
.rt-apply-btn--confirm { background: var(--green-600, #16a34a); border-color: var(--green-600, #16a34a); }
.rt-apply-btn--confirm:hover { filter: brightness(0.92); background: var(--green-600, #16a34a); }
.rt-apply-btn--ghost { background: transparent; color: var(--text); border-color: var(--border); }
.rt-apply-btn--ghost:hover { background: var(--surface-sunken); border-color: var(--border); }
.rt-drawer-overlay {
    position: fixed; inset: 0; background: rgba(15, 15, 15, 0.42); z-index: 90; cursor: pointer;
}
.rt-drawer {
    position: fixed; top: 0; right: 0; bottom: 0; width: min(560px, 92vw); z-index: 91;
    background: var(--surface); border-left: 1px solid var(--border);
    box-shadow: -18px 0 48px rgba(0, 0, 0, 0.18);
    display: flex; flex-direction: column;
    animation: rt-drawer-in 0.18s ease-out;
}
@keyframes rt-drawer-in { from { transform: translateX(24px); opacity: 0; } to { transform: none; opacity: 1; } }
.rt-drawer-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 20px; border-bottom: 1px solid var(--border); flex: 0 0 auto;
}
.rt-drawer-title { margin: 0; font-size: 15px; font-weight: 700; }
.rt-drawer-close {
    border: none; background: transparent; color: var(--text-muted); font-size: 22px; line-height: 1;
    cursor: pointer; padding: 2px 6px; border-radius: 6px;
}
.rt-drawer-close:hover { background: var(--surface-sunken); color: var(--text); }
.rt-drawer-body { padding: 16px 20px; overflow-y: auto; flex: 1 1 auto; }
.rt-drawer-section-label {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--accent-hover); margin: 14px 0 6px;
}
.rt-drawer-section-label:first-child { margin-top: 0; }
.rt-drawer-agent { font-family: var(--font-mono); font-size: 13px; }
.rt-drawer-area {
    display: flex; flex-direction: column; gap: 3px;
    background: var(--surface-sunken); border: 1px solid var(--border);
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
.rt-drawer-success { font-size: 14px; line-height: 1.6; }
.rt-drawer-note { font-size: 12px; color: var(--text-muted); line-height: 1.55; }
.rt-diff {
    font-family: var(--font-mono); font-size: 11.5px; line-height: 1.5; margin: 0;
    background: var(--surface-sunken); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px; overflow-x: auto; display: flex; flex-direction: column;
}
.rt-diff-line { white-space: pre; }
.rt-diff-add { color: var(--green-600, #16a34a); background: color-mix(in srgb, #16a34a 9%, transparent); }
.rt-diff-del { color: var(--sev-high, #dc2626); background: color-mix(in srgb, #dc2626 8%, transparent); }
.rt-diff-hunk { color: var(--accent-hover); }
.rt-diff-file { color: var(--text-faint); }
.rt-drawer-footer {
    display: flex; align-items: center; gap: 10px; padding: 14px 20px;
    border-top: 1px solid var(--border); flex: 0 0 auto;
}
.rt-drawer-footnote { font-size: 11px; color: var(--text-faint); margin-left: auto; text-align: right; }
.rt-drawer-body--loading {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; gap: 12px; padding: 48px 28px;
}
.rt-drawer-loading-title { font-size: 14px; font-weight: 600; margin: 0; }
.rt-drawer-spinner {
    width: 26px; height: 26px; border-radius: 999px;
    border: 3px solid var(--border); border-top-color: var(--accent);
    animation: rt-spin 0.8s linear infinite;
}
@keyframes rt-spin { to { transform: rotate(360deg); } }
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
)
