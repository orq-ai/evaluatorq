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
_TAB_RULES = "".join(
    f".tabs > .tab-radio:nth-of-type({i}):checked ~ .tab-bar > .tab-label:nth-child({i}) "
    "{ color: var(--text-strong); border-bottom-color: var(--teal-600); }\n"
    f".tabs > .tab-radio:nth-of-type({i}):checked ~ .tab-panels > .tab-panel:nth-child({i}) "
    "{ display: block; }\n"
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

DASHBOARD_CSS = _DASHBOARD_CSS_HEAD + _TAB_RULES + _DASHBOARD_CSS_TAIL
