"""Dashboard chrome CSS.

``load_css()`` (``common.reports``) styles only the *report body* sections and
defines the ``:root`` brand tokens.  It carries no rules for the dashboard
shell — header, nav, the index card grid, the filter/body split, panels.  This
module supplies exactly those, reusing the brand tokens already in the page.

Inlined as a second ``<style>`` block by ``shell.page()`` after the report CSS,
so its rules win on equal specificity and all ``var(--…)`` references resolve.
"""

from __future__ import annotations

# ponytail: one stylesheet for all chrome; reuses :root vars from load_css().
DASHBOARD_CSS = """
/* ---- shell ---------------------------------------------------------- */
.dashboard-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 32px;
    background: var(--surface);
    border-bottom: var(--border);
    position: sticky;
    top: 0;
    z-index: 20;
}
.dashboard-header .brand-link {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    text-decoration: none;
    color: var(--slate);
}
.dashboard-header .nav-logo svg,
.dashboard-header .nav-logo img { height: 30px; width: auto; display: block; }
.dashboard-header .brand-name {
    font-size: 13px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--gray-500);
}

.dashboard-layout { max-width: 1100px; margin: 0 auto; padding: 0 32px; }

.dashboard-layout > nav { border-bottom: var(--border); margin: 0 0 28px; }
.dashboard-layout > nav ul {
    display: flex;
    gap: 4px;
    list-style: none;
    margin: 0;
    padding: 14px 0 0;
}
.dashboard-layout > nav a {
    display: inline-block;
    padding: 10px 18px;
    text-decoration: none;
    color: var(--gray-700);
    font-weight: 600;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}
.dashboard-layout > nav a:hover { color: var(--slate); }
.dashboard-layout > nav a.active {
    color: var(--slate);
    border-bottom-color: var(--orq-teal);
}

.dashboard-main { padding-bottom: 80px; }
.dashboard-main > section > h1 { margin-top: 0; }

/* ---- index card grid ----------------------------------------------- */
.report-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px;
    margin-top: 20px;
}
.report-card-item {
    position: relative;
    display: flex;
    flex-direction: column;
    background: var(--surface);
    border: var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: box-shadow 0.12s ease, transform 0.12s ease;
}
.report-card-item:hover {
    box-shadow: 0 6px 20px rgba(20, 20, 19, 0.08);
    transform: translateY(-2px);
}
.report-card-link {
    display: block;
    padding: 16px 18px 12px;
    text-decoration: none;
    color: var(--slate);
    flex: 1;
}
.report-card-surface {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--rust);
    background: var(--clay-tint);
    padding: 2px 9px;
    border-radius: 999px;
    margin-bottom: 10px;
}
.report-card-name { font-size: 16px; font-weight: 600; line-height: 1.35; }
.report-card-meta { margin-top: 6px; font-size: 13px; color: var(--gray-500); }
.report-card-export {
    display: block;
    padding: 9px 18px;
    border-top: var(--border);
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
    color: var(--gray-700);
    background: var(--gray-100);
}
.report-card-export:hover { background: var(--oat); color: var(--slate); }
.card-error {
    margin-left: 8px;
    font-size: 11px;
    font-weight: 700;
    color: #fff;
    background: var(--c-fail);
    padding: 1px 7px;
    border-radius: 999px;
}
.empty-state { color: var(--gray-500); }

/* ---- report view: filter sidebar + body ---------------------------- */
.filter-swap-container {
    display: flex;
    align-items: flex-start;
    gap: 28px;
}
.filter-form { flex: 0 0 230px; position: sticky; top: 96px; }
.report-body-area { flex: 1 1 auto; min-width: 0; }

.filter-sidebar,
.download-sidebar,
.rt-panel {
    background: var(--surface);
    border: var(--border);
    border-radius: var(--radius-panel);
    padding: 18px;
}
.filter-title, .download-title { margin: 0 0 14px; font-size: 15px; }
.filter-group { margin-bottom: 16px; }
.filter-label {
    display: block;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--gray-500);
    margin-bottom: 6px;
}
.filter-checkbox, .filter-radio {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 13px;
    padding: 3px 0;
    cursor: pointer;
}

.download-sidebar { margin-top: 20px; }
.download-link {
    display: inline-block;
    margin-right: 8px;
    margin-top: 4px;
    padding: 6px 14px;
    border: var(--border);
    border-radius: var(--radius-row);
    text-decoration: none;
    font-size: 13px;
    font-weight: 600;
    color: var(--gray-700);
    background: var(--surface);
}
.download-link:hover { background: var(--gray-100); color: var(--slate); }

/* ---- interactive panels -------------------------------------------- */
.rt-interactive-panels, .sim-interactive-panels { margin-top: 32px; }
.rt-panel { margin-bottom: 22px; }
.rt-panel-title { margin: 0 0 14px; font-size: 17px; }
.rt-panel-loading { color: var(--gray-500); font-style: italic; }

@media (max-width: 760px) {
    .filter-swap-container { flex-direction: column; }
    .filter-form { position: static; flex-basis: auto; width: 100%; }
}
"""
