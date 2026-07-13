// Measure a persona×scenario heatmap for design-parity checks.
// Scoped to ONE component instance and reports container chrome — not just cell
// typography — per docs/wireframe/visual-validation-pitfalls.md (trap #3, #9).
// Paste into the console. Default root = the panel wrapping the first heatmap;
// pass an explicit selector to target a specific instance:
//   measureHeatmap('#some-scope .rk-panel:has(.rk-heatmap)')
const measureHeatmap = (rootSel = '.rk-panel:has(.rk-heatmap)') => {
  const root = document.querySelector(rootSel) || document.querySelector('.rk-heatmap');
  if (!root) return JSON.stringify({ error: 'no root matched', rootSel }, null, 0);
  const table = root.matches('.rk-heatmap') ? root : root.querySelector('.rk-heatmap');
  if (!table) return JSON.stringify({ error: 'no .rk-heatmap under root', rootSel }, null, 0);

  // Every lookup is scoped to `root`/`table` — never document-wide — so all
  // reported metrics come from the same component instance.
  const q = s => (s === '.rk-heatmap' ? table : root.querySelector(s));
  const cs = (el, p) => (el ? getComputedStyle(el)[p] : null);
  const rect = el => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) };
  };
  // One border side as "width style color" (the chrome trap #3 is about).
  const bd = (el, side) =>
    el ? `${cs(el, `border${side}Width`)} ${cs(el, `border${side}Style`)} ${cs(el, `border${side}Color`)}` : null;
  const borders = el =>
    el ? { top: bd(el, 'Top'), right: bd(el, 'Right'), bottom: bd(el, 'Bottom'), left: bd(el, 'Left') } : null;

  const cell = q('.rk-heat-cell:not(.rk-heat-empty)') || q('.rk-heat-cell');
  const col = q('.rk-heat-col');
  const row = q('.rk-heat-row');
  const title = root.querySelector('.rk-panel-title')
    || [...root.querySelectorAll('h2, h3')].find(e => /persona\s*.\s*scenario|Goal completion/i.test(e.textContent));

  // Zebra bleed: compare the first two body-row backgrounds. A stripe the mockup
  // doesn't have shows up as a diff here. (`:hover` can't be read statically.)
  const bodyRows = [...table.querySelectorAll('tbody tr')];

  return JSON.stringify(
    {
      container: {
        border: borders(table),
        background: cs(table, 'backgroundColor'),
        borderRadius: cs(table, 'borderRadius'),
        boxShadow: cs(table, 'boxShadow'),
        padding: cs(table, 'padding'),
        margin: cs(table, 'margin'),
        borderSpacing: cs(table, 'borderSpacing'),
        borderCollapse: cs(table, 'borderCollapse'),
      },
      cell: cell
        ? {
            w: rect(cell).w,
            h: rect(cell).h,
            radius: cs(cell, 'borderRadius'),
            font: cs(cell, 'fontSize'),
            family: (cs(cell, 'fontFamily') || '').slice(0, 12),
            background: cs(cell, 'backgroundColor'),
            border: borders(cell),
          }
        : null,
      colHeader: { font: cs(col, 'fontSize'), family: (cs(col, 'fontFamily') || '').slice(0, 12), underline: bd(col, 'Bottom'), background: cs(col, 'backgroundColor') },
      rowLabel: { font: cs(row, 'fontSize'), divider: bd(row, 'Right'), background: cs(row, 'backgroundColor') },
      zebra: { row0: cs(bodyRows[0], 'backgroundColor'), row1: cs(bodyRows[1], 'backgroundColor') },
      hover: 'manual — trigger :hover on a cell/row and re-check background',
      title: { font: cs(title, 'fontSize'), family: (cs(title, 'fontFamily') || '').slice(0, 12), rect: rect(title) },
      // Sanity (trap #9): confirm these boxes are tight/plausible before trusting numbers.
      matched: { root: { tag: root.tagName, cls: root.className, rect: rect(root) }, table: rect(table) },
    },
    null,
    0,
  );
};
measureHeatmap();
