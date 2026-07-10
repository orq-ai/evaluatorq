/**
 * dashboard.js — ORQ evaluatorq dashboard runtime helpers.
 *
 * Vega re-embed after HTMX swap
 * ─────────────────────────────
 * htmx does NOT execute <script> tags inside swapped content, so the
 * per-chart IIFE emitted by render_embed() only runs on initial page load.
 * On a filter swap the chart <div> and its <script type="application/json">
 * data island are replaced inside #filter-swap, but vegaEmbed is never
 * called for the new fragment.
 *
 * This handler listens for htmx:afterSwap, scopes its scan to the swapped
 * fragment only (evt.detail.target), and for every [data-vega-for] island it
 * finds:
 *   1. Finalises the prior embed result (window.__orqVegaViews[id].finalize())
 *      to tear down vega-embed's injected DOM nodes and event listeners.
 *      NOTE: finalize() must be called on the embed RESULT (r), not r.view —
 *      r.view.finalize() alone leaks vega-embed's injected wrappers.
 *   2. Re-embeds the chart into the replacement <div> using the updated spec
 *      from the JSON island.
 *   3. Stores the new embed result back into window.__orqVegaViews[id].
 *
 * Unchanged charts outside the swapped fragment are left untouched.
 */

(function () {
  window.__orqVegaViews = window.__orqVegaViews || {};

  // htmx 2.x executes inline <script> in swapped content by default, which
  // would run render_embed()'s per-chart IIFE AND this afterSwap handler on the
  // same node -> double-embed + detached-view leak. Disable inline-script
  // execution so this handler is the single embed path for swapped fragments.
  // (Charts in the initial full-page load still embed via their IIFE, executed
  // normally by the browser, not by htmx.)
  document.addEventListener('htmx:config', function () {
    if (window.htmx) window.htmx.config.allowScriptTags = false;
  });
  if (window.htmx) window.htmx.config.allowScriptTags = false;

  document.body.addEventListener('htmx:afterSwap', function (evt) {
    var scope = evt.detail.target;
    if (!scope || !window.vegaEmbed) return;

    scope.querySelectorAll('[data-vega-for]').forEach(function (tag) {
      var id = tag.getAttribute('data-vega-for');
      if (!id) return;

      var el = scope.querySelector('#' + CSS.escape(id));
      if (!el) return;

      // Tear down the prior embed result (embed-level, not just view-level).
      var prior = window.__orqVegaViews[id];
      if (prior && prior.finalize) {
        prior.finalize();
      }
      delete window.__orqVegaViews[id];

      var spec;
      try {
        spec = JSON.parse(tag.textContent);
      } catch (e) {
        return;
      }

      window.vegaEmbed(el, spec, { actions: false }).then(function (r) {
        window.__orqVegaViews[id] = r;
      });
    });
  });

  // ⌘K / Ctrl+K focuses the global report search; Escape clears + blurs it.
  document.addEventListener('keydown', function (evt) {
    if ((evt.metaKey || evt.ctrlKey) && (evt.key === 'k' || evt.key === 'K')) {
      var input = document.querySelector('.search-input');
      if (input) {
        evt.preventDefault();
        input.focus();
        input.select();
      }
    } else if (evt.key === 'Escape') {
      var active = document.querySelector('.search-input');
      var results = document.getElementById('search-results');
      if (results) results.innerHTML = '';
      if (active && document.activeElement === active) active.blur();
    }
  });

  // Resize-on-tab-show: CSS-only report tabs render their Vega charts while the
  // panel is display:none (zero width), so charts come up tiny. When a tab is
  // selected, resize the now-visible panel's tracked views to fit (RES-1021).
  document.body.addEventListener('change', function (evt) {
    var t = evt.target;
    if (!t || !t.classList || !t.classList.contains('tab-radio')) return;
    var tabs = t.closest('.tabs');
    if (!tabs || !window.__orqVegaViews) return;
    requestAnimationFrame(function () {
      tabs.querySelectorAll('.tab-panel').forEach(function (panel) {
        if (panel.offsetParent === null) return; // still hidden
        panel.querySelectorAll('[data-vega-for]').forEach(function (tag) {
          var id = tag.getAttribute('data-vega-for');
          var r = id ? window.__orqVegaViews[id] : null;
          if (r && r.view && r.view.resize) {
            try {
              r.view.resize().run();
            } catch (e) {
              /* view finalized/detached — ignore */
            }
          }
        });
      });
    });
  });
})();
