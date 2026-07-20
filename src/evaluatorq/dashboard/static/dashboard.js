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

  // Agent-simulation entity details: persona/scenario templates and lazy
  // conversation transcripts share one dialog. j/k steps through the entity
  // list that opened the drawer; Escape is handled natively by <dialog>.
  (function () {
    var activeState = null;
    var drillStack = [];

    function currentDialog() {
      var dialog = document.querySelector('.sim-entity-dialog');
      return dialog && dialog.showModal ? dialog : null;
    }

    function contentNode() {
      var dialog = currentDialog();
      return dialog ? dialog.querySelector('[data-sim-entity-content]') : null;
    }

    function dialogIsOpen() {
      var dialog = currentDialog();
      return !!(dialog && dialog.open);
    }

    function openDialog() {
      var dialog = currentDialog();
      if (!dialog) return;
      if (!dialog.open) dialog.showModal();
    }

    function formValues() {
      var form = document.getElementById('filter-form');
      return form ? new FormData(form) : {};
    }

    function isEditable(element) {
      if (!element) return false;
      var tag = element.tagName;
      return element.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
    }

    function matchingTriggers() {
      if (!activeState || !activeState.origin) return [];
      return Array.prototype.slice.call(
        activeState.origin.querySelectorAll(
          '[data-sim-entity-trigger][data-entity-kind="' + activeState.kind + '"]'
        )
      );
    }

    function updateActions() {
      var dialog = currentDialog();
      if (!dialog) return;
      var back = dialog.querySelector('[data-sim-entity-back]');
      var prev = dialog.querySelector('[data-sim-entity-prev]');
      var next = dialog.querySelector('[data-sim-entity-next]');
      var canStep = matchingTriggers().length > 1;
      if (back) back.hidden = drillStack.length === 0;
      if (prev) prev.disabled = !canStep;
      if (next) next.disabled = !canStep;
    }

    function loadConversation() {
      var content = contentNode();
      if (!activeState || !activeState.url || !content || !window.htmx) return;
      content.innerHTML = '<p class="sim-drawer-loading">Loading conversation…</p>';
      window.htmx.ajax('GET', activeState.url, {
        target: content,
        swap: 'innerHTML',
        values: formValues()
      });
    }

    function openConversation(trigger, pushCurrent) {
      var url = trigger.getAttribute('data-drawer-url');
      var content = contentNode();
      if (!url || !content || !window.htmx) return;
      if (pushCurrent && activeState) drillStack.push(activeState);
      activeState = { kind: 'conversation', url: url, origin: trigger.parentElement };
      openDialog();
      updateActions();
      loadConversation();
    }

    function openTemplate(kind, id, pushCurrent, origin) {
      var template = document.querySelector(
        '[data-sim-entity-template][data-entity-kind="' + kind + '"][data-entity-id="' + id + '"]'
      );
      var content = contentNode();
      if (!template || !content) return;
      if (pushCurrent && activeState) drillStack.push(activeState);
      activeState = { kind: kind, id: id, origin: origin || template.parentElement };
      content.innerHTML = template.innerHTML;
      openDialog();
      updateActions();
    }

    function step(delta) {
      if (!activeState) return;
      var triggers = matchingTriggers();
      if (!triggers.length) return;
      var current = triggers.findIndex(function (trigger) {
        return activeState.kind === 'conversation'
          ? trigger.getAttribute('data-drawer-url') === activeState.url
          : trigger.getAttribute('data-entity-id') === activeState.id;
      });
      if (current < 0) return;
      var next = (current + delta + triggers.length) % triggers.length;
      var trigger = triggers[next];
      if (activeState.kind === 'conversation') {
        openConversation(trigger, false);
      } else {
        openTemplate(activeState.kind, trigger.getAttribute('data-entity-id'), false, activeState.origin);
      }
    }

    function restoreState(state) {
      if (!state) return;
      activeState = state;
      if (state.kind === 'conversation') {
        openDialog();
        updateActions();
        loadConversation();
        return;
      }
      var template = document.querySelector(
        '[data-sim-entity-template][data-entity-kind="' + state.kind + '"][data-entity-id="' + state.id + '"]'
      );
      var content = contentNode();
      if (!template || !content) return;
      content.innerHTML = template.innerHTML;
      openDialog();
      updateActions();
    }

    function activateTrigger(trigger) {
      var kind = trigger.getAttribute('data-entity-kind');
      var fromDrawer = !!trigger.closest('.sim-entity-dialog');
      if (kind === 'conversation') {
        if (!fromDrawer) drillStack = [];
        openConversation(trigger, fromDrawer);
        return;
      }
      drillStack = [];
      openTemplate(kind, trigger.getAttribute('data-entity-id'), false, trigger.parentElement);
    }

    document.body.addEventListener('click', function (evt) {
      if (evt.target.closest('[data-no-drawer]')) return;
      var trigger = evt.target.closest('[data-sim-entity-trigger]');
      if (!trigger) return;
      evt.preventDefault();
      activateTrigger(trigger);
    });

    document.body.addEventListener('click', function (evt) {
      var dialog = currentDialog();
      if (!dialog || !dialog.open) return;
      if (evt.target === dialog || evt.target.closest('[data-sim-entity-close]')) {
        dialog.close();
      } else if (evt.target.closest('[data-sim-entity-back]')) {
        restoreState(drillStack.pop());
      } else if (evt.target.closest('[data-sim-entity-prev]')) {
        step(-1);
      } else if (evt.target.closest('[data-sim-entity-next]')) {
        step(1);
      }
    });

    document.body.addEventListener('keydown', function (evt) {
      if (evt.target.closest('[data-no-drawer]')) return;
      var trigger = evt.target.closest('[data-sim-entity-trigger]');
      if (trigger && (evt.key === 'Enter' || evt.key === ' ')) {
        evt.preventDefault();
        activateTrigger(trigger);
        return;
      }
      if (!dialogIsOpen() || isEditable(document.activeElement)) return;
      if (evt.key === 'j' || evt.key === 'J') {
        evt.preventDefault();
        step(1);
      } else if (evt.key === 'k' || evt.key === 'K') {
        evt.preventDefault();
        step(-1);
      }
    });
  })();
})();

// Persist open <details> filter dropdowns across HTMX filter swaps.
// The filter POST outer-swaps #filter-swap, which would snap dropdowns shut.
// Record which are open before the swap, re-open the same ids after.
(function () {
  var openIds = [];
  document.body.addEventListener('htmx:beforeSwap', function () {
    var form = document.getElementById('filter-form');
    if (!form) return;
    openIds = Array.prototype.slice
      .call(form.querySelectorAll('details[id^="filter-dd"][open]'))
      .map(function (d) { return d.id; });
  });
  document.body.addEventListener('htmx:afterSwap', function () {
    openIds.forEach(function (id) {
      var d = document.getElementById(id);
      if (d) d.open = true;
    });
    openIds = [];
  });
})();

// Filter dropdowns behave as an accordion: opening one closes the others.
// `toggle` does not bubble, so listen in the capture phase. The "More filters"
// expander (.filter-dd-more) is excluded — it wraps the nested dropdowns and
// must stay open while one of its children is used.
(function () {
  document.addEventListener('toggle', function (evt) {
    var d = evt.target;
    if (!d.open || !d.matches || !d.matches('details.filter-dd')) return;
    if (d.classList.contains('filter-dd-more')) return;
    var scope = d.closest('#filter-form') || document;
    scope.querySelectorAll('details.filter-dd[open]').forEach(function (o) {
      if (o === d || o.classList.contains('filter-dd-more') || o.contains(d)) return;
      o.open = false;
    });
  }, true);
})();

// Top failure modes panel — min-count slider filters bars client-side.
(function () {
  document.body.addEventListener('input', function (evt) {
    var slider = evt.target.closest('[data-fm-slider]');
    if (!slider) return;
    var panel = slider.closest('[data-fm-panel]');
    if (!panel) return;
    var threshold = parseInt(slider.value, 10);
    var out = panel.querySelector('[data-fm-out]');
    if (out) out.textContent = threshold;
    var visible = 0;
    panel.querySelectorAll('.sim-fm-row').forEach(function (row) {
      var show = parseInt(row.getAttribute('data-count'), 10) >= threshold;
      row.hidden = !show;
      if (show) visible++;
    });
    var empty = panel.querySelector('[data-fm-empty]');
    if (empty) empty.hidden = visible > 0;
  });
})();

// Tab history: CSS-radio report tabs don't change the URL, so browser Back
// would jump past every tab switch to the last full page load (the homepage).
// Push a hash per user tab click and restore the matching radio on Back/Forward.
(function () {
  var restoring = false; // true while we set radios programmatically (no push)
  function selectRadio(id) {
    var radio = id && document.getElementById(id);
    if (radio && radio.classList.contains('tab-radio') && !radio.checked) {
      restoring = true;
      radio.checked = true;
      // Setting .checked in JS skips the 'change' event the Vega-resize handler
      // listens for, so dispatch one so restored charts still size correctly.
      radio.dispatchEvent(new Event('change', { bubbles: true }));
      restoring = false;
    }
  }
  // User clicks a tab -> push its id as a hash (a real history entry).
  document.body.addEventListener('change', function (evt) {
    var t = evt.target;
    if (restoring || !t || !t.classList || !t.classList.contains('tab-radio') || !t.id) return;
    if (('#' + t.id) === location.hash) return;
    history.pushState(null, '', '#' + t.id);
  });
  // Back/Forward -> restore the hashed tab, or reset each group to its first tab.
  window.addEventListener('popstate', function () {
    var id = location.hash.slice(1);
    if (id) {
      selectRadio(id);
    } else {
      document.querySelectorAll('.tabs .tab-radio:first-of-type').forEach(function (r) {
        selectRadio(r.id);
      });
    }
  });
  // Honor a tab hash on initial load / refresh.
  if (location.hash) selectRadio(location.hash.slice(1));
})();
