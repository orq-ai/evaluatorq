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

  // Delegated so the finder handlers survive HTMX fragment swaps.
  document.body.addEventListener('focusin', function (evt) {
    const query = evt.target.closest('.finder-command-textarea[data-finder-placeholders]');
    if (!query) return;
    query.dataset.finderPlaceholderDismissed = 'true';
    query.placeholder = '';
  });
  window.setInterval(function () {
    document.querySelectorAll('.finder-command-textarea[data-finder-placeholders]').forEach(function (query) {
      if (query.dataset.finderPlaceholderDismissed === 'true' || query.value || document.activeElement === query) return;
      try {
        const placeholders = JSON.parse(query.dataset.finderPlaceholders || '[]');
        if (!Array.isArray(placeholders) || placeholders.length < 2) return;
        const index = (Number(query.dataset.finderPlaceholderIndex || 0) + 1) % placeholders.length;
        query.dataset.finderPlaceholderIndex = String(index);
        query.placeholder = placeholders[index];
      } catch (_error) {
        query.dataset.finderPlaceholderDismissed = 'true';
      }
    });
  }, 4000);

  document.body.addEventListener('click', function (evt) {
    const example = evt.target.closest('[data-finder-example]');
    if (example) {
      const query = document.querySelector('#finder-query-form textarea[name="query"]');
      if (query) {
        query.value = example.getAttribute('data-finder-example') || '';
        query.focus();
        query.dispatchEvent(new Event('input', { bubbles: true }));
      }
      return;
    }

    const item = evt.target.closest('.facet-item');
    if (item) { showFacet(item); return; }
    const addFilter = evt.target.closest('.finder-controls .add');
    if (addFilter) {
      const ownMenu = addFilter.parentElement.querySelector('.finder-facets');
      if (ownMenu) { ownMenu.style.left = ''; ownMenu.style.top = ''; ownMenu.classList.toggle('open'); }
      return;
    }
    const filtersButton = evt.target.closest('[data-explorer-filters]');
    if (filtersButton) {
      const menu = document.querySelector('.finder-controls .finder-facets');
      if (menu && menu.classList.contains('open')) closeFacetMenus();
      else if (menu) {
        openFacetMenu(menu, filtersButton);
        if (menu.hasAttribute('data-refresh-on-open') && window.htmx) {
          menu.setAttribute('hx-vals', '{"open":"1"}');
          window.htmx.trigger(menu, 'refreshFacets');
        }
      }
      return;
    }
    const chipOpen = evt.target.closest('[data-chip-open]');
    if (chipOpen) {
      const menu = document.querySelector('.finder-controls .finder-facets');
      const target = menu && menu.querySelector('.facet-item[data-facet="' + chipOpen.getAttribute('data-chip-open') + '"]');
      if (target) {
        // Anchor the menu under the clicked chip instead of under + Filter.
        openFacetMenu(menu, chipOpen.closest('.chip'));
        showFacet(target);
      }
      return;
    }
    if (!evt.target.closest('.finder-controls .addwrap') && !evt.target.closest('.chip-open')) closeFacetMenus();

    const remove = evt.target.closest('[data-finder-remove]');
    if (!remove) return;
    const name = remove.getAttribute('data-finder-remove');
    if (!name) return;
    const value = remove.getAttribute('data-finder-value');
    document.querySelectorAll('input[name="' + name + '"]').forEach(function (input) {
      if (value === null || input.value === value) {
        if (input.type === 'checkbox') input.checked = false;
        else if (input.type === 'hidden') input.remove();
        else input.value = '';
      }
    });
    const fromToolbar = remove.closest('.xr-chips');
    remove.closest('.chip').remove();
    if (fromToolbar) loadExplorer();
  });

  // On /traces the Filters button opens the facet menu itself; picks apply with one reload when it closes.
  let filtersDirty = false;
  let pendingExplorerControl = null;
  function loadExplorer() {
    const form = document.getElementById('explorer-load-form');
    if (form && form.requestSubmit) form.requestSubmit();
  }
  function openFacetMenu(menu, anchor) {
    const box = anchor.getBoundingClientRect();
    const wrap = menu.parentElement.getBoundingClientRect();
    menu.style.left = (box.left - wrap.left) + 'px';
    menu.style.top = (box.bottom - wrap.top + 6) + 'px';
    menu.classList.add('open');
    document.querySelectorAll('[data-explorer-filters]').forEach(function (b) { b.setAttribute('aria-expanded', 'true'); });
  }
  document.body.addEventListener('htmx:afterSwap', function (evt) {
    const target = evt.detail && evt.detail.target;
    if (!target || !target.matches || !target.matches('.finder-facets')) return;
    const menu = document.querySelector('.finder-controls .finder-facets.open');
    const anchor = document.querySelector('[data-explorer-filters]');
    if (menu && anchor) openFacetMenu(menu, anchor);
  });
  function closeFacetMenus() {
    document.querySelectorAll('.finder-facets.open').forEach(function (menu) { menu.classList.remove('open'); });
    document.querySelectorAll('[data-explorer-filters]').forEach(function (b) { b.setAttribute('aria-expanded', 'false'); });
    if (filtersDirty) { filtersDirty = false; loadExplorer(); }
  }
  document.body.addEventListener('change', function (evt) {
    if (evt.target.closest('.finder-facets') && !evt.target.matches('.facet-search') && document.querySelector('[data-explorer-filters]')) filtersDirty = true;
  });
  // A control click applies the pending facet edits first, then repeats the
  // selected rows request so its choice is applied to the newly loaded rows.
  document.addEventListener('click', function (evt) {
    if (!filtersDirty) return;
    const control = evt.target.closest('[hx-get][hx-target="#explorer-results"]');
    if (!control) return;
    const url = control.getAttribute('hx-get');
    if (!url) return;
    evt.preventDefault();
    evt.stopPropagation();
    pendingExplorerControl = url;
    filtersDirty = false;
    document.querySelectorAll('.finder-facets.open').forEach(function (menu) { menu.classList.remove('open'); });
    document.querySelectorAll('[data-explorer-filters]').forEach(function (button) { button.setAttribute('aria-expanded', 'false'); });
    loadExplorer();
  }, true);
  function runPendingExplorerControl() {
    if (!pendingExplorerControl || !window.htmx) return;
    const url = pendingExplorerControl;
    pendingExplorerControl = null;
    window.htmx.ajax('GET', url, { target: '#explorer-results', swap: 'outerHTML' });
  }
  document.body.addEventListener('htmx:afterSettle', function (evt) {
    const target = evt.detail?.target;
    if (!pendingExplorerControl || target?.id !== 'explorer-results') return;
    const loadFailed = !!document.querySelector('#explorer-results .finder-form-error');
    if (loadFailed) {
      pendingExplorerControl = null;
      return;
    }
    runPendingExplorerControl();
  });
  ['htmx:sendError', 'htmx:responseError', 'htmx:timeout'].forEach(function (name) {
    document.body.addEventListener(name, function (evt) {
      if (evt.detail?.elt?.id === 'explorer-load-form') pendingExplorerControl = null;
    });
  });
  // A swap of the whole filter row (Load, Clear, a new run) replaces the menu and its unsaved ticks with it.
  function filterRowReplaced(evt) {
    const id = evt.detail && evt.detail.target && evt.detail.target.id;
    if (id !== 'finder-body' && id !== 'finder-controls') return;
    filtersDirty = false;
    if (!document.querySelector('.finder-facets.open')) {
      document.querySelectorAll('[data-explorer-filters]').forEach(function (b) { b.setAttribute('aria-expanded', 'false'); });
    }
  }
  document.body.addEventListener('htmx:afterSwap', filterRowReplaced);
  document.body.addEventListener('htmx:oobAfterSwap', filterRowReplaced);

  function showFacet(item) {
    const menu = item.closest('.finder-facets');
    if (!menu) return;
    const name = item.getAttribute('data-facet');
    menu.querySelectorAll('.facet-item').forEach(function (other) {
      const on = other === item;
      other.classList.toggle('is-active', on);
      other.setAttribute('aria-expanded', on ? 'true' : 'false');
    });
    menu.querySelectorAll('.facet-sub').forEach(function (sub) {
      const on = sub.getAttribute('data-facet-sub') === name;
      sub.classList.toggle('is-active', on);
      sub.hidden = !on;
    });
  }

  document.body.addEventListener('mouseover', function (evt) {
    const item = evt.target.closest('.facet-item');
    if (item && !item.classList.contains('is-active')) showFacet(item);
  });

  document.body.addEventListener('input', function (evt) {
    const search = evt.target.closest('.finder-facets .facet-search');
    if (!search) return;
    const sub = search.closest('.facet-sub');
    const query = search.value.trim().toLocaleLowerCase();
    let visible = 0;
    sub.querySelectorAll('.facet-values label').forEach(function (option) {
      const label = option.querySelector('input + span');
      const matches = (label ? label.textContent : option.textContent).toLocaleLowerCase().includes(query);
      option.hidden = !matches;
      if (matches) visible += 1;
    });
    sub.querySelector('.facet-no-results').hidden = visible !== 0;
  });

  // Unticking a value in the menu must also drop the hidden input the chip row submits.
  document.body.addEventListener('change', function (evt) {
    const box = evt.target;
    if (!box.matches('.finder-facets input[type="checkbox"]') || box.checked) return;
    document.querySelectorAll('input[type="hidden"][name="' + box.name + '"]').forEach(function (hidden) {
      if (hidden.value === box.value) hidden.remove();
    });
    document.querySelectorAll('.chip[data-chip-name="' + box.name + '"]').forEach(function (chip) {
      if (chip.getAttribute('data-finder-value') === box.value) chip.remove();
    });
  });

  document.addEventListener('keydown', function (evt) {
    if (evt.key === 'Escape') closeFacetMenus();
    if (!(evt.metaKey || evt.ctrlKey) || evt.key !== 'Enter') return;
    const query = evt.target.closest('#finder-query-form textarea[name="query"]');
    if (!query) return;
    evt.preventDefault();
    const form = query.form || document.getElementById('finder-query-form');
    if (form && form.requestSubmit) form.requestSubmit();
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

  // Live filter-slider readout: update the number next to a range slider while
  // it is being dragged (`input`), before HTMX fires the `change` round-trip.
  // Delegated on document so it survives the HTMX form swap.
  document.addEventListener('input', function (evt) {
    var slider = evt.target;
    if (!slider || !slider.classList || !slider.classList.contains('filter-slider')) return;
    var row = slider.closest('.filter-slider-row');
    var readout = row && row.querySelector('.filter-slider-readout');
    if (!readout) return;
    var glyph = slider.getAttribute('data-glyph') || '';
    readout.textContent = (glyph ? glyph + ' ' : '') + slider.value;
    // Engaged = moved off the no-op default bound.
    var def = slider.getAttribute('data-default');
    var engaged = def !== null && parseFloat(slider.value) !== parseFloat(def);
    readout.classList.toggle('is-engaged', engaged);
  });

  // Agent-simulation entity details: persona/scenario templates and lazy
  // conversation transcripts share one dialog. j/k steps through the entity
  // list that opened the drawer; Escape is handled natively by <dialog>.
  (function () {
    var activeState = null;
    // Each drawer view (conversation / persona / scenario) is a real browser
    // history entry, so Back/Forward walk the drill path. `drawerDepth` mirrors
    // how many drawer entries sit above the page entry (0 = closed); it is read
    // back from history.state on popstate so it survives Back/Forward. Pushes are
    // suppressed while applying a popstate so we don't re-enter history.
    var drawerDepth = 0;
    var suppressPush = false;

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
      // Native Escape closes <dialog> without going through dismiss(); unwind the
      // drawer's history entries so Back/Forward stay consistent. Attach lazily
      // (the dialog may be injected after this script runs) and only once.
      if (!dialog.dataset.simCloseBound) {
        dialog.dataset.simCloseBound = '1';
        dialog.addEventListener('close', function () {
          if (drawerDepth > 0 && !suppressPush) history.go(-drawerDepth);
        });
      }
      dialog.classList.remove('sim-entity-dialog--closing');
      if (!dialog.open) dialog.showModal();
    }

    function closeDrawer() {
      var dialog = currentDialog();
      if (!dialog || !dialog.open || dialog.classList.contains('sim-entity-dialog--closing')) return;
      dialog.classList.add('sim-entity-dialog--closing');
      var finished = false;
      var fallback;
      function finishClose() {
        if (finished) return;
        finished = true;
        window.clearTimeout(fallback);
        dialog.classList.remove('sim-entity-dialog--closing');
        dialog.close();
      }
      dialog.addEventListener('animationend', finishClose, { once: true });
      fallback = window.setTimeout(finishClose, 220);
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
      if (back) back.hidden = drawerDepth <= 1;
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

    function triggerSerial(trigger, kind) {
      return kind === 'conversation'
        ? { kind: 'conversation', url: trigger.getAttribute('data-drawer-url') }
        : { kind: kind, id: trigger.getAttribute('data-entity-id') };
    }

    // Re-find a view's originating trigger group on the page so j/k stepping
    // still works after a Back/Forward that dropped the click-time origin.
    function originForSerial(serial) {
      var sel = serial.kind === 'conversation'
        ? '[data-sim-entity-trigger][data-drawer-url="' + serial.url + '"]'
        : '[data-sim-entity-trigger][data-entity-kind="' + serial.kind + '"][data-entity-id="' + serial.id + '"]';
      var t = document.querySelector(sel);
      return t ? t.parentElement : null;
    }

    // Render a drawer view from its serialized form. Does NOT touch history —
    // callers decide whether the move is a push, replace, or popstate restore.
    function applySerial(serial, origin) {
      openDialog();
      var content = contentNode();
      if (!content) return;
      activeState = { kind: serial.kind, origin: origin || originForSerial(serial) };
      if (serial.kind === 'conversation') {
        if (!serial.url || !window.htmx) return;
        activeState.url = serial.url;
        loadConversation();
      } else {
        var template = document.querySelector(
          '[data-sim-entity-template][data-entity-kind="' + serial.kind + '"][data-entity-id="' + serial.id + '"]'
        );
        if (!template) return;
        activeState.id = serial.id;
        content.innerHTML = template.innerHTML;
      }
      updateActions();
    }

    // A clickthrough/drill: a new history entry above the current one.
    function pushDrawer(serial, origin) {
      drawerDepth += 1;
      if (!suppressPush) history.pushState({ simDrawer: serial, drawerDepth: drawerDepth }, '');
      applySerial(serial, origin);
    }

    // A lateral move (j/k within the same list): update the current entry in
    // place so the history stack doesn't grow one item per arrow press.
    function replaceDrawer(serial, origin) {
      if (!suppressPush) history.replaceState({ simDrawer: serial, drawerDepth: drawerDepth }, '');
      applySerial(serial, origin);
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
      replaceDrawer(triggerSerial(trigger, activeState.kind), activeState.origin);
    }

    // Close by unwinding every drawer entry back to the page, so Forward doesn't
    // silently reopen and the URL matches the visible state.
    function dismiss() {
      if (drawerDepth > 0) {
        history.go(-drawerDepth);
      } else {
        closeDrawer();
      }
    }

    function activateTrigger(trigger) {
      var kind = trigger.getAttribute('data-entity-kind');
      pushDrawer(triggerSerial(trigger, kind), trigger.parentElement);
    }

    window.addEventListener('popstate', function (evt) {
      var serial = evt.state && evt.state.simDrawer;
      if (serial) {
        suppressPush = true;
        drawerDepth = evt.state.drawerDepth || 1;
        applySerial(serial, null);
        suppressPush = false;
      } else {
        drawerDepth = 0;
        if (dialogIsOpen()) closeDrawer();
      }
    });

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
        dismiss();
      } else if (evt.target.closest('[data-sim-entity-back]')) {
        history.back();
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

/**
 * Apply-recommendations drawer: instant loading state (RES-1143)
 * ──────────────────────────────────────────────────────────────
 * The preview endpoint runs an LLM merge of the agent's full instructions,
 * which takes tens of seconds. Without this, the click gives no feedback
 * until the response lands. On beforeRequest for any request targeting the
 * drawer mount, inject the drawer shell with a spinner immediately; the
 * HTMX swap then replaces it with the real content (or an error drawer).
 */
(function () {
  var DRAWER = 'rt-apply-drawer';

  // The loading state's styles ride along with the injection instead of the
  // server-rendered page CSS, so a cached page can never show it unstyled.
  function ensureLoadingCss() {
    if (document.getElementById('rt-apply-anim-css')) return;
    var style = document.createElement('style');
    style.id = 'rt-apply-anim-css';
    style.textContent =
      '@keyframes rt-spin { to { transform: rotate(360deg); } }' +
      '@keyframes rt-shimmer { 0% { opacity: 0.45; } 50% { opacity: 1; } 100% { opacity: 0.45; } }' +
      '@keyframes rt-dots { 0%, 20% { content: ""; } 40% { content: "."; } ' +
      '60% { content: ".."; } 80%, 100% { content: "..."; } }' +
      '.rt-drawer-body--loading { display: flex; flex-direction: column; align-items: center; ' +
      'justify-content: center; text-align: center; gap: 14px; padding: 56px 28px 40px; }' +
      // Literal colors: this theme's --border is a full shorthand (not a
      // color), which silently invalidated the earlier var()-based border and
      // background declarations and left the spinner and skeleton invisible.
      '.rt-drawer-spinner { display: block; width: 30px; height: 30px; border-radius: 999px; ' +
      'border: 3px solid #dad8d2; border-top-color: var(--accent, #ff8f34); ' +
      'animation: rt-spin 0.8s linear infinite; }' +
      '.rt-drawer-loading-title { font-size: 14px; font-weight: 600; margin: 0; }' +
      '.rt-drawer-loading-title::after { display: inline-block; width: 1.2em; text-align: left; ' +
      'content: "..."; animation: rt-dots 1.6s steps(1) infinite; }' +
      '.rt-drawer-skeleton { width: 100%; max-width: 380px; display: flex; flex-direction: column; ' +
      'gap: 8px; margin-top: 18px; }' +
      '.rt-drawer-skeleton span { display: block; height: 10px; border-radius: 5px; ' +
      'background: #dad8d2; animation: rt-shimmer 1.4s ease-in-out infinite; }' +
      '.rt-drawer-skeleton span:nth-child(2) { width: 82%; animation-delay: 0.15s; }' +
      '.rt-drawer-skeleton span:nth-child(3) { width: 91%; animation-delay: 0.3s; }' +
      '.rt-drawer-skeleton span:nth-child(4) { width: 68%; animation-delay: 0.45s; }';
    document.head.appendChild(style);
  }

  function loadingDrawer(message) {
    ensureLoadingCss();
    return (
      '<div class="rt-drawer-overlay"></div>' +
      '<aside class="rt-drawer" role="dialog" aria-modal="true" aria-busy="true">' +
      '<div class="rt-drawer-head"><h3 class="rt-drawer-title">Preview changes</h3></div>' +
      '<div class="rt-drawer-body rt-drawer-body--loading">' +
      '<span class="rt-drawer-spinner" aria-hidden="true"></span>' +
      '<p class="rt-drawer-loading-title">' + message + '</p>' +
      '<p class="rt-drawer-note">This rewrites the agent instructions with an LLM and usually ' +
      'takes 10–30 seconds. This is a read-only preview: the agent on the platform is ' +
      'not modified unless you click Apply on the next screen.</p>' +
      '<div class="rt-drawer-skeleton" aria-hidden="true"><span></span><span></span><span></span><span></span></div>' +
      '</div></aside>'
    );
  }

  document.body.addEventListener('htmx:beforeRequest', function (evt) {
    var src = evt.detail.elt;
    if (!src || !src.closest) return;
    var form = src.closest('.rt-apply-form, .rt-focus-rec-apply');
    if (!form) return;
    var mount = document.getElementById(DRAWER);
    if (!mount) return;
    var single = form.classList.contains('rt-focus-rec-apply');
    mount.innerHTML = loadingDrawer(
      single
        ? 'Merging this recommendation into the agent instructions'
        : 'Merging the pending recommendations into the agent instructions'
    );
  });

  // A transport failure would otherwise leave the spinner up forever.
  ['htmx:sendError', 'htmx:responseError', 'htmx:timeout'].forEach(function (name) {
    document.body.addEventListener(name, function (evt) {
      var src = evt.detail.elt;
      if (!src || !src.closest || !src.closest('.rt-apply-form, .rt-focus-rec-apply, .rt-drawer')) return;
      var mount = document.getElementById(DRAWER);
      if (!mount || !mount.querySelector('.rt-drawer-body--loading')) return;
      mount.innerHTML =
        '<div class="rt-drawer-overlay"></div>' +
        '<aside class="rt-drawer" role="dialog" aria-modal="true">' +
        '<div class="rt-drawer-head"><h3 class="rt-drawer-title">Apply recommendations</h3></div>' +
        '<div class="rt-drawer-body"><p class="rt-drawer-error">The request failed before a preview ' +
        'came back. Check the dashboard terminal for details and try again.</p></div></aside>';
    });
  });

  // Clicking the injected overlay (no hx- attributes on the client-side shell)
  // closes the drawer, matching the server-rendered overlay behavior.
  document.body.addEventListener('click', function (evt) {
    if (!evt.target || !evt.target.classList || !evt.target.classList.contains('rt-drawer-overlay')) return;
    var mount = document.getElementById(DRAWER);
    if (mount && mount.querySelector('.rt-drawer-body--loading')) mount.innerHTML = '';
  });
  // Explorer: browser timezone, local From/To, presets.
  function explorerEndpointOffset(name) {
    const dateInput = document.getElementById('explorer-' + name);
    const timeInput = document.getElementById('explorer-' + name + '-time');
    if (!dateInput || !timeInput || !dateInput.value || !timeInput.value) return null;
    const local = new Date(dateInput.value + 'T' + timeInput.value);
    return Number.isNaN(local.getTime()) ? null : String(local.getTimezoneOffset());
  }
  function explorerUpdateOffsets() {
    ['from', 'to'].forEach((name) => {
      const dateInput = document.getElementById('explorer-' + name);
      const timeInput = document.getElementById('explorer-' + name + '-time');
      if (!dateInput || !timeInput) return;
      const fieldName = name + '_tz_offset';
      let field = document.querySelector('[name="' + fieldName + '"][form="explorer-load-form"]');
      if (!field) {
        field = document.createElement('input');
        field.type = 'hidden';
        field.name = fieldName;
        field.setAttribute('form', 'explorer-load-form');
        dateInput.parentNode.appendChild(field);
      }
      const offset = explorerEndpointOffset(name);
      field.value = offset === null ? '' : offset;
    });
  }
  function explorerLocal(dateInput) {
    const utc = dateInput.getAttribute('data-utc');
    const prefix = dateInput.id === 'explorer-from' ? 'explorer-from' : 'explorer-to';
    const timeInput = document.getElementById(prefix + '-time');
    if (!utc || dateInput.dataset.localised || !timeInput || timeInput.dataset.localised) return;
    const d = new Date(utc + 'Z');
    const pad = (n) => String(n).padStart(2, '0');
    dateInput.value = d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
    timeInput.value = pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
    dateInput.dataset.localised = '1';
    timeInput.dataset.localised = '1';
  }
  function explorerInit() {
    document.querySelectorAll('[data-explorer-tz]').forEach((el) => { el.value = String(new Date().getTimezoneOffset()); });
    document.querySelectorAll('#explorer-from, #explorer-to').forEach(explorerLocal);
    const exact = document.querySelector('[data-explorer-range-mode]')?.value === 'exact';
    const seconds = document.querySelector('[data-explorer-range-seconds]')?.value;
    document.querySelectorAll('[data-explorer-preset]').forEach((b) => {
      b.setAttribute('aria-pressed', String(!exact && b.getAttribute('data-explorer-preset') === seconds));
    });
    const custom = document.querySelector('.xr-exact');
    if (custom) {
      // Runs after every swap, polls included: open the exact-range panel, never close one the user opened.
      if (exact) custom.open = true;
      custom.querySelector('summary')?.toggleAttribute('data-active', exact);
    }
    const tz = document.querySelector('[data-explorer-tz-label]');
    if (tz) {
      const off = -new Date().getTimezoneOffset();
      const pad = (n) => String(Math.floor(Math.abs(n))).padStart(2, '0');
      tz.textContent = 'Local time (UTC' + (off >= 0 ? '+' : '−') + pad(off / 60) + ':' + pad(off % 60) + ')';
    }
    explorerUpdateRangeLabel();
    document.querySelectorAll('#explorer-from, #explorer-to, #explorer-from-time, #explorer-to-time').forEach((el) => { el.required = exact; });
    explorerUpdateOffsets();
  }
  let customRangeWasOpen = false;
  document.body.addEventListener('htmx:beforeSwap', function (evt) {
    if (evt.detail?.target?.id !== 'explorer-results') return;
    customRangeWasOpen = !!document.querySelector('.xr-exact')?.open;
  });
  document.body.addEventListener('htmx:afterSwap', function (evt) {
    if (evt.detail?.target?.id !== 'explorer-results' || !customRangeWasOpen) return;
    const custom = document.querySelector('.xr-exact');
    if (custom) custom.open = true;
    customRangeWasOpen = false;
  });
  function explorerUpdateRangeLabel(presetLabel) {
    const label = document.querySelector('[data-explorer-range-label]');
    if (!label) return;
    const mode = document.querySelector('[data-explorer-range-mode]');
    if (mode?.value === 'exact') {
      const read = (id) => new Date((document.getElementById(id)?.value || '') + 'T' + (document.getElementById(id + '-time')?.value || '00:00:00'));
      const from = read('explorer-from'), to = read('explorer-to');
      if (isNaN(from) || isNaN(to)) { label.textContent = 'Custom range'; return; }
      const year = new Date().getFullYear();
      const showYear = from.getFullYear() !== year || to.getFullYear() !== year;
      const day = (d) => d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: showYear ? 'numeric' : undefined });
      const time = (d) => d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false });
      const sameDay = day(from) === day(to);
      // ponytail: ranges of a day or more show dates only; the exact seconds live in the tooltip and the inputs.
      label.textContent = to - from >= 86400000 ? day(from) + ' – ' + day(to)
        : sameDay ? day(from) + ' ' + time(from) + '–' + time(to)
        : day(from) + ' ' + time(from) + ' – ' + day(to) + ' ' + time(to);
      label.title = from.toLocaleString() + ' – ' + to.toLocaleString();
      return;
    }
    if (presetLabel) {
      const labels = { '15m': '15 minutes', '1h': '1 hour', '24h': '24 hours', '7d': '7 days', '30d': '30 days' };
      label.textContent = 'Last ' + (labels[presetLabel] || presetLabel);
      return;
    }
    const seconds = document.querySelector('[data-explorer-range-seconds]')?.value || '604800';
    const match = document.querySelector('[data-explorer-preset="' + seconds + '"]');
    if (match) { explorerUpdateRangeLabel(match.textContent.trim()); return; }
    const days = Number(seconds) / 86400;
    label.textContent = 'Last ' + (Number.isInteger(days) ? days : days.toFixed(1)) + ' days';
  }
  function explorerRefreshRelativeRange() {
    const mode = document.querySelector('[data-explorer-range-mode]');
    if (!mode || mode.value !== 'relative') return;
    const seconds = Number(document.querySelector('[data-explorer-range-seconds]')?.value || 0);
    if (!seconds) return;
    const to = new Date();
    const from = new Date(to.getTime() - seconds * 1000);
    const pad = (n) => String(n).padStart(2, '0');
    [['from', from], ['to', to]].forEach(([name, date]) => {
      const dateInput = document.getElementById('explorer-' + name);
      const timeInput = document.getElementById('explorer-' + name + '-time');
      if (!dateInput || !timeInput) return;
      dateInput.value = date.getFullYear() + '-' + pad(date.getMonth() + 1) + '-' + pad(date.getDate());
      timeInput.value = pad(date.getHours()) + ':' + pad(date.getMinutes()) + ':' + pad(date.getSeconds());
    });
    explorerUpdateOffsets();
  }
  document.addEventListener('submit', function (evt) {
    const form = evt.target;
    if (!form || form.id !== 'explorer-load-form') return;
    explorerRefreshRelativeRange();
  }, true);
  // Apply filters only sends its own hx-post, bypassing the form submit event.
  // Refresh relative dates before htmx gathers the request parameters.
  document.addEventListener('click', function (evt) {
    if (evt.target.closest('[hx-post="/find/load"]')) explorerRefreshRelativeRange();
  }, true);
  document.addEventListener('submit', function (evt) {
    const form = evt.target;
    if (!form || form.id !== 'finder-query-form') return;
    const rows = document.getElementById('explorer-rows');
    const limit = document.getElementById('finder-limit-query');
    if (rows && limit) limit.value = rows.value;
    const dateTime = (name) => {
      const date = document.getElementById('explorer-' + name)?.value;
      const time = document.getElementById('explorer-' + name + '-time')?.value;
      return date && time ? date + 'T' + time : '';
    };
    const start = dateTime('from'), end = dateTime('to');
    const endTime = document.getElementById('explorer-to-time');
    if (!start || !end || start >= end) {
      if (endTime) {
        endTime.setCustomValidity(!start || !end ? 'Enter both ends of the time range' : 'End must be after start');
        endTime.reportValidity();
      }
      evt.preventDefault();
      evt.stopImmediatePropagation();
      return;
    }
    if (endTime) endTime.setCustomValidity('');
    const startDate = new Date(start), endDate = new Date(end);
    document.querySelectorAll('#finder-controls [data-new-range]').forEach((field) => {
      if (field.name === 'new_from') field.value = start;
      else if (field.name === 'new_to') field.value = end;
      else if (field.name === 'new_from_tz_offset') field.value = String(startDate.getTimezoneOffset());
      else if (field.name === 'new_to_tz_offset') field.value = String(endDate.getTimezoneOffset());
    });
  }, true);
  document.addEventListener('DOMContentLoaded', explorerInit);
  document.body.addEventListener('htmx:afterSettle', explorerInit);
  document.body.addEventListener('change', function (evt) {
    if (evt.target && (evt.target.id === 'explorer-from' || evt.target.id === 'explorer-to' || evt.target.id === 'explorer-from-time' || evt.target.id === 'explorer-to-time')) explorerUpdateOffsets();
  });
  document.addEventListener('click', function (evt) {
    if (evt.target.closest('[data-explorer-apply]')) {
      const val = (id) => (document.getElementById(id)?.value || '') + 'T' + (document.getElementById(id + '-time')?.value || '');
      const toInput = document.getElementById('explorer-to-time');
      if (toInput) {
        toInput.setCustomValidity(val('explorer-from') >= val('explorer-to') ? 'End must be after start' : '');
        if (!toInput.reportValidity()) return;
      }
      const mode = document.querySelector('[data-explorer-range-mode]');
      if (mode) mode.value = 'exact';
      document.querySelectorAll('[data-explorer-preset]').forEach((b) => b.setAttribute('aria-pressed', 'false'));
      document.querySelectorAll('#explorer-from, #explorer-to, #explorer-from-time, #explorer-to-time').forEach((el) => { el.required = true; });
      explorerUpdateRangeLabel();
      const menu = evt.target.closest('.xr-time-menu');
      if (menu) menu.open = false;
      document.getElementById('explorer-load-form')?.requestSubmit();
      return;
    }
    const preset = evt.target.closest('[data-explorer-preset]');
    if (!preset) return;
    const mode = document.querySelector('[data-explorer-range-mode]');
    const seconds = document.querySelector('[data-explorer-range-seconds]');
    if (mode) mode.value = 'relative';
    if (seconds) seconds.value = preset.getAttribute('data-explorer-preset');
    document.querySelectorAll('[data-explorer-preset]').forEach((b) => b.setAttribute('aria-pressed', String(b === preset)));
    explorerUpdateRangeLabel(preset.textContent.trim());
    document.querySelectorAll('#explorer-from, #explorer-to, #explorer-from-time, #explorer-to-time').forEach((el) => { el.required = false; });
    const to = new Date();
    const from = new Date(to.getTime() - Number(preset.getAttribute('data-explorer-preset')) * 1000);
    [['#explorer-from', from], ['#explorer-to', to]].forEach(([sel, d]) => {
      const input = document.querySelector(sel);
      const prefix = sel === '#explorer-from' ? 'explorer-from' : 'explorer-to';
      const timeInput = document.getElementById(prefix + '-time');
      if (!input || !timeInput) return;
      const utc = d.toISOString().slice(0, 19);
      input.setAttribute('data-utc', utc);
      timeInput.setAttribute('data-utc', utc);
      delete input.dataset.localised;
      delete timeInput.dataset.localised;
      explorerLocal(input);
    });
    explorerUpdateOffsets();
    const exact = preset.closest('.xr-time-options')?.querySelector('.xr-exact');
    if (exact) exact.open = false;
    const menu = preset.closest('.xr-time-menu');
    if (menu) menu.open = false;
    document.getElementById('explorer-load-form')?.requestSubmit();
  });

  document.addEventListener('keydown', function (evt) {
    if (evt.key === 'Enter' && evt.target.matches?.('.xr-exact input')) {
      evt.preventDefault();
      document.querySelector('[data-explorer-apply]')?.click();
      return;
    }
    if (evt.key !== 'Escape') return;
    const menu = evt.target.closest?.('.xr-time-menu[open], .xr-cols[open], .xr-sort[open], .xr-top-menu[open]');
    if (!menu) return;
    menu.open = false;
    menu.querySelector('summary')?.focus();
  });

  // The Columns menu lives inside #explorer-results, which every tick swaps.
  // Remember whether it was open (and which box had focus) and restore both.
  let colsWasOpen = false;
  let colsFocus = null;
  document.body.addEventListener('htmx:beforeSwap', function (evt) {
    if (evt.detail.target?.id !== 'explorer-results') return;
    const menu = document.getElementById('explorer-cols');
    colsWasOpen = !!menu?.open;
    const active = document.activeElement;
    colsFocus = colsWasOpen && active && menu.contains(active) ? (active.value || null) : null;
  });
  document.body.addEventListener('htmx:afterSwap', function (evt) {
    if (evt.detail.target?.id !== 'explorer-results' && !document.getElementById('explorer-results')) return;
    if (!colsWasOpen) return;
    colsWasOpen = false;
    const menu = document.getElementById('explorer-cols');
    if (!menu) return;
    menu.open = true;
    if (colsFocus !== null) {
      const box = Array.from(menu.querySelectorAll('input')).find((i) => i.value === colsFocus);
      box?.focus({ preventScroll: true });
    }
    colsFocus = null;
  });
  document.addEventListener('click', function (evt) {
    const menu = document.getElementById('explorer-cols');
    if (menu?.open && !menu.contains(evt.target)) menu.open = false;
  });

  // Poll renders (data-poll) of #explorer-results and #explorer-toolbar: drop one older than the
  // table on screen or identical to it; otherwise keep what the user had open, scrolled, selected
  // and focused. The server half is described above find_poll in dashboard/trace_finder/routes.py.
  const pollKept = {};
  function pollState(root) {
    const details = {};
    root.querySelectorAll('details').forEach(function (d) {
      const key = d.className;
      details[key] = details[key] || [];
      details[key].push(d.open);
    });
    const scroll = {};
    root.querySelectorAll('.xr-table-wrap, .tv-rows').forEach(function (el) {
      scroll[el.className] = [el.scrollLeft, el.scrollTop];
    });
    const active = document.activeElement;
    let focus = null;
    if (active && root.contains(active)) {
      focus = active.id ? '#' + CSS.escape(active.id)
        : active.hasAttribute('data-tv-row') ? '[data-tv-row="' + CSS.escape(active.getAttribute('data-tv-row')) + '"]'
        : active.hasAttribute('data-xr-sort') ? '[data-xr-sort="' + CSS.escape(active.getAttribute('data-xr-sort')) + '"]'
        : active.name ? '[name="' + CSS.escape(active.name) + '"][value="' + CSS.escape(active.value) + '"]'
        : null;
    }
    const sel = root.querySelector('[data-tv-row].sel');
    // OOB toolbar swaps bypass hx-preserve for controls inside the toolbar. Keep values the user
    // may still be editing (custom range and requested row count) until they choose Load/Apply.
    const controls = Array.from(root.querySelectorAll('input[id], select[id], textarea[id]')).map(function (el) {
      return {
        id: el.id,
        value: el.value,
        checked: 'checked' in el ? el.checked : null,
        utc: el.getAttribute('data-utc'),
        localised: el.getAttribute('data-localised'),
        required: el.required
      };
    });
    return {
      details: details,
      scroll: scroll,
      focus: focus,
      sel: sel ? sel.getAttribute('data-tv-row') : null,
      controls: controls
    };
  }
  function pollRestore(root, state) {
    const seen = {};
    root.querySelectorAll('details').forEach(function (d) {
      const key = d.className;
      const index = seen[key] = (seen[key] || 0) + 1;
      const was = state.details[key];
      if (was && index <= was.length) d.open = was[index - 1];
    });
    root.querySelectorAll('.xr-table-wrap, .tv-rows').forEach(function (el) {
      const at = state.scroll[el.className];
      if (at) { el.scrollLeft = at[0]; el.scrollTop = at[1]; }
    });
    if (state.sel) root.querySelector('[data-tv-row="' + CSS.escape(state.sel) + '"]')?.classList.add('sel');
    (state.controls || []).forEach(function (saved) {
      const el = root.querySelector('#' + CSS.escape(saved.id));
      if (!el) return;
      el.value = saved.value;
      if (saved.checked !== null) el.checked = saved.checked;
      if (saved.utc === null) el.removeAttribute('data-utc');
      else el.setAttribute('data-utc', saved.utc);
      if (saved.localised === null) el.removeAttribute('data-localised');
      else el.setAttribute('data-localised', saved.localised);
      el.required = saved.required;
    });
    if (state.focus) root.querySelector(state.focus)?.focus({ preventScroll: true });
  }
  document.body.addEventListener('htmx:oobBeforeSwap', function (evt) {
    const target = evt.detail.target;
    const incoming = evt.detail.fragment && evt.detail.fragment.firstElementChild;
    if (!target || !incoming || !incoming.hasAttribute('data-poll')) return;
    const shown = Number(document.getElementById('explorer-results')?.getAttribute('data-view-version') || 0);
    const version = Number(incoming.getAttribute('data-view-version') || 0);
    if (version < shown) { evt.detail.shouldSwap = false; return; }
    if (incoming.getAttribute('data-render-key') === target.getAttribute('data-render-key')) {
      target.setAttribute('data-view-version', String(version));
      evt.detail.shouldSwap = false;
      return;
    }
    pollKept[target.id] = pollState(target);
  });
  function pollAfterSwap(id) {
    const state = pollKept[id];
    delete pollKept[id];
    const root = document.getElementById(id);
    if (state && root) pollRestore(root, state);
  }
  document.body.addEventListener('htmx:oobAfterSwap', function (evt) {
    const id = evt.detail.target && evt.detail.target.id;
    if (id && pollKept[id]) pollAfterSwap(id);
    const results = id === 'explorer-results' ? document.getElementById('explorer-results') : null;
    if (results?.hasAttribute('data-initial-load')) {
      const scope = document.getElementById('finder-scope');
      const within = scope?.querySelector('input[name="scope"][value="within"]');
      const fresh = scope?.querySelector('input[name="scope"][value="new"]');
      if (scope?.getAttribute('data-auto-scope') === 'pending' && within && fresh) {
        within.disabled = false;
        within.checked = true;
        fresh.checked = false;
        scope.removeAttribute('data-auto-scope');
      }
    }
  });
  document.body.addEventListener('click', function (evt) {
    if (evt.target.closest('#finder-scope')) {
      document.getElementById('finder-scope')?.removeAttribute('data-auto-scope');
    }
  });
  // The run-status slot is the Ask AI poll's main target; keep its open criteria cards across ticks.
  document.body.addEventListener('htmx:beforeSwap', function (evt) {
    if (evt.detail.target?.id === 'finder-run-status') pollKept['finder-run-status'] = pollState(evt.detail.target);
  });
  document.body.addEventListener('htmx:afterSwap', function (evt) {
    if (evt.detail.target?.id === 'finder-run-status') pollAfterSwap('finder-run-status');
  });

  // Trajectories: one tooltip, positioned from the hovered segment's data-* attributes.
  document.addEventListener('mouseover', function (evt) {
    const seg = evt.target.closest('.tv-segs i[data-tv-msg], .tv-segs i[data-tv-tools]');
    const tv = evt.target.closest('.tv');
    const tip = tv && tv.querySelector('.tv-tip');
    if (!tip) return;
    if (!seg) { tip.hidden = true; return; }
    const d = seg.dataset;
    tip.replaceChildren();
    const h = document.createElement('div'); h.className = 'h';
    const sw = document.createElement('i'); sw.className = seg.className;
    const k = document.createElement('b'); k.textContent = d.tvKind;
    const n = document.createElement('span'); n.className = 'n'; n.textContent = d.tvN;
    h.append(sw, k, n);
    const sub = document.createElement('div'); sub.className = 'sub';
    if (d.tvTool) { const t = document.createElement('span'); t.className = 'tool'; t.textContent = d.tvTool; sub.append(t); }
    const tk = document.createElement('span'); tk.className = 'tk'; tk.textContent = d.tvTok; sub.append(tk);
    const pre = document.createElement('pre'); pre.textContent = d.tvP;
    const foot = document.createElement('div'); foot.className = 'c'; foot.textContent = d.tvTools ? 'Captured tool schemas are shown as one block.' : 'Click to open this message ↗';
    tip.append(h, sub, pre, foot);
    tip.hidden = false;
    const box = tv.getBoundingClientRect(); const r = seg.getBoundingClientRect();
    const left = Math.min(Math.max(8, r.left - box.left + r.width / 2 - 160), box.width - 328);
    tip.style.left = left + 'px';
    tip.style.top = (r.bottom - box.top + 10) + 'px';
  });

  // Segment click opens the drawer at that message; the row's own hx-get handles every other click.
  document.addEventListener('click', function (evt) {
    const seg = evt.target.closest('.tv-segs i[data-tv-msg]');
    const row = evt.target.closest('[data-tv-row]');
    if (row) {
      document.querySelectorAll('[data-tv-row].sel').forEach((el) => el.classList.remove('sel'));
      row.classList.add('sel');
    }
    if (!seg || !row) return;
    const tip = seg.closest('.tv').querySelector('.tv-tip');
    if (tip) tip.hidden = true;
    evt.stopPropagation();
    htmx.ajax('GET', '/find/trace/' + encodeURIComponent(row.getAttribute('data-tv-row')) + '?msg=' + seg.getAttribute('data-tv-msg'), { target: '#finder-drawer', swap: 'innerHTML' });
  }, true);

  // Drawer: scroll the thread (not the page) to the selected message, and move the selection locally.
  let latestFinderDrawerXhr = null;
  document.body.addEventListener('htmx:beforeRequest', function (evt) {
    if (evt.detail.target?.id === 'finder-drawer') latestFinderDrawerXhr = evt.detail.xhr;
  });
  document.body.addEventListener('htmx:beforeSwap', function (evt) {
    if (evt.detail.target?.id === 'finder-drawer' && evt.detail.xhr !== latestFinderDrawerXhr) {
      evt.preventDefault();
    }
  });
  function drawerMarkSelected(root, index) {
    root.querySelectorAll('.fd-msg').forEach((el) => {
      const on = el.getAttribute('data-msg') === String(index);
      el.classList.toggle('on', on);
    });
    root.querySelectorAll('.fd-mini i').forEach((el) => el.classList.toggle('on', el.getAttribute('data-mini-msg') === String(index)));
  }
  function drawerScrollTo(root, index) {
    const list = root.querySelector('#fd-thread');
    const target = root.querySelector('#msg-' + index);
    if (list && target) list.scrollTop = target.offsetTop - list.offsetTop - list.clientHeight / 2 + target.clientHeight / 2;
  }
  function drawerSelect(root, index) {
    drawerMarkSelected(root, index);
    root.querySelectorAll('.fd-msg').forEach((el) => { el.open = el.getAttribute('data-msg') === String(index); });
    drawerScrollTo(root, index);
  }
  document.body.addEventListener('htmx:afterSwap', function (evt) {
    if (evt.detail.target && evt.detail.target.id === 'finder-drawer') {
      const on = evt.detail.target.querySelector('.fd-msg.on');
      if (on) drawerScrollTo(evt.detail.target, on.getAttribute('data-msg'));
    }
  });
  document.addEventListener('click', function (evt) {
    const mini = evt.target.closest('.fd-mini i[data-mini-msg]');
    const summary = evt.target.closest('.fd-msg > summary');
    const root = document.getElementById('finder-drawer');
    if (!root || (!mini && !summary)) return;
    if (mini) {
      drawerSelect(root, mini.getAttribute('data-mini-msg'));
      return;
    }
    const details = summary.parentElement;
    const index = details.getAttribute('data-msg');
    drawerMarkSelected(root, index);
    if (!details.open) drawerScrollTo(root, index);
  });

  // Finder rows open the trace drawer from the keyboard; the drawer behaves as a modal dialog:
  // focus moves in on open, Tab stays inside, Escape closes, focus returns to the opening row,
  // and j/k (or ArrowDown/ArrowUp) step to the next/previous trace like the simulation drawer.
  let finderOpenId = null;
  const finderDialog = () => document.querySelector('#finder-drawer [role="dialog"]');
  const shortcutGuide = () => document.querySelector('#finder-shortcut-guide');
  const finderRows = () => Array.from(document.querySelectorAll('#explorer-results [data-tv-row]'));
  const finderRow = (id) => finderRows().find((row) => row.getAttribute('data-tv-row') === id) || null;
  function finderEditable(el) {
    return !!el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
  }
  function traceShortcutBlocked(evt) {
    const target = evt.target;
    const active = document.activeElement;
    const details = target.closest?.('details, summary') || active?.closest?.('details, summary');
    const otherModal = Array.from(document.querySelectorAll('dialog[open], [role="dialog"][aria-modal="true"]'))
      .some((modal) => modal !== finderDialog() && modal !== shortcutGuide());
    const shiftedShortcut = ['?', 'J', 'K'].includes(evt.key) && evt.shiftKey && !evt.ctrlKey && !evt.altKey && !evt.metaKey;
    return evt.ctrlKey || evt.altKey || evt.metaKey || (evt.shiftKey && !shiftedShortcut) ||
      finderEditable(target) || finderEditable(active) || details || otherModal;
  }
  function ensureShortcutGuide() {
    let guide = shortcutGuide();
    if (guide) return guide;
    guide = document.createElement('dialog');
    guide.id = 'finder-shortcut-guide';
    guide.setAttribute('aria-labelledby', 'finder-shortcut-title');
    guide.style.cssText = 'border:1px solid var(--line,#d7dce2);border-radius:14px;padding:22px;max-width:440px;width:calc(100% - 40px);color:var(--ink,#18202a);background:var(--surface,#fff);box-shadow:0 24px 80px #0004';
    guide.innerHTML = '<h2 id="finder-shortcut-title">Keyboard shortcuts</h2>' +
      '<dl style="display:grid;grid-template-columns:140px 1fr;gap:8px 12px;margin:18px 0">' +
      '<dt><kbd>/</kbd></dt><dd style="margin:0">Ask AI about traces</dd>' +
      '<dt><kbd>?</kbd></dt><dd style="margin:0">Show this guide</dd>' +
      '<dt><kbd>j</kbd> / <kbd>k</kbd> or arrows</dt><dd style="margin:0">Next or previous trace</dd>' +
      '<dt><kbd>o</kbd></dt><dd style="margin:0">Open this trace in Orq</dd>' +
      '<dt><kbd>c</kbd></dt><dd style="margin:0">Copy this trace ID</dd>' +
      '<dt><kbd>Esc</kbd></dt><dd style="margin:0">Close the guide or trace</dd></dl>' +
      '<button type="button" class="btn-secondary" data-shortcut-guide-close>Close</button>';
    guide.addEventListener('click', function (evt) {
      if (evt.target === guide) guide.close();
    });
    guide.querySelector('[data-shortcut-guide-close]').addEventListener('click', () => guide.close());
    document.body.appendChild(guide);
    return guide;
  }
  function handleTraceShortcut(evt) {
    if (window.location.pathname !== '/traces') return false;
    if (traceShortcutBlocked(evt)) return false;
    if (evt.key === '/') {
      const input = document.querySelector('.finder-command-textarea, .finder-command textarea[name="query"], #finder-query-form textarea[name="query"]');
      if (!input || input.disabled) return false;
      evt.preventDefault();
      input.focus();
      input.select();
      return true;
    }
    if (evt.key === '?') {
      evt.preventDefault();
      const guide = ensureShortcutGuide();
      if (!guide.open) guide.showModal();
      guide.querySelector('[data-shortcut-guide-close]').focus();
      return true;
    }
    if (evt.key === 'o' && finderDialog() && finderOpenId) {
      const link = Array.from(finderDialog().querySelectorAll('a[href]'))
        .find((anchor) => anchor.textContent.includes('Open in Orq'));
      if (!link) return false;
      evt.preventDefault();
      link.click();
      return true;
    }
    if (evt.key === 'c' && finderDialog() && finderOpenId) {
      const button = Array.from(finderDialog().querySelectorAll('.rt-drawer-footer button[data-trace-id]'))
        .find((candidate) => candidate.getAttribute('data-trace-id') === finderOpenId);
      if (!button) return false;
      evt.preventDefault();
      button.click();
      return true;
    }
    return false;
  }
  function finderStep(delta) {
    const rows = finderRows();
    const current = rows.findIndex((row) => row.getAttribute('data-tv-row') === finderOpenId);
    if (current < 0 || !rows.length) return;
    const next = rows[(current + delta + rows.length) % rows.length];
    next.scrollIntoView({ block: 'nearest' });
    next.click();
  }
  document.addEventListener('click', function (evt) {
    const row = evt.target.closest('#explorer-results [data-tv-row]');
    if (row) finderOpenId = row.getAttribute('data-tv-row');
  }, true);
  document.addEventListener('keydown', function (evt) {
    const guide = shortcutGuide();
    if (evt.key === 'Escape' && guide?.open) {
      evt.preventDefault();
      guide.close();
      return;
    }
    if (guide?.open) return;
    if (handleTraceShortcut(evt)) return;
    const dialog = finderDialog();
    if (!dialog) {
      const row = evt.target.closest?.('#explorer-results [data-tv-row]');
      if (row && evt.target === row && (evt.key === 'Enter' || evt.key === ' ')) {
        evt.preventDefault();
        row.click();
      }
      return;
    }
    if (evt.key === 'Escape') {
      evt.preventDefault();
      dialog.querySelector('.rt-drawer-close')?.click();
      return;
    }
    if (evt.key === 'Tab') {
      const items = Array.from(dialog.querySelectorAll('a[href], button:not([disabled]), input, select, textarea, summary, [tabindex]:not([tabindex="-1"])'))
        .filter((el) => el.offsetParent !== null);
      if (!items.length) { evt.preventDefault(); dialog.focus(); return; }
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement;
      if (!dialog.contains(active) || (evt.shiftKey && (active === first || active === dialog))) {
        evt.preventDefault();
        (evt.shiftKey ? last : first).focus();
      } else if (!evt.shiftKey && active === last) {
        evt.preventDefault();
        first.focus();
      }
      return;
    }
    if (evt.metaKey || evt.ctrlKey || evt.altKey || finderEditable(document.activeElement)) return;
    if (window.location.pathname === '/traces' && traceShortcutBlocked(evt)) return;
    if (evt.key === 'j' || evt.key === 'J' || evt.key === 'ArrowDown') {
      evt.preventDefault();
      finderStep(1);
    } else if (evt.key === 'k' || evt.key === 'K' || evt.key === 'ArrowUp') {
      evt.preventDefault();
      finderStep(-1);
    }
  });
  document.body.addEventListener('htmx:afterSettle', function (evt) {
    if (evt.detail.target?.id !== 'finder-drawer') return;
    const dialog = finderDialog();
    if (dialog) {
      if (!dialog.contains(document.activeElement)) dialog.focus({ preventScroll: true });
      return;
    }
    const row = finderOpenId && finderRow(finderOpenId);
    if (row) row.focus({ preventScroll: true });
  });

  // Explorer swaps replace the rows and headers; put focus back on the row or sort button that had it.
  let explorerFocus = null;
  document.body.addEventListener('htmx:beforeRequest', function (evt) {
    const sortButton = evt.detail.elt?.closest?.('[data-xr-sort]');
    if (sortButton) explorerFocus = ['data-xr-sort', sortButton.getAttribute('data-xr-sort')];
  });
  document.body.addEventListener('htmx:beforeSwap', function (evt) {
    if (evt.detail.target?.id !== 'explorer-results' || explorerFocus) return;
    // Polls re-render the results while a load finishes; they must not drop focus either.
    const active = document.activeElement;
    const row = active?.closest?.('#explorer-results [data-tv-row]');
    if (row === active) explorerFocus = ['data-tv-row', row.getAttribute('data-tv-row')];
    const sortButton = active?.closest?.('#explorer-results [data-xr-sort]');
    if (sortButton) explorerFocus = ['data-xr-sort', sortButton.getAttribute('data-xr-sort')];
  });
  document.body.addEventListener('htmx:afterSettle', function () {
    if (!explorerFocus) return;
    const [attr, value] = explorerFocus;
    explorerFocus = null;
    const target = Array.from(document.querySelectorAll('#explorer-results [' + attr + ']')).find((el) => el.getAttribute(attr) === value);
    if (target && !finderDialog()) target.focus({ preventScroll: true });
  });
})();
