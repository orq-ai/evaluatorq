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

  function loadingDrawer(message) {
    return (
      '<div class="rt-drawer-overlay"></div>' +
      '<aside class="rt-drawer" role="dialog" aria-modal="true" aria-busy="true">' +
      '<div class="rt-drawer-head"><h3 class="rt-drawer-title">Preview changes</h3></div>' +
      '<div class="rt-drawer-body rt-drawer-body--loading">' +
      '<span class="rt-drawer-spinner" aria-hidden="true"></span>' +
      '<p class="rt-drawer-loading-title">' + message + '</p>' +
      '<p class="rt-drawer-note">This rewrites the agent instructions with an LLM and usually ' +
      'takes 10–30 seconds. Nothing is written to the agent.</p>' +
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
        ? 'Merging this recommendation into the agent instructions…'
        : 'Merging the pending recommendations into the agent instructions…'
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
})();
