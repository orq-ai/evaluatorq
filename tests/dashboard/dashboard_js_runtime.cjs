const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(
  path.resolve(__dirname, '../../src/evaluatorq/dashboard/static/dashboard.js'),
  'utf8'
);

function emitter() {
  const listeners = new Map();
  return {
    addEventListener(name, callback) {
      if (!listeners.has(name)) listeners.set(name, []);
      listeners.get(name).push(callback);
    },
    emit(name, event = { detail: { target: null } }) {
      for (const callback of listeners.get(name) || []) callback(event);
    },
  };
}

function classList() {
  const values = new Set();
  return {
    add(name) { values.add(name); },
    remove(name) { values.delete(name); },
    contains(name) { return values.has(name); },
  };
}

function loadDashboard({
  elements = new Map(), query = () => null, queryAll = () => [], pathname = '/traces', dateClass = Date,
  htmx = null,
} = {}) {
  const body = Object.assign(emitter(), {
    children: [],
    appendChild(child) { this.children.push(child); child.parentNode = this; },
  });
  const documentEvents = emitter();
  const windowEvents = emitter();
  const scheduled = new Map();
  const intervals = new Map();
  let shortcutGuide = null;
  let nextTimer = 1;
  const document = {
    body,
    addEventListener: documentEvents.addEventListener,
    getElementById(id) { return elements.get(id) || null; },
    querySelector(selector) {
      if (selector === '#finder-shortcut-guide' && shortcutGuide) return shortcutGuide;
      return query(selector);
    },
    querySelectorAll: queryAll,
    createElement(tagName) {
      const listeners = new Map();
      const closeButton = {
        focus() { this.focused = true; },
        addEventListener(name, callback) {
          if (!listeners.has(`close:${name}`)) listeners.set(`close:${name}`, []);
          listeners.get(`close:${name}`).push(callback);
        },
        click() {
          for (const callback of listeners.get('close:click') || []) callback({ target: closeButton });
        },
      };
      const element = {
        tagName: tagName.toUpperCase(), style: {}, dataset: {}, open: false,
        append() {}, replaceChildren() {},
        setAttribute() {},
        addEventListener(name, callback) {
          if (!listeners.has(name)) listeners.set(name, []);
          listeners.get(name).push(callback);
        },
        querySelector(selector) {
          return selector === '[data-shortcut-guide-close]' ? closeButton : null;
        },
        showModal() { this.open = true; },
        close() { this.open = false; },
        focus() { this.focused = true; },
      };
      if (tagName === 'dialog') shortcutGuide = element;
      return element;
    },
    activeElement: null,
  };
  const window = {
    location: { pathname },
    htmx,
    addEventListener: windowEvents.addEventListener,
    setTimeout(callback) {
      const id = nextTimer++;
      scheduled.set(id, callback);
      return id;
    },
    clearTimeout(id) { scheduled.delete(id); },
    setInterval(callback, delay) {
      const id = nextTimer++;
      intervals.set(id, { callback, delay });
      return id;
    },
    clearInterval(id) { intervals.delete(id); },
  };
  const entries = [{ state: null }];
  let index = 0;
  const history = {
    get state() { return entries[index].state; },
    pushState(state) {
      entries.splice(index + 1, entries.length, { state });
      index += 1;
    },
    replaceState(state) { entries[index] = { state }; },
    back() { this.go(-1); },
    go(delta) {
      index += delta;
      assert.ok(index >= 0 && index < entries.length);
      windowEvents.emit('popstate', { state: entries[index].state });
    },
  };
  vm.runInNewContext(source, {
    document, window, history, location: { hash: '' },
    FormData: class {}, Date: dateClass, CSS: { escape: value => value },
    Event: class { constructor(type, init) { this.type = type; Object.assign(this, init); } },
  });
  return { body, document, documentEvents, window, history, entries, scheduled, intervals };
}

function frozenDate(now) {
  const NativeDate = Date;
  return class extends NativeDate {
    constructor(...args) { super(...(args.length ? args : [now])); }
  };
}

function requestBody(form, includedById) {
  const includedForms = form.hxInclude.split(',').map(selector => includedById.get(selector.trim().slice(1)));
  assert.ok(includedForms.every(Boolean), 'every hx-include selector should resolve to a form/control group');
  const fields = [...new Set([...form.elements, ...includedForms.flatMap(included => included.elements)])];
  const params = new URLSearchParams();
  for (const field of fields) {
    if (!field.name || field.disabled || field.form && field.form !== form &&
        !includedForms.some(included => included.id === field.form.id)) continue;
    if (['checkbox', 'radio'].includes(field.type) && !field.checked) continue;
    params.append(field.name, field.value);
  }
  return params;
}

function submitExplorerQuery({ mode, seconds = '900', now, start, end }) {
  const dateClass = frozenDate(now);
  const input = (name, value, id = '') => ({ name, value, id, disabled: false });
  const fields = [
    input('new_from', ''), input('new_to', ''),
    input('new_from_tz_offset', '0'), input('new_to_tz_offset', '0'),
  ];
  const from = input('from', start.slice(0, 10), 'explorer-from');
  const fromTime = input('from_time', start.slice(11), 'explorer-from-time');
  const to = input('to', end.slice(0, 10), 'explorer-to');
  const toTime = input('to_time', end.slice(11), 'explorer-to-time');
  fromTime.setCustomValidity = message => { fromTime.validationMessage = message; };
  fromTime.reportValidity = () => {};
  toTime.setCustomValidity = message => { toTime.validationMessage = message; };
  toTime.reportValidity = () => {};
  const rangeMode = input('range_mode', mode);
  const rangeSeconds = input('range_seconds', seconds);
  const rangeFromOffset = input('from_tz_offset', '0');
  const rangeToOffset = input('to_tz_offset', '0');
  const explorerElements = [from, fromTime, to, toTime, rangeMode, rangeSeconds, rangeFromOffset, rangeToOffset];
  const controls = { id: 'finder-controls', elements: fields };
  const explorerForm = { id: 'explorer-load-form', elements: explorerElements };
  explorerElements.forEach(control => { control.form = explorerForm; });
  const unchecked = { ...input('include_archived', 'yes'), type: 'checkbox', checked: false, form: explorerForm };
  const checked = { ...input('include_errors', 'yes'), type: 'checkbox', checked: true, form: explorerForm };
  explorerForm.elements.push(unchecked, checked);
  const queryForm = {
    id: 'finder-query-form',
    hxInclude: '#finder-controls, #explorer-load-form',
    elements: [input('query', 'slow traces'), input('scope', 'new'), input('mode', 'immediate')],
  };
  queryForm.elements.forEach(control => { control.form = queryForm; });
  const unrelatedForm = { id: 'unrelated-form' };
  controls.elements.push({ ...input('foreign', 'wrong form'), form: unrelatedForm });
  const elements = new Map([
    ['explorer-load-form', explorerForm], ['explorer-from', from], ['explorer-from-time', fromTime],
    ['explorer-to', to], ['explorer-to-time', toTime],
  ]);
  const app = loadDashboard({
    elements,
    dateClass,
    query(selector) {
      if (selector === '[data-explorer-range-mode]') return rangeMode;
      if (selector === '[data-explorer-range-seconds]') return rangeSeconds;
      if (selector === '[name="from_tz_offset"][form="explorer-load-form"]') return rangeFromOffset;
      if (selector === '[name="to_tz_offset"][form="explorer-load-form"]') return rangeToOffset;
      return null;
    },
    queryAll(selector) { return selector === '#finder-controls [data-new-range]' ? fields : []; },
  });
  const evt = {
    target: queryForm,
    prevented: false,
    stopped: false,
    preventDefault() { this.prevented = true; },
    stopImmediatePropagation() { this.stopped = true; },
  };
  app.documentEvents.emit('submit', evt);
  const body = requestBody(queryForm, new Map([['finder-controls', controls], ['explorer-load-form', explorerForm]]));
  return { app, body, evt, fields, queryForm, fromTime, toTime };
}

function keyEvent(key, options = {}) {
  let prevented = false;
  return {
    key,
    target: { tagName: 'BODY', closest() { return null; } },
    preventDefault() { prevented = true; },
    get defaultPrevented() { return prevented; },
    ...options,
  };
}

test('trace shortcuts respect editable targets, modifiers, modal state, and route', () => {
  let focuses = 0;
  let selections = 0;
  const input = {
    tagName: 'TEXTAREA', disabled: false,
    focus() { focuses += 1; },
    select() { selections += 1; },
  };
  let modalOpen = false;
  const app = loadDashboard({
    query: selector => selector.includes('textarea') ? input : null,
    queryAll: selector => selector === 'dialog[open], [role="dialog"][aria-modal="true"]' && modalOpen
      ? [{ open: true }]
      : [],
  });
  const press = (key, options) => {
    const event = keyEvent(key, options);
    app.documentEvents.emit('keydown', event);
    return event;
  };

  assert.equal(press('/').defaultPrevented, true);
  assert.equal(focuses, 1);
  assert.equal(selections, 1);

  const editable = { tagName: 'INPUT', isContentEditable: false };
  app.document.activeElement = editable;
  assert.equal(press('/').defaultPrevented, false);
  app.document.activeElement = null;
  assert.equal(press('/', { target: editable }).defaultPrevented, false);
  assert.equal(press('/', { ctrlKey: true }).defaultPrevented, false);

  const guideEvent = press('?');
  assert.equal(guideEvent.defaultPrevented, true);
  const guide = app.document.querySelector('#finder-shortcut-guide');
  assert.equal(guide.open, true);
  assert.ok(app.body.children.includes(guide));
  guide.querySelector('[data-shortcut-guide-close]').click();
  assert.equal(guide.open, false);
  assert.equal(guide.querySelector('[data-shortcut-guide-close]').focused, true);
  assert.equal(press('?').defaultPrevented, true);
  assert.equal(guide.open, true);
  assert.equal(press('/').defaultPrevented, false);
  assert.equal(press('Escape').defaultPrevented, true);
  assert.equal(guide.open, false);

  modalOpen = true;
  assert.equal(press('/').defaultPrevented, false);
  modalOpen = false;
  app.window.location.pathname = '/reports';
  assert.equal(press('/').defaultPrevented, false);
  assert.equal(focuses, 1);
});

test('finder placeholder stays fixed when focusing the question', () => {
  const query = { placeholder: 'Ask a question about your traces…' };
  const app = loadDashboard();
  app.body.emit('focusin', { target: { closest: () => query } });
  assert.equal(query.placeholder, 'Ask a question about your traces…');
  assert.equal(app.intervals.size, 0);
});

test('trace Ask AI submits custom bounds and endpoint timezone offsets', () => {
  const app = submitExplorerQuery({
    mode: 'exact', now: Date.parse('2026-09-30T12:00:00Z'),
    start: '2026-09-30T10:15:00', end: '2026-09-30T11:45:30',
  });
  assert.equal(app.evt.prevented, false);
  assert.equal(app.body.get('new_from'), '2026-09-30T10:15:00');
  assert.equal(app.body.get('new_to'), '2026-09-30T11:45:30');
  assert.equal(app.body.get('new_from_tz_offset'), String(new Date('2026-09-30T10:15:00').getTimezoneOffset()));
  assert.equal(app.body.get('new_to_tz_offset'), String(new Date('2026-09-30T11:45:30').getTimezoneOffset()));
  assert.equal(app.body.get('from'), '2026-09-30');
  assert.equal(app.body.get('from_time'), '10:15:00');
  assert.equal(app.body.get('to'), '2026-09-30');
  assert.equal(app.body.get('to_time'), '11:45:30');
  assert.equal(app.body.get('range_mode'), 'exact');
  assert.equal(app.body.get('query'), 'slow traces');
  assert.deepEqual(app.body.getAll('from'), ['2026-09-30']);
  assert.deepEqual(app.body.getAll('include_archived'), []);
  assert.deepEqual(app.body.getAll('include_errors'), ['yes']);
  assert.deepEqual(app.body.getAll('foreign'), []);
});

test('trace Ask AI refreshes a relative range at submit time and submits it', () => {
  const now = Date.parse('2026-09-30T12:00:00Z');
  const result = submitExplorerQuery({
    mode: 'relative', seconds: '3600', now,
    start: '2026-09-30T09:00:00', end: '2026-09-30T10:00:00',
  });
  const DateAtSubmit = frozenDate(now);
  const end = new DateAtSubmit();
  const start = new DateAtSubmit(now - 3600 * 1000);
  const localValue = date => {
    const pad = value => String(value).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
      `T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  };
  assert.equal(result.evt.prevented, false);
  assert.equal(result.body.get('new_from'), localValue(start));
  assert.equal(result.body.get('new_to'), localValue(end));
  assert.equal(result.body.get('new_from_tz_offset'), String(start.getTimezoneOffset()));
  assert.equal(result.body.get('new_to_tz_offset'), String(end.getTimezoneOffset()));
  assert.equal(result.body.get('range_seconds'), '3600');
});

test('search form submission on /find skips explorer range validation', () => {
  const queryForm = { id: 'finder-query-form', elements: [] };
  const app = loadDashboard({ pathname: '/find' });
  const evt = {
    target: queryForm,
    prevented: false,
    stopImmediatePropagation() { this.stopped = true; },
    preventDefault() { this.prevented = true; },
  };
  app.documentEvents.emit('submit', evt);
  assert.equal(evt.prevented, false);
  assert.equal(evt.stopped, undefined);
});

test('open filter dropdown survives a fragment swap', () => {
  const old = { id: 'filter-dd-persona', open: true };
  const replacement = { id: old.id, open: false };
  const untouched = { id: 'filter-dd-scenario', open: false };
  const elements = new Map([
    ['filter-form', { querySelectorAll() { return [old]; } }],
  ]);
  const app = loadDashboard({ elements });

  app.body.emit('htmx:beforeSwap', { detail: { target: null } });
  elements.set(old.id, replacement);
  elements.set(untouched.id, untouched);
  app.body.emit('htmx:afterSwap', { detail: { target: null } });

  assert.equal(replacement.open, true);
  assert.equal(untouched.open, false);

  // The next swap has no open dropdown; an old snapshot must not reopen it.
  elements.set('filter-form', { querySelectorAll() { return []; } });
  replacement.open = false;
  app.body.emit('htmx:beforeSwap', { detail: { target: null } });
  app.body.emit('htmx:afterSwap', { detail: { target: null } });
  assert.equal(replacement.open, false);
});

test('overlapping explorer OOB swaps restore each captured table state in order', () => {
  function resultsRoot({ open, left, top, selected, version, sequence, renderKey }) {
    const detail = { className: 'xr-exact', open };
    const scroll = { className: 'xr-table-wrap', scrollLeft: left, scrollTop: top };
    const rows = new Map(['row-first', 'row-second'].map(id => [id, {
      getAttribute(name) { return name === 'data-tv-row' ? id : null; },
      classList: classList(),
    }]));
    if (selected) rows.get(selected)?.classList.add('sel');
    const attributes = {
      'data-view-version': String(version),
      'data-poll-sequence': String(sequence),
      'data-render-key': renderKey,
    };
    return {
      id: 'explorer-results',
      querySelectorAll(selector) {
        if (selector === 'details') return [detail];
        if (selector === '.xr-table-wrap, .tv-rows') return [scroll];
        if (selector === 'input[id], select[id], textarea[id]') return [];
        return [];
      },
      querySelector(selector) {
        if (selector === '[data-tv-row].sel') return selected ? rows.get(selected) : null;
        const match = selector.match(/^\[data-tv-row="([^"]+)"\]$/);
        if (match) return rows.get(match[1]) || null;
        return null;
      },
      contains() { return false; },
      getAttribute(name) { return attributes[name] ?? null; },
      setAttribute(name, value) { attributes[name] = String(value); },
      hasAttribute(name) { return Object.hasOwn(attributes, name); },
      detail, scroll, rows,
    };
  }
  const elements = new Map();
  const app = loadDashboard({ elements });
  const firstRequestTarget = resultsRoot({
    open: true, left: 4, top: 120, selected: 'row-first', version: 7, sequence: 10, renderKey: 'first',
  });
  elements.set('explorer-results', firstRequestTarget);

  // A suppressed duplicate advances the sequence but must not add a saved state.
  const duplicate = { getAttribute(name) { return ({
    'data-poll': '', 'data-view-version': '7', 'data-poll-sequence': '11', 'data-render-key': 'first',
  })[name] ?? null; }, hasAttribute(name) { return name === 'data-poll'; } };
  const duplicateEvent = { detail: { target: firstRequestTarget, fragment: { firstElementChild: duplicate } } };
  app.body.emit('htmx:oobBeforeSwap', duplicateEvent);
  assert.equal(duplicateEvent.detail.shouldSwap, false);

  const firstIncoming = { getAttribute(name) { return ({
    'data-poll': '', 'data-view-version': '7', 'data-poll-sequence': '12', 'data-render-key': 'second',
  })[name] ?? null; }, hasAttribute(name) { return name === 'data-poll'; } };
  app.body.emit('htmx:oobBeforeSwap', {
    detail: { target: firstRequestTarget, fragment: { firstElementChild: firstIncoming } },
  });

  const secondRequestTarget = resultsRoot({
    open: false, left: 19, top: 560, selected: 'row-second', version: 7, sequence: 12, renderKey: 'second',
  });
  elements.set('explorer-results', secondRequestTarget);
  const secondIncoming = { getAttribute(name) { return ({
    'data-poll': '', 'data-view-version': '7', 'data-poll-sequence': '13', 'data-render-key': 'third',
  })[name] ?? null; }, hasAttribute(name) { return name === 'data-poll'; } };
  app.body.emit('htmx:oobBeforeSwap', {
    detail: { target: secondRequestTarget, fragment: { firstElementChild: secondIncoming } },
  });

  const firstReplacement = resultsRoot({
    open: false, left: 0, top: 0, selected: null, version: 7, sequence: 12, renderKey: 'third',
  });
  elements.set('explorer-results', firstReplacement);
  app.body.emit('htmx:oobAfterSwap', { detail: { target: firstReplacement } });
  assert.equal(firstReplacement.detail.open, true);
  assert.equal(firstReplacement.scroll.scrollLeft, 4);
  assert.equal(firstReplacement.scroll.scrollTop, 120);
  assert.equal(firstReplacement.rows.get('row-first').classList.contains('sel'), true);

  const secondReplacement = resultsRoot({
    open: true, left: 0, top: 0, selected: null, version: 7, sequence: 13, renderKey: 'fourth',
  });
  elements.set('explorer-results', secondReplacement);
  app.body.emit('htmx:oobAfterSwap', { detail: { target: secondReplacement } });
  assert.equal(secondReplacement.detail.open, false);
  assert.equal(secondReplacement.scroll.scrollLeft, 19);
  assert.equal(secondReplacement.scroll.scrollTop, 560);
  assert.equal(secondReplacement.rows.get('row-second').classList.contains('sel'), true);
});

test('initial load enables Within results even after a click on it while it was disabled', () => {
  function scopeGroup() {
    const attributes = { 'data-auto-scope': 'pending' };
    const within = { disabled: true, checked: false };
    const fresh = { disabled: false, checked: true };
    return {
      within, fresh,
      getAttribute(name) { return attributes[name] ?? null; },
      removeAttribute(name) { delete attributes[name]; },
      querySelector(selector) { return selector.includes('"within"') ? within : selector.includes('"new"') ? fresh : null; },
    };
  }
  const results = { id: 'explorer-results', hasAttribute: name => name === 'data-initial-load' };
  const inScope = { matches: () => false, closest: selector => (selector === '#finder-scope' ? {} : null) };

  // A click on the disabled radio is not a choice: the auto-select still applies.
  let scope = scopeGroup();
  let app = loadDashboard({ elements: new Map([['finder-scope', scope], ['explorer-results', results]]) });
  app.body.emit('click', { target: inScope });
  app.body.emit('htmx:oobAfterSwap', { detail: { target: results } });
  assert.deepEqual([scope.within.disabled, scope.within.checked, scope.fresh.checked], [false, true, false]);

  // A real choice of New search keeps it, but Within results still becomes usable.
  scope = scopeGroup();
  app = loadDashboard({ elements: new Map([['finder-scope', scope], ['explorer-results', results]]) });
  app.body.emit('change', { target: inScope });
  app.body.emit('htmx:oobAfterSwap', { detail: { target: results } });
  assert.deepEqual([scope.within.disabled, scope.within.checked, scope.fresh.checked], [false, false, true]);
});

test('drawer history restores a prior view and closes after its exit animation', () => {
  const dialogEvents = emitter();
  const content = { innerHTML: '' };
  const dialog = {
    ...dialogEvents,
    open: false,
    dataset: {},
    classList: classList(),
    showModal() { this.open = true; },
    close() { this.open = false; dialogEvents.emit('close'); },
    querySelector(selector) {
      return selector === '[data-sim-entity-content]' ? content : null;
    },
  };
  const origin = { querySelectorAll() { return [alice, bob]; } };
  function trigger(id) {
    return {
      parentElement: origin,
      getAttribute(name) {
        return { 'data-entity-kind': 'persona', 'data-entity-id': id }[name] || null;
      },
    };
  }
  const alice = trigger('alice');
  const bob = trigger('bob');
  const templates = new Map([['alice', { innerHTML: 'Alice details' }], ['bob', { innerHTML: 'Bob details' }]]);
  const app = loadDashboard({
    query(selector) {
      if (selector === '.sim-entity-dialog') return dialog;
      if (selector.includes('data-sim-entity-template')) {
        return templates.get(selector.includes('alice') ? 'alice' : 'bob');
      }
      if (selector.includes('data-sim-entity-trigger')) return alice;
      return null;
    },
  });
  function click(match) {
    app.body.emit('click', {
      target: {
        classList: classList(),
        closest(selector) { return match(selector); },
      },
      preventDefault() {},
    });
  }

  click(selector => selector === '[data-sim-entity-trigger]' ? alice : null);
  assert.equal(content.innerHTML, 'Alice details');
  assert.equal(app.entries.length, 2);
  click(selector => selector === '[data-sim-entity-trigger]' ? bob : null);
  assert.equal(content.innerHTML, 'Bob details');
  assert.equal(app.entries.length, 3);

  click(selector => selector === '[data-sim-entity-back]' ? {} : null);
  assert.equal(content.innerHTML, 'Alice details');
  assert.equal(app.entries.length, 3);
  app.history.go(1);
  assert.equal(content.innerHTML, 'Bob details');
  app.history.back();
  assert.equal(content.innerHTML, 'Alice details');

  click(selector => selector === '[data-sim-entity-close]' ? {} : null);
  assert.equal(dialog.open, true);
  assert.equal(dialog.classList.contains('sim-entity-dialog--closing'), true);
  assert.equal(app.scheduled.size, 1);
  dialog.emit('animationend');
  assert.equal(dialog.open, false);
  assert.equal(dialog.classList.contains('sim-entity-dialog--closing'), false);
  assert.equal(app.scheduled.size, 0);

  click(selector => selector === '[data-sim-entity-trigger]' ? alice : null);
  assert.equal(app.history.state.simDrawer.id, 'alice');
  dialog.close(); // Native Escape closes the dialog directly.
  assert.equal(app.history.state, null);
});

test('trajectory tooltip stays within narrow and wide containers', () => {
  const tip = {
    hidden: true,
    style: {},
    append() {},
    replaceChildren() {},
    get offsetWidth() { return Math.min(320, Number.parseFloat(this.style.maxWidth) || 320); },
  };
  const seg = {
    className: 'msg',
    dataset: { tvKind: 'Message', tvN: '1', tvTok: '10 tokens', tvP: 'hello', tvTools: '' },
    getBoundingClientRect() { return { left: 220, bottom: 20, width: 10 }; },
  };
  let containerWidth = 250;
  const tv = {
    querySelector() { return tip; },
    getBoundingClientRect() { return { left: 0, top: 0, width: containerWidth }; },
  };
  const app = loadDashboard();
  app.documentEvents.emit('mouseover', {
    target: { closest(selector) { return selector.includes('.tv-segs') ? seg : tv; } },
  });

  assert.equal(tip.hidden, false);
  assert.equal(tip.style.maxWidth, '234px');
  assert.equal(tip.style.left, '8px');

  containerWidth = 600;
  app.documentEvents.emit('mouseover', {
    target: { closest(selector) { return selector.includes('.tv-segs') ? seg : tv; } },
  });
  assert.equal(tip.style.maxWidth, '584px');
  assert.equal(tip.style.left, '65px');
});

test('pending explorer control waits for its main result settle and coalesces a second control click', () => {
  let submits = 0;
  const form = { id: 'explorer-load-form', requestSubmit() { submits += 1; } };
  const filterButton = { setAttribute() {} };
  const first = { getAttribute: () => '/traces?limit=25' };
  const second = { getAttribute: () => '/traces?limit=50' };
  const elements = new Map([['explorer-load-form', form]]);
  const calls = [];
  const app = loadDashboard({
    elements,
    htmx: { config: {}, ajax(method, url) { calls.push([method, url]); } },
    query(selector) {
      return selector === '[data-explorer-filters]' ? filterButton : null;
    },
    queryAll() { return []; },
  });
  app.body.emit('change', { target: { closest: selector => selector === '.finder-facets' ? {} : null, matches: () => false } });
  const clickControl = control => {
    const event = {
      target: { closest: selector => selector === '[hx-get][hx-target="#explorer-results"]' ? control : null },
      preventDefault() { this.prevented = true; },
      stopPropagation() { this.stopped = true; },
    };
    app.documentEvents.emit('click', event);
    return event;
  };
  clickControl(first);
  assert.equal(submits, 1);

  const requestConfig = {};
  app.body.emit('htmx:beforeRequest', { detail: { elt: form, requestConfig } });
  app.body.emit('htmx:afterSettle', {
    detail: { target: { id: 'explorer-results' }, elt: { id: 'finder-controls' }, requestConfig },
  });
  assert.deepEqual(calls, []);
  app.body.emit('htmx:afterSettle', {
    detail: { target: { id: 'explorer-results' }, elt: { id: 'explorer-results' }, requestConfig: {} },
  });
  assert.deepEqual(calls, []);

  const secondClick = clickControl(second);
  assert.equal(secondClick.prevented, true);
  app.body.emit('htmx:afterSettle', {
    detail: { target: { id: 'explorer-results' }, elt: { id: 'explorer-results' }, requestConfig },
  });
  assert.deepEqual(calls, [['GET', '/traces?limit=50']]);
});

function withNewYorkTime(callback) {
  const previous = process.env.TZ;
  process.env.TZ = 'America/New_York';
  try {
    callback();
  } finally {
    if (previous === undefined) delete process.env.TZ;
    else process.env.TZ = previous;
  }
}

test('explorer submission rejects a nonexistent local time during the spring DST jump', () => withNewYorkTime(() => {
  const { app, evt, fromTime } = submitExplorerQuery({
    mode: 'exact',
    now: '2026-03-08T12:00:00',
    start: '2026-03-08T02:30:00',
    end: '2026-03-08T04:00:00',
  });
  assert.equal(evt.prevented, true);
  assert.equal(evt.stopped, true);
  assert.equal(fromTime.validationMessage, 'Enter a valid local time');
  assert.ok(app);
}));

function invalidExplorerRangeFixture() {
  const validity = {};
  const reports = [];
  const field = (id, value) => ({
    id, value,
    setCustomValidity(message) { validity[id] = message; },
    reportValidity() { reports.push(id); return !validity[id]; },
  });
  const from = field('explorer-from', '2026-03-08');
  const fromTime = field('explorer-from-time', '02:30:00');
  const to = field('explorer-to', '2026-03-08');
  const toTime = field('explorer-to-time', '04:00:00');
  const form = { id: 'explorer-load-form' };
  const mode = { value: 'exact' };
  const app = loadDashboard({
    elements: new Map([
      ['explorer-load-form', form], ['explorer-from', from], ['explorer-from-time', fromTime],
      ['explorer-to', to], ['explorer-to-time', toTime],
    ]),
    query: selector => selector === '[data-explorer-range-mode]' ? mode : null,
    queryAll: selector => selector === '[data-explorer-tz]' ? [] : [],
  });
  return { app, form, fromTime, toTime, validity, reports };
}

test('direct explorer load rejects nonexistent local time and reports the invalid field', () => withNewYorkTime(() => {
  const { app, form, fromTime, validity, reports } = invalidExplorerRangeFixture();
  const event = {
    target: form,
    preventDefault() { this.prevented = true; },
    stopImmediatePropagation() { this.stopped = true; },
  };
  app.documentEvents.emit('submit', event);
  assert.equal(event.prevented, true);
  assert.equal(event.stopped, true);
  assert.equal(validity[fromTime.id], 'Enter a valid local time');
  assert.deepEqual(reports, [fromTime.id]);
}));

test('Apply filters rejects nonexistent local time before the hx-post can submit', () => withNewYorkTime(() => {
  const { app, fromTime, validity, reports } = invalidExplorerRangeFixture();
  const button = { closest: selector => selector === '[hx-post="/find/load"]' ? button : null };
  const event = {
    target: button,
    preventDefault() { this.prevented = true; },
    stopImmediatePropagation() { this.stopped = true; },
  };
  app.documentEvents.emit('click', event);
  assert.equal(event.prevented, true);
  assert.equal(event.stopped, true);
  assert.equal(validity[fromTime.id], 'Enter a valid local time');
  assert.deepEqual(reports, [fromTime.id]);
}));

test('Picking a model from a list tells the form the hidden value changed', () => {
  const app = loadDashboard();
  const events = [];
  const hidden = { value: '', dispatchEvent(event) { events.push([event.type, event.bubbles, this.value]); } };
  const button = { textContent: '', focus() {} };
  const sub = { getAttribute: () => 'model' };
  const classes = { contains: () => true, toggle() {} };
  const option = {
    hasAttribute: () => false,
    classList: classes,
    getAttribute: name => (name === 'data-model' ? 'gpt-5.6-luna' : null),
    setAttribute() {},
    closest: selector => ({ '.model-pick .model-option': option, '.model-pick': pick, '.facet-sub': sub })[selector] ?? null,
  };
  const pick = {
    querySelector: selector => ({ 'input[type="hidden"]': hidden, '.model-pick-btn': button })[selector] ?? null,
    querySelectorAll: () => [],
  };
  const target = { closest: selector => (selector === '.model-pick .model-option' ? option : null) };
  app.body.emit('click', { target });
  assert.equal(hidden.value, 'gpt-5.6-luna');
  assert.deepEqual(events, [['change', true, 'gpt-5.6-luna']]);
});
