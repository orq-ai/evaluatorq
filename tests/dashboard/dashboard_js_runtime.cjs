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
} = {}) {
  const body = Object.assign(emitter(), { appendChild() {} });
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
    FormData: class {}, Event: class {}, Date: dateClass, CSS: { escape: value => value },
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
  const fields = [...form.elements, ...includedForms.flatMap(included => included.elements)];
  const params = new URLSearchParams();
  for (const field of fields) {
    if (field.name && !field.disabled) params.append(field.name, field.value);
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
  toTime.setCustomValidity = () => {};
  toTime.reportValidity = () => {};
  const rangeMode = input('range_mode', mode);
  const rangeSeconds = input('range_seconds', seconds);
  const rangeFromOffset = input('from_tz_offset', '0');
  const rangeToOffset = input('to_tz_offset', '0');
  const explorerElements = [from, fromTime, to, toTime, rangeMode, rangeSeconds, rangeFromOffset, rangeToOffset];
  const controls = { id: 'finder-controls', elements: fields };
  const explorerForm = { id: 'explorer-load-form', elements: explorerElements };
  const queryForm = {
    id: 'finder-query-form',
    hxInclude: '#finder-controls, #explorer-load-form',
    elements: [input('query', 'slow traces'), input('scope', 'new'), input('mode', 'immediate')],
  };
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
  return { app, body, evt, fields, queryForm };
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

test('finder placeholders rotate on the interval and stop after dismissal', () => {
  const query = {
    dataset: { finderPlaceholders: '["First example", "Second example"]' },
    value: '',
    placeholder: '',
  };
  const app = loadDashboard({ queryAll: () => [query] });

  assert.equal(app.intervals.size, 1);
  const [{ callback, delay }] = app.intervals.values();
  assert.equal(delay, 4000);
  callback();
  assert.equal(query.placeholder, 'Second example');
  query.dataset.finderPlaceholderDismissed = 'true';
  callback();
  assert.equal(query.placeholder, 'Second example');
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
