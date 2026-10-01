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
    emit(name, event = {}) {
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

function loadDashboard({ elements = new Map(), query = () => null } = {}) {
  const body = emitter();
  const documentEvents = emitter();
  const windowEvents = emitter();
  const scheduled = new Map();
  let nextTimer = 1;
  const document = {
    body,
    addEventListener: documentEvents.addEventListener,
    getElementById(id) { return elements.get(id) || null; },
    querySelector: query,
    querySelectorAll() { return []; },
  };
  const window = {
    addEventListener: windowEvents.addEventListener,
    setTimeout(callback) {
      const id = nextTimer++;
      scheduled.set(id, callback);
      return id;
    },
    clearTimeout(id) { scheduled.delete(id); },
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
    FormData: class {}, Event: class {}, CSS: { escape: value => value },
  });
  return { body, document, history, entries, scheduled };
}

test('open filter dropdown survives a fragment swap', () => {
  const old = { id: 'filter-dd-persona', open: true };
  const replacement = { id: old.id, open: false };
  const untouched = { id: 'filter-dd-scenario', open: false };
  const elements = new Map([
    ['filter-form', { querySelectorAll() { return [old]; } }],
  ]);
  const app = loadDashboard({ elements });

  app.body.emit('htmx:beforeSwap');
  elements.set(old.id, replacement);
  elements.set(untouched.id, untouched);
  app.body.emit('htmx:afterSwap', { detail: { target: null } });

  assert.equal(replacement.open, true);
  assert.equal(untouched.open, false);

  // The next swap has no open dropdown; an old snapshot must not reopen it.
  elements.set('filter-form', { querySelectorAll() { return []; } });
  replacement.open = false;
  app.body.emit('htmx:beforeSwap');
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
