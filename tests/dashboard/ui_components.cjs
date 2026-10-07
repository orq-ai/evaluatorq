'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {Document} = require('./minidom.cjs');

const source = fs.readFileSync(
  path.resolve(__dirname, '../../src/evaluatorq/dashboard/static/ui-components.js'),
  'utf8'
);
const window = {};
vm.runInContext(source, vm.createContext({window}));
const components = window.EvaluatorqComponents;
assert.ok(components, 'the shared components API is exposed');

function parse(html) {
  const root = new Document('<main>' + html + '</main>').querySelector('main');
  // The shared minidom parses details markup but does not model the native open property.
  root.querySelectorAll('details').forEach(details => {
    let open = details.hasAttribute('open');
    Object.defineProperty(details, 'open', {
      get() { return open; },
      set(value) { open = Boolean(value); if (open) details.setAttribute('open', ''); else details.removeAttribute('open'); },
    });
  });
  return root;
}

assert.equal(components.escapeHtml(`<>&"'`), '&lt;&gt;&amp;&quot;&#39;');

{
  const markup = components.foldout({
    key: ' hostile&" ', className: 'extra', summaryHtml: '<b>Trusted summary</b>',
    bodyHtml: '<em>Trusted body</em>', data: {section: 'a&"<'},
  });
  const root = parse(markup);
  const details = root.querySelector('details');
  assert.ok(details, 'foldout renders native details');
  assert.equal(details.open, false, 'foldout defaults closed');
  assert.equal(details.getAttribute('data-ui-key'), ' hostile&" ');
  assert.equal(details.getAttribute('data-section'), 'a&"<');
  assert.equal(details.getAttribute('data-bad key'), null, 'unrequested unsafe attributes are absent');
  assert.ok(markup.includes('Trusted summary'), 'summary fragments are trusted HTML');
  assert.ok(markup.includes('Trusted body'), 'body fragments are trusted HTML');
  assert.ok(markup.includes('data-ui-key=" hostile&amp;&quot; "'), 'key attributes are escaped');
  assert.ok(markup.includes('data-section="a&amp;&quot;&lt;"'), 'data values are escaped');
  assert.ok(details.classList.contains('eq-foldout'));
  assert.ok(details.classList.contains('extra'));
  assert.equal(parse(components.foldout({key: 'open-me', summaryHtml: 'Summary', open: true})).querySelector('details').open, true);
}

{
  assert.throws(() => components.foldout({key: 'x', summaryHtml: 'x', data: {'onclick"': 'bad'}}), /data names/i);
  assert.throws(() => components.foldout({key: 'x', summaryHtml: 'x', data: {'ui-key': 'override'}}), /ui-key/i);
}

{
  const html = components.resultFeedback({value: 0, outcome: '0', tone: 'positive', reason: '<script>x</script>', bodyHtml: '<b>Details</b>'});
  const root = parse(html);
  const result = root.querySelector('.eq-result');
  assert.ok(result.classList.contains('eq-result--positive'));
  assert.equal(root.querySelector('.eq-result-value').textContent, '0');
  assert.equal(root.querySelector('.eq-result-status'), null, 'duplicate outcome is omitted');
  assert.equal(root.querySelector('[title]'), null);
  assert.ok(html.includes('&lt;script&gt;x&lt;/script&gt;'), 'reason text is escaped');
  assert.ok(html.includes('<b>Details</b>'), 'body fragments are trusted HTML');
  const falseValue = parse(components.resultFeedback({value: false, outcome: 'negative', tone: 'warning'}));
  assert.equal(falseValue.querySelector('.eq-result-value').textContent, 'false');
  assert.equal(falseValue.querySelector('.eq-result-status').textContent, 'negative');
  assert.throws(() => components.resultFeedback({value: 'x', tone: 'danger'}), /tone/i);
  assert.throws(() => components.resultFeedback({value: 'x', tone: 'info onclick=1'}), /tone/i);
}

{
  const root = parse(components.preconditionList([
    {name: 'ready', met: true, required: true, detail: '<script>bad</script>'},
    {name: 'missing', met: false, required: true},
    {name: 'maybe', met: 'partial', required: false},
  ]));
  const list = root.querySelector('.eq-preconditions');
  assert.ok(list);
  assert.ok(list.querySelector('.met'));
  assert.ok(list.querySelector('.not-met'));
  assert.ok(list.querySelector('.partial'));
  assert.equal(list.querySelectorAll('li').length, 3);
  assert.ok(list.textContent.includes('<script>bad</script>'), 'detail is escaped and shown as text');
  const empty = parse(components.preconditionList([]));
  assert.ok(empty.textContent.includes('No preconditions recorded.'));
}

{
  const container = parse('<details data-ui-key="first" open></details><details data-ui-key="second"></details><details open></details><section><details data-ui-key="nested" open></details></section>');
  const state = components.captureOpen(container);
  assert.deepEqual(Array.from(state), ['first', 'nested']);
  const nodes = container.querySelectorAll('details');
  nodes.forEach(node => { node.open = false; });
  // A key absent from the saved state closes keyed details; unkeyed details remain untouched.
  nodes[2].open = true;
  components.restoreOpen(container, ['second', 'unknown']);
  assert.equal(nodes[0].open, false);
  assert.equal(nodes[1].open, true);
  assert.equal(nodes[2].open, true);
  assert.equal(nodes[3].open, false);
  components.restoreOpen(container, ['first'], 'data-other-key');
}

process.stdout.write('ui components behavior passed\n');
