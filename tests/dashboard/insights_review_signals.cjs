const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.js', 'utf8');
const start = source.indexOf('function signalValue(value) {');
const end = source.indexOf('function tracePanel(t) {', start);
assert.ok(start >= 0 && end > start, 'signal rendering functions should exist');
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const requests = [], pending = [];
let renders = 0;
const context = {
  esc,
  S: {sel: null},
  renderDrawer() { renders++; },
  fetch(url, options) { requests.push({url, options}); return new Promise(resolve => pending.push({resolve})); },
  AbortController: class { constructor() { this.signal = {aborted: false}; } abort() { this.signal.aborted = true; } },
};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);

const missing = context.signalsPanel({has_signals: false, signals: null});
assert.match(missing, /Signals were not measured for this trace/);

const legacy = context.signalsPanel({has_signals: true, has_signal_details: false, signals: {results: {}}});
assert.match(legacy, /0 recorded/);
assert.match(legacy, /Signals/);

const summary = {
  has_signals: true,
  has_signal_details: true,
  id: 'trace-a',
  trace_id: 'otel-trace-a',
  span_id: 'span-a',
  signal_detail_url: '/detail/a',
  signals: {config_version: 'v1', results: {
    retry: {name: 'retry', group: 'B', value: 2, approximate: true, reason: 'retried', rule_version: 'r1'},
    unavailable: {name: 'unavailable', group: 'C', value: null, no_basis: 'unsupported source', approximate: false},
    tags: {name: 'tags', group: 'D', value: {coding: true}},
  }},
};
const panel = context.signalsPanel(summary);
assert.match(panel, /3 recorded/);
assert.match(panel, /Approximate/);
assert.match(panel, /No basis/);
assert.match(panel, /Tags/);
assert.match(panel, /Group D/);
assert.match(panel, /Config version/);
assert.match(panel, /Signal details are not loaded/);
for (const forbidden of ['Step 5', 'Preconditions', 'Source coverage']) assert.ok(!panel.includes(forbidden));

async function flush() { for (let i = 0; i < 8; i++) await Promise.resolve(); }
async function main() {
  context.S.sel = {kind: 'trace', id: 'trace-a'};
  context.loadSignalDetails(summary);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, '/detail/a');

  const traceB = {...summary, id: 'trace-b', trace_id: 'otel-trace-b', span_id: 'span-b', signal_detail_url: '/detail/b'};
  context.S.sel = {kind: 'trace', id: 'trace-b'};
  context.loadSignalDetails(traceB);
  assert.equal(requests[0].options.signal.aborted, true, 'selection change aborts old request');
  assert.equal(requests.length, 2);

  pending[1].resolve({ok: true, async json() { return {trace_id: 'otel-trace-b', span_id: 'span-b', signals: {results: {}}, source_coverage: {otel: 'partial'}}; }});
  await flush();
  assert.equal(context.signalDetailsFor('trace-b').source_coverage.otel, 'partial');
  assert.equal(renders, 1, 'completed selected trace rerenders');

  pending[0].resolve({ok: true, async json() { return {trace_id: 'otel-trace-a', span_id: 'span-a', signals: {results: {}}}; }});
  await flush();
  assert.equal(context.signalDetailsFor('trace-a'), null, 'aborted stale response is not cached');
  assert.equal(renders, 1, 'stale response does not rerender selected trace');
  context.loadSignalDetails(traceB);
  assert.equal(requests.length, 2, 'cached details avoid another request');

  const traceC = {...summary, id: 'trace-c', trace_id: 'otel-trace-c', span_id: 'span-c', signal_detail_url: '/detail/c'};
  context.S.sel = {kind: 'trace', id: 'trace-c'};
  context.loadSignalDetails(traceC);
  pending[2].resolve({ok: false, status: 503});
  await flush();
  const failed = context.signalsPanel(traceC);
  assert.match(failed, /Could not load signal details/);
  assert.match(failed, /Retry/);
  context.loadSignalDetails(traceC);
  assert.equal(requests.length, 3, 'failed request is not automatically repeated on rerender');
  context.loadSignalDetails(traceC, true);
  assert.equal(requests.length, 4, 'retry starts a new request');
  pending[3].resolve({ok: true, async json() { return {trace_id: 'otel-trace-c', span_id: 'span-c', signals: {results: {retry: {evidence: [{step_id: 7}], preconditions: [{name: 'tool', met: true}]}}}}; }});
  await flush();
  const detailed = context.signalsPanel(traceC);
  assert.match(detailed, /Step 7/);
  assert.match(detailed, /Preconditions/);
  console.log('signal summaries, lazy loading, failure, cache, and stale selection passed');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
