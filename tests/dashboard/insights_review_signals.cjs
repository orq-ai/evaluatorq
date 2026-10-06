const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const signalsSource = fs.readFileSync('src/evaluatorq/dashboard/static/insights-signals.js', 'utf8');
const signalsContext = {window: {}};
vm.createContext(signalsContext);
vm.runInContext(signalsSource, signalsContext);
const component = signalsContext.window.EvaluatorqSignals;
assert.ok(component, 'the signal renderer should be exposed as window.EvaluatorqSignals');

const report = {config_version: 'v1', results: {
  retry: {name: 'retry', group: 'B', value: 0, approximate: true, reason: 'retried', rule_version: 'r1'},
  autonomy: {name: 'autonomy', group: 'C', value: false, approximate: false},
  unknown_result: {name: 'unknown_result', group: 'C', value: 'unknown'},
  unavailable: {name: 'unavailable', group: 'C', value: null, no_basis: 'unsupported source'},
  tool_loop: {name: 'tool_loop', group: 'D', value: true, reason: 'loop detected'},
  unnecessary_tool: {name: 'unnecessary_tool', group: 'D', value: false},
  unsupported_tag: {name: 'unsupported_tag', group: 'D', value: null, no_basis: 'missing tool activity'},
  approximate_tag: {name: 'approximate_tag', group: 'B', value: 1, approximate: true},
  tag_score: {name: 'tag_score', group: 'D', value: 0},
}};

function render(level, detail = null) {
  return component.render({report, detail, level});
}

const defaultPanel = render('L4');
assert.match(defaultPanel, /L4 Tags/);
for (const [level, label] of [['L1', 'Structure'], ['L2', 'Tools'], ['L3', 'Autonomy'], ['L4', 'Tags']]) {
  assert.ok(defaultPanel.includes(`data-signal-level="${level}"`), `level selector includes ${level}`);
  assert.ok(defaultPanel.includes(`title="${label}"`), `level selector names ${label}`);
}
for (const level of ['L1', 'L2', 'L3', 'L4', 'all']) {
  assert.ok(defaultPanel.includes(`data-signal-level="${level}"`), `level selector includes ${level}`);
}
assert.match(defaultPanel, /tool_loop/);
assert.match(defaultPanel, /Flagged/);
assert.match(defaultPanel, /unnecessary_tool/);
assert.match(defaultPanel, /Clear/);
assert.match(defaultPanel, /unsupported_tag/);
assert.match(defaultPanel, /No basis/);
assert.match(defaultPanel, /tag_score/);
assert.match(defaultPanel, />0</);
assert.doesNotMatch(defaultPanel, /unsupported source/);
assert.doesNotMatch(defaultPanel, /retry/);

const toolsPanel = render('L2');
assert.match(toolsPanel, /retry/);
assert.match(toolsPanel, />0</, 'numeric zero is preserved as a signal value');
assert.match(toolsPanel, /Approximate/);
assert.match(toolsPanel, /approximate tag/i);
assert.doesNotMatch(toolsPanel, /tool_loop/);

const mounted = {
  innerHTML: '',
  buttons: null,
  querySelectorAll(selector) {
    assert.equal(selector, '[data-signal-level]');
    this.buttons ||= [...this.innerHTML.matchAll(/<button[^>]*data-signal-level="([^"]+)"/g)]
      .map(match => ({dataset: {signalLevel: match[1]}, onclick: null}));
    return this.buttons;
  },
};
component.mount(mounted, {report, level: 'L2'});
mounted.querySelectorAll('[data-signal-level]').find(button => button.dataset.signalLevel === 'L3').onclick();
assert.match(mounted.innerHTML, /data-signal-level="L3" aria-pressed="true"/);
assert.match(mounted.innerHTML, /unknown_result/);

const autonomyPanel = render('L3');
assert.match(autonomyPanel, /autonomy/);
assert.match(autonomyPanel, />false</, 'boolean false is preserved as a signal value');
assert.match(autonomyPanel, /unknown/);
assert.match(autonomyPanel, /unsupported source/);
assert.match(autonomyPanel, /No basis/);
assert.doesNotMatch(autonomyPanel, /retry/);

assert.match(render('L1'), /No L1 signals were recorded/);
const detailsPanel = render('L4', {signals: {results: {
  tool_loop: {evidence: [{step_id: 7, reason: 'loop'}], preconditions: [
    {name: 'tool activity', met: true, required: true, detail: 'present'},
    {name: 'minimum turns', met: false, required: true, detail: 'not reached'},
    {name: 'trajectory coverage', met: 'partial', required: false, detail: 'partial source'},
  ]},
}}});
for (const phrase of ['tool_loop', 'Step 7', 'Preconditions', 'Met', 'Not met', 'Partial']) {
  assert.ok(detailsPanel.includes(phrase), `expanded signal details include ${phrase}`);
}
assert.match(detailsPanel, /<code class="signal-code">tool_loop<\/code>/);
assert.match(detailsPanel, /<li class="met">/);
assert.match(detailsPanel, /<li class="not-met">/);
assert.match(detailsPanel, /<li class="partial">/);

const unsafeName = 'bad_<img src=x onerror=alert(1)>';
const unsafeReport = {results: {
  [unsafeName]: {group: 'D', value: true, reason: '<script>alert("signal")</script>'},
}};
const escapedPanel = component.render({report: unsafeReport, level: 'L4'});
assert.ok(escapedPanel.includes('data-signal-name="bad_&lt;img src=x onerror=alert(1)&gt;"'));
assert.ok(escapedPanel.includes('<code class="signal-code">bad_&lt;img src=x onerror=alert(1)&gt;</code>'));
assert.ok(escapedPanel.includes('&lt;script&gt;alert(&quot;signal&quot;)&lt;/script&gt;'));
assert.doesNotMatch(escapedPanel, /<script>alert\("signal"\)<\/script>/);

const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.js', 'utf8');
const start = source.indexOf('function signalValue(value) {');
const end = source.indexOf('function tracePanel(t) {', start);
assert.ok(start >= 0 && end > start, 'review signal controller functions should exist');
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const requests = [], pending = [];
let renders = 0;
const context = {
  esc,
  window: {EvaluatorqSignals: component},
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
assert.match(legacy, /No signals were recorded/);
assert.match(legacy, /L4 Tags/);

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
    tool_loop: {name: 'tool_loop', group: 'D', value: true},
  }},
};
const panel = context.signalsPanel(summary);
assert.match(panel, /L4 Tags/);
assert.match(panel, /tool_loop/);
assert.doesNotMatch(panel, /retry/);
assert.doesNotMatch(panel, /unsupported source/);
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
  pending[3].resolve({ok: true, async json() { return {trace_id: 'otel-trace-c', span_id: 'span-c', signals: {results: {tool_loop: {evidence: [{step_id: 7}], preconditions: [{name: 'tool', met: true}]}}}}; }});
  await flush();
  const detailed = context.signalsPanel(traceC);
  assert.match(detailed, /Step 7/);
  assert.match(detailed, /Preconditions/);
  console.log('signal level selection, visual status, details, lazy loading, failure, cache, and stale selection passed');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
