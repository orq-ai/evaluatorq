const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {Document} = require('./minidom.cjs');

const signalsSource = fs.readFileSync('src/evaluatorq/dashboard/static/insights-signals.js', 'utf8');
const signalsContext = {window: {}};
vm.createContext(signalsContext);
vm.runInContext(fs.readFileSync('src/evaluatorq/dashboard/static/ui-components.js', 'utf8'), signalsContext);
vm.runInContext(signalsSource, signalsContext);
const component = signalsContext.window.EvaluatorqSignals;
assert.ok(component, 'the shared signal view should be exposed');

const report = {config_version: 'v1', results: {
  tool_counts: {name: 'tool_counts', group: 'A', value: {exec: 2, '<img src=x onerror=alert(1)>': 0}},
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
const detail = {signals: {results: {
  retry: {evidence: [{step_id: 4, reason: 'retry evidence'}], preconditions: [{name: 'retry condition', met: true, required: true, detail: 'passed'}]},
  tool_loop: {evidence: [{step_id: 7, reason: 'loop evidence'}], preconditions: [
    {name: 'tool activity', met: true, required: true, detail: 'present'},
    {name: 'minimum turns', met: false, required: true, detail: 'not reached'},
    {name: 'trajectory coverage', met: 'partial', required: false, detail: 'partial source'},
  ]},
}}};

const panel = component.render({report, detail});
assert.match(panel, /class="signal-browser eq-foldout"[^>]*data-signal-key="browser"/);
assert.match(panel, /<details class="signal-browser eq-foldout"[^>]*data-signal-key="browser">/,
  'the complete signal browser starts collapsed');
assert.match(panel, /data-signal-focus="tool_loop"/);
assert.doesNotMatch(panel, /data-signal-focus="(?:retry|approximate_tag|unnecessary_tool|unsupported_tag|tag_score)"/,
  'compact drilldown buttons are reserved for known true D flags');
assert.match(panel, /unknown_result/);
assert.match(panel, /Flagged/);
assert.match(panel, /Clear/);
assert.match(panel, /No basis/);
assert.match(panel, /tag_score/);
assert.match(panel, />0</, 'numeric zero remains a recorded signal value');
assert.match(panel, />false</, 'boolean false remains a recorded signal value');
assert.match(panel, /Approximate/);
assert.match(panel, /class="eq-result [^"]+"/);

for (const level of ['L1', 'L2', 'L3', 'L4']) {
  assert.match(panel, new RegExp(`<details class="signal-level [^"]*"[^>]*data-signal-level="${level}" data-signal-key="level:${level}"`),
    `${level} is an independently collapsible level`);
}
assert.match(panel, /<details class="signal-level [^"]*"[^>]*data-signal-level="L1" data-signal-key="level:L1">/,
  'levels start collapsed');
assert.match(panel, /data-signal-key="row:tool_loop"/);
assert.match(panel, /data-signal-key="row:retry"/);
assert.match(panel, /data-signal-key="evidence:tool_loop"/);
assert.match(panel, /<details class="signal-evidence-section eq-foldout"[^>]*data-signal-key="evidence:tool_loop">/,
  'evidence starts collapsed independently from its signal row');
assert.match(panel, /class="signal-row[^"]*"[^>]*data-signal-name="tool_loop"/);
const dom = new Document(panel);
const browser = dom.querySelector('details[data-signal-key="browser"]');
assert.ok(browser);
assert.equal(browser.hasAttribute('open'), false);
for (const level of ['L1', 'L2', 'L3', 'L4']) {
  const section = dom.querySelector(`details[data-signal-key="level:${level}"]`);
  assert.ok(section);
  assert.equal(section.closest('details.signal-browser'), browser, `${level} sits inside the browser`);
  assert.equal(section.hasAttribute('open'), false, `${level} starts folded`);
}
const toolLoopRow = dom.querySelector('details[data-signal-key="row:tool_loop"]');
const toolLoopLevel = dom.querySelector('details[data-signal-key="level:L4"]');
assert.equal(toolLoopRow.closest('details.signal-level'), toolLoopLevel);
for (const row of toolLoopLevel.querySelectorAll('.signal-row')) {
  assert.equal(row.querySelector('code.signal-code'), null, 'L4 rows do not repeat each tag name as a code header');
}
const zeroTagRow = dom.querySelector('details[data-signal-key="row:tag_score"]');
assert.equal(zeroTagRow.querySelector('.eq-result-value').textContent, '0');
assert.ok(zeroTagRow.querySelector('.signal-badge.measured'));
const countsRow = dom.querySelector('details[data-signal-key="row:tool_counts"]');
assert.equal(countsRow.querySelector('.eq-result-value').textContent, '2 entries');
assert.ok(panel.includes('&lt;img src=x onerror=alert(1)&gt;'), 'object keys are escaped in the breakdown');
assert.doesNotMatch(panel, /<img src=x onerror=alert\(1\)>/);
const countKeys = countsRow.querySelectorAll('dt').map(node => node.textContent);
const countValues = countsRow.querySelectorAll('dd').map(node => node.textContent);
assert.deepEqual(countKeys, ['exec', '<img src=x onerror=alert(1)>']);
assert.deepEqual(countValues, ['2', '0'], 'object breakdown preserves nonzero and numeric zero values');
const toolLoopEvidence = dom.querySelector('details[data-signal-key="evidence:tool_loop"]');
assert.equal(toolLoopEvidence.closest('details.signal-row'), toolLoopRow);
assert.equal(toolLoopEvidence.hasAttribute('open'), false, 'evidence has its own folded state');

const rowStart = panel.indexOf('data-signal-key="row:tool_loop"');
const evidenceStart = panel.indexOf('data-signal-key="evidence:tool_loop"', rowStart);
const rowMarkup = panel.slice(rowStart, evidenceStart);
assert.match(rowMarkup, /Preconditions/);
assert.match(rowMarkup, /Met/);
assert.match(rowMarkup, /Not met/);
assert.match(rowMarkup, /Partial/);
assert.doesNotMatch(rowMarkup, /Step 7|loop evidence/,
  'preconditions remain visible while evidence is folded');

const noFlags = component.render({report: {results: {
  approximate: {group: 'B', value: 1, approximate: true},
  clear: {group: 'D', value: false},
  unknown: {group: 'D', value: null},
}}});
assert.doesNotMatch(noFlags, /data-signal-focus=/);
assert.match(noFlags, /No flagged L4 signals|No signals were recorded|No signal flags|No flags/i);
assert.match(noFlags, /unknown/);

const missingValuePanel = new Document(component.render({report: {results: {missing: {group: 'B', value: null}}}}));
assert.equal(missingValuePanel.querySelector('.eq-result-value').textContent, 'No value',
  'the primary result keeps an absent value distinct from an explicit no-basis reason');

const emptyPanel = component.render({report: {results: {}}});
assert.match(emptyPanel, /No flagged L4 signals|No signals were recorded|No signal flags|No flags/i);
const missingPanel = component.render({report: null});
assert.match(missingPanel, /Signals were not measured for this trace/);

const unsafeName = 'bad_<img src=x onerror=alert(1)>';
const unsafePanel = component.render({report: {results: {
  [unsafeName]: {group: 'D', value: true, reason: '<script>alert("signal")</script>'},
}}});
assert.ok(unsafePanel.includes('data-signal-name="bad_&lt;img src=x onerror=alert(1)&gt;"'));
assert.ok(unsafePanel.includes('&lt;script&gt;alert(&quot;signal&quot;)&lt;/script&gt;'));
assert.doesNotMatch(unsafePanel, /<script>alert\("signal"\)<\/script>/);

function fakeDetails(openKeys) {
  const nodes = [...openKeys].map(key => ({
    dataset: {signalKey: key}, open: true,
    getAttribute(name) { return name === 'data-signal-key' ? key : null; },
  }));
  return {nodes, querySelectorAll(selector) {
    assert.ok(['details[data-signal-key][open]', 'details[data-signal-key]'].includes(selector));
    return selector.endsWith('[open]') ? this.nodes.filter(node => node.open) : this.nodes;
  }};
}
const stateContainer = fakeDetails(['browser', 'level:L4', 'row:tool_loop']);
const openState = component.captureOpen(stateContainer);
assert.deepEqual([...openState].sort(), ['browser', 'level:L4', 'row:tool_loop']);
stateContainer.nodes.forEach(node => { node.open = false; });
component.restoreOpen(stateContainer, openState);
assert.deepEqual(stateContainer.nodes.filter(node => node.open).map(node => node.dataset.signalKey).sort(),
  ['browser', 'level:L4', 'row:tool_loop']);

const browserNode = {tagName: 'DETAILS', open: false, parentElement: null};
const levelNode = {tagName: 'DETAILS', open: false, parentElement: browserNode};
const rowSummary = {focused: false, focusOptions: null, focus(options) { this.focused = true; this.focusOptions = options; }};
const rowNode = {
  tagName: 'DETAILS', open: false, dataset: {signalName: 'tool_loop'}, parentElement: levelNode,
  querySelector(selector) { assert.equal(selector, 'summary'); return rowSummary; },
  scrollIntoView(options) { this.scrollOptions = options; },
};
const focusButton = {dataset: {signalFocus: 'tool_loop'}, onclick: null};
const bindContainer = {
  querySelectorAll(selector) {
    if (selector === '[data-signal-focus]') return [focusButton];
    if (selector === '.signal-row') return [rowNode];
    throw new Error(`unexpected selector ${selector}`);
  },
};
component.bind(bindContainer);
assert.equal(typeof focusButton.onclick, 'function');
focusButton.onclick();
assert.equal(browserNode.open, true);
assert.equal(levelNode.open, true);
assert.equal(rowNode.open, true);
assert.equal(rowSummary.focused, true);
assert.equal(rowSummary.focusOptions.preventScroll, true);
assert.equal(rowNode.scrollOptions.block, 'nearest');

const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.js', 'utf8');
const start = source.indexOf('function signalValue(value) {');
const end = source.indexOf('function tracePanel(t) {', start);
assert.ok(start >= 0 && end > start, 'review signal loading functions should exist');
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

const legacy = context.signalsPanel({has_signals: true, has_signal_details: false, signals: {results: {}}});
assert.match(legacy, /No flagged L4 signals|No signals were recorded|No signal flags|No flags/i);
assert.match(legacy, /L4/);

const summary = {
  has_signals: true,
  has_signal_details: true,
  id: 'trace-a', trace_id: 'otel-trace-a', span_id: 'span-a', signal_detail_url: '/detail/a',
  signals: {config_version: 'v1', results: {
    retry: {name: 'retry', group: 'B', value: 2, approximate: true, reason: 'retried', rule_version: 'r1'},
    unavailable: {name: 'unavailable', group: 'C', value: null, no_basis: 'unsupported source', approximate: false},
    tool_loop: {name: 'tool_loop', group: 'D', value: true},
  }},
};
const drawerPanel = context.signalsPanel(summary);
assert.match(drawerPanel, /L4/);
assert.match(drawerPanel, /tool_loop/);
assert.doesNotMatch(drawerPanel, /data-signal-focus="retry"/);
assert.match(drawerPanel, /<details class="signal-browser eq-foldout"[^>]*data-signal-key="browser">/,
  'the review drawer keeps the signal hierarchy folded until opened');
assert.doesNotMatch(drawerPanel, /Step 5|Source coverage/);

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
  console.log('signal hierarchy, flag drilldown, state restoration, details loading, failure, cache, and stale selection passed');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
