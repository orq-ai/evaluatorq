const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.js', 'utf8').replace(/\r\n/g, '\n');
const context = {
  DIMS: [],
  S: {dim: null, view: 'themes', sel: null, more: 20, q: '', filters: [], cmp: 'l:outcome'},
  D: {dims: {}},
  T: [{id: 'trace-1', ts: '2026-09-30T10:00:00Z', request: 'Fix the build', l: {outcome: ['done']}, errors: [], has_summary: true, has_tool_stats: true, tools: {}, skills: {}, commands: {}}],
  LABELS: [{name: 'outcome', kind: 'choice', criteria: {done: 'Done', partial: 'Partial'}}],
  LBY: {outcome: {name: 'outcome', kind: 'choice', criteria: {done: 'Done', partial: 'Partial'}}},
  C: {}, byId: {'trace-1': null}, N: 1, TOP_K: 8, OTHER: '#bdbbb5', ERR_YES: '#df5325', ERR_NO: '#9fcfc7',
};
context.byId['trace-1'] = context.T[0];
const canvas = {innerHTML: '', querySelectorAll: () => []};
const heads = {innerHTML: '', querySelectorAll: () => []};
const tlist = {innerHTML: '', querySelectorAll: () => []};
const query = {oninput: null};
context.$ = id => id === 'canvas' ? canvas : id === 'heads' ? heads : id === 'tlist' ? tlist : id === 'q' ? query : null;
context.visible = () => context.T;
context.baseVisible = () => context.T;
context.stats = () => ({n: 1, errShare: null, errN: 0, summN: 1, doneShare: 1, riskyShare: null, unfixedN: 0, unfixedShare: null, frusMean: null, frusMeasured: 0, frusN: 0, frusShare: null});
context.MIN_N = 5; context.PD = null; context.OUTL = context.LBY.outcome; context.FRUSL = null; context.TASKL = null;
context.isOn = () => false;
context.clusterOf = () => { throw new Error('dimensionless traces must not be grouped'); };
context.esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
context.chip = () => 'Done'; context.outcomeHtml = () => 'Done'; context.frusHtml = () => '—'; context.FLAG_DEFS = [];
context.frus = () => null; context.flagCell = () => '—';
context.hasErr = t => t.errors.length > 0; context.isOn = () => false;
context.lvals = () => ['done', 'partial']; context.ldisp = (_name, value) => value === 'done' ? 'Done' : value; context.lcolor = () => '#2ebd85';
context.lval = (trace, name) => trace.l[name]?.[0] ?? null;
context.categoryColumns = values => values; context.withOther = values => values; context.bases = () => [];
context.visible = () => context.T; context.compareEligible = () => true; context.pct = (a, b) => b ? Math.round(a / b * 100) : 0;
context.esc = context.esc; context.human = value => value; context.commit = () => {};
vm.createContext(context);
const datesStart = source.indexOf('const reviewDate = value => {');
const datesEnd = source.indexOf('const costLabel =', datesStart);
assert.ok(datesStart >= 0 && datesEnd > datesStart, 'review date formatting helpers are present');
vm.runInContext(source.slice(datesStart, datesEnd), context);
const validDate = '2026-09-30T10:00:00Z';
assert.equal(vm.runInContext('fmtDate(null)', context), 'Time unavailable');
assert.equal(vm.runInContext("fmtDate('')", context), 'Time unavailable');
assert.equal(vm.runInContext("fmtDate('not a timestamp')", context), 'Time unavailable');
assert.equal(vm.runInContext('fmtTime(null)', context), 'Time unavailable');
assert.equal(vm.runInContext("fmtTime('')", context), 'Time unavailable');
assert.equal(vm.runInContext("fmtTime('not a timestamp')", context), 'Time unavailable');
assert.equal(
  vm.runInContext(`fmtDate('${validDate}')`, context),
  new Date(validDate).toLocaleDateString('en-GB', {day: 'numeric', month: 'short'}),
);
assert.equal(
  vm.runInContext(`fmtTime('${validDate}')`, context),
  new Date(validDate).toLocaleString('en-GB', {day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'}),
);

function extract(startMarker, endMarker) {
  const start = source.indexOf(startMarker), end = source.indexOf(endMarker, start);
  assert.ok(start >= 0 && end > start, `${startMarker} is present`);
  vm.runInContext(source.slice(start, end), context);
}
extract('function renderThemes() {', 'function pickCluster(');
extract('const CANDIDATES = [', 'let HEADS = [];');
extract('let HEADS = [];', '/* ---------- filter bar ---------- */');
assert.equal(typeof context.renderHeadlines, 'function', 'production renderHeadlines is evaluated from the source');
extract('function cmpCols() {', '/* ---------- trace list ---------- */');
extract('function renderTraces() {', '/* ---------- right panel ---------- */');

context.renderThemes();
assert.match(canvas.innerHTML, /No themes in this run/);
assert.match(canvas.innerHTML, /labels and traces are still available/);
context.renderHeadlines();
assert.match(heads.innerHTML, /Tasks were completed/);
assert.doesNotMatch(heads.innerHTML, /undefined/);
context.renderTraces();
assert.match(tlist.innerHTML, /Not grouped/);
assert.match(tlist.innerHTML, /Fix the build/);
assert.doesNotMatch(tlist.innerHTML, /undefined theme/);
context.T[0].ts = null;
context.renderTraces();
assert.match(tlist.innerHTML, /Time unavailable/);
assert.doesNotMatch(tlist.innerHTML, /1 Jan/);
context.renderCompare();
assert.match(canvas.innerHTML, /How all traces break down/);
assert.match(canvas.innerHTML, /All traces/);
assert.match(canvas.innerHTML, /Done/);
const startup = source.slice(source.lastIndexOf('\nfromUrl();\nrenderHeader();'), source.lastIndexOf('\n})();'));
assert.ok(startup.includes('render();'), 'review startup is present');
Object.assign(context, {
  fromUrl() {}, renderHeader() {}, bindMock() {}, render() {}, openRerun() {},
  location: {hash: ''}, document: {}, tops() { throw new Error('no dimension exists'); },
});
vm.runInContext(startup, context);
console.log('Dimensionless Insights review checks passed');

const tipsStart = source.indexOf('function bindTips(root) {');
const tipsEnd = source.indexOf('let toastT;', tipsStart);
extract('function bindTips(root) {', 'let toastT;');
const listeners = {};
const tip = {style: {}, classList: {add() {}, remove() {}}, replaceChildren(...nodes) { this.nodes = nodes; }};
const element = {dataset: {tt: '<img src=x onerror=alert(1)>', tt2: '<script>bad()</script>'}, addEventListener(name, fn) { listeners[name] = fn; }};
context.$ = id => id === 'tip' ? tip : null;
context.document = {createElement(tag) { return {tag, className: '', textContent: ''}; }};
context.innerWidth = 1000;
context.bindTips({querySelectorAll: () => [element]});
listeners.mousemove({clientX: 10, clientY: 20});
assert.equal(tip.nodes[0].textContent, '<img src=x onerror=alert(1)>');
assert.equal(tip.nodes[1].textContent, '<script>bad()</script>');
assert.equal(tip.nodes[0].innerHTML, undefined, 'trace supplied tooltip text is inserted as text');
assert.equal(tipsStart >= 0 && tipsEnd > tipsStart, true);
