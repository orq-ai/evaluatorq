const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.js', 'utf8');
const start = source.indexOf('function fromUrl() {');
const end = source.indexOf('function commit()', start);
assert.ok(start >= 0 && end > start, 'URL restoration implementation is present');
const state = {
  dim: 'intent', view: 'themes', filters: [], sel: null, colorBy: 'l:sentiment', cmp: 'd:failure',
  activityKind: 'tools', activityQuery: 'old', activitySort: 'calls', activityItem: null,
  activityHelpers: true, activityExpanded: new Set(['git']), activityShowAll: true, q: 'old',
};
const context = {
  URLSearchParams,
  D: {dims: {intent: {}, failure: {}}},
  LBY: {outcome: {}, sentiment: {}},
  OUTL: {name: 'outcome'},
  S: state,
  C: { 'base-2': {id: 'base-2'} },
  byId: Object.create(null),
  XF: Object.create(null),
  T: [],
  decF: () => null,
  location: {hash: '#view=map&dim=intent&color=l%3Asentiment&cmp=d%3Afailure'},
  lvals: () => ['yes', 'no'],
};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);

context.fromUrl();
assert.equal(state.colorBy, 'l:sentiment', 'explicit map color restores');
assert.equal(state.cmp, 'd:failure', 'explicit comparison restores');

context.location.hash = '#view=themes&dim=intent';
context.fromUrl();
assert.equal(state.colorBy, 'cluster', 'omitted color resets to the map default');
assert.equal(state.cmp, 'l:outcome', 'omitted compare axis resets to the default');
console.log('URL state restoration checks passed');

const compareStart = source.indexOf('function cmpCols() {');
const compareEnd = source.indexOf('/* ---------- trace list ---------- */', compareStart);
assert.ok(compareStart >= 0 && compareEnd > compareStart, 'Map Compare implementation is present');
const compareCanvas = {innerHTML: '', querySelectorAll: () => []};
Object.assign(context, {
  TOP_K: 8,
  OTHER: '#bdbbb5',
  ERR_YES: '#df5325',
  D: {dims: {intent: {clusters: []}}},
  LBY: {sentiment: {kind: 'choice'}},
  S: {dim: 'intent', cmp: 'l:sentiment', filters: []},
  C: {},
  T: [
    {id: 'span:1', l: {sentiment: ['positive']}, has_summary: true, errors: []},
    {id: 'span:2', l: {}, has_summary: true, errors: []},
    {id: 'span:3', l: {sentiment: ['positive']}, has_summary: true, errors: []},
  ],
  bases: () => [{id: 'intent-a', name: 'Intent A', color: '#111111', members: new Set(['span:1', 'span:2', 'span:3'])}],
  withOther: values => values,
  visible: () => context.T,
  lvals: () => ['positive', 'neutral', 'negative'],
  lval: (trace, name) => trace.l[name]?.[0] ?? null,
  ldisp: (_name, value) => value,
  lcolor: () => '#2ebd85',
  hasErr: trace => trace.errors.length > 0,
  esc: value => String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char])),
  human: value => value,
  hexA: () => 'rgba(0,0,0,.5)',
  $: id => id === 'canvas' ? compareCanvas : null,
  fkey: () => '',
  commit: () => {},
  toast: () => {},
});
vm.runInContext(source.slice(compareStart, compareEnd), context);
context.renderCompare();
assert.match(compareCanvas.innerHTML, /2 eligible<\/small>/, 'traces missing either compared value leave the denominator');
assert.match(compareCanvas.innerHTML, /100% · 2/, 'cell share uses only traces with both values');
assert.match(compareCanvas.innerHTML, /style="background:rgba\(46,189,133,0\.52\)"/, 'a full-share cell receives a strong tint from its column');
assert.match(compareCanvas.innerHTML, /No traces with both values/, 'zero cells explain missing intersections');

context.S.cmp = 'err';
context.T = Array.from({length: 8}, (_, index) => ({id: `span:${index}`, has_summary: true, errors: index === 0 ? ['error'] : []}));
context.bases = () => [{id: 'intent-a', name: 'Intent A', color: '#111111', members: new Set(context.T.map(trace => trace.id))}];
context.renderCompare();
assert.match(compareCanvas.innerHTML, /style="background:rgba\(223,83,37,0\.17\)"[^>]*>1<small>13% · 8/, 'a 13% error cell uses a light error tint');
assert.match(compareCanvas.innerHTML, /style="background:rgba\(41,157,143,0\.47\)"[^>]*>7<small>88% · 8/, 'an 88% no-error cell uses a stronger success tint');

const folded = context.categoryColumns(Array.from({length: 12}, (_, index) => ({
  label: `value-${index}`,
  color: '#111111',
  test: trace => trace.value === index,
  f: {t: 'l', name: 'label', v: String(index)},
})));
assert.equal(folded.length, 9, 'comparison categories keep eight values and a folded Other column');
assert.match(folded[8].label, /Other \(4 categories\)/);
assert.equal(folded[8].f, null, 'the aggregated Other column cannot create an incorrect single-value filter');
console.log('Map Compare denominator and category checks passed');

// Folded cells still filter both axes. With one folded axis, the concrete
// axis stays a normal cluster filter and `b` stores the folded membership.
const compareDims = ['intent', 'review'];
const compareTraces = [];
const compareClusters = Object.fromEntries(compareDims.map(dim => [dim, Array.from({length: 12}, (_, index) => ({
  id: `${dim}-${index}`, dim, name: `${dim} ${index}`, color: '#111111', level: 'base', members: new Set(),
}))]));
for (const row of compareClusters.intent) for (const col of compareClusters.review) {
  const id = `${row.id}:${col.id}`;
  row.members.add(id); col.members.add(id);
  compareTraces.push({id, a: {intent: row.id, review: col.id}, has_summary: true, errors: []});
}
const compareById = Object.fromEntries(compareTraces.map(trace => [trace.id, trace]));
const compareState = {dim: 'intent', cmp: 'd:review', filters: []};
let clickedCell;
const compareCell = {dataset: {}, onclick: null, onkeydown: null};
const foldedCanvas = {innerHTML: '', querySelectorAll: () => { compareCell.dataset = {r: clickedCell.row, j: String(clickedCell.col)}; return [compareCell]; }};
Object.assign(context, {
  TOP_K: 8,
  T: compareTraces,
  byId: compareById,
  C: Object.fromEntries(compareDims.flatMap(dim => compareClusters[dim].map(cluster => [cluster.id, cluster]))),
  D: {dims: {intent: {clusters: compareClusters.intent}, review: {clusters: compareClusters.review}}},
  S: compareState,
  bases: dim => compareClusters[dim],
  withOther: values => values.length <= 9 ? values : [...values.slice(0, 8), {
    id: 'other', other: true, dim: values[0].dim, name: `Other (${values.length - 8} themes)`, color: '#bdbbb5',
    members: new Set(values.slice(8).flatMap(value => [...value.members])),
  }],
  clusterOf: (trace, dim) => context.C[trace.a[dim]],
  visible: () => compareTraces,
  fkey: filter => filter.t === 'c' ? `c:${context.C[filter.id]?.dim || ''}` : filter.t,
  commit: () => {},
  $: id => id === 'canvas' ? foldedCanvas : null,
});
vm.runInContext(source.slice(compareStart, compareEnd), context);
const clickCompareCell = (row, col) => {
  clickedCell = {row, col};
  context.renderCompare();
  const canvas = context.$('canvas');
  const cell = canvas.querySelectorAll()[0];
  assert.equal(typeof cell.onclick, 'function', 'compare cell is clickable');
  cell.onclick();
};

clickCompareCell('intent-0', 8);
assert.deepEqual(compareState.filters.map(filter => filter.t), ['c', 'b'], 'concrete row plus folded column adds both axes');
assert.equal(compareState.filters[0].id, 'intent-0');
assert.equal(compareState.filters[1].ids.size, 4 * 12, 'folded column set includes all member trace keys');
assert.ok([...compareState.filters[1].ids].every(id => id.endsWith(':review-8') || id.endsWith(':review-9') || id.endsWith(':review-10') || id.endsWith(':review-11')));

clickCompareCell('other', 0);
assert.deepEqual(compareState.filters.map(filter => filter.t), ['c', 'b'], 'folded row plus concrete column adds both axes');
assert.equal(compareState.filters[0].id, 'review-0');
assert.equal(compareState.filters[1].ids.size, 4 * 12, 'folded row set includes all member trace keys');
assert.ok([...compareState.filters[1].ids].every(id => id.startsWith('intent-8:') || id.startsWith('intent-9:') || id.startsWith('intent-10:') || id.startsWith('intent-11:')));

clickCompareCell('other', 8);
assert.deepEqual(compareState.filters.map(filter => filter.t), ['b'], 'two folded axes are represented by their intersection');
assert.equal(compareState.filters[0].ids.size, 16, 'two folded axes intersect to the 4 by 4 trace keys');
assert.ok([...compareState.filters[0].ids].every(id => /intent-(8|9|10|11):review-(8|9|10|11)$/.test(id)));
assert.match(foldedCanvas.innerHTML, /16<small>33% · 48<\/small>/, 'folded row and column cell count uses the row eligible denominator');

const filterHelpersStart = source.indexOf('function encodeTraceKey(id) {');
const filterHelpersEnd = source.indexOf('function toUrl()', filterHelpersStart);
vm.runInContext(source.slice(filterHelpersStart, filterHelpersEnd), context);
const encoded = context.encF(compareState.filters[0]);
const restored = context.decF(encoded);
assert.equal(restored.t, 'b', 'URL filter restores as a set filter');
assert.deepEqual([...restored.ids].sort(), [...compareState.filters[0].ids].sort(), 'URL restore keeps the complete two-axis intersection');
console.log('Map Compare folded cell filters and URL restoration checks passed');

const actionStart = source.indexOf('function viewClusterTraces(c) {');
const actionEnd = source.indexOf('function tracePanel(t) {', actionStart);
assert.ok(actionStart >= 0 && actionEnd > actionStart, 'cluster drawer actions are present');
let commits = 0;
const sheetOpens = [];
const actionState = {dim: 'failure', view: 'map', filters: [{t: 'c', id: 'old'}], q: 'old', more: 1, sel: {kind: 'cluster'}, activityItem: {name: 'old'}};
Object.assign(context, {
  S: actionState,
  NEW_RUN_URL: '/insights/new?mount=dialog',
  commit: () => { commits++; },
  openSheet: (...args) => { sheetOpens.push(args); },
});
vm.runInContext(source.slice(actionStart, actionEnd), context);
const savedCluster = {id: 'top-1', dim: 'intent', name: 'Scope creep', members: new Set(['snapshot:trace:1', 'snapshot:trace:2'])};
context.viewClusterTraces(savedCluster);
assert.equal(actionState.dim, 'intent');
assert.equal(actionState.view, 'themes');
assert.equal(actionState.filters.length, 1, 'view action clears conflicting filters');
assert.deepEqual([...actionState.filters[0].ids], ['snapshot:trace:1', 'snapshot:trace:2'], 'view action uses exact saved member keys');
assert.equal(actionState.q, '', 'view action clears search so every matching member appears');
assert.equal(actionState.sel, null);
assert.equal(actionState.more, 20);
assert.equal(commits, 1, 'view action commits the filtered list state');

const draftedQuestions = [];
context.InsightsRunForm = {draftQuestion: (form, draft) => draftedQuestions.push([form, draft])};
context.turnClusterIntoQuestion(savedCluster);
assert.equal(sheetOpens.length, 1, 'question action opens the run form dialog');
assert.equal(sheetOpens[0][0], '/insights/new?mount=dialog');
sheetOpens[0][2]('form');
assert.deepEqual(JSON.parse(JSON.stringify(draftedQuestions)), [['form', {
  name: 'about_scope_creep', kind: 'noul', text: 'Did the agent handle Scope creep well?',
}]], 'the dialog opens the custom question editor with a yes/no draft about the cluster');
assert.match(source, /data-act="cluster-traces"/);
assert.match(source, /data-act="cluster-question"/);
console.log('Coding question labels and cluster drawer actions checks passed');
