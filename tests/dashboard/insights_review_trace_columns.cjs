'use strict';
/* The Insights review page shows a question's column, tile or stat only when the run asked it on at least one trace. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {Document} = require('./minidom.cjs');
const makeExtract = require('./extract.cjs');

const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.js', 'utf8').replace(/\r\n/g, '\n');

const FRUS_CRITERIA = ['1 calm', '2 mild', '3 irritated', '4 frustrated', '5 angry'];
const CORR_CRITERIA = ['0 none', '1 once', '2 twice', '3 three or more'];
const label = (name, kind, criteria, counts = {}, n_failed = 0) => ({name, kind, criteria, counts, n_failed, n_not_asked: 0});
const FRUS_ONLY = () => [label('user_frustration', 'score', FRUS_CRITERIA, {1: 2, 4: 1})];
const UNASKED_CODING = () => [
  label('task_type', 'choice', {bugfix: 'Bug fix', feature: 'Feature'}),
  label('outcome', 'choice', {done: 'Done', partial: 'Partial'}),
  label('risky_action', 'choice', {none: 'None', deleted: 'Deleted files'}),
  label('verified', 'choice', {verified: 'v', claimed_without_check: 'c', unverified: 'u', not_applicable: 'n'}),
  label('unfixed_error', 'noul', {}),
  label('user_frustration', 'score', FRUS_CRITERIA),
  label('user_corrections', 'score', CORR_CRITERIA),
];
const FULL_CODING = () => [
  label('task_type', 'choice', {bugfix: 'Bug fix', feature: 'Feature'}, {bugfix: 4}),
  label('outcome', 'choice', {done: 'Done', partial: 'Partial'}, {done: 3, partial: 1}),
  label('risky_action', 'choice', {none: 'None', deleted: 'Deleted files'}, {none: 2, deleted: 1}, 1),
  label('verified', 'choice', {verified: 'v', claimed_without_check: 'c', unverified: 'u', not_applicable: 'n'}, {verified: 2, unverified: 1, not_applicable: 1}),
  label('unfixed_error', 'noul', {}, {yes: 1, no: 3}),
  label('user_frustration', 'score', FRUS_CRITERIA, {1: 3, 4: 1}),
  label('user_corrections', 'score', CORR_CRITERIA, {2: 1, 3: 1, 0: 2}),
];
const DIM = {theme: {clusters: [
  {id: 'c1', level: 'base', name: 'Builds', description: '', example_trace_ids: []},
  {id: 'c2', level: 'base', name: 'Docs', description: '', example_trace_ids: []},
], n_no_signal: 0, n_noise: 0}};
const trace = (id, extra = {}) => ({
  id, trace_id: id, ts: '2026-09-30T10:00:00Z', request: `Request ${id}`, l: {}, label_errors: {}, a: {theme: id === 't2' ? 'c2' : 'c1'},
  errors: [], has_summary: true, has_tool_stats: true, tools: {}, skills: {}, commands: {}, ...extra,
});

function load({labels, traces, dims = DIM}) {
  const els = {};
  for (const id of ['tlist', 'canvas']) els[id] = {innerHTML: '', querySelectorAll: () => [], style: {setProperty(name, value) { this[name] = value; }}};
  const context = {
    D: {run: {priority: [], priority_dimension: null}, dims: JSON.parse(JSON.stringify(dims)), labels, traces},
    document: {getElementById: id => id === 'q' ? {} : els[id] || null},
    S: {dim: null, filters: [], shut: new Set(), sel: null, more: 20, q: ''},
    isOn: () => false, questionGroups: () => '', tagCounts: () => '', signalsPanel: () => '', safeTraceUrl: () => null, safeOrqUrl: () => null,
  };
  vm.createContext(context);
  const extract = makeExtract(source, context);
  extract('const T = D.traces', '/* ---------- state, mirrored into the URL ---------- */');
  extract('function hexA(', 'function renderCompare(');
  vm.runInContext('S.dim = PD; S.sel = {kind: "trace", id: T[0].id}; var visible = () => T;', context);
  extract('/* ---------- theme tree ---------- */', 'function pickCluster(');
  extract('function renderTraces() {', '/* ---------- right panel ---------- */');
  extract('function errorBreakdown(', 'function viewClusterTraces(');
  extract('function tracePanel(t) {', '/* ---------- shared bits ---------- */');
  const run = code => vm.runInContext(code, context);
  return {context, els, run};
}

const elements = node => node.children.filter(child => child.tagName);
const keys = nodes => nodes.map(n => n.getAttribute('data-col'));
const tracks = value => value.trim().split(/\s+/);
const statLabels = html => new Document(html).querySelectorAll('.stat .l').map(n => n.textContent);

function checkList(env, expected, {tracksOf = expected.length} = {}) {
  env.run('renderTraces()');
  const doc = new Document(env.els.tlist.innerHTML);
  const head = doc.querySelector('.trh');
  assert.deepEqual(keys(head.querySelectorAll('[data-col]')), expected, 'header columns');
  assert.deepEqual([...env.run('visibleCols().map(c => c.key)')], expected, 'spec columns');
  assert.equal(tracks(env.els.tlist.style['--tr-cols']).length, tracksOf, 'grid track count');
  const rows = doc.querySelectorAll('button.tr');
  assert.ok(rows.length > 0, 'rows are rendered');
  for (const row of rows) {
    assert.deepEqual(keys(row.querySelectorAll('[data-col]')), expected, 'row columns match the header');
    assert.equal(elements(row).length, tracksOf, 'one grid item per track');
    assert.equal(row.querySelectorAll('.req').length, 1, 'no nested request span');
    assert.equal(row.querySelectorAll('.when').length, 1, 'no nested date span');
  }
  return {doc, rows};
}

function checkTree(env, expected) {
  env.run('renderThemes()');
  const doc = new Document(env.els.canvas.innerHTML);
  assert.deepEqual(keys(doc.querySelectorAll('.cols span[data-col]')), expected, 'tree header columns');
  const tail = tracks(/--tree-tail:([^"]*)/.exec(env.els.canvas.innerHTML)[1]);
  assert.equal(tail.length, expected.length, 'tree grid tail has one track per column');
  const rows = doc.querySelectorAll('.row');
  assert.ok(rows.length > 0, 'tree rows are rendered');
  for (const row of rows) assert.equal(elements(row).length, 2 + expected.length, 'icon, name, then one cell per column');
  return doc;
}

const panelStats = env => statLabels(env.run('clusterPanel(C.c1)'));
const drawer = env => { const html = env.run('tracePanel(T[0])'); return {html, tiles: statLabels(html)}; };

/* lval: yes/no answers arrive as 'yes'/'no' strings from label_key */
{
  const env = load({labels: [label('unfixed_error', 'noul', {}, {yes: 1, no: 1}), label('user_corrections', 'score', CORR_CRITERIA)], traces: [trace('t1')]});
  const read = answer => env.run(`lval({l: {unfixed_error: ${JSON.stringify(answer)}}}, 'unfixed_error')`);
  assert.equal(read(['yes']), 'yes');
  assert.equal(read(['no']), 'no');
  assert.equal(read(['maybe']), null, 'an unknown string is no answer');
  assert.equal(read([null]), null);
  assert.equal(env.run("isYes({l: {unfixed_error: ['yes']}}, 'unfixed_error')"), true);
  assert.equal(env.run("isYes({l: {unfixed_error: ['no']}}, 'unfixed_error')"), false);
}

/* frustration-only run: only frustration (and the always-on columns) appear everywhere */
{
  const traces = [trace('t1', {l: {user_frustration: ['1']}}), trace('t2', {l: {user_frustration: ['4']}}), trace('t3', {l: {user_frustration: ['1']}})];
  const env = load({labels: FRUS_ONLY(), traces});
  checkList(env, ['when', 'req', 'theme', 'frus', 'errors']);
  checkTree(env, ['n', 'err', 'frus']);
  assert.deepEqual(panelStats(env), ['Traces', 'Assistant errors', 'Frustration']);
  assert.deepEqual(drawer(env).tiles, ['Frustration']);
  assert.match(drawer(env).html, /<h5>Conversation<\/h5>/);
  assert.doesNotMatch(drawer(env).html, /Coding work/);
}

/* the same run with no themes: no Theme column, no tree */
{
  const env = load({labels: FRUS_ONLY(), traces: [trace('t1', {l: {user_frustration: ['1']}})], dims: {}});
  checkList(env, ['when', 'req', 'frus', 'errors']);
  assert.doesNotMatch(env.els.tlist.innerHTML, /data-col="theme"/);
  assert.doesNotMatch(env.els.tlist.innerHTML, /Not grouped/);
}

/* no trace has a summary: no Errors column */
{
  const env = load({labels: FRUS_ONLY(), traces: [trace('t1', {has_summary: false, l: {user_frustration: ['1']}})]});
  checkList(env, ['when', 'req', 'theme', 'frus']);
}

/* coding labels configured but never asked: nothing of them shows anywhere */
{
  const env = load({labels: UNASKED_CODING(), traces: [trace('t1'), trace('t2')]});
  checkList(env, ['when', 'req', 'theme', 'errors']);
  checkTree(env, ['n', 'err']);
  assert.deepEqual(panelStats(env), ['Traces', 'Assistant errors']);
  const {html, tiles} = drawer(env);
  assert.deepEqual(tiles, []);
  assert.doesNotMatch(html, /Coding work/);
  assert.doesNotMatch(html, /<h5>Flags<\/h5>/);
}

/* a label whose every trace failed still counts as asked, and the cell says failed */
{
  const failedOnly = label('risky_action', 'choice', {none: 'None', deleted: 'Deleted files'}, {}, 2);
  const env = load({labels: [failedOnly], traces: [trace('t1', {label_errors: {risky_action: 'timeout'}}), trace('t2', {label_errors: {risky_action: 'timeout'}})]});
  const {rows} = checkList(env, ['when', 'req', 'theme', 'risky', 'errors']);
  assert.match(rows[0].querySelector('[data-col="risky"]').innerHTML, /failed/);
}

/* full coding run */
{
  const traces = [
    trace('t1', {l: {risky_action: ['none'], unfixed_error: ['no'], verified: ['verified'], user_corrections: ['2'], task_type: ['bugfix'], outcome: ['done'], user_frustration: ['1']}}),
    trace('t2', {l: {risky_action: ['deleted'], unfixed_error: ['yes'], verified: ['unverified'], user_corrections: ['3'], task_type: ['bugfix'], outcome: ['partial'], user_frustration: ['4']}}),
    trace('t3', {l: {risky_action: ['none'], unfixed_error: ['no'], verified: ['not_applicable'], user_corrections: ['0'], task_type: ['bugfix'], outcome: ['done'], user_frustration: ['1']}}),
    trace('t4', {l: {unfixed_error: ['no'], verified: ['claimed_without_check'], user_corrections: ['0'], task_type: ['bugfix'], outcome: ['done'], user_frustration: ['1']}, label_errors: {risky_action: 'timeout'}}),
    trace('t5', {has_summary: false, l: {unfixed_error: ['no'], user_corrections: ['0'], task_type: ['bugfix'], outcome: ['done'], user_frustration: ['1']}}),
    trace('t6', {errors: ['one', 'two'], l: {risky_action: ['none'], unfixed_error: ['no'], verified: ['verified'], user_corrections: ['0'], task_type: ['bugfix'], outcome: ['done'], user_frustration: ['1']}}),
  ];
  const env = load({labels: FULL_CODING(), traces});
  const all = ['when', 'req', 'theme', 'task', 'outcome', 'frus', 'unfixed', 'risky', 'unver', 'corr', 'errors'];
  assert.equal(all.length, 11);
  const {rows} = checkList(env, all);
  const cell = (row, key) => row.querySelector(`[data-col="${key}"]`);
  const text = (row, key) => cell(row, key).textContent.trim();
  const flags = ['unfixed', 'risky', 'unver', 'corr'];
  assert.deepEqual(flags.map(k => text(rows[0], k)), ['No', 'No', 'No', 'No'], 'answered no reads No');
  assert.ok(flags.every(k => cell(rows[0], k).querySelector('.bool.no')));
  assert.deepEqual(flags.map(k => text(rows[1], k)), ['Yes', 'Yes', 'Yes', 'Yes']);
  assert.equal(text(rows[2], 'unver'), 'n/a', 'nothing to verify');
  assert.equal(text(rows[3], 'unver'), 'Yes', 'claimed_without_check is flagged');
  assert.equal(text(rows[3], 'risky'), 'failed');
  assert.equal(cell(rows[3], 'risky').querySelector('.bool'), null);
  assert.equal(cell(rows[4], 'risky').innerHTML, env.run('NA'), 'no answer and no error is not asked');
  assert.equal(text(rows[4], 'errors'), '—', 'no summary');
  assert.equal(text(rows[0], 'errors'), '0');
  assert.ok(cell(rows[0], 'errors').querySelector('.errb.zero'));
  assert.equal(text(rows[5], 'errors'), '2');
  checkTree(env, ['n', 'err', 'done', 'risky', 'frus']);
  assert.deepEqual(panelStats(env), ['Traces', 'Assistant errors', 'Done', 'Frustration']);
  assert.deepEqual(drawer(env).tiles, ['Task', 'Outcome', 'Verified', 'Frustration', 'User corrections']);
  for (const [index, t] of traces.entries()) {
    const listed = flags.filter(k => text(rows[index], k) === 'Yes').length;
    assert.equal(new Document(env.run(`flagsText(T[${index}])`)).querySelectorAll('.flagtxt').length, listed, `drawer flags match the list for ${t.id}`);
  }
}

/* drawer: an errored answer reads failed, an unanswered one reads not measured */
{
  const env = load({labels: FULL_CODING(), traces: [trace('t1', {label_errors: {outcome: 'timeout'}, l: {task_type: ['bugfix']}})]});
  const html = env.run('tracePanel(T[0])');
  const outcomeTile = new Document(html).querySelectorAll('.stat').find(n => n.querySelector('.l').textContent === 'Outcome');
  assert.match(outcomeTile.innerHTML, /failed/);
  const verifiedTile = new Document(html).querySelectorAll('.stat').find(n => n.querySelector('.l').textContent === 'Verified');
  assert.doesNotMatch(verifiedTile.innerHTML, /failed/);
}

console.log('Insights trace column checks passed');
