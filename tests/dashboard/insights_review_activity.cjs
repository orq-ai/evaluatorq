const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.js', 'utf8');
const styles = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.css', 'utf8');
const start = source.indexOf('/* ---------- activity ---------- */');
const end = source.indexOf('function colorOf(t) {', start);
assert.ok(start >= 0 && end > start, 'Activity implementation is present');
const canvas = {innerHTML: '', querySelectorAll: () => []};
const results = {scrollTop: 37};
const focused = {dataset: {kind: 'commands', name: 'rm file'}, focus: options => { focused.options = options; }};
const document = {
  querySelector: selector => selector === '.activity-results' ? results : null,
  querySelectorAll: selector => selector === '[data-activity-item]' ? [focused] : [],
};
const elements = {canvas, activitySort: {}, activityQuery: {}};
const context = {
  D: {activity: {shell_tools: ['Bash', 'exec_command']}},
  T: [],
  baseVisible: () => context.T,
  visible: () => context.T.filter(t => !context.S.activityItem || (t[context.S.activityItem.kind] || {})[context.S.activityItem.name] > 0),
  S: {view: 'activity', activityKind: 'skills', activityQuery: '', activitySort: 'coverage', activityItem: null, activityHelpers: false, activityExpanded: new Set(), activityShowAll: false, filters: [], q: ''},
  HELPERS: new Set(['grep', 'sed', 'echo', 'cat', 'ls']),
  $: id => elements[id] || null,
  document,
  history: {replaceState: () => {}},
  location: {hash: ''},
  toUrl: () => '#view=activity',
  commit: () => context.run('renderActivity()'),
  esc: value => String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char])),
  pct: (a, b) => b ? Math.round(a / b * 100) : 0,
  isRisky: command => command === 'git reset' || command.startsWith('rm '),
};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);
context.run = expression => vm.runInContext(expression, context);

const traces = [
  {id: 'sun', ts: '2026-09-27T23:59:59Z', has_tool_stats: true, skills: {review: 2, helper: 1}, tools: {Bash: 2, Read: 1}, commands: {'git status': 2, 'grep pattern': 1}},
  {id: 'mon', ts: '2026-09-28T00:00:00Z', has_tool_stats: true, skills: {review: 1}, tools: {Bash: 1, Read: 1}, commands: {'git status': 1}},
  {id: 'later', ts: '2026-10-12T13:00:00Z', has_tool_stats: true, skills: {later: 1}, tools: {Read: 1}, commands: {'rm file': 1}},
  {id: 'untimed', ts: null, has_tool_stats: true, skills: {review: 1}, tools: {Bash: 1}, commands: {'git status': 1}},
  {id: 'missing', ts: '2026-10-19T00:00:00Z', has_tool_stats: false, skills: {review: 99}, tools: {Write: 99}, commands: {'rm file': 99}},
  {id: 'empty', ts: '2026-10-19T01:00:00Z', has_tool_stats: true, skills: {}, tools: {}, commands: {}},
];
for (let i = 0; i < 12; i++) traces.push({id: `skill-${i}`, ts: '2026-09-28T04:00:00Z', has_tool_stats: true, skills: {[`skill-${i}`]: 1}, tools: {}, commands: {}});
context.T = traces;

const run = context.run;
const skill = run("activityRows(T, 'skills').find(row => row.name === 'review')");
assert.equal(skill.traces, 3, 'missing stats never count as use');
assert.equal(skill.calls, 4, 'skill loads aggregate without becoming a separate visible metric');
assert.equal(run("activityEligible(T).length"), 17, 'empty ToolStats records are eligible');
assert.equal(run("activityRows(T, 'tools').find(row => row.name === 'Bash').calls"), 4);

const weeks = run("activityWeeks(T, 'skills', 'review')");
assert.deepEqual(JSON.parse(JSON.stringify(weeks.map(w => [w.key, w.eligible, w.using]))), [
  ['2026-09-21', 1, 1], ['2026-09-28', 13, 1], ['2026-10-05', 0, 0],
  ['2026-10-12', 1, 0], ['2026-10-19', 1, 0],
]);
assert.equal(weeks.reduce((sum, week) => sum + week.eligible, 0), 16, 'null timestamps do not inflate weekly denominators');
assert.equal(run("activityWeekKey('2026-09-27T23:59:59Z')"), '2026-09-21');
assert.equal(run("activityWeekKey('2026-09-28T00:00:00Z')"), '2026-09-28');
assert.equal(run('activityWeekKey(null)'), null);
context.S.activityItem = {kind: 'skills', name: 'review'};
context.T = traces.filter(t => t.id === 'sun');
assert.match(run('activityDetail()'), /1 of 1 eligible traces in this week/);
context.T = traces;
context.S.activityItem = null;

const commandPairs = run("activityPairs(T, 'commands', 'git status')");
assert.equal(commandPairs.using, 3);
assert.equal(commandPairs.groups.tools.some(pair => pair.name === 'Bash'), false, 'shell tool overlap is excluded from command pairings');
const bashPairs = run("activityPairs(T, 'tools', 'Bash')");
assert.equal(bashPairs.groups.commands.some(pair => pair.name === 'git status'), false, 'a shell tool never lists its own shell commands');
context.T = [{id: 'alone', ts: null, has_tool_stats: true, skills: {}, tools: {Read: 1}, commands: {}}];
context.S.activityItem = {kind: 'tools', name: 'Read'};
assert.equal((run('activityDetail()').match(/No co-occurrences recorded\./g) || []).length, 3, 'empty pair groups explain the absence of co-occurrences');
context.T = traces;
context.S.activityItem = null;

run('renderActivity()');
assert.match(canvas.innerHTML, /class="activity-summary skills"/);
assert.match(canvas.innerHTML, /17 eligible traces · 1 missing tool stats/);
assert.doesNotMatch(canvas.innerHTML, /Total loads|recorded loads|id="activitySort"/);
assert.match(canvas.innerHTML, /Show all 15 skills/);
assert.match(canvas.innerHTML, /3<small> \/ 17 · 18%<\/small>/);
assert.match(canvas.innerHTML, /Bar and percentage: share of 17 eligible traces/);
assert.match(canvas.innerHTML, /metric-label">Traces using<\/span>/);
assert.match(styles, /\.activity-results\{max-height:460px;overflow-y:auto/);
assert.match(styles, /\.activity-head\{position:sticky/);
assert.match(styles, /@media\(max-width:600px\)/);
assert.match(styles, /nth-child\(n\+11\):not\(\.on\)/, 'mobile initially reveals ten rows and keeps the selected row visible');
assert.match(styles, /\.activity-row \.metric-label[^}]*display:block/);
assert.match(styles, /\.activity-row:focus-visible[^}]*outline:2px solid/, 'keyboard focus remains visible');

context.S.activityKind = 'commands';
context.S.activityQuery = 'grep';
run('renderActivity()');
assert.match(canvas.innerHTML, /grep pattern/, 'search finds hidden command helpers');
context.S.activityQuery = '';
context.S.activityKind = 'tools';
run('renderActivity()');
assert.match(canvas.innerHTML, /Total calls/);
assert.match(canvas.innerHTML, /Sort <select/);
assert.match(canvas.innerHTML, /relative to the largest visible item/);
context.S.activitySort = 'calls';
run('renderActivity()');
assert.ok(canvas.innerHTML.indexOf('Bash') < canvas.innerHTML.indexOf('Read'), 'tool calls can be sorted by calls');

context.S.activityKind = 'commands';
context.S.activitySort = 'coverage';
context.S.activityItem = {kind: 'commands', name: 'git status'};
run('renderActivity()');
assert.match(canvas.innerHTML, /data-activity-program="git"[^>]*aria-expanded="true"/, 'selected command keeps its program open');
context.S.activityQuery = 'rm file';
run('renderActivity()');
assert.match(canvas.innerHTML, /Potentially state-changing/, 'heuristic label applies to a risky command name');
assert.match(canvas.innerHTML, /Bar length: traces using is relative to 17 eligible traces/);
assert.match(canvas.innerHTML, /largest visible command or program within its level/);
assert.doesNotMatch(canvas.innerHTML, /Potentially state-changing[^<]*.*trace changed state/i);
context.S.activityQuery = '';
const details = run('activityDetail()');
assert.match(details, /Often together/);
assert.match(details, /\/ 3 traces · /, 'pair percentage denominator is selected-item trace count');
assert.ok((details.match(/data-activity-pair/g) || []).length <= 15, 'at most five pairings per category');

const largeSkills = Array.from({length: 12}, (_, i) => ({id: `s${i}`, has_tool_stats: true, skills: {[`s${i}`]: 1}, tools: {}, commands: {}}));
context.T = largeSkills;
context.S.activityKind = 'skills';
context.S.activityItem = {kind: 'skills', name: 's11'};
run('renderActivity()');
assert.match(canvas.innerHTML, /class="activity-row skills[^\"]*on[^\"]*" data-activity-item[^>]*data-name="s11"/);
assert.equal(run("visible().length"), 1, 'selected activity filters only the matching trace cohort');
assert.equal(run("activityRows(baseVisible(), 'skills').length"), 12, 'rankings retain the globally filtered cohort');

context.T = [];
context.S.activityItem = null;
context.S.filters = [{t: 'cluster'}];
run('renderActivity()');
assert.match(canvas.innerHTML, /No traces match the current run filters/);
context.T = [{id: 'none', has_tool_stats: true, skills: {}, tools: {}, commands: {}}];
run('renderActivity()');
assert.match(canvas.innerHTML, /No skills were recorded/);
context.T = [{id: 'unknown', has_tool_stats: false, skills: {}, tools: {}, commands: {}}];
run('renderActivity()');
assert.match(canvas.innerHTML, /Activity data was not recorded/);
context.T = [{id: 'filtered', has_tool_stats: true, skills: {'was-recorded': 1}, tools: {}, commands: {}}];
context.S.filters = [{t: 'cluster'}];
const recordedElsewhere = context.T;
context.T = [...recordedElsewhere, {id: 'filtered-empty', has_tool_stats: true, skills: {}, tools: {}, commands: {}}];
context.baseVisible = () => context.T.slice(1);
run('renderActivity()');
assert.match(canvas.innerHTML, /No items match the current run filters/);

context.T = traces;
context.baseVisible = () => context.T;
context.S.filters = [];
context.S.activityKind = 'commands';
context.S.activityItem = {kind: 'commands', name: 'git status'};
context.S.activityExpanded = new Set();
context.commit = () => {};
context.run("activitySelect('commands', 'rm file', true)");
assert.equal(results.scrollTop, 37, 'selection preserves desktop ranking scroll');
assert.equal(focused.options.preventScroll, true, 'selection retains keyboard focus');
const evil = '</button><img src=x onerror=alert(1)>';
context.T = [{id: 'evil', ts: null, has_tool_stats: true, skills: {[evil]: 1}, tools: {}, commands: {}}];
Object.assign(context.S, {activityKind: 'skills', activityQuery: '', activityItem: null});
run('renderActivity()');
assert.doesNotMatch(canvas.innerHTML, /<img/, 'trace-derived activity names are escaped');
assert.match(canvas.innerHTML, /&lt;img src=x/);
console.log('Insights Activity synthetic aggregation and rendering checks passed');
