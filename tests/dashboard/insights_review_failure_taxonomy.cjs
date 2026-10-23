const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.js', 'utf8').replace(/\r\n/g, '\n');
const helperStart = source.indexOf('function scoreKey(criteria, index)');
const helperEnd = source.indexOf('\nfunction frus(', helperStart);
const filterStart = source.indexOf('function fids(f)', helperEnd);
const filterEnd = source.indexOf('\nfunction flabel(', filterStart);
const decodeStart = source.indexOf('function decodePart(value)', filterEnd);
const decodeEnd = source.indexOf('\nfunction toUrl()', decodeStart);
const renderStart = source.indexOf('function labelBlock(l, ts)', decodeEnd);
const renderEnd = source.indexOf('\nfunction tagCounts', renderStart);
assert.ok(helperStart >= 0 && helperEnd > helperStart, 'label value helpers are present');
assert.ok(filterStart >= 0 && filterEnd > filterStart, 'label filter helper is present');
assert.ok(decodeStart >= 0 && decodeEnd > decodeStart, 'filter decoder is present');
assert.ok(renderStart >= 0 && renderEnd > renderStart, 'labelBlock is present');

const context = {
  esc: value => String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char])),
  human: value => value.replace(/_/g, ' ').replace(/^./, char => char.toUpperCase()),
  QUAL: ['#025558', '#9fcfc7', '#df5325'],
  SEM: {},
  BAD_YES: new Set(),
  OTHER: '#9fcfc7',
  RISK_H: {},
  LBY: {},
  T: [],
  isOn: () => false,
};
vm.createContext(context);
vm.runInContext(source.slice(helperStart, helperEnd), context);
vm.runInContext(source.slice(filterStart, filterEnd), context);
vm.runInContext(source.slice(decodeStart, decodeEnd), context);
vm.runInContext(source.slice(renderStart, renderEnd), context);

const legacy = payload.labels.find(label => label.name === 'failure_taxonomy');
assert.ok(legacy);
legacy.counts_warning = 'Counts may include <clean> & ordinary traces.';
const trace = {id: 'failure', l: {failure_taxonomy: ['Hallucination']}};
const cleanTrace = {id: 'clean', l: {failure_taxonomy: ['No failure']}};
context.LBY.failure_taxonomy = legacy;
context.T = [trace, cleanTrace];

const legacyHtml = context.labelBlock(legacy, [trace]);
assert.match(legacyHtml, /<p class="note label-counts-warning" role="status">Counts may include &lt;clean&gt; &amp; ordinary traces\.<\/p>/);
assert.ok(legacyHtml.indexOf('class="lh"') < legacyHtml.indexOf('role="status"'));
assert.ok(legacyHtml.indexOf('role="status"') < legacyHtml.indexOf('class="sbar"'));
assert.match(context.labelBlock(legacy, []), /Counts may include &lt;clean&gt; &amp; ordinary traces\./);
assert.match(legacyHtml, /data-v="Hallucination"/);
assert.deepEqual(Array.from(context.lvals(legacy)), Object.keys(legacy.criteria));
assert.equal(context.decF('l:failure_taxonomy:Hallucination').v, 'Hallucination');
assert.equal(context.decF('l:failure_taxonomy:No%20failure'), null, 'legacy counts must not invent a clean choice');
assert.deepEqual(Array.from(context.lvals({kind: 'noul'})), ['yes', 'no']);
assert.deepEqual(Array.from(context.lvals({kind: 'score', criteria: ['1: Low', '2: High']})), ['1', '2']);

for (const criteria of [undefined, null, ['Hallucination']]) {
  const saved = {...legacy, criteria, counts: {'Hallucination': 1, 'No failure': 1}};
  context.LBY.failure_taxonomy = saved;
  assert.deepEqual(Array.from(context.lvals(saved)), ['Hallucination', 'No failure']);
  assert.match(context.labelBlock(saved, [cleanTrace]), /data-v="No failure"/);
  const cleanFilter = context.decF('l:failure_taxonomy:No%20failure');
  assert.equal(cleanFilter.v, 'No failure');
  const matching = Array.from(context.fids(cleanFilter), id => id);
  assert.deepEqual(matching, ['clean']);
}

const revised = {...legacy, name: 'revised_taxonomy', criteria: {'No failure': 'Clean', Hallucination: 'A failure'}, counts_warning: null};
const unrelated = {...legacy, name: 'other_choice', criteria: {yes: 'Yes'}, counts_warning: null};
context.LBY.revised_taxonomy = revised;
assert.doesNotMatch(context.labelBlock(revised, [cleanTrace]), /role="status"/);
assert.doesNotMatch(context.labelBlock(unrelated, [trace]), /role="status"/);
assert.deepEqual(Array.from(context.lvals(revised)), ['No failure', 'Hallucination']);
assert.equal(context.decF('l:revised_taxonomy:No%20failure').v, 'No failure');
