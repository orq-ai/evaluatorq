const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.js', 'utf8').replace(/\r\n/g, '\n');
const start = source.indexOf('function labelBlock(l, ts)');
const end = source.indexOf('\nfunction tagCounts', start);
assert.ok(start >= 0 && end > start, 'labelBlock is present');

const context = {
  esc: value => String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char])),
  human: value => value.replace(/_/g, ' ').replace(/^./, char => char.toUpperCase()),
  lvals: label => Object.keys(label.criteria || {}),
  lval: (trace, name) => trace.l?.[name]?.[0] ?? null,
  ldisp: (_name, value) => value,
  lcolor: () => '#025558',
  isOn: () => false,
};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);

const legacy = payload.labels.find(label => label.name === 'failure_taxonomy');
assert.ok(legacy);
legacy.counts_warning = 'Counts may include <clean> & ordinary traces.';
const revised = {...legacy, name: 'revised_taxonomy', counts_warning: null};
const unrelated = {...legacy, name: 'other_choice', counts_warning: null};
const trace = {l: {failure_taxonomy: ['Hallucination'], revised_taxonomy: ['No failure'], other_choice: ['yes']}};

const legacyHtml = context.labelBlock(legacy, [trace]);
assert.match(legacyHtml, /<p class="note" role="status">Counts may include &lt;clean&gt; &amp; ordinary traces\.<\/p>/);
assert.ok(legacyHtml.indexOf('class="lh"') < legacyHtml.indexOf('role="status"'));
assert.ok(legacyHtml.indexOf('role="status"') < legacyHtml.indexOf('class="sbar"'));
assert.match(context.labelBlock(legacy, []), /Counts may include &lt;clean&gt; &amp; ordinary traces\./);
assert.doesNotMatch(context.labelBlock(revised, [trace]), /role="status"/);
assert.doesNotMatch(context.labelBlock(unrelated, [trace]), /role="status"/);
