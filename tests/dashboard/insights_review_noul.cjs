const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

// Reads a review payload built by build_review_payload and checks that lval reads its noul answers.
const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.js', 'utf8').replace(/\r\n/g, '\n');
const start = source.indexOf('function lvals(l)');
const end = source.indexOf('\n}\n', source.indexOf('function lval(t, name)')) + 3;
assert.ok(start >= 0 && end > start, 'lvals and lval are present');
const context = {LBY: Object.fromEntries(payload.labels.map(label => [label.name, label]))};
vm.createContext(context);
vm.runInContext(`const scoreKey = (criteria, i) => String(i);\n${source.slice(start, end)}`, context);

const counts = {};
for (const trace of payload.traces) {
  const value = context.lval(trace, 'made_errors');
  counts[value] = (counts[value] || 0) + 1;
}
const label = payload.labels.find(item => item.name === 'made_errors');
assert.deepEqual(counts, label.counts);
