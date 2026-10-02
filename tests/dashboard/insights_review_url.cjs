const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.js', 'utf8');
const start = source.indexOf('function safeTraceUrl(value) {');
const end = source.indexOf('function tracePanel(t) {', start);
assert.ok(start >= 0 && end > start, 'URL validation helpers are present');
const context = {URL, location: {origin: 'https://dashboard.example'}};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);

assert.equal(context.safeTraceUrl('/insights/run/trace?trace_id=123'), 'https://dashboard.example/insights/run/trace?trace_id=123');
assert.equal(context.safeTraceUrl('javascript:alert(1)'), '');
assert.equal(context.safeTraceUrl('data:text/html,<script>alert(1)</script>'), '');
assert.equal(context.safeTraceUrl('//attacker.example/trace'), '');
assert.equal(context.safeOrqUrl('https://orq.example/workspace/traces'), 'https://orq.example/workspace/traces');
assert.equal(context.safeOrqUrl('http://orq.example/workspace/traces'), '');
assert.equal(context.safeOrqUrl('javascript:alert(1)'), '');
assert.equal(context.safeOrqUrl('data:text/html,<script>alert(1)</script>'), '');
