const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-run-common.js', 'utf8');
const calls = [];
const context = {
  URLSearchParams,
  FormData,
  fetch: async (url, options) => {
    calls.push({url, options});
    return {ok: true, json: async () => ({path: '/tmp/source.json'})};
  },
  window: {},
};
vm.createContext(context);
vm.runInContext(source, context);
const common = context.window.InsightsRunCommon;

assert.equal(common.normalizeSource('question'), 'query');
assert.equal(common.normalizeSource('file'), 'snapshot');
assert.equal(common.normalizeSource('recent'), 'recent');
assert.equal(common.validateSource({source: 'question', query: '  '}), 'query');
assert.equal(common.validateSource({source: 'file', snapshotPath: ''}), 'snapshot');
assert.equal(common.validateSource({source: 'recent', windowValid: false}), 'window');
assert.equal(common.validateSource({source: 'recent', facetsLoading: true}), 'facets');
assert.equal(common.validateSource({source: 'recent', windowValid: true, limitValid: true}), '');
assert.equal(common.validateSelection({selectionCount: 0, parallelismValid: true}), 'selection');
assert.equal(common.validateSelection({selectionCount: 1, parallelismValid: false}), 'parallelism');
assert.equal(common.validateSelection({selectionCount: 1, parallelismValid: true}), '');

(async () => {
  const facets = await common.requestFacets(14, {facet_agent: ['A', 'B'], facet_project: ['P']}, null);
  assert.equal(facets.ok, true);
  assert.equal(calls[0].url, '/insights/facets?window_days=14&facet_agent=A&facet_agent=B&facet_project=P');
  await common.requestFacets(14, {}, null, true);
  assert.equal(calls[1].url, '/insights/facets?window_days=14&retry=1');
  const preview = await common.requestSnapshotPreview('/tmp/a.json', 'csrf-token');
  assert.equal(preview.ok, true);
  assert.equal(calls[2].url, '/insights/snapshot-preview');
  assert.equal(calls[2].options.body.get('csrf'), 'csrf-token');
  assert.equal(calls[2].options.body.get('snapshot_path'), '/tmp/a.json');
  const uploaded = await common.uploadSource('snapshot', new Blob(['{}']), {
    querySelector: () => ({value: 'form-token'}),
  });
  assert.deepEqual(uploaded, {path: '/tmp/source.json'});
  assert.equal(calls[3].url, '/insights/uploads');
  assert.equal(calls[3].options.body.get('csrf'), 'form-token');
  assert.equal(calls[3].options.body.get('kind'), 'snapshot');
  console.log('shared Insights run helper checks passed');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
