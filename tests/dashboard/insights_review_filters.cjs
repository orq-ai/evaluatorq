const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync('src/evaluatorq/dashboard/static/insights-review.js', 'utf8');
const start = source.indexOf('const TPL = [');
const end = source.indexOf('function buttonBusy(input, busy)', start);
assert.ok(start >= 0 && end > start, 'new run sheet implementation is present');
assert.match(source.slice(start, end), /<label>Question<\/label>/, 'the question source field has the visible Question label');

class Element {
  constructor(dataset = {}) {
    this.dataset = dataset;
    this.checked = false;
    this.value = '';
    this.disabled = false;
    this.hidden = false;
    this.listeners = {};
    this.textContent = '';
    this.files = [];
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  change() { this.listeners.change?.(); }
  click() {
    this.onclick?.({preventDefault() {}});
    this.parent?.listeners.click?.({target: {closest: selector => selector === '[data-remove-facet]' ? this : null}, stopPropagation() {}});
  }
}

const facetInputs = [
  new Element({}),
  new Element({}),
  new Element({}),
];
Object.assign(facetInputs[0], {name: 'facet_agent_name', value: 'agent-a'});
Object.assign(facetInputs[1], {name: 'facet_agent_name', value: 'agent-b'});
Object.assign(facetInputs[2], {name: 'facet_agent_name', value: 'agent-c'});
let selectedFacets;
const sheet = {
  markup: '',
  removeButtons: [],
  rendered: false,
  set innerHTML(markup) {
    this.markup = markup;
    this.rendered = true;
    this.removeButtons = [];
    const owner = this;
    selectedFacets = new Element();
    selectedFacets.parent = this;
    selectedFacets.listeners = {};
    Object.defineProperty(selectedFacets, 'innerHTML', {set(html) {
      selectedFacets.markup = html;
      owner.removeButtons = [...html.matchAll(/<button\b(?=[^>]*data-remove-facet="([^"]+)")(?=[^>]*data-facet-value="([^"]+)")[^>]*>/g)]
        .map(([, name, value]) => new Element({removeFacet: name, facetValue: value}));
      owner.removeButtons.forEach(button => { button.parent = selectedFacets; });
    }});
  },
  get innerHTML() { return this.markup; },
  querySelectorAll(selector) {
    if (selector === 'input[name^="facet_"]') return this.rendered ? facetInputs : [];
    if (selector === 'input[name^="facet_"]:checked') return this.rendered ? facetInputs.filter(input => input.checked) : [];
    if (selector === '[data-remove-facet]') return this.removeButtons;
    return [];
  },
  querySelector(selector) { return selector === '.insights-selected-facets' ? selectedFacets : null; },
};
const ids = {
  sheet,
  closeSheet: new Element(),
  startRun: new Element(),
  days: Object.assign(new Element(), {value: '7'}),
  limit: Object.assign(new Element(), {value: '200'}),
  runName: Object.assign(new Element(), {value: 'Filter test'}),
};
const submitted = [];
class FormDataFake {
  constructor() { this.values = new Map(); }
  set(name, value) { this.values.set(name, [String(value)]); }
  append(name, value) { this.values.set(name, [...(this.values.get(name) || []), String(value)]); }
  getAll(name) { return this.values.get(name) || []; }
}
const context = {
  shouldPrefillRerun: false,
  D: {},
  $: id => ids[id] || null,
  esc: value => String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char])),
  human: value => value.replace(/_/g, ' ').replace(/^./, char => char.toUpperCase()),
  bindMock() {},
  closeSheet() {},
  InsightsRunCommon: {
    normalizeSource: source => source === 'question' ? 'query' : source,
    validateSource: () => '',
    validateSelection: () => '',
    csrfToken: async () => 'csrf-test',
  },
  FormData: FormDataFake,
  fetch: async (_url, options) => {
    submitted.push(options.body);
    return {redirected: false, text: async () => '<p role="alert">Submitted</p>'};
  },
  DOMParser: class { parseFromString() { return {querySelector: () => ({textContent: 'Submitted'})}; } },
  location: {assign() {}},
};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);
context.run = expression => vm.runInContext(expression, context);
context.run("NR.facets = {facet_agent_name: ['agent-a', 'agent-b']}; NR.facetMarkup = '<div class=\"finder-controls insights-filter-picker\"><div class=\"insights-selected-facets\" aria-live=\"polite\"></div><label><input type=\"checkbox\" name=\"facet_agent_name\" value=\"agent-a\">Agent A</label><label><input type=\"checkbox\" name=\"facet_agent_name\" value=\"agent-b\">Agent B</label><label><input type=\"checkbox\" name=\"facet_agent_name\" value=\"agent-c\">Agent C</label></div>';   NR.labels = new Set(['made_errors']);");

(async () => {
  context.run('renderSheet()');
  assert.match(sheet.markup, /class="insights-selected-facets"/, 'selected filter chips have a visible container');
  assert.match(selectedFacets.markup, /data-remove-facet="facet_agent_name"[^>]*data-facet-value="agent-a"|data-facet-value="agent-a"[^>]*data-remove-facet="facet_agent_name"/);
  assert.match(selectedFacets.markup, /agent.*agent-a/i);
  assert.deepEqual(facetInputs.map(input => input.checked), [true, true, false], 'checkboxes reflect NR.facets after render');

  facetInputs[2].checked = true;
  facetInputs[2].change();
  assert.deepEqual(JSON.parse(JSON.stringify(context.run('NR.facets.facet_agent_name'))), ['agent-a', 'agent-b', 'agent-c'], 'checking a facet updates NR.facets');
  assert.match(selectedFacets.markup, /data-facet-value="agent-c"/, 'a newly checked facet immediately appears as a chip');

  context.run("NR.facets.facet_agent_name = ['<img src=x onerror=alert(1)>']; renderSelectedFacets();");
  assert.match(selectedFacets.markup, /&lt;img src=x onerror=alert\(1\)&gt;/, 'facet values are escaped in chip markup');
  assert.doesNotMatch(selectedFacets.markup, /<img/, 'facet values cannot inject markup');
  context.run('captureFacets(); renderSelectedFacets()');

  const removeAgentA = sheet.removeButtons.find(button => button.dataset.facetValue === 'agent-a');
  assert.ok(selectedFacets.listeners.click, 'the selected chip container handles removal clicks');
  removeAgentA.click();
  assert.deepEqual(JSON.parse(JSON.stringify(context.run('NR.facets.facet_agent_name'))), ['agent-b', 'agent-c'], 'removing a chip updates NR.facets');
  assert.deepEqual(facetInputs.map(input => input.checked), [false, true, true], 'removing a chip unchecks its matching checkbox');
  assert.doesNotMatch(selectedFacets.markup, /data-facet-value="agent-a"/, 'removed facet disappears from the chip list');

  context.run('renderSheet()');
  assert.deepEqual(facetInputs.map(input => input.checked), [false, true, true], 'checkbox state survives rerender after chip removal');
  assert.doesNotMatch(selectedFacets.markup, /data-facet-value="agent-a"/, 'rerender preserves the removed chip state');
  assert.match(selectedFacets.markup, /data-facet-value="agent-b"/);
  assert.match(selectedFacets.markup, /data-facet-value="agent-c"/);

  await ids.startRun.onclick();
  assert.equal(submitted.length, 1, 'starting the run submits the selected filters');
  assert.deepEqual(submitted[0].getAll('facet_agent_name'), ['agent-b', 'agent-c'], 'submitted filters include new selections and exclude the removed value');
  console.log('Insights review selected facet state and submission checks passed');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
