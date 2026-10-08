'use strict';
/* Drives insights-run-form.js against the markup render_run_form produces (passed in by the Python test). */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {Document, DOMParser} = require('./minidom.cjs');

// Read the fixtures from stdin, not an env var: on Linux a single env string is
// capped at 128KB (MAX_ARG_STRLEN), and the rendered form markup exceeds it, so
// exec("node") fails with E2BIG ("Argument list too long"). stdin has no such cap.
const fixtures = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = fs.readFileSync(path.resolve(__dirname, '../../src/evaluatorq/dashboard/static/insights-run-form.js'), 'utf8');

const FACET_MARKUP = '<div class="finder-controls"><span class="addwrap"><div class="finder-facets" id="menu">'
  + '<label><input form="insights-new-form" type="checkbox" name="facet_agent_name" value="support-bot"><span>support-bot</span></label>'
  + '</div></span></div>';

function response(body, {status = 200, redirected = false, url = ''} = {}) {
  return {ok: status < 400, status, redirected, url, text: async () => body, json: async () => JSON.parse(body)};
}

function boot(handlers) {
  const document = new Document(fixtures.form);
  const calls = [];
  const location = {assigned: null, assign(url) { this.assigned = url; }};
  const processed = [];
  const window = {location, htmx: {process: element => processed.push(element)}};
  const fetch = async (url, options = {}) => {
    calls.push({url: String(url), options});
    const pathname = String(url).split('?')[0];
    const handler = handlers[pathname];
    if (!handler) throw new Error('unexpected request ' + url);
    return handler(String(url), options);
  };
  const context = vm.createContext({
    window, document, fetch, DOMParser, URLSearchParams, FormData, AbortController, File, console, setTimeout, clearTimeout, Promise,
  });
  vm.runInContext(source, context);
  const form = document.getElementById('insights-new-form');
  return {document, form, calls, window, processed, location};
}

const tick = () => new Promise(resolve => setImmediate(resolve));
const visibleStep = form => form.querySelectorAll('[data-step]').filter(section => !section.hidden).map(section => section.dataset.step);
const checked = (form, name) => form.querySelectorAll('input[name="' + name + '"]:checked').map(input => input.value);
const baseHandlers = () => ({
  '/insights/facets': () => response(FACET_MARKUP),
  '/insights/new/plan': () => response('<ol class="irf-stages"><li>Filter recent traces</li></ol>'),
});

async function main() {
  {
    const app = boot(baseHandlers());
    await tick();
    assert.equal(app.processed.length, 1, 'mounting processes the form with htmx');
    assert.deepEqual(visibleStep(app.form), ['1'], 'the form opens on step 1');
    assert.equal(app.calls.filter(call => call.url.startsWith('/insights/facets')).length, 1, 'the picker loads on mount');
    assert.ok(app.form.querySelector('#menu'), 'the picker markup lands in the form');
  }

  {
    const app = boot(baseHandlers());
    await tick();
    const question = app.form.querySelector('textarea[name="query"]');
    const compiler = app.form.querySelector('[data-compiler-model]');
    assert.equal(app.form.querySelector('[data-source="orq"]').hidden, false, 'the Orq fields show on the first tab');
    assert.equal(compiler.hidden, true, 'the compiler picker is hidden while the question is blank');
    question.value = 'refunds';
    question.dispatchEvent({type: 'input', bubbles: true, target: question});
    assert.equal(compiler.hidden, false, 'the compiler picker shows once there is a question');
    question.value = '';
    question.dispatchEvent({type: 'input', bubbles: true, target: question});
    assert.equal(compiler.hidden, true);
    app.form.querySelector('[data-irf-next]').click();
    await tick();
    assert.deepEqual(visibleStep(app.form), ['2'], 'a blank question is allowed: it analyzes every trace in the window');
    app.form.querySelector('[data-irf-back]').click();
    app.form.querySelector('input[name="source"][value="file"]').click();
    assert.equal(app.form.querySelector('[data-source="orq"]').hidden, true, 'the Orq fields hide on the file tab');
    assert.equal(app.form.querySelector('[data-source="file"]').hidden, false);
    app.form.querySelector('[data-irf-next]').click();
    await tick();
    const error = app.form.querySelector('#insights-run-error');
    assert.equal(error.textContent, 'Browse to choose a trace file first.');
    assert.deepEqual(visibleStep(app.form), ['1'], 'Continue is blocked until a file is chosen');
  }

  {
    const app = boot(baseHandlers());
    await tick();
    const preset = app.form.querySelector('[data-preset="coding"]');
    preset.click();
    const wanted = {
      dimensions: preset.dataset.dimensions.split(' '),
      labels: preset.dataset.labels.split(' '),
      coding_labels: preset.dataset.codingLabels.split(' '),
    };
    for (const [name, values] of Object.entries(wanted)) {
      assert.deepEqual(checked(app.form, name).sort(), [...values].sort(), name + ' matches the preset exactly');
    }
    assert.equal(app.form.querySelector('input[name="preset"]').value, 'coding');
    app.form.querySelector('input[name="labels"][value="sentiment"]').click();
    assert.equal(app.form.querySelector('input[name="preset"]').value, '', 'a manual change clears the preset');
  }

  {
    const app = boot(baseHandlers());
    await tick();
    app.form.querySelector('[data-cq-open]').click();
    const editor = app.form.querySelector('[data-cq-editor]');
    assert.equal(editor.hidden, false);
    const kind = app.form.querySelector('[data-cq="kind"]');
    kind.value = 'score';
    kind.dispatchEvent({type: 'change', bubbles: true, target: kind});
    assert.equal(app.form.querySelector('[data-cq-criteria]').hidden, false);
    const fill = (name, text, criteria) => {
      app.form.querySelector('[data-cq="name"]').value = name;
      app.form.querySelector('[data-cq="text"]').value = text;
      app.form.querySelector('[data-cq="criteria"]').value = criteria;
      app.form.querySelector('[data-cq-add]').click();
    };
    const error = app.form.querySelector('[data-cq-error]');
    fill('needs_follow_up', 'Should a human follow up?', '1: a\n2: b\n3: c\n4: d');
    assert.equal(error.textContent, 'Add exactly five score criteria, one per line.');
    assert.equal(app.form.querySelector('input[name="custom_labels_json"]').value, '[]');
    fill('needs_follow_up', 'Should a human follow up?', '1: a\n2: b\n3: c\n4: d\n5: e');
    const saved = JSON.parse(app.form.querySelector('input[name="custom_labels_json"]').value);
    assert.deepEqual(saved.map(spec => spec.name), ['needs_follow_up']);
    assert.equal(saved[0].criteria.length, 5);
    assert.ok(app.form.querySelector('[data-custom="needs_follow_up"]'), 'the question shows as a chip');
    assert.equal(editor.hidden, true);
    app.form.querySelector('[data-cq-open]').click();
    fill('needs_follow_up', 'Again?', '');
    assert.equal(error.textContent, 'That question name is already in use.');
    fill('sentiment', 'Again?', '');
    assert.equal(error.textContent, 'That question name is already in use.');
    app.form.querySelector('[data-cq-cancel]').click();
    app.form.querySelector('[data-cq-remove="needs_follow_up"]').click();
    assert.equal(app.form.querySelector('input[name="custom_labels_json"]').value, '[]');
  }

  {
    const app = boot(baseHandlers());
    await tick();
    app.window.InsightsRunForm.draftQuestion(app.form, {name: 'about_refunds', kind: 'noul', text: 'Did the agent handle refunds well?'});
    assert.deepEqual(visibleStep(app.form), ['2'], 'a drafted question opens the analysis step');
    assert.equal(app.form.querySelector('[data-cq-editor]').hidden, false);
    assert.equal(app.form.querySelector('[data-cq="name"]').value, 'about_refunds');
    app.form.querySelector('[data-cq-add]').click();
    const added = JSON.parse(app.form.querySelector('input[name="custom_labels_json"]').value);
    assert.deepEqual(added.map(spec => spec.name), ['about_refunds']);
  }

  {
    const app = boot(baseHandlers());
    await tick();
    app.calls.length = 0;
    app.form.querySelector('#menu').dispatchEvent({type: 'facets:closed', bubbles: true});
    await tick();
    assert.equal(app.calls.length, 0, 'closing the menu without a change fetches nothing');
    app.form.querySelector('input[name="facet_agent_name"]').checked = true;
    app.form.querySelector('#menu').dispatchEvent({type: 'facets:closed', bubbles: true});
    await tick();
    const requests = app.calls.filter(call => call.url.startsWith('/insights/facets'));
    assert.equal(requests.length, 1, 'one refetch after a change');
    const params = new URLSearchParams(requests[0].url.split('?')[1]);
    assert.deepEqual(params.getAll('facet_agent_name'), ['support-bot']);
    assert.equal(params.get('window_days'), '7');
  }

  {
    const app = boot({
      ...baseHandlers(),
      '/insights/uploads': () => response(JSON.stringify({kind: 'finder', path: '/runs/.uploads/finder-1.json'}), {status: 201}),
    });
    await tick();
    const input = app.form.querySelector('input[data-file]');
    input.files = [new File(['{}'], 'export.json', {type: 'application/json'})];
    input.dispatchEvent({type: 'change', bubbles: true, target: input});
    await tick();
    await tick();
    const traceFile = app.form.querySelector('input[name="trace_file"]');
    assert.equal(traceFile.value, '/runs/.uploads/finder-1.json');
    assert.equal(traceFile.dataset.kind, 'finder', 'the upload route says which kind of file it read');
    assert.ok(!app.calls.some(call => call.url === '/insights/snapshot-preview'), 'a Finder export is not measured');
    assert.equal(app.form.querySelector('[data-file-name]').value, 'export.json');
    assert.equal(app.form.querySelector('[data-file-status]').textContent, 'Finder export is ready.');
    const upload = app.calls.find(call => call.url === '/insights/uploads');
    assert.equal(upload.options.body.get('csrf'), 'token', "the upload carries the form's own token");
    assert.equal(upload.options.body.get('kind'), null, 'the browser no longer says what kind of file it is sending');
  }

  {
    const app = boot({
      ...baseHandlers(),
      '/insights/uploads': () => response(JSON.stringify({kind: 'snapshot', path: '/tmp/snap.json'}), {status: 201}),
      '/insights/snapshot-preview': () => response('<p>3 traces</p>'),
    });
    await tick();
    const traceFile = app.form.querySelector('input[name="trace_file"]');
    traceFile.value = '/old/finder.json';
    traceFile.dataset.kind = 'finder';
    const input = app.form.querySelector('input[data-file]');
    input.files = [new File(['{}'], 'snap.json', {type: 'application/json'})];
    input.dispatchEvent({type: 'change', bubbles: true, target: input});
    await tick();
    await tick();
    assert.equal(traceFile.value, '/tmp/snap.json', 'a new upload replaces the earlier file');
    assert.equal(traceFile.dataset.kind, 'snapshot');
    assert.equal(app.form.querySelector('[data-file-status]').textContent, 'Trace snapshot is ready.');
    assert.ok(app.calls.some(call => call.url === '/insights/snapshot-preview'), 'a snapshot is measured');
  }

  {
    const app = boot({
      ...baseHandlers(),
      '/insights/runs': () => response(fixtures.rejected, {status: 422}),
    });
    await tick();
    app.form.querySelector('[data-irf-next]').click();
    await tick();
    app.form.querySelector('[data-irf-next]').click();
    await tick();
    assert.deepEqual(visibleStep(app.form), ['3']);
    const original = app.form;
    app.form.querySelector('[data-irf-start]').click();
    await tick();
    await tick();
    const replaced = app.document.getElementById('insights-new-form');
    assert.notEqual(replaced, original, 'the 422 response replaces the form');
    assert.deepEqual(visibleStep(replaced), ['3'], 'the user stays on the step they were on');
    const error = replaced.querySelector('#insights-run-error');
    assert.equal(error.hidden, false);
    assert.match(error.textContent, /Browse to choose a trace file first/);
    assert.ok(replaced.classList.contains('irf-ready'), 'the replacement is mounted');
    assert.equal(app.processed.includes(replaced), true);
    const post = app.calls.find(call => call.url === '/insights/runs');
    assert.equal(post.options.body.get('csrf'), 'token');
    assert.equal(post.options.body.get('mount'), 'page');
  }

  {
    const app = boot({
      ...baseHandlers(),
      '/insights/runs': () => response('', {status: 200, redirected: true, url: '/insights/run-1'}),
    });
    await tick();
    app.form.querySelector('[data-irf-next]').click();
    await tick();
    app.form.querySelector('[data-irf-next]').click();
    await tick();
    app.form.querySelector('[data-irf-start]').click();
    await tick();
    assert.equal(app.location.assigned, '/insights/run-1', 'a started run opens its page');
  }

  {
    const FULL = '<div data-part="estimate"><section>FULL ESTIMATE</section></div>'
      + '<div data-part="compact">full says 100 traces</div><div data-part="plan"><ol class="irf-stages"><li>Stage</li></ol></div>';
    const app = boot({
      ...baseHandlers(),
      '/insights/new/plan': url => response(url.includes('compact=1') ? 'compact says 100 traces' : FULL),
    });
    await tick();
    const compact = app.form.querySelector('#insights-run-compact');
    const planCalls = () => app.calls.filter(call => call.url.startsWith('/insights/new/plan'));
    assert.equal(planCalls().length, 1, 'the step bar line loads on mount');
    assert.ok(planCalls()[0].url.includes('compact=1'));
    await tick();
    assert.equal(compact.textContent, 'compact says 100 traces');

    app.calls.length = 0;
    const limit = app.form.querySelector('input[name="limit"]');
    limit.value = '50';
    limit.dispatchEvent({type: 'change', bubbles: true, target: limit});
    limit.value = '60';
    limit.dispatchEvent({type: 'change', bubbles: true, target: limit});
    await tick();
    assert.equal(planCalls().length, 0, 'nothing is requested before the debounce');
    await new Promise(resolve => setTimeout(resolve, 600));
    assert.equal(planCalls().length, 1, 'two quick changes make one compact request');
    assert.ok(planCalls()[0].url.includes('compact=1'));
    assert.equal(new URLSearchParams(planCalls()[0].url.split('?')[1]).get('limit'), '60');

    app.calls.length = 0;
    assert.equal(app.form.querySelector('.irf-review').hidden, true, 'the review panel stays hidden until a plan is requested');
    app.form.querySelector('[data-irf-next]').click();
    await tick();
    app.form.querySelector('[data-irf-next]').click();
    await tick();
    await tick();
    assert.deepEqual(visibleStep(app.form), ['3']);
    assert.equal(app.form.querySelector('.irf-review').hidden, false);
    const full = planCalls().filter(call => !call.url.includes('compact=1'));
    assert.equal(full.length, 1, 'opening the review step loads the full estimate once');
    assert.match(app.form.querySelector('#insights-run-estimate').innerHTML, /FULL ESTIMATE/);
    assert.match(app.form.querySelector('#insights-run-plan').innerHTML, /Stage/);
    assert.equal(compact.textContent, 'full says 100 traces', 'the full response also refreshes the step bar line');

    app.calls.length = 0;
    const model = app.form.querySelector('#insights-run-models input');
    model.dispatchEvent({type: 'change', bubbles: true, target: model});
    await tick();
    await tick();
    assert.equal(planCalls().filter(call => !call.url.includes('compact=1')).length, 1, 'a model change reloads the full estimate');
  }
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
