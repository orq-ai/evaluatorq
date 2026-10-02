/* Controller for the Insights run form. The server renders every control and list; this file only drives them. */
(function (global) {
  'use strict';

  const POPULATION_SOURCES = ['recent', 'query'];
  const NAME_PATTERN = /^[a-z][a-z0-9_]*$/;
  const contexts = new WeakMap();

  function esc(text) {
    return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function field(form, name) {
    return form.querySelector('[name="' + name + '"]');
  }

  function checkedValues(form, name) {
    return Array.from(form.querySelectorAll('input[name="' + name + '"]:checked')).map(function (input) { return input.value; });
  }

  function activeSource(form) {
    const checked = form.querySelector('input[name="source"]:checked');
    return checked ? checked.value : 'recent';
  }

  function inRange(input, min, max) {
    const text = String(input.value).trim();
    const value = Number(text);
    return text !== '' && Number.isInteger(value) && value >= min && value <= max;
  }

  function customLabels(form) {
    try {
      const parsed = JSON.parse(field(form, 'custom_labels_json').value || '[]');
      return Array.isArray(parsed) ? parsed : [];
    } catch (failure) {
      return [];
    }
  }

  function words(text) {
    return String(text || '').split(' ').filter(Boolean);
  }

  function formPairs(form) {
    const active = activeSource(form);
    const pairs = [];
    form.querySelectorAll('input[name], textarea[name], select[name]').forEach(function (element) {
      const type = element.type;
      if (element.disabled || type === 'file' || type === 'button' || type === 'submit') return;
      if ((type === 'checkbox' || type === 'radio') && !element.checked) return;
      if (element.name.indexOf('facet_') === 0 && POPULATION_SOURCES.indexOf(active) < 0) return;
      pairs.push([element.name, element.value]);
    });
    const chosen = form.querySelector('[data-file-name="' + active + '"]');
    pairs.push(['source_name', chosen ? chosen.value : '']);
    return pairs;
  }

  function showError(ctx, message) {
    const element = ctx.form.querySelector('#insights-run-error');
    element.textContent = message || '';
    element.hidden = !message;
  }

  function updateSource(ctx) {
    const active = activeSource(ctx.form);
    ctx.form.querySelectorAll('[data-source]').forEach(function (element) {
      element.hidden = words(element.dataset.source).indexOf(active) < 0;
    });
  }

  function showStep(ctx, step, keepError) {
    const form = ctx.form;
    ctx.step = step;
    form.querySelectorAll('[data-step]').forEach(function (section) {
      section.hidden = Number(section.dataset.step) !== step;
    });
    form.querySelectorAll('.irf-steps li').forEach(function (item, index) {
      item.classList.toggle('active', index + 1 === step);
      item.classList.toggle('completed', index + 1 < step);
      if (index + 1 === step) item.setAttribute('aria-current', 'step');
      else item.removeAttribute('aria-current');
    });
    form.querySelector('[data-irf-back]').hidden = step === 1;
    form.querySelector('[data-irf-next]').hidden = step === 3;
    form.querySelector('[data-irf-start]').hidden = step !== 3;
    if (!keepError) showError(ctx, '');
    updateSource(ctx);
    if (step === 3) loadPlan(ctx);
  }

  function validate(ctx, step) {
    const form = ctx.form;
    const active = activeSource(form);
    if (step === 1) {
      if (active === 'query' && !field(form, 'query').value.trim()) return 'Enter a question to find matching traces.';
      if (active === 'finder' && !field(form, 'finder_export').value.trim()) return 'Browse to choose a Finder export first.';
      if (active === 'snapshot' && !field(form, 'snapshot_path').value.trim()) return 'Browse to choose a trace snapshot file first.';
      if (POPULATION_SOURCES.indexOf(active) >= 0) {
        if (!inRange(field(form, 'window_days'), 1, 90) || !inRange(field(form, 'limit'), 1, 5000)) return 'Enter a valid window and trace limit.';
        if (form.querySelector('#insights-facet-options').getAttribute('aria-busy') === 'true') return 'Wait for the filter values to load.';
      }
    }
    if (step === 2) {
      if (!form.querySelector('[data-cq-editor]').hidden) return 'Finish or cancel the custom question first.';
      const chosen = checkedValues(form, 'dimensions').length + checkedValues(form, 'labels').length
        + checkedValues(form, 'coding_labels').length + customLabels(form).length;
      if (!chosen) return 'Select at least one question or grouping.';
    }
    if (step === 3 && !inRange(field(form, 'parallelism'), 1, 200)) return 'Enter a valid number of parallel requests.';
    return '';
  }

  async function measureSnapshot(ctx) {
    const form = ctx.form;
    const path = field(form, 'snapshot_path').value.trim();
    const host = form.querySelector('#insights-snapshot-preview');
    if (!path) {
      host.textContent = '';
      return false;
    }
    if (ctx.measuredPath === path) return true;
    if (ctx.snapshotAbort) ctx.snapshotAbort.abort();
    const abort = new AbortController();
    ctx.snapshotAbort = abort;
    host.textContent = 'Measuring the projected input…';
    try {
      const body = new URLSearchParams({csrf: field(form, 'csrf').value, snapshot_path: path});
      const response = await fetch('/insights/snapshot-preview', {method: 'POST', body: body, signal: abort.signal});
      const markup = await response.text();
      if (ctx.snapshotAbort !== abort) return false;
      host.innerHTML = markup;
      if (response.ok) ctx.measuredPath = path;
      return response.ok;
    } catch (failure) {
      if (failure.name !== 'AbortError' && ctx.snapshotAbort === abort) host.textContent = 'Could not measure this trace file.';
      return false;
    } finally {
      if (ctx.snapshotAbort === abort) ctx.snapshotAbort = null;
    }
  }

  function facetKey(form) {
    const selected = Array.from(form.querySelectorAll('input[name^="facet_"]:checked')).map(function (input) {
      return input.name + '=' + input.value;
    });
    return field(form, 'window_days').value.trim() + '|' + selected.sort().join('&');
  }

  async function refreshFacets(ctx, retry) {
    const form = ctx.form;
    if (POPULATION_SOURCES.indexOf(activeSource(form)) < 0) return;
    if (!inRange(field(form, 'window_days'), 1, 90)) return;
    const key = facetKey(form);
    if (!retry && key === ctx.facetKey) return;
    const params = new URLSearchParams({window_days: field(form, 'window_days').value.trim()});
    if (retry) params.set('retry', '1');
    form.querySelectorAll('input[name^="facet_"]:checked').forEach(function (input) { params.append(input.name, input.value); });
    if (ctx.facetAbort) ctx.facetAbort.abort();
    const abort = new AbortController();
    ctx.facetAbort = abort;
    const host = form.querySelector('#insights-facet-options');
    host.setAttribute('aria-busy', 'true');
    try {
      const response = await fetch('/insights/facets?' + params.toString(), {signal: abort.signal});
      if (!response.ok) throw new Error('Filter values could not be loaded for this window.');
      const markup = await response.text();
      if (ctx.facetAbort !== abort) return;
      host.innerHTML = markup;
      ctx.facetKey = key;
    } catch (failure) {
      if (ctx.facetAbort === abort && failure.name !== 'AbortError') {
        host.insertAdjacentHTML('afterbegin', '<p class="insights-facet-unavailable" role="status">' + esc(failure.message)
          + ' <button type="button" data-retry-facets>Retry</button>. Your selected filters are kept.</p>');
      }
    } finally {
      if (ctx.facetAbort === abort) {
        ctx.facetAbort = null;
        host.setAttribute('aria-busy', 'false');
      }
    }
  }

  async function loadPlan(ctx) {
    const form = ctx.form;
    const host = form.querySelector('#insights-run-plan');
    const sequence = ++ctx.planSequence;
    const params = new URLSearchParams();
    formPairs(form).forEach(function (pair) {
      if (pair[0] !== 'csrf' && pair[0] !== 'mount') params.append(pair[0], pair[1]);
    });
    host.textContent = 'Working out the stages…';
    try {
      const response = await fetch('/insights/new/plan?' + params.toString());
      const markup = await response.text();
      if (sequence === ctx.planSequence) host.innerHTML = markup;
    } catch (failure) {
      if (sequence === ctx.planSequence) host.textContent = 'The expected stages could not be loaded.';
    }
  }

  function markPreset(ctx, id) {
    field(ctx.form, 'preset').value = id || '';
    ctx.form.querySelectorAll('[data-preset]').forEach(function (button) {
      const on = button.dataset.preset === id;
      button.classList.toggle('on', on);
      button.setAttribute('aria-pressed', String(on));
    });
  }

  function updateMistakeHint(ctx) {
    ctx.form.querySelector('[data-hint-made-errors]').hidden = checkedValues(ctx.form, 'labels').indexOf('made_errors') >= 0;
  }

  function applyPreset(ctx, button) {
    const chosen = {
      dimensions: words(button.dataset.dimensions),
      labels: words(button.dataset.labels),
      coding_labels: words(button.dataset.codingLabels),
    };
    Object.keys(chosen).forEach(function (name) {
      ctx.form.querySelectorAll('input[name="' + name + '"]').forEach(function (input) {
        input.checked = chosen[name].indexOf(input.value) >= 0;
      });
    });
    markPreset(ctx, button.dataset.preset);
    updateMistakeHint(ctx);
  }

  function renderCustom(ctx, specs) {
    field(ctx.form, 'custom_labels_json').value = JSON.stringify(specs);
    ctx.form.querySelector('[data-custom-list]').innerHTML = specs.map(function (spec) {
      const name = esc(spec.name);
      return '<span class="tg on" data-custom="' + name + '"><span class="tg-title">' + name + '</span><small>custom</small>'
        + '<button type="button" data-cq-remove="' + name + '" aria-label="Remove question ' + name + '">&times;</button></span>';
    }).join('');
  }

  function editor(ctx) {
    const form = ctx.form;
    return {
      box: form.querySelector('[data-cq-editor]'),
      name: form.querySelector('[data-cq="name"]'),
      kind: form.querySelector('[data-cq="kind"]'),
      text: form.querySelector('[data-cq="text"]'),
      criteria: form.querySelector('[data-cq="criteria"]'),
      criteriaField: form.querySelector('[data-cq-criteria]'),
      error: form.querySelector('[data-cq-error]'),
    };
  }

  function resetEditor(ctx) {
    const e = editor(ctx);
    e.name.value = '';
    e.kind.value = 'noul';
    e.text.value = '';
    e.criteria.value = '';
    e.criteriaField.hidden = true;
    e.error.hidden = true;
    e.box.hidden = true;
  }

  function addCustomQuestion(ctx) {
    const e = editor(ctx);
    const name = e.name.value.trim();
    const kind = e.kind.value;
    const instructions = e.text.value.trim();
    const lines = e.criteria.value.split('\n').map(function (line) { return line.trim(); }).filter(Boolean);
    const specs = customLabels(ctx.form);
    const used = specs.map(function (spec) { return spec.name; });
    ctx.form.querySelectorAll('input[name="labels"], input[name="coding_labels"]').forEach(function (input) { used.push(input.value); });
    let problem = '';
    if (!name || !instructions) problem = 'Enter a name and a question.';
    else if (!NAME_PATTERN.test(name)) problem = 'Use lowercase letters, digits and underscores in the name, starting with a letter.';
    else if (kind === 'choice' && lines.length < 2) problem = 'Add at least two answer options, one per line.';
    else if (kind === 'score' && lines.length !== 5) problem = 'Add exactly five score criteria, one per line.';
    else if (used.indexOf(name) >= 0) problem = 'That question name is already in use.';
    if (problem) {
      e.error.textContent = problem;
      e.error.hidden = false;
      return;
    }
    const spec = {name: name, kind: kind, instructions: instructions};
    if (kind === 'choice') spec.criteria = Object.fromEntries(lines.map(function (line) { return [line, line]; }));
    if (kind === 'score') spec.criteria = lines;
    specs.push(spec);
    renderCustom(ctx, specs);
    resetEditor(ctx);
  }

  async function upload(ctx, input) {
    const form = ctx.form;
    const kind = input.dataset.file;
    const file = input.files && input.files[0];
    if (!file) return;
    const status = form.querySelector('[data-file-status="' + kind + '"]');
    status.textContent = 'Uploading…';
    try {
      const body = new FormData();
      body.set('csrf', field(form, 'csrf').value);
      body.set('kind', kind);
      body.set('file', file);
      const response = await fetch('/insights/uploads', {method: 'POST', body: body});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Upload failed.');
      field(form, kind === 'finder' ? 'finder_export' : 'snapshot_path').value = result.path;
      form.querySelector('[data-file-name="' + kind + '"]').value = file.name;
      status.textContent = kind === 'finder' ? 'Finder export is ready.' : 'Trace snapshot is ready.';
      showError(ctx, '');
      if (kind === 'snapshot') {
        ctx.measuredPath = null;
        await measureSnapshot(ctx);
      }
    } catch (failure) {
      status.textContent = failure.message || 'Upload failed.';
    } finally {
      input.value = '';
    }
  }

  async function goNext(ctx) {
    const message = validate(ctx, ctx.step);
    if (message) {
      showError(ctx, message);
      return;
    }
    if (ctx.step === 1 && activeSource(ctx.form) === 'snapshot') {
      const next = ctx.form.querySelector('[data-irf-next]');
      next.disabled = true;
      const measured = await measureSnapshot(ctx);
      next.disabled = false;
      if (!measured) {
        showError(ctx, 'Could not measure this trace file. Browse to choose a valid snapshot.');
        return;
      }
    }
    showStep(ctx, ctx.step + 1);
  }

  async function submit(ctx) {
    const form = ctx.form;
    for (const step of [1, 2, 3]) {
      const message = validate(ctx, step);
      if (message) {
        showStep(ctx, step);
        showError(ctx, message);
        return;
      }
    }
    const start = form.querySelector('[data-irf-start]');
    const previousStep = ctx.step;
    start.disabled = true;
    start.textContent = 'Starting…';
    try {
      const response = await fetch('/insights/runs', {method: 'POST', body: new URLSearchParams(formPairs(form))});
      if (response.redirected) {
        global.location.assign(response.url);
        return;
      }
      const doc = new DOMParser().parseFromString(await response.text(), 'text/html');
      const replacement = doc.getElementById('insights-new-form');
      if (!replacement) throw new Error('The run could not be started. Your selections are still here.');
      form.replaceWith(replacement);
      mountAt(replacement, previousStep, true);
    } catch (failure) {
      showError(ctx, failure.message || 'The run could not be started. Your selections are still here.');
      start.disabled = false;
      start.textContent = 'Start run';
    }
  }

  function onClick(ctx, event) {
    const target = event.target;
    const hit = function (selector) { return target.closest(selector); };
    let element;
    if (hit('[data-irf-next]')) goNext(ctx);
    else if (hit('[data-irf-back]')) showStep(ctx, Math.max(1, ctx.step - 1));
    else if ((element = hit('[data-preset]'))) applyPreset(ctx, element);
    else if ((element = hit('[data-browse]'))) ctx.form.querySelector('input[data-file="' + element.dataset.browse + '"]').click();
    else if (hit('[data-cq-open]')) {
      const e = editor(ctx);
      e.box.hidden = false;
      e.name.focus();
    } else if (hit('[data-cq-add]')) addCustomQuestion(ctx);
    else if (hit('[data-cq-cancel]')) resetEditor(ctx);
    else if ((element = hit('[data-cq-remove]'))) {
      renderCustom(ctx, customLabels(ctx.form).filter(function (spec) { return spec.name !== element.dataset.cqRemove; }));
    } else if (hit('[data-retry-facets]')) refreshFacets(ctx, true);
  }

  function onChange(ctx, event) {
    const target = event.target;
    if (target.name === 'source') {
      showError(ctx, '');
      updateSource(ctx);
      refreshFacets(ctx);
    } else if (target.name === 'window_days') {
      refreshFacets(ctx);
    } else if (target.name === 'dimensions' || target.name === 'labels' || target.name === 'coding_labels') {
      markPreset(ctx, '');
      updateMistakeHint(ctx);
    } else if (target.matches('input[data-file]')) {
      upload(ctx, target);
    } else if (target.matches('[data-cq="kind"]')) {
      const e = editor(ctx);
      e.criteriaField.hidden = target.value === 'noul';
      e.criteria.setAttribute('placeholder', target.value === 'score' ? '1: Lowest\n2: Low\n3: Moderate\n4: High\n5: Highest' : 'option_one\noption_two');
    }
  }

  function onKeydown(ctx, event) {
    if (event.key !== 'Enter') return;
    const target = event.target;
    if (target.matches('[data-cq]') && target.type !== 'textarea') {
      event.preventDefault();
      addCustomQuestion(ctx);
    } else if (target.matches('input[type="search"]')) {
      event.preventDefault();
    }
  }

  function mountAt(form, step, keepError) {
    if (!form) return null;
    if (contexts.has(form)) return contexts.get(form);
    const ctx = {form: form, step: 1, facetKey: null, facetAbort: null, snapshotAbort: null, measuredPath: null, planSequence: 0};
    contexts.set(form, ctx);
    form.addEventListener('click', function (event) { onClick(ctx, event); });
    form.addEventListener('change', function (event) { onChange(ctx, event); });
    form.addEventListener('keydown', function (event) { onKeydown(ctx, event); });
    form.addEventListener('facets:closed', function () { refreshFacets(ctx); });
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      submit(ctx);
    });
    if (global.htmx && typeof global.htmx.process === 'function') global.htmx.process(form);
    form.classList.add('irf-ready');
    showStep(ctx, step || 1, keepError);
    refreshFacets(ctx);
    return ctx;
  }

  function mount(form) {
    return mountAt(form, 1, false);
  }

  function draftQuestion(form, draft) {
    const ctx = mountAt(form, 1, false);
    showStep(ctx, 2);
    const e = editor(ctx);
    e.box.hidden = false;
    e.name.value = draft.name;
    e.kind.value = draft.kind;
    e.text.value = draft.text;
    e.criteriaField.hidden = draft.kind === 'noul';
    e.name.focus();
  }

  function autoMount() {
    mount(document.getElementById('insights-new-form'));
  }

  global.InsightsRunForm = {mount: mount, draftQuestion: draftQuestion};
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', autoMount);
  else autoMount();
})(window);
