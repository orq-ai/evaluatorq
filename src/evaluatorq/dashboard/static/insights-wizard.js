/* Progressive controls for the dashboard's Insights run form. */
(function () {
  const form = document.getElementById('insights-new-form');
  if (!form) return;
  const sections = Array.from(form.querySelectorAll('[data-step]'));
  const stepLabels = Array.from(document.querySelectorAll('.insights-wizard-steps span'));
  const back = form.querySelector('[data-wizard-back]');
  const next = form.querySelector('[data-wizard-next]');
  const start = form.querySelector('[data-wizard-start]');
  const error = document.getElementById('insights-wizard-error');
  const preview = document.getElementById('insights-run-preview');
  const snapshotPreview = document.getElementById('insights-snapshot-preview');
  const facetOptions = document.getElementById('insights-facet-options');
  let facetRequest = null;
  let facetLoadedWindow = null;
  let step = 1;

  function setFacetLoading(loading) {
    facetOptions.setAttribute('aria-busy', String(loading));
    facetOptions.querySelectorAll('input[name^="facet_"]').forEach(function (input) {
      input.disabled = loading || selected('source')[0] === 'finder';
    });
    let status = facetOptions.querySelector('.insights-facet-loading');
    if (loading && !status) {
      status = document.createElement('p');
      status.className = 'insights-facet-loading';
      status.setAttribute('role', 'status');
      status.textContent = 'Refreshing facet values…';
      facetOptions.prepend(status);
    } else if (!loading && status) {
      status.remove();
    }
  }

  function selected(name) {
    return Array.from(form.querySelectorAll('input[name="' + name + '"]:checked')).map(function (input) { return input.value; });
  }

  function showError(message) {
    error.textContent = message;
    error.hidden = !message;
  }

  function updateSource() {
    const source = selected('source')[0];
    form.querySelectorAll('[data-source]').forEach(function (field) {
      field.hidden = !field.dataset.source.split(' ').includes(source);
    });
    facetOptions.querySelectorAll('input').forEach(function (input) { input.disabled = source === 'finder' || facetOptions.getAttribute('aria-busy') === 'true'; });
    if (source === 'finder' && facetRequest) {
      facetRequest.abort();
      facetRequest = null;
      setFacetLoading(false);
    }
  }

  async function refreshFacets(retry) {
    if (selected('source')[0] === 'finder') return;
    const windowDays = form.elements.window_days.value;
    const forced = retry === true;
    if (!forced && facetRequest && facetRequest.windowDays === windowDays) return;
    if (facetRequest) {
      facetRequest.abort();
      facetRequest = null;
    }
    if (!form.elements.window_days.checkValidity() || (!forced && facetLoadedWindow === windowDays)) {
      setFacetLoading(false);
      return;
    }
    const request = new AbortController();
    request.windowDays = windowDays;
    facetRequest = request;
    const params = new URLSearchParams({ window_days: windowDays });
    if (forced) params.set('retry', '1');
    facetOptions.querySelectorAll('input[name^="facet_"]:checked').forEach(function (input) {
      params.append(input.name, input.value);
    });
    setFacetLoading(true);
    try {
      const response = await fetch('/insights/facets?' + params.toString(), { signal: request.signal });
      if (!response.ok) throw new Error('Facet values could not be loaded for this window.');
      const markup = await response.text();
      if (facetRequest !== request) return;
      facetOptions.innerHTML = markup;
      facetLoadedWindow = windowDays;
      updateSource();
    } catch (failure) {
      if (facetRequest === request && failure.name !== 'AbortError') {
        const message = '<p class="insights-facet-unavailable" role="status">Could not load filter choices from Orq. In <a href="/settings" target="_blank" rel="noopener">Settings → Authentication</a>, select an Orq profile with trace access and Save. If you use the environment API key, set <code>ORQ_API_KEY</code> for the dashboard and restart it. Then click <button type="button" data-retry-facets>Retry</button>. Your selected filters are kept.</p>';
        facetOptions.querySelector('.insights-facet-unavailable')?.remove();
        if (facetOptions.querySelector('input[name^="facet_"]')) facetOptions.insertAdjacentHTML('afterbegin', message);
        else facetOptions.innerHTML = message;
      }
    } finally {
      if (facetRequest === request) {
        facetRequest = null;
        setFacetLoading(false);
      }
    }
  }

  async function refreshSnapshotPreview() {
    const path = form.elements.snapshot_path.value.trim();
    if (snapshotRequest) snapshotRequest.abort();
    if (selected('source')[0] !== 'snapshot' || !path) {
      snapshotPreview.replaceChildren();
      if (step === 3) updatePreview();
      return false;
    }
    const request = new AbortController();
    snapshotRequest = request;
    snapshotPreview.textContent = 'Measuring projected input…';
    if (step === 3) updatePreview();
    const body = new FormData();
    body.set('csrf', form.elements.csrf.value);
    body.set('snapshot_path', path);
    try {
      const response = await fetch('/insights/snapshot-preview', { method: 'POST', body, signal: request.signal });
      const html = await response.text();
      if (snapshotRequest !== request) return false;
      snapshotPreview.innerHTML = html;
      if (step === 3) updatePreview();
      return response.ok;
    } catch (failure) {
      if (failure.name !== 'AbortError' && snapshotRequest === request) {
        snapshotPreview.textContent = 'Could not measure this trace file. Check the path and try again.';
        if (step === 3) updatePreview();
      }
      return false;
    } finally {
      if (snapshotRequest === request) snapshotRequest = null;
    }
  }

  function plan() {
    const source = selected('source')[0];
    const dimensions = selected('dimensions');
    const labels = effectiveLabels(dimensions);
    const populationStage = {
      finder: 'Load Finder matches',
      query: 'Find matching traces',
      recent: 'Load recent traces'
    }[source];
    const stages = [populationStage];
    let classifyStage = 'Prepare traces';
    if (source === 'query') {
      classifyStage = labels.length ? 'Match and classify traces' : 'Match traces';
    } else if (labels.length) {
      classifyStage = 'Classify traces';
    }
    stages.push(classifyStage);
    stages.push('Summarize traces');
    dimensions.forEach(function (name) { stages.push('Cluster and map ' + name); });
    stages.push(dimensions.includes('intent') && labels.includes('customer_satisfaction') ? 'Build priority matrix' : 'Check priority matrix');
    stages.push('Save run');
    return stages;
  }

  function updatePreview() {
    const source = selected('source')[0];
    const dimensions = selected('dimensions');
    const labels = effectiveLabels(dimensions);
    const displayedLabels = labels.map(function (name) {
      return name === 'sentiment' && !selected('labels').includes(name) ? 'sentiment (added automatically)' : name;
    });
    const summary = document.createElement('p');
    let sourceLabel = 'Recent traces';
    if (source === 'query') sourceLabel = 'Search by question';
    else if (source === 'finder') sourceLabel = 'Finder export';
    summary.textContent = 'Source: ' + sourceLabel +
      ' · Labels: ' + (displayedLabels.join(', ') || 'none') + ' · Dimensions: ' + (dimensions.join(', ') || 'none');
    const facets = document.createElement('p');
    const selectedFacets = Array.from(facetOptions.querySelectorAll('input[name^="facet_"]:checked')).map(function (input) {
      return input.name.slice(6).replaceAll('_', ' ') + ' = ' + input.value;
    });
    facets.textContent = source === 'finder' ? 'Facets: fixed by Finder export' : source === 'snapshot' ? 'Facets: fixed by local file' : 'Facets: ' + (selectedFacets.join(', ') || 'all values');
    const coverage = document.createElement('div');
    if (source === 'snapshot') {
      coverage.innerHTML = snapshotPreview.innerHTML || '<p>Measuring projected input…</p>';
    } else {
      coverage.textContent = 'Truncation can be measured after the traces load from Orq. The run page will show the exact amount.';
    }
    const heading = document.createElement('h4');
    heading.textContent = 'Expected stages';
    const list = document.createElement('ol');
    plan().forEach(function (name) {
      const item = document.createElement('li');
      item.textContent = name;
      list.appendChild(item);
    });
    preview.replaceChildren(summary, facets, coverage, heading, list);
  }

  function effectiveLabels(dimensions) {
    const labels = selected('labels');
    if (dimensions.includes('sentiment') && !labels.includes('sentiment')) labels.push('sentiment');
    return labels;
  }

  function showStep(value) {
    step = value;
    form.classList.add('wizard-ready');
    sections.forEach(function (section) { section.hidden = Number(section.dataset.step) !== step; });
    stepLabels.forEach(function (label, index) {
      label.classList.toggle('active', index + 1 === step);
      label.classList.toggle('completed', index + 1 < step);
      if (index + 1 === step) label.setAttribute('aria-current', 'step');
      else label.removeAttribute('aria-current');
    });
    back.hidden = step === 1;
    next.hidden = step === 3;
    start.hidden = step !== 3;
    showError('');
    updateSource();
    if (step === 3) updatePreview();
  }

  function validateStep() {
    const source = selected('source')[0];
    if (step === 1 && source === 'query' && !form.elements.query.value.trim()) return 'Enter a question to find matching traces.';
    if (step === 1 && source === 'finder' && !form.elements.finder_export.value.trim()) return 'Enter a Finder JSON path.';
    if (step === 1 && source === 'snapshot' && !form.elements.snapshot_path.value.trim()) return 'Enter a trace snapshot JSON path.';
    if (step === 1 && source !== 'finder' && source !== 'snapshot' && (!form.elements.window_days.checkValidity() || !form.elements.limit.checkValidity())) return 'Enter a valid window and trace limit.';
    if (step === 1 && source !== 'finder' && source !== 'snapshot' && facetOptions.getAttribute('aria-busy') === 'true') return 'Wait for the facet values to load.';
    if (step === 2 && !selected('labels').length && !selected('dimensions').length && !form.elements.coding_analysis.checked) return 'Select at least one label or dimension.';
    if (step === 2 && !form.elements.parallelism.checkValidity()) return 'Enter a valid parallel request count.';
    return '';
  }

  back.addEventListener('click', function () { showStep(Math.max(1, step - 1)); });
  next.addEventListener('click', async function () {
    const message = validateStep();
    if (message) { showError(message); return; }
    if (step === 1 && selected('source')[0] === 'snapshot') {
      clearTimeout(snapshotTimer);
      next.disabled = true;
      const valid = await refreshSnapshotPreview();
      next.disabled = false;
      if (!valid) { showError('Could not load the trace file preview. Check the path and try again.'); return; }
    }
    showStep(Math.min(3, step + 1));
  });
  form.elements.snapshot_path.addEventListener('input', function () {
    clearTimeout(snapshotTimer);
    showError('');
    snapshotTimer = setTimeout(refreshSnapshotPreview, 450);
  });
  form.addEventListener('change', function (event) {
    updateSource();
    if (event.target.name === 'window_days' || event.target.name === 'source') refreshFacets();
    if (event.target.name === 'source') { showError(''); refreshSnapshotPreview(); }
    if (event.target.matches('input[name^="facet_"]')) updateFacetChips();
    if (step === 3) updatePreview();
  });
  form.addEventListener('input', function (event) {
    if (event.target.name === 'window_days') refreshFacets();
  });
  facetOptions.addEventListener('click', function (event) {
    if (event.target.closest('[data-retry-facets]')) refreshFacets(true);
  });
  form.addEventListener('submit', function (event) {
    for (const value of [1, 2]) {
      step = value;
      const message = validateStep();
      if (message) { event.preventDefault(); showStep(value); showError(message); return; }
    }
    step = 3;
    start.disabled = true;
    start.textContent = 'Starting…';
  });
  showStep(1);
  refreshFacets();
})();
