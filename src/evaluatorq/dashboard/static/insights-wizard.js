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
  let step = 1;

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
  }

  function plan() {
    const source = selected('source')[0];
    const dimensions = selected('dimensions');
    const labels = effectiveLabels(dimensions);
    const stages = [source === 'finder' ? 'Load Finder matches' : source === 'query' ? 'Find matching traces' : 'Load recent traces'];
    stages.push(source === 'query' && labels.length ? 'Match and classify traces' : source === 'query' ? 'Match traces' : labels.length ? 'Classify traces' : 'Prepare traces');
    stages.push('Summarize traces');
    dimensions.forEach(function (name) { stages.push('Cluster ' + name); });
    stages.push(dimensions.includes('intent') && labels.includes('customer_satisfaction') ? 'Build priority matrix' : 'Check priority matrix');
    stages.push('Save run');
    return stages;
  }

  function updatePreview() {
    const source = selected('source')[0];
    const dimensions = selected('dimensions');
    const labels = effectiveLabels(dimensions);
    const summary = document.createElement('p');
    summary.textContent = 'Source: ' + (source === 'query' ? 'Search by question' : source === 'finder' ? 'Finder export' : 'Recent traces') +
      ' · Labels: ' + (labels.join(', ') || 'none') + ' · Dimensions: ' + (dimensions.join(', ') || 'none');
    const heading = document.createElement('h4');
    heading.textContent = 'Expected stages';
    const list = document.createElement('ol');
    plan().forEach(function (name) {
      const item = document.createElement('li');
      item.textContent = name;
      list.appendChild(item);
    });
    preview.replaceChildren(summary, heading, list);
  }

  function effectiveLabels(dimensions) {
    const labels = selected('labels');
    if (dimensions.includes('sentiment') && !labels.includes('sentiment')) labels.push('sentiment (added for sentiment dimension)');
    return labels;
  }

  function showStep(value) {
    step = value;
    form.classList.add('wizard-ready');
    sections.forEach(function (section) { section.hidden = Number(section.dataset.step) !== step; });
    stepLabels.forEach(function (label, index) { label.classList.toggle('active', index + 1 === step); });
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
    if (step === 1 && source !== 'finder' && (!form.elements.window_days.checkValidity() || !form.elements.limit.checkValidity())) return 'Enter a valid window and trace limit.';
    if (step === 2 && !selected('labels').length && !selected('dimensions').length) return 'Select at least one label or dimension.';
    if (step === 2 && !form.elements.parallelism.checkValidity()) return 'Enter a valid parallel request count.';
    return '';
  }

  back.addEventListener('click', function () { showStep(Math.max(1, step - 1)); });
  next.addEventListener('click', function () {
    const message = validateStep();
    if (message) { showError(message); return; }
    showStep(Math.min(3, step + 1));
  });
  form.addEventListener('change', function () { updateSource(); if (step === 3) updatePreview(); });
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
})();
