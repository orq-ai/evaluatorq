/* Shared saved-signal view for the review drawer and full trace page. */
(function () {
  'use strict';
  const levels = {A: 'L1', B: 'L2', C: 'L3', D: 'L4'};
  const labels = {L1: 'Structure', L2: 'Tools', L3: 'Autonomy', L4: 'Tags'};
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const value = item => item == null ? 'No value recorded' : typeof item === 'string' ? item : JSON.stringify(item);
  const title = name => name.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
  function state(signal) {
    if (signal.no_basis) return {kind: 'no-basis', label: 'No basis', icon: '—'};
    if (signal.value == null) return {kind: 'no-basis', label: 'No value', icon: '—'};
    if (signal.group === 'D' && signal.value === true) return {kind: 'flagged', label: 'Flagged', icon: '!'};
    if (signal.group === 'D' && signal.value === false) return {kind: 'clear', label: 'Clear', icon: '✓'};
    if (signal.approximate) return {kind: 'approximate', label: 'Approximate', icon: '≈'};
    return {kind: 'measured', label: 'Measured', icon: '•'};
  }
  function preconditions(items) {
    if (!items.length) return '<p class="signal-empty">No preconditions recorded.</p>';
    const metCount = items.filter(item => item.met === true).length;
    return `<div class="signal-check-count">${metCount} of ${items.length} met<meter min="0" max="${items.length}" value="${metCount}" aria-label="Preconditions met"></meter></div><ul class="signal-preconditions">${items.map(item => {
      const kind = item.met === true ? 'met' : item.met === 'partial' ? 'partial' : 'not-met';
      const label = kind === 'met' ? 'Met' : kind === 'partial' ? 'Partial' : 'Not met';
      const icon = kind === 'met' ? '✓' : kind === 'partial' ? '≈' : '×';
      return `<li class="${kind}"><span class="signal-check-icon" aria-label="${label}">${icon}</span><div><div class="signal-check-title"><b>${esc(item.name)}</b><span>${label} · ${item.required ? 'required' : 'optional'}</span></div>${item.detail ? `<p>${esc(item.detail)}</p>` : ''}</div></li>`;
    }).join('')}</ul>`;
  }
  function evidence(items) {
    if (!items.length) return '<p class="signal-empty">No evidence references.</p>';
    return `<ul class="signal-evidence">${items.map(ref => {
      const path = ref.agent_path?.length ? `<span>Agent ${ref.agent_path.map(esc).join(' / ')}</span>` : '';
      const call = ref.call_id ? `<code>${esc(ref.call_id)}</code>` : '';
      const related = ref.related?.length ? `<span>Related steps: ${ref.related.map(item => esc(JSON.stringify(item))).join(', ')}</span>` : '';
      const calls = ref.related_call_ids?.length ? `<span>Related calls: ${ref.related_call_ids.map(esc).join(', ')}</span>` : '';
      return `<li><span class="signal-step">Step ${esc(ref.step_id)}</span><div>${path}${call}${ref.reason ? `<p>${esc(ref.reason)}</p>` : ''}${ref.subgroup ? `<span>${esc(ref.subgroup)}</span>` : ''}${related}${calls}</div></li>`;
    }).join('')}</ul>`;
  }
  function row(name, signal, detail, options) {
    const status = state(signal);
    const metric = signal.group !== 'D' && signal.value != null && !signal.no_basis
      ? `<span class="signal-summary-value">${esc(typeof signal.value === 'object' ? `${Object.keys(signal.value).length} entries` : value(signal.value))}</span>` : '';
    const displayValue = signal.group === 'D' && typeof signal.value === 'boolean' ? status.label : value(signal.value);
    const breakdown = signal.value != null && typeof signal.value === 'object' && !signal.no_basis
      ? `<dl class="signal-value-entries">${Object.entries(signal.value).map(([key, item]) => `<div><dt>${esc(key)}</dt><dd>${esc(value(item))}</dd></div>`).join('')}</dl>` : '';
    const resultValue = signal.no_basis || signal.value == null ? status.label
      : typeof signal.value === 'object' ? `${Object.keys(signal.value).length} entries` : displayValue;
    const result = `<div class="signal-feedback ${status.kind}" role="group" aria-label="Result"><div class="signal-result-heading"><strong class="signal-result-value">${esc(resultValue)}</strong>${resultValue !== status.label ? `<span class="signal-result-status">${status.label}</span>` : ''}</div>${signal.no_basis ? `<p>${esc(signal.no_basis)}</p>` : signal.reason ? `<p>${esc(signal.reason)}</p>` : ''}${breakdown}</div>`;
    const checks = detail
      ? preconditions(detail.preconditions || [])
      : `<p class="signal-empty">${options.loading ? 'Loading preconditions…' : options.error ? 'Preconditions could not be loaded.' : 'Preconditions are not loaded.'}</p>`;
    const refs = detail?.evidence || [];
    const technicalName = signal.group === 'D' ? '' : `<code class="signal-code">${esc(name)}</code>`;
    const evidenceDetails = detail ? `<details class="signal-evidence-section" data-signal-key="evidence:${esc(name)}"><summary>Evidence<span>${refs.length}</span></summary>${evidence(refs)}</details>` : '';
    return `<details class="signal-row ${status.kind}" data-signal-name="${esc(name)}" data-signal-key="row:${esc(name)}"><summary><span class="signal-state-icon" aria-hidden="true">${status.icon}</span><span class="signal-title">${esc(title(name))}</span><span class="signal-summary-meta">${metric}<span class="signal-badge ${status.kind}">${status.label}</span></span></summary><div class="signal-body">${result}${technicalName}<section class="signal-checks"><h6>Preconditions${detail ? `<span>${(detail.preconditions || []).length}</span>` : ''}</h6>${checks}</section>${evidenceDetails}${signal.rule_version ? `<div class="signal-rule">Rule ${esc(signal.rule_version)}</div>` : ''}</div></details>`;
  }
  function render(options) {
    const report = options.report;
    if (!report) return '<section class="signal-panel"><h5>Signals</h5><p class="signal-empty">Signals were not measured for this trace.</p></section>';
    const results = report.results || {}, names = Object.keys(results);
    const fullResults = options.detail?.signals?.results || {};
    const flagged = names.filter(name => state(results[name]).kind === 'flagged').sort();
    const unknownTags = names.filter(name => results[name].group === 'D' && state(results[name]).kind === 'no-basis').length;
    const highlights = `<div class="signal-highlights"><div class="signal-highlights-heading"><h5>Flagged signals</h5><span>${flagged.length}</span></div>${flagged.length ? `<div class="signal-tags">${flagged.map(name => `<button type="button" class="signal-tag" data-signal-focus="${esc(name)}" aria-label="View ${esc(title(name))}"><span aria-hidden="true">!</span>${esc(title(name))}<span aria-hidden="true">›</span></button>`).join('')}</div>` : '<p class="signal-empty">No flagged L4 signals.</p>'}${unknownTags ? `<p class="signal-coverage-note">${unknownTags} L4 ${unknownTags === 1 ? 'tag has' : 'tags have'} no basis.</p>` : ''}</div>`;
    const error = options.error ? `<p class="signal-empty" role="status">Could not load signal details: ${esc(options.error)} <button type="button" class="btn" data-signal-retry="${esc(options.retryId || '')}">Retry</button></p>` : options.loading ? '<p class="signal-empty" role="status">Loading signal details…</p>' : '';
    const groups = Object.keys(labels).map(level => {
      const groupNames = names.filter(name => levels[results[name].group] === level);
      groupNames.sort((a, b) => {
        const rank = signal => state(signal).kind === 'flagged' ? 0 : state(signal).kind === 'no-basis' ? 2 : 1;
        return rank(results[a]) - rank(results[b]) || a.localeCompare(b);
      });
      const unavailable = groupNames.filter(name => state(results[name]).kind === 'no-basis').length;
      const flags = groupNames.filter(name => state(results[name]).kind === 'flagged').length;
      const counts = `${flags ? `<span class="signal-badge flagged">${flags} flagged</span>` : ''}${unavailable ? `<span class="signal-badge no-basis">${unavailable} no basis</span>` : ''}`;
      return `<details class="signal-level ${level.toLowerCase()}" data-signal-level="${level}" data-signal-key="level:${level}"><summary><span class="signal-level-number">${level}</span><span class="signal-level-name">${labels[level]}<small>${groupNames.length} ${groupNames.length === 1 ? 'signal' : 'signals'}</small></span><span class="signal-level-status">${counts}</span></summary><div class="signal-level-body">${groupNames.map(name => row(name, results[name], fullResults[name], options)).join('') || `<p class="signal-empty">No ${level} signals were recorded for this trace.</p>`}</div></details>`;
    }).join('');
    const coverage = options.detail?.source_coverage || {};
    const metadata = `<details class="signal-config" data-signal-key="config"><summary>Report details</summary><div>Config version <code>${esc(report.config_version || 'Unknown')}</code></div>${Object.keys(coverage).length ? `<h6>Source coverage</h6><pre>${esc(JSON.stringify(coverage, null, 2))}</pre>` : ''}</details>`;
    return `<section class="signal-panel">${highlights}<details class="signal-browser" data-signal-key="browser"><summary><span><b>Signals</b><small>Explore all four levels</small></span><span class="signal-total">${names.length} recorded</span></summary><div class="signal-browser-body">${error}${!names.length ? '<p class="signal-empty">No signals were recorded for this trace.</p>' : ''}${groups}${metadata}</div></details></section>`;
  }
  function captureOpen(container) {
    return [...container.querySelectorAll('details[data-signal-key][open]')].map(node => node.dataset.signalKey);
  }
  function restoreOpen(container, keys) {
    const open = new Set(keys);
    container.querySelectorAll('details[data-signal-key]').forEach(node => { node.open = open.has(node.dataset.signalKey); });
  }
  function bind(container) {
    container.querySelectorAll('[data-signal-focus]').forEach(button => {
      button.onclick = () => {
        const row = [...container.querySelectorAll('.signal-row')].find(node => node.dataset.signalName === button.dataset.signalFocus);
        if (!row) return;
        for (let node = row; node && node !== container; node = node.parentElement) {
          if (node.tagName === 'DETAILS') node.open = true;
        }
        row.querySelector('summary').focus({preventScroll: true});
        row.scrollIntoView({block: 'nearest'});
      };
    });
  }
  function mount(container, options) {
    container.innerHTML = render(options);
    bind(container);
  }
  function mountSaved() {
    document.querySelectorAll('[data-saved-signals]').forEach(container => mount(container, JSON.parse(container.dataset.savedSignals)));
  }
  window.EvaluatorqSignals = {render, mount, mountSaved, value, captureOpen, restoreOpen, bind};
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountSaved);
    else mountSaved();
  }
})();
