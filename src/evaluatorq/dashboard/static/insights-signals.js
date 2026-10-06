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
    const status = state(signal), level = levels[signal.group] || 'Unknown';
    const metric = signal.group !== 'D' && signal.value != null && !signal.no_basis
      ? `<span class="signal-summary-value">${esc(typeof signal.value === 'object' ? `${Object.keys(signal.value).length} entries` : value(signal.value))}</span>` : '';
    const result = signal.no_basis
      ? `<div class="signal-feedback no-basis"><span class="signal-result-icon">—</span><div><strong>Not measurable</strong><p>${esc(signal.no_basis)}</p></div></div>`
      : `<div class="signal-feedback ${status.kind}"><span class="signal-result-value">${esc(value(signal.value))}</span><div><strong>${status.label}</strong>${signal.reason ? `<p>${esc(signal.reason)}</p>` : ''}${signal.approximate ? '<p>Approximate measurement</p>' : ''}</div></div>`;
    const details = detail
      ? `<h6>Preconditions <span>${(detail.preconditions || []).length}</span></h6>${preconditions(detail.preconditions || [])}<h6>Evidence <span>${(detail.evidence || []).length}</span></h6>${evidence(detail.evidence || [])}`
      : `<p class="signal-empty">${options.loading ? 'Loading signal details…' : options.error ? 'Signal details could not be loaded.' : 'Signal details are not loaded.'}</p>`;
    return `<details class="signal-row ${status.kind}" data-signal-name="${esc(name)}"><summary><span class="signal-state-icon" aria-hidden="true">${status.icon}</span><span class="signal-title">${esc(title(name))}${options.level === 'all' ? `<small>${level}</small>` : ''}</span><span class="signal-summary-meta">${metric}<span class="signal-badge ${status.kind}">${status.label}</span></span></summary><div class="signal-body"><code class="signal-code">${esc(name)}</code>${result}${details}${signal.rule_version ? `<div class="signal-rule">Rule ${esc(signal.rule_version)}</div>` : ''}</div></details>`;
  }
  function render(options) {
    const report = options.report;
    if (!report) return '<section class="signal-panel"><h5>Signals</h5><p class="signal-empty">Signals were not measured for this trace.</p></section>';
    const level = options.level || 'L4', results = report.results || {};
    const names = Object.keys(results).filter(name => level === 'all' || levels[results[name].group] === level);
    names.sort((a, b) => {
      const rank = signal => signal.no_basis || signal.value == null ? 2 : signal.group === 'D' && signal.value === true ? 0 : 1;
      return rank(results[a]) - rank(results[b]) || a.localeCompare(b);
    });
    const selected = {...options, level}, fullResults = options.detail?.signals?.results || {};
    const count = names.filter(name => results[name].group === 'D' && results[name].value === true && !results[name].no_basis).length;
    const selector = [...Object.keys(labels), 'all'].map(item => `<button type="button" data-signal-level="${item}" aria-pressed="${item === level}" title="${labels[item] || 'All levels'}">${item === 'all' ? 'All' : item}</button>`).join('');
    const error = options.error ? `<p class="signal-empty" role="status">Could not load signal details: ${esc(options.error)} <button type="button" class="btn" data-signal-retry="${esc(options.retryId || '')}">Retry</button></p>` : options.loading ? '<p class="signal-empty" role="status">Loading signal details…</p>' : '';
    const coverage = options.detail?.source_coverage || {};
    const metadata = `<details class="signal-config"><summary>Report details</summary><div>Config version <code>${esc(report.config_version || 'Unknown')}</code></div>${Object.keys(coverage).length ? `<h6>Source coverage</h6><pre>${esc(JSON.stringify(coverage, null, 2))}</pre>` : ''}</details>`;
    return `<section class="signal-panel"><div class="signal-heading"><h5>Signals</h5><span>${names.length} shown · ${Object.keys(results).length} recorded</span></div><div class="signal-levels" role="group" aria-label="Signal level">${selector}</div><div class="signal-level-caption">${level === 'all' ? 'All levels' : `${level} ${labels[level] || ''}`}${level === 'L4' ? `<span>${count} flagged</span>` : ''}</div>${error}${names.map(name => row(name, results[name], fullResults[name], selected)).join('') || `<p class="signal-empty">${Object.keys(results).length ? `No ${esc(level === 'all' ? '' : level + ' ')}signals were recorded for this trace.` : 'No signals were recorded for this trace.'}</p>`}${metadata}</section>`;
  }
  function mount(container, options) {
    container.innerHTML = render(options);
    container.querySelectorAll('[data-signal-level]').forEach(button => {
      button.onclick = () => mount(container, {...options, level: button.dataset.signalLevel});
    });
  }
  function mountSaved() {
    document.querySelectorAll('[data-saved-signals]').forEach(container => mount(container, JSON.parse(container.dataset.savedSignals)));
  }
  window.EvaluatorqSignals = {render, mount, mountSaved, value};
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountSaved);
    else mountSaved();
  }
})();
