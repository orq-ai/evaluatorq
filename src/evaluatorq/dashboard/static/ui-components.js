/* Dependency-free dashboard primitives. HTML slots accept trusted caller-built fragments. */
(function () {
  'use strict';
  const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const tones = new Set(['info', 'positive', 'negative', 'warning', 'neutral']);
  function foldout({key, className = '', summaryHtml, bodyHtml = '', data = {}, open = false}) {
    const attrs = Object.entries(data).map(([name, value]) => {
      if (!/^[a-z][a-z0-9-]*$/.test(name)) throw new TypeError('Foldout data names must contain lowercase letters, digits or hyphens.');
      if (name === 'ui-key') throw new TypeError('Use the foldout key option for data-ui-key.');
      return ` data-${name}="${escapeHtml(value)}"`;
    }).join('');
    const keyAttr = key == null ? '' : ` data-ui-key="${escapeHtml(key)}"`;
    return `<details class="${escapeHtml(className ? `${className} eq-foldout` : 'eq-foldout')}"${keyAttr}${attrs}${open ? ' open' : ''}><summary>${summaryHtml}</summary>${bodyHtml}</details>`;
  }
  function resultFeedback({value, outcome = '', tone = 'info', reason = '', bodyHtml = '', numeric = false, label = 'Result'}) {
    if (!tones.has(tone)) throw new TypeError(`Unknown feedback tone: ${tone}`);
    const status = outcome && String(value) !== String(outcome) ? `<span class="eq-result-status">${escapeHtml(outcome)}</span>` : '';
    return `<div class="eq-result eq-result--${tone}" role="group" aria-label="${escapeHtml(label)}"><div class="eq-result-heading"><strong class="eq-result-value${numeric ? ' eq-result-value--numeric' : ''}">${escapeHtml(value)}</strong>${status}</div>${reason ? `<p>${escapeHtml(reason)}</p>` : ''}${bodyHtml}</div>`;
  }
  function preconditionList(items, {emptyText = 'No preconditions recorded.'} = {}) {
    if (!items.length) return `<p class="eq-empty">${escapeHtml(emptyText)}</p>`;
    const metCount = items.filter(item => item.met === true).length;
    return `<div class="eq-check-count">${metCount} of ${items.length} met<meter min="0" max="${items.length}" value="${metCount}" aria-label="Preconditions met"></meter></div><ul class="eq-preconditions">${items.map(item => {
      const kind = item.met === true ? 'met' : item.met === 'partial' ? 'partial' : 'not-met';
      const label = kind === 'met' ? 'Met' : kind === 'partial' ? 'Partial' : 'Not met';
      const icon = kind === 'met' ? '✓' : kind === 'partial' ? '≈' : '×';
      return `<li class="${kind}"><span class="eq-check-icon" aria-hidden="true">${icon}</span><div><div class="eq-check-title"><b>${escapeHtml(item.name)}</b><span>${label} · ${item.required ? 'required' : 'optional'}</span></div>${item.detail ? `<p>${escapeHtml(item.detail)}</p>` : ''}</div></li>`;
    }).join('')}</ul>`;
  }
  function captureOpen(container, keyAttribute = 'data-ui-key') {
    return [...container.querySelectorAll(`details[${keyAttribute}][open]`)].map(node => node.getAttribute(keyAttribute));
  }
  function restoreOpen(container, keys, keyAttribute = 'data-ui-key') {
    const open = new Set(keys);
    container.querySelectorAll(`details[${keyAttribute}]`).forEach(node => { node.open = open.has(node.getAttribute(keyAttribute)); });
  }
  window.EvaluatorqComponents = {escapeHtml, foldout, resultFeedback, preconditionList, captureOpen, restoreOpen};
})();
