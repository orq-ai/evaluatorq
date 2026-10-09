/* Controller for the Insights run form. The server renders every control and list; this file only drives them. */
(function (global) {
  'use strict';

  const REFRESH_DELAY_MS = 400;
  const NAME_PATTERN = /^[a-z][a-z0-9_]*$/;
  const MAX_SESSIONS = 1000;
  const FILE_HINT = 'A Finder export or a local trace file (JSON).';
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
    return checked ? checked.value : 'orq';
  }

  function isCurrentSource(ctx, source, token) {
    return ctx.sourceToken === token && activeSource(ctx.form) === source;
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

  function randomId() {
    const bytes = new Uint8Array(8);
    if (global.crypto && global.crypto.getRandomValues) global.crypto.getRandomValues(bytes);
    else bytes.forEach(function (_byte, index) { bytes[index] = Math.floor(Math.random() * 256); });
    return Array.from(bytes, function (byte) { return byte.toString(16).padStart(2, '0'); }).join('');
  }

  function words(text) {
    return String(text || '').split(' ').filter(Boolean);
  }

  function isSessionField(name) {
    return name === 'session' || name === 'selected' || name.indexOf('session_') === 0;
  }

  function formPairs(form) {
    const active = activeSource(form);
    const pairs = [];
    form.querySelectorAll('input[name], textarea[name], select[name]').forEach(function (element) {
      const type = element.type;
      if (element.disabled || type === 'file' || type === 'button' || type === 'submit') return;
      if ((type === 'checkbox' || type === 'radio') && !element.checked) return;
      if (element.name.indexOf('facet_') === 0 && active !== 'orq') return;
      // The frozen trace_file carries the selection; sending every row would overflow a GET request line.
      if (isSessionField(element.name)) return;
      pairs.push([element.name, element.value]);
    });
    const chosen = form.querySelector('[data-file-name="' + active + '"]');
    pairs.push(['source_name', chosen ? chosen.value : '']);
    return pairs;
  }

  // One hidden trace_file field holds the chosen upload; data-kind is the kind the server read from its contents.
  function setTraceFile(form, path, kind) {
    const traceFile = field(form, 'trace_file');
    traceFile.value = path;
    traceFile.dataset.kind = kind;
  }

  function clearTraceFile(form) {
    setTraceFile(form, '', '');
  }

  function showError(ctx, message) {
    const element = ctx.form.querySelector('#insights-run-error');
    element.textContent = message || '';
    element.hidden = !message;
  }

  // One trace_file field serves the Trace file and Local sessions tabs; clear it when the active tab does not own it.
  function reconcileTraceFileOwner(ctx) {
    const active = activeSource(ctx.form);
    if (active !== 'file' && active !== 'sessions') return;
    if (!field(ctx.form, 'trace_file').value.trim() || ctx.traceFileOwner === active) return;
    if (ctx.snapshotAbort) ctx.snapshotAbort.abort();
    clearTraceFile(ctx.form);
    ctx.traceFileOwner = null;
    ctx.measuredKey = null;
    ctx.form.querySelector('#insights-snapshot-preview').textContent = '';
    ctx.form.querySelector('[data-file-name="file"]').value = '';
    ctx.form.querySelector('[data-file-name="sessions"]').value = '';
    ctx.form.querySelector('[data-file-status]').textContent = FILE_HINT;
    if (active === 'sessions') ctx.sessionsDirty = true;
  }

  function updateSource(ctx) {
    const active = activeSource(ctx.form);
    ctx.form.querySelectorAll('[data-source]').forEach(function (element) {
      element.hidden = words(element.dataset.source).indexOf(active) < 0;
    });
    updateCompilerModel(ctx);
    if (active === 'file') loadRecentFiles(ctx);
  }

  function updateCompilerModel(ctx) {
    const picker = ctx.form.querySelector('[data-compiler-model]');
    if (picker) picker.hidden = activeSource(ctx.form) !== 'orq' || !field(ctx.form, 'query').value.trim();
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
      if (active === 'file' && !field(form, 'trace_file').value.trim()) {
        return 'Browse to choose a trace file first.';
      }
      if (active === 'sessions') {
        const picked = checkedValues(form, 'session').length;
        if (!picked && !field(form, 'trace_file').value.trim()) return 'Select at least one session.';
        if (picked > MAX_SESSIONS) return 'Select at most ' + MAX_SESSIONS + ' sessions.';
      }
      if (active === 'orq') {
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

  function filePreviewKey(form, path, kind) {
    if (kind !== 'snapshot') return JSON.stringify([kind, path]);
    return JSON.stringify([
      kind, path, field(form, 'classifier_model').value, field(form, 'trace_input_chars').value,
      checkedValues(form, 'dimensions'), checkedValues(form, 'labels'), checkedValues(form, 'coding_labels'),
      field(form, 'custom_labels_json').value,
    ]);
  }

  async function measureSelectedFile(ctx) {
    clearTimeout(ctx.previewTimer);
    ctx.previewTimer = null;
    const form = ctx.form;
    const traceFile = field(form, 'trace_file');
    const kind = traceFile.dataset.kind;
    const path = kind === 'snapshot' || kind === 'finder' ? traceFile.value.trim() : '';
    const owner = ctx.traceFileOwner;
    const sourceToken = ctx.sourceToken;
    const host = form.querySelector('#insights-snapshot-preview');
    if (!path || (owner !== 'file' && owner !== 'sessions') || !isCurrentSource(ctx, owner, sourceToken)) {
      if (ctx.snapshotAbort) ctx.snapshotAbort.abort();
      ctx.snapshotAbort = null;
      ctx.measuredKey = null;
      ++ctx.previewSequence;
      host.textContent = '';
      return false;
    }
    const key = filePreviewKey(form, path, kind);
    if (ctx.measuredKey === key) return true;
    if (ctx.snapshotAbort) ctx.snapshotAbort.abort();
    const abort = new AbortController();
    ctx.snapshotAbort = abort;
    const selection = ctx.fileSelectionSequence;
    const sequence = ++ctx.previewSequence;
    host.textContent = kind === 'finder' ? 'Checking the Finder export…' : 'Measuring model input characters…';
    try {
      const body = new URLSearchParams({csrf: field(form, 'csrf').value});
      body.set(kind === 'finder' ? 'finder_path' : 'snapshot_path', path);
      if (kind === 'snapshot') {
        body.set('trace_input_chars', field(form, 'trace_input_chars').value);
        body.set('classifier_model', field(form, 'classifier_model').value);
        ['dimensions', 'labels', 'coding_labels'].forEach(function (name) {
          checkedValues(form, name).forEach(function (value) { body.append(name, value); });
        });
        body.set('custom_labels_json', field(form, 'custom_labels_json').value);
      }
      const endpoint = kind === 'finder' ? '/insights/finder-preview' : '/insights/snapshot-preview';
      const response = await fetch(endpoint, {method: 'POST', body: body, signal: abort.signal});
      const markup = await response.text();
      if (ctx.snapshotAbort !== abort || !isCurrentSource(ctx, owner, sourceToken)
          || ctx.traceFileOwner !== owner || field(form, 'trace_file').value.trim() !== path
          || field(form, 'trace_file').dataset.kind !== kind || selection !== ctx.fileSelectionSequence
          || sequence !== ctx.previewSequence || key !== filePreviewKey(form, path, kind)) return false;
      host.innerHTML = markup;
      if (response.ok) ctx.measuredKey = key;
      return response.ok;
    } catch (failure) {
      if (failure.name !== 'AbortError' && ctx.snapshotAbort === abort
          && isCurrentSource(ctx, owner, sourceToken) && ctx.traceFileOwner === owner
          && selection === ctx.fileSelectionSequence && sequence === ctx.previewSequence
          && field(form, 'trace_file').value.trim() === path
          && key === filePreviewKey(form, path, kind)) {
        host.textContent = 'Could not check this trace file.';
      }
      return false;
    } finally {
      if (ctx.snapshotAbort === abort) ctx.snapshotAbort = null;
    }
  }

  function localiseTimes(root) {
    root.querySelectorAll('time[data-utc]').forEach(function (element) {
      const when = new Date(element.getAttribute('data-utc') + 'Z');
      if (!Number.isNaN(when.getTime())) element.textContent = when.toLocaleString();
    });
  }

  function sessionsStatus(ctx, message) {
    ctx.form.querySelector('[data-sessions-status]').textContent = message;
  }

  function closeFolderBrowser(ctx) {
    if (ctx.folderAbort) ctx.folderAbort.abort();
    ctx.folderAbort = null;
    ctx.folderToken += 1;
    const browser = ctx.form.querySelector('[data-folder-browser]');
    browser.hidden = true;
    browser.setAttribute('aria-busy', 'false');
  }

  function cancelSourceWork(ctx) {
    if (ctx.sessionsAbort) ctx.sessionsAbort.abort();
    ctx.sessionsAbort = null;
    const sessions = ctx.form.querySelector('#insights-sessions');
    sessions.setAttribute('aria-busy', 'false');
    sessionsStatus(ctx, '');
    closeFolderBrowser(ctx);
    if (ctx.freezeAbort) ctx.freezeAbort.abort();
    ctx.freezeAbort = null;
    ctx.uploadSequence += 1;
    if (ctx.uploadAbort) ctx.uploadAbort.abort();
    ctx.uploadAbort = null;
    const fileInput = ctx.form.querySelector('input[data-file]');
    if (fileInput) fileInput.value = '';
    const fileStatus = ctx.form.querySelector('[data-file-status]');
    if (fileStatus.textContent === 'Uploading…') fileStatus.textContent = FILE_HINT;
    if (ctx.snapshotAbort) ctx.snapshotAbort.abort();
    ctx.snapshotAbort = null;
    ctx.measuredKey = null;
    ctx.form.querySelector('#insights-snapshot-preview').textContent = '';
  }

  function selectionChanged(ctx) {
    const form = ctx.form;
    const rows = Array.from(form.querySelectorAll('input[name="session"]'));
    const picked = rows.filter(function (input) { return input.checked; }).length;
    const count = form.querySelector('[data-sessions-selected]');
    if (count) count.textContent = picked + ' selected';
    const all = form.querySelector('[data-sessions-all]');
    if (all) all.checked = rows.length > 0 && picked === rows.length;
    ctx.selectionToken += 1;
    ctx.sessionsDirty = true;
    ctx.measuredKey = null;
    ++ctx.fileSelectionSequence;
    ++ctx.previewSequence;
    if (ctx.snapshotAbort) ctx.snapshotAbort.abort();
    ctx.snapshotAbort = null;
    clearTimeout(ctx.previewTimer);
    ctx.previewTimer = null;
    ctx.traceFileOwner = null;
    clearTraceFile(form);
    form.querySelector('[data-file-name="sessions"]').value = '';
    form.querySelector('#insights-snapshot-preview').textContent = '';
  }

  async function searchSessions(ctx) {
    const form = ctx.form;
    if (activeSource(form) !== 'sessions') return;
    const host = form.querySelector('#insights-sessions');
    if (ctx.sessionsAbort) ctx.sessionsAbort.abort();
    const abort = new AbortController();
    const sourceToken = ctx.sourceToken;
    const selectionToken = ctx.selectionToken;
    ctx.sessionsAbort = abort;
    const body = new URLSearchParams({csrf: field(form, 'csrf').value, tz_offset: String(new Date().getTimezoneOffset()), session_tab: ctx.tabId});
    ['session_from', 'session_to', 'session_project_dir', 'session_text'].forEach(function (name) {
      body.set(name, field(form, name).value);
    });
    checkedValues(form, 'session_source').forEach(function (value) { body.append('session_source', value); });
    checkedValues(form, 'session').forEach(function (value) { body.append('selected', value); });
    host.setAttribute('aria-busy', 'true');
    sessionsStatus(ctx, 'Searching local sessions…');
    try {
      const response = await fetch('/insights/sessions/search', {method: 'POST', body: body, signal: abort.signal});
      const markup = await response.text();
      if (ctx.sessionsAbort !== abort || !isCurrentSource(ctx, 'sessions', sourceToken)) return;
      if (selectionToken !== ctx.selectionToken) {
        sessionsStatus(ctx, 'The selection changed while searching. Search again to refresh results.');
        return;
      }
      host.innerHTML = markup;
      localiseTimes(host);
      sessionsStatus(ctx, '');
      selectionChanged(ctx);
    } catch (failure) {
      if (failure.name !== 'AbortError' && ctx.sessionsAbort === abort && isCurrentSource(ctx, 'sessions', sourceToken)) {
        sessionsStatus(ctx, 'Could not search local sessions.');
      }
    } finally {
      if (ctx.sessionsAbort === abort) {
        ctx.sessionsAbort = null;
        host.setAttribute('aria-busy', 'false');
      }
    }
  }

  async function freezeSessions(ctx) {
    const form = ctx.form;
    const token = ctx.selectionToken;
    const sourceToken = ctx.sourceToken;
    const generic = 'Could not prepare the selected sessions.';
    const abort = new AbortController();
    ctx.freezeAbort = abort;
    const body = new URLSearchParams({csrf: field(form, 'csrf').value});
    checkedValues(form, 'session').forEach(function (value) { body.append('session', value); });
    sessionsStatus(ctx, 'Preparing the selected sessions…');
    try {
      const response = await fetch('/insights/sessions/snapshot', {method: 'POST', body: body, signal: abort.signal});
      let result = {};
      try {
        result = await response.json();
      } catch (failure) {
        result = {};
      }
      if (ctx.freezeAbort !== abort || !isCurrentSource(ctx, 'sessions', sourceToken)) return false;
      if (token !== ctx.selectionToken) {
        sessionsStatus(ctx, '');
        showError(ctx, 'The selection changed while it was being prepared. Press Continue again.');
        return false;
      }
      if (!response.ok || !result.path) throw new Error(result.error || generic);
      setTraceFile(form, result.path, 'snapshot');
      form.querySelector('[data-file-name="sessions"]').value = result.n_sessions + (result.n_sessions === 1 ? ' local session' : ' local sessions');
      ctx.traceFileOwner = 'sessions';
      ctx.sessionsDirty = false;
      ctx.measuredKey = null;
      ++ctx.fileSelectionSequence;
      const failed = result.failed || [];
      sessionsStatus(ctx, failed.length
        ? failed.length + ' selected session' + (failed.length === 1 ? ' was' : 's were') + ' left out: '
          + failed.map(function (item) { return item.error; }).join('; ')
        : '');
      return true;
    } catch (failure) {
      if (ctx.freezeAbort === abort && isCurrentSource(ctx, 'sessions', sourceToken)) {
        sessionsStatus(ctx, '');
        showError(ctx, failure.message || generic);
      }
      return false;
    } finally {
      if (ctx.freezeAbort === abort) ctx.freezeAbort = null;
    }
  }

  function localDate(daysAgo) {
    const when = new Date();
    when.setDate(when.getDate() - daysAgo);
    const pad = function (n) { return String(n).padStart(2, '0'); };
    return when.getFullYear() + '-' + pad(when.getMonth() + 1) + '-' + pad(when.getDate());
  }

  function updateSessionRangeLabel(ctx) {
    const form = ctx.form;
    const label = form.querySelector('[data-session-range-label]');
    const preset = form.querySelector('.irf-session-time-menu').dataset.sessionRange || '7';
    if (preset === 'all') label.textContent = 'All dates';
    else if (preset === '7') label.textContent = 'Last 7 days';
    else if (preset === '30') label.textContent = 'Last 30 days';
    else {
      const from = field(form, 'session_from').value;
      const to = field(form, 'session_to').value;
      label.textContent = from || to ? (from || 'Any start') + ' – ' + (to || 'Any end') : 'Custom range';
    }
  }

  function setSessionPreset(ctx, preset) {
    const form = ctx.form;
    const menu = form.querySelector('.irf-session-time-menu');
    menu.dataset.sessionRange = preset;
    form.querySelectorAll('[data-session-preset]').forEach(function (button) {
      button.setAttribute('aria-pressed', String(button.dataset.sessionPreset === preset));
    });
    const from = field(form, 'session_from');
    const to = field(form, 'session_to');
    if (preset === 'all') {
      from.value = '';
      to.value = '';
    } else if (preset === '7' || preset === '30') {
      from.value = localDate(Number(preset) - 1);
      to.value = localDate(0);
    }
    updateSessionRangeLabel(ctx);
  }

  function defaultSessionDates(ctx) {
    const menu = ctx.form.querySelector('.irf-session-time-menu');
    if (!menu.dataset.sessionRange) setSessionPreset(ctx, '7');
    else updateSessionRangeLabel(ctx);
  }
  function folderCurrent(ctx, sourceToken, folderToken, abort) {
    return ctx.folderAbort === abort && ctx.folderToken === folderToken
      && isCurrentSource(ctx, 'sessions', sourceToken)
      && !ctx.form.querySelector('[data-folder-browser]').hidden;
  }

  function renderFolderPage(ctx, result, append) {
    const form = ctx.form;
    const browser = form.querySelector('[data-folder-browser]');
    const crumbs = result.breadcrumbs.map(function (crumb) {
      return '<button type="button" class="irf-folder-crumb" data-folder-nav data-path="' + esc(crumb.path) + '">' + esc(crumb.name) + '</button>';
    }).join('<span aria-hidden="true">/</span>');
    const rows = result.directories.map(function (directory) {
      return '<div role="listitem"><button type="button" class="irf-folder-entry" data-folder-nav data-path="' + esc(directory.path) + '">'
        + '<span aria-hidden="true">▸</span><span>' + esc(directory.name) + '</span></button></div>';
    }).join('');
    form.querySelector('[data-folder-breadcrumbs]').innerHTML = crumbs;
    form.querySelector('[data-folder-current]').textContent = result.path;
    const list = form.querySelector('[data-folder-list]');
    list.innerHTML = append ? list.innerHTML + rows : rows || '<p class="irf-folder-empty">No subfolders in this directory.</p>';
    const parent = form.querySelector('[data-folder-parent]');
    parent.disabled = !result.parent;
    parent.dataset.path = result.parent || '';
    const more = form.querySelector('[data-folder-more]');
    more.hidden = !result.has_more;
    more.dataset.offset = String(result.offset + result.limit);
    const use = form.querySelector('[data-folder-use]');
    use.disabled = false;
    use.dataset.path = result.path;
    browser.dataset.loaded = 'true';
    browser.dataset.parent = result.parent || '';
    form.querySelector('[data-folder-error]').hidden = true;
  }

  async function loadFolderPage(ctx, path, offset, append) {
    const form = ctx.form;
    const browser = form.querySelector('[data-folder-browser]');
    if (ctx.folderAbort) ctx.folderAbort.abort();
    const abort = new AbortController();
    const sourceToken = ctx.sourceToken;
    const folderToken = ++ctx.folderToken;
    ctx.folderAbort = abort;
    browser.hidden = false;
    browser.setAttribute('aria-busy', 'true');
    const error = form.querySelector('[data-folder-error]');
    error.hidden = true;
    const use = form.querySelector('[data-folder-use]');
    if (!append) {
      use.disabled = true;
      browser.dataset.loaded = 'false';
      form.querySelector('[data-folder-current]').textContent = path || 'Your home directory';
      form.querySelector('[data-folder-list]').textContent = 'Loading folders…';
      form.querySelector('[data-folder-breadcrumbs]').textContent = '';
      form.querySelector('[data-folder-more]').hidden = true;
      form.querySelector('[data-folder-parent]').disabled = true;
    }
    const body = new URLSearchParams({
      csrf: field(form, 'csrf').value,
      path: path || '',
      offset: String(offset),
    });
    try {
      const response = await fetch('/insights/sessions/directories', {method: 'POST', body: body, signal: abort.signal});
      let result;
      try {
        result = await response.json();
      } catch (failure) {
        result = {};
      }
      if (!folderCurrent(ctx, sourceToken, folderToken, abort)) return;
      if (!response.ok) throw new Error(result.error || 'Directory listing failed.');
      if (!result.path || !Array.isArray(result.directories) || !Array.isArray(result.breadcrumbs)) {
        throw new Error('The directory response was incomplete.');
      }
      renderFolderPage(ctx, result, append);
    } catch (failure) {
      if (failure.name !== 'AbortError' && folderCurrent(ctx, sourceToken, folderToken, abort)) {
        error.textContent = (failure.message || 'Could not list this directory.')
          + ' Check the path, try another folder, or cancel to keep the typed filter.';
        error.hidden = false;
        if (!append) {
          use.disabled = true;
          browser.dataset.loaded = 'false';
        }
        if (!append) form.querySelector('[data-folder-list]').textContent = 'No folder loaded.';
      }
    } finally {
      if (ctx.folderAbort === abort) {
        ctx.folderAbort = null;
        browser.setAttribute('aria-busy', 'false');
      }
    }
  }

  function openFolderBrowser(ctx) {
    const typed = field(ctx.form, 'session_project_dir').value;
    ctx.form.querySelector('[data-folder-browser]').hidden = false;
    loadFolderPage(ctx, typed || '', 0, false);
  }

  function useFolder(ctx) {
    const browser = ctx.form.querySelector('[data-folder-browser]');
    const use = ctx.form.querySelector('[data-folder-use]');
    if (browser.dataset.loaded !== 'true' || use.disabled || !use.dataset.path) return;
    const input = field(ctx.form, 'session_project_dir');
    input.value = use.dataset.path;
    input.dispatchEvent(new global.Event('input', {bubbles: true}));
    input.dispatchEvent(new global.Event('change', {bubbles: true}));
    closeFolderBrowser(ctx);
  }

  function selectSessionDateRange(ctx) {
    setSessionPreset(ctx, 'custom');
  }


  function scheduleFilePreview(ctx) {
    const traceFile = field(ctx.form, 'trace_file');
    const active = activeSource(ctx.form);
    if ((active !== 'file' && active !== 'sessions') || ctx.traceFileOwner !== active) return;
    const kind = traceFile.dataset.kind;
    const path = kind === 'snapshot' ? traceFile.value.trim() : '';
    if (!path || ctx.measuredKey === filePreviewKey(ctx.form, path, kind)) return;
    if (ctx.snapshotAbort) ctx.snapshotAbort.abort();
    ctx.snapshotAbort = null;
    ++ctx.previewSequence;
    ctx.form.querySelector('#insights-snapshot-preview').textContent = 'Updating model input characters…';
    clearTimeout(ctx.previewTimer);
    ctx.previewTimer = setTimeout(function () {
      ctx.previewTimer = null;
      measureSelectedFile(ctx);
    }, REFRESH_DELAY_MS);
  }
  function facetKey(form) {
    const selected = Array.from(form.querySelectorAll('input[name^="facet_"]:checked')).map(function (input) {
      return input.name + '=' + input.value;
    });
    return field(form, 'window_days').value.trim() + '|' + selected.sort().join('&');
  }

  async function refreshFacets(ctx, retry) {
    const form = ctx.form;
    if (activeSource(form) !== 'orq') return;
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

  function planParams(form, compact) {
    const params = new URLSearchParams();
    formPairs(form).forEach(function (pair) {
      if (pair[0] !== 'csrf' && pair[0] !== 'mount') params.append(pair[0], pair[1]);
    });
    if (compact) params.set('compact', '1');
    return params;
  }

  async function loadCompact(ctx) {
    const host = ctx.form.querySelector('#insights-run-compact');
    const sequence = ++ctx.compactSequence;
    const source = activeSource(ctx.form);
    const sourceToken = ctx.sourceToken;
    try {
      const response = await fetch('/insights/new/plan?' + planParams(ctx.form, true).toString());
      const markup = await response.text();
      if (sequence === ctx.compactSequence && isCurrentSource(ctx, source, sourceToken)) host.innerHTML = markup;
    } catch (failure) {
      if (sequence === ctx.compactSequence && isCurrentSource(ctx, source, sourceToken)) {
        host.textContent = 'The estimate could not be loaded.';
      }
    }
  }

  function scheduleRefresh(ctx) {
    clearTimeout(ctx.refreshTimer);
    ctx.refreshTimer = setTimeout(function () {
      ctx.refreshTimer = null;
      if (ctx.step === 3) loadPlan(ctx);
      else loadCompact(ctx);
    }, REFRESH_DELAY_MS);
  }

  async function loadPlan(ctx) {
    const form = ctx.form;
    clearTimeout(ctx.refreshTimer);
    ctx.refreshTimer = null;
    const host = form.querySelector('#insights-run-plan');
    const estimate = form.querySelector('#insights-run-estimate');
    const compact = form.querySelector('#insights-run-compact');
    const sequence = ++ctx.planSequence;
    const source = activeSource(form);
    const sourceToken = ctx.sourceToken;
    ctx.compactSequence += 1;
    host.closest('.irf-review').hidden = false;
    host.textContent = 'Working out the stages…';
    estimate.textContent = 'Working out the estimate…';
    try {
      const response = await fetch('/insights/new/plan?' + planParams(form, false).toString());
      const markup = await response.text();
      if (sequence !== ctx.planSequence || !isCurrentSource(ctx, source, sourceToken)) return;
      if (!response.ok) {
        host.innerHTML = markup;
        estimate.textContent = '';
        return;
      }
      const parts = new DOMParser().parseFromString(markup, 'text/html');
      const part = function (name) { return parts.querySelector('[data-part="' + name + '"]'); };
      host.innerHTML = part('plan') ? part('plan').innerHTML : '';
      estimate.innerHTML = part('estimate') ? part('estimate').innerHTML : '';
      if (part('compact')) compact.innerHTML = part('compact').innerHTML;
    } catch (failure) {
      if (sequence === ctx.planSequence && isCurrentSource(ctx, source, sourceToken)) {
        host.textContent = 'The expected stages could not be loaded.';
        estimate.textContent = 'The estimate could not be loaded.';
      }
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
    scheduleFilePreview(ctx);
  }

  function renderCustom(ctx, specs) {
    field(ctx.form, 'custom_labels_json').value = JSON.stringify(specs);
    ctx.form.querySelector('[data-custom-list]').innerHTML = specs.map(function (spec) {
      const name = esc(spec.name);
      return '<span class="tg on" data-custom="' + name + '"><span class="tg-title">' + name + '</span><small>custom</small>'
        + '<button type="button" data-cq-remove="' + name + '" aria-label="Remove question ' + name + '">&times;</button></span>';
    }).join('');
    scheduleFilePreview(ctx);
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

  async function loadRecentFiles(ctx, force) {
    if (ctx.recentFilesLoaded && !force) return;
    ctx.recentFilesLoaded = true;
    const sequence = ++ctx.recentFilesSequence;
    const host = ctx.form.querySelector('[data-recent-files]');
    try {
      const response = await fetch('/insights/files');
      if (!response.ok) throw new Error('Recent trace files could not be loaded.');
      const markup = await response.text();
      if (sequence === ctx.recentFilesSequence) host.innerHTML = markup;
    } catch (failure) {
      if (sequence === ctx.recentFilesSequence) {
        ctx.recentFilesLoaded = false;
        host.innerHTML = '<p class="insights-muted" role="status">Recent trace files could not be loaded. '
          + '<button type="button" class="irf-btn" data-retry-files>Retry</button></p>';
      }
    }
  }

  function chooseFile(ctx, path, kind, name) {
    if (activeSource(ctx.form) !== 'file') return;
    ++ctx.fileSelectionSequence;
    if (ctx.snapshotAbort) ctx.snapshotAbort.abort();
    ctx.snapshotAbort = null;
    ctx.measuredKey = null;
    clearTimeout(ctx.previewTimer);
    ctx.previewTimer = null;
    ++ctx.previewSequence;
    const form = ctx.form;
    const traceFile = field(form, 'trace_file');
    traceFile.value = path;
    traceFile.dataset.kind = kind;
    ctx.traceFileOwner = 'file';
    form.querySelector('[data-file-name="file"]').value = name;
    form.querySelector('[data-file-status]').textContent = kind === 'finder' ? 'Finder export is ready.' : 'Trace snapshot is ready.';
    form.querySelector('#insights-snapshot-preview').textContent = '';
    showError(ctx, '');
    scheduleRefresh(ctx);
    measureSelectedFile(ctx);
  }

  async function upload(ctx, input) {
    const form = ctx.form;
    const file = input.files && input.files[0];
    if (!file || activeSource(form) !== 'file') return;
    if (ctx.uploadAbort) ctx.uploadAbort.abort();
    const abort = new AbortController();
    const sequence = ++ctx.uploadSequence;
    const selection = ++ctx.fileSelectionSequence;
    const sourceToken = ctx.sourceToken;
    ctx.uploadAbort = abort;
    const current = function () {
      return ctx.uploadAbort === abort && ctx.uploadSequence === sequence
        && selection === ctx.fileSelectionSequence && isCurrentSource(ctx, 'file', sourceToken);
    };
    const status = form.querySelector('[data-file-status]');
    status.textContent = 'Uploading…';
    try {
      const body = new FormData();
      body.set('csrf', field(form, 'csrf').value);
      body.set('file', file);
      const response = await fetch('/insights/uploads', {method: 'POST', body: body, signal: abort.signal});
      if (!current()) return;
      let result;
      try { result = await response.json(); } catch (failure) { throw new Error('Upload failed.'); }
      if (!current()) return;
      if (!response.ok) throw new Error(result.error || 'Upload failed.');
      chooseFile(ctx, result.path, result.kind, file.name);
      loadRecentFiles(ctx, true);
    } catch (failure) {
      if (current()) status.textContent = failure.message || 'Upload failed.';
    } finally {
      if (ctx.uploadSequence === sequence) {
        input.value = '';
        if (ctx.uploadAbort === abort) ctx.uploadAbort = null;
      }
    }
  }

  async function goNext(ctx) {
    const active = activeSource(ctx.form);
    const sourceToken = ctx.sourceToken;
    const current = function () { return isCurrentSource(ctx, active, sourceToken); };
    const message = validate(ctx, ctx.step);
    if (message) {
      showError(ctx, message);
      return;
    }
    if (ctx.step === 1 && (active === 'file' || active === 'sessions')) {
      const next = ctx.form.querySelector('[data-irf-next]');
      next.disabled = true;
      try {
        if (active === 'sessions' && (ctx.sessionsDirty || !field(ctx.form, 'trace_file').value.trim())) {
          if (!(await freezeSessions(ctx)) || !current()) return;
        }
        if (!current()) return;
        if (!(await measureSelectedFile(ctx))) {
          if (current()) {
            showError(ctx, active === 'sessions'
              ? 'Could not measure the selected sessions. Search again and reselect them.'
              : 'Could not check this trace file. Browse to choose a valid file.');
          }
          return;
        }
        if (!current()) return;
      } finally {
        next.disabled = false;
      }
    }
    if (current()) showStep(ctx, ctx.step + 1);
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
    else if ((element = hit('[data-session-preset]'))) setSessionPreset(ctx, element.dataset.sessionPreset);
    else if (hit('[data-session-apply]')) {
      selectSessionDateRange(ctx);
      ctx.form.querySelector('.irf-session-time-menu').open = false;
    } else if (hit('[data-folder-open]')) openFolderBrowser(ctx);
    else if (hit('[data-folder-cancel]')) closeFolderBrowser(ctx);
    else if (hit('[data-folder-use]')) useFolder(ctx);
    else if (hit('[data-folder-home]')) loadFolderPage(ctx, '', 0, false);
    else if ((element = hit('[data-folder-parent]'))) {
      if (!element.disabled) loadFolderPage(ctx, element.dataset.path, 0, false);
    } else if ((element = hit('[data-folder-more]'))) {
      if (!element.hidden && !element.disabled) {
        element.disabled = true;
        loadFolderPage(ctx, ctx.form.querySelector('[data-folder-current]').textContent, Number(element.dataset.offset), true)
          .finally(function () { element.disabled = false; });
      }
    } else if ((element = hit('[data-folder-nav]'))) loadFolderPage(ctx, element.dataset.path, 0, false);
    else if ((element = hit('[data-browse]'))) ctx.form.querySelector('input[data-file]').click();
    else if ((element = hit('[data-recent-file]'))) chooseFile(ctx, element.dataset.path, element.dataset.kind, element.dataset.displayName);
    else if (hit('[data-retry-files]')) loadRecentFiles(ctx, true);
    else if (hit('[data-cq-open]')) {
      const e = editor(ctx);
      e.box.hidden = false;
      e.name.focus();
    } else if (hit('[data-cq-add]')) addCustomQuestion(ctx);
    else if (hit('[data-cq-cancel]')) resetEditor(ctx);
    else if ((element = hit('[data-cq-remove]'))) {
      renderCustom(ctx, customLabels(ctx.form).filter(function (spec) { return spec.name !== element.dataset.cqRemove; }));
    } else if (hit('[data-retry-facets]')) refreshFacets(ctx, true);
    else if (hit('[data-sessions-search]')) searchSessions(ctx);
  }

  function affectsEstimate(target) {
    return !target.matches('[data-cq], input[data-file], input[type="search"], input[name="session"], input[name^="session_"], [data-sessions-all]');
  }

  function onChange(ctx, event) {
    const target = event.target;
    if (target.closest('#insights-run-models')) {
      if (target.name === 'classifier_model') scheduleFilePreview(ctx);
      loadPlan(ctx);
      return;
    }
    if (affectsEstimate(target)) scheduleRefresh(ctx);
    if (target.name === 'source') {
      ctx.sourceToken += 1;
      cancelSourceWork(ctx);
      ++ctx.fileSelectionSequence;
      clearTimeout(ctx.previewTimer);
      ctx.previewTimer = null;
      ++ctx.previewSequence;
      showError(ctx, '');
      reconcileTraceFileOwner(ctx);
      updateSource(ctx);
      refreshFacets(ctx);
    } else if (target.name === 'window_days') {
      refreshFacets(ctx);
    } else if (target.name === 'session_from' || target.name === 'session_to') {
      selectSessionDateRange(ctx);
    } else if (target.matches('[data-sessions-all]')) {
      ctx.form.querySelectorAll('input[name="session"]').forEach(function (input) { input.checked = target.checked; });
      selectionChanged(ctx);
    } else if (target.name === 'session') {
      selectionChanged(ctx);
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
    if (['dimensions', 'labels', 'coding_labels', 'trace_input_chars'].includes(target.name)) scheduleFilePreview(ctx);
  }

  function onKeydown(ctx, event) {
    if (event.key !== 'Enter') return;
    const target = event.target;
    if (target.matches('[data-cq]') && target.type !== 'textarea') {
      event.preventDefault();
      addCustomQuestion(ctx);
    } else if (target.matches('input[type="search"]')) {
      event.preventDefault();
    } else if (target.matches('input[name^="session_"]')) {
      event.preventDefault();
      searchSessions(ctx);
    }
  }

  function mountAt(form, step, keepError) {
    if (!form) return null;
    if (contexts.has(form)) return contexts.get(form);
    const ctx = {
      form: form, step: 1, facetKey: null, facetAbort: null, snapshotAbort: null, measuredKey: null,
      tabId: randomId(), sessionsAbort: null, freezeAbort: null, uploadAbort: null, uploadSequence: 0,
      sessionsDirty: false, selectionToken: 0, sourceToken: 0, traceFileOwner: null,
      folderAbort: null, folderToken: 0,
      planSequence: 0, compactSequence: 0, refreshTimer: null, previewTimer: null, previewSequence: 0,
      fileSelectionSequence: 0, recentFilesLoaded: false, recentFilesSequence: 0,
    };
    contexts.set(form, ctx);
    form.addEventListener('click', function (event) { onClick(ctx, event); });
    form.addEventListener('change', function (event) { onChange(ctx, event); });
    form.addEventListener('input', function (event) {
      if (event.target.name === 'query') updateCompilerModel(ctx);
      if (event.target.name === 'session_from' || event.target.name === 'session_to') selectSessionDateRange(ctx);
      if (event.target.name === 'trace_input_chars') scheduleFilePreview(ctx);
      if (affectsEstimate(event.target)) scheduleRefresh(ctx);
    });
    form.addEventListener('keydown', function (event) { onKeydown(ctx, event); });
    form.addEventListener('facets:closed', function () { refreshFacets(ctx); });
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      submit(ctx);
    });
    if (global.htmx && typeof global.htmx.process === 'function') global.htmx.process(form);
    form.classList.add('irf-ready');
    const mountedSource = activeSource(form);
    if (field(form, 'trace_file').value.trim() && (mountedSource === 'file' || mountedSource === 'sessions')) ctx.traceFileOwner = mountedSource;
    defaultSessionDates(ctx);
    showStep(ctx, step || 1, keepError);
    refreshFacets(ctx);
    if (ctx.step !== 3) loadCompact(ctx);
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
