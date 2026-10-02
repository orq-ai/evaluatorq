/* Shared request and validation rules for the Insights run wizard and review sheet. */
(function (global) {
  function normalizeSource(source) {
    if (source === 'question') return 'query';
    if (source === 'file') return 'snapshot';
    return source;
  }

  function validateSource(options) {
    const source = normalizeSource(options.source);
    if (options.query !== undefined && source === 'query' && !options.query.trim()) return 'query';
    if (options.finderPath !== undefined && source === 'finder' && !options.finderPath.trim()) return 'finder';
    if (options.snapshotPath !== undefined && source === 'snapshot' && !options.snapshotPath.trim()) return 'snapshot';
    if (options.windowValid === false || options.limitValid === false) return 'window';
    if (options.facetsLoading) return 'facets';
    return '';
  }

  function validateSelection(options) {
    if (!options.selectionCount) return 'selection';
    if (options.parallelismValid === false) return 'parallelism';
    return '';
  }

  function facetUrl(windowDays, facets, retry) {
    const params = new URLSearchParams({window_days: String(windowDays)});
    if (retry) params.set('retry', '1');
    Object.entries(facets || {}).forEach(function (entry) {
      const name = entry[0];
      const values = entry[1];
      (Array.isArray(values) ? values : [values]).forEach(function (value) {
        if (value !== undefined && value !== null && value !== '') params.append(name, value);
      });
    });
    return '/insights/facets?' + params.toString();
  }

  function requestFacets(windowDays, facets, signal, retry) {
    return fetch(facetUrl(windowDays, facets, retry), {signal: signal});
  }

  async function requestSnapshotPreview(path, csrf, signal) {
    const body = new FormData();
    body.set('csrf', csrf);
    body.set('snapshot_path', path);
    return fetch('/insights/snapshot-preview', {method: 'POST', body: body, signal: signal});
  }

  async function csrfToken(form) {
    const input = form && form.querySelector('input[name="csrf"]');
    if (input && input.value) return input.value;
    const response = await fetch('/insights/new', {headers: {'Accept': 'text/html'}});
    if (!response.ok) throw new Error('Could not load the run form security token. Reload the page and try again.');
    const doc = new DOMParser().parseFromString(await response.text(), 'text/html');
    const token = doc.querySelector('input[name="csrf"]')?.value;
    if (!token) throw new Error('Could not load the run form security token. Reload the page and try again.');
    return token;
  }

  async function uploadSource(kind, file, form) {
    const body = new FormData();
    body.set('csrf', await csrfToken(form));
    body.set('kind', kind);
    body.set('file', file);
    const response = await fetch('/insights/uploads', {method: 'POST', body: body});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Upload failed.');
    return result;
  }

  global.InsightsRunCommon = {
    normalizeSource: normalizeSource,
    validateSource: validateSource,
    validateSelection: validateSelection,
    requestFacets: requestFacets,
    requestSnapshotPreview: requestSnapshotPreview,
    csrfToken: csrfToken,
    uploadSource: uploadSource,
  };
})(window);
