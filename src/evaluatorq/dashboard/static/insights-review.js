
(async function startInsightsReview() {
  const root = document.querySelector('.insights-review-root');
  const loading = document.getElementById('review-loading');
  const error = document.getElementById('review-error');
  const progress = JSON.parse(root.dataset.runProgress || 'null');
  let D;
  try {
    const response = await fetch(root.dataset.reviewUrl, {headers: {'Accept': 'application/json'}});
    if (!response.ok) throw new Error(response.status === 404 ? 'This run could not be found or read.' : 'Request failed (' + response.status + ').');
    D = await response.json();
    if (!D || !D.run || !D.dims || !Array.isArray(D.traces) || !Array.isArray(D.labels)) throw new Error('This run returned an unreadable review payload.');
    loading.hidden = true;
  } catch (err) {
    loading.hidden = true;
    error.hidden = false;
    error.textContent = (err.message || 'Could not load this run.') + ' ';
    const retry = document.createElement('button'); retry.type = 'button'; retry.textContent = 'Retry';
    retry.onclick = () => location.reload(); error.append(retry);
    return;
  }
const T = D.traces || [], N = T.length;
const byId = Object.assign(Object.create(null), Object.fromEntries(T.map(t => [t.id, t])));
const byTraceId = new Map();
for (const t of T) {
  if (!byTraceId.has(t.trace_id)) byTraceId.set(t.trace_id, []);
  byTraceId.get(t.trace_id).push(t.id);
}
const QUAL = ['#2ebd85', '#ff8f34', '#7e22ce', '#025558', '#2f80ed', '#df5325', '#00b8a0', '#4fa8d8'];
const ERR_YES = '#df5325', ERR_NO = '#9fcfc7', OTHER = '#bdbbb5';
const MIN_N = 5;              // smallest group a headline may speak about
const TOP_K = 8;              // charts show this many themes, the rest fold into "Other"
const DIMS = Object.keys(D.dims);
const PD = D.run.priority_dimension && D.dims[D.run.priority_dimension] ? D.run.priority_dimension : (DIMS[0] || null);
const LABELS = D.labels, LBY = Object.assign(Object.create(null), Object.fromEntries(LABELS.map(l => [l.name, l])));
const FRUSL = LBY.user_frustration || null, OUTL = LBY.outcome || null, TASKL = LBY.task_type || null;
const FRUS_HI = 4;            // level 4 or 5 counts as frustrated
const BAD_YES = new Set(['unfixed_error', 'scope_creep']);   // for these, "yes" is the bad answer
const RISKY_CMD = ['git reset', 'git clean', 'rm', 'gh api', 'kubectl', 'npm publish', 'terraform'];
const HELPERS = new Set(['ls', 'cat', 'echo', 'head', 'tail', 'wc', 'grep', 'rg', 'sed', 'awk', 'cut', 'sort', 'uniq', 'jq', 'find', 'pwd', 'cd', 'which', 'sleep', 'printf', 'mkdir', 'cp', 'mv', 'touch', 'test', 'true', 'break', 'sync', 'tr', 'xargs', 'date', 'env', 'export']);   // everyday shell plumbing, hidden by default
const RISK_H = {none: 'No risky action', deleted: 'Deleted files', history_rewrite: 'Rewrote history', merged_or_closed: 'Merged/closed PR', published: 'Published', infra_change: 'Changed infra', secret_exposed: 'Exposed secret'};
const riskKind = t => { const v = lval(t, 'risky_action'); return v == null || v === 'none' ? null : v; };   // risky = any answer but "none"
const isRiskyT = t => riskKind(t) != null;
const riskyCmds = t => Object.entries(t.commands || {}).filter(([k]) => isRisky(k)).sort((a, b) => b[1] - a[1]);
const riskyText = t => { const r = riskyCmds(t), k = riskKind(t); return (RISK_H[k] || k) + (r.length ? ' · ran: ' + r.map(([c, n]) => `${c} ×${n}`).join(', ') : ''); };
const isRisky = c => RISKY_CMD.some(p => c === p || c.startsWith(p + ' ')) || c.includes('--force');

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const pct = (a, b) => b ? Math.round(100 * a / b) : 0;
const human = s => s.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
const NA = '<span class="na" title="Not measured for these traces">—</span>';
const fmtShare = v => v == null ? NA : Math.round(v * 100) + '%';
const hasErr = t => (t.errors || []).length > 0;
const reviewDate = value => {
  if (value === null || value === undefined || value === '') return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
};
const fmtDate = value => {
  const date = reviewDate(value);
  return date ? date.toLocaleDateString('en-GB', {day: 'numeric', month: 'short'}) : 'Time unavailable';
};
const fmtTime = value => {
  const date = reviewDate(value);
  return date
    ? date.toLocaleString('en-GB', {day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'})
    : 'Time unavailable';
};
const costLabel = run => run.cost == null ? 'Cost unavailable' : run.cost_is_partial ? `Partial cost: $${run.cost.toFixed(3)}` : `$${run.cost.toFixed(3)}`;
const $ = id => document.getElementById(id);
const FRAMP = ['#1f8f82', '#6aae9f', '#c9a13f', '#e0682f', '#c23a1c'];   // teal (calm) to red (angry)
const framp = k => FRAMP[Math.max(0, Math.min(4, Math.round(k * 4)))];
const SEM = {
  risky_action: {none: '#bdbbb5', deleted: '#e0a23a', merged_or_closed: '#e8b96a', published: '#d98a3d', infra_change: '#d9622b', history_rewrite: '#c23a1c', secret_exposed: '#8f1d1d'},
  outcome: {done: '#1f8f82', partial: '#e0a23a', not_done: '#c23a1c', cut_off: '#8e8c96', inconclusive: '#bdbbb5'},
  verified: {verified: '#299D8F', claimed_without_check: '#c23a1c', unverified: '#ec7a3f', not_applicable: '#bdbbb5'},
};

/* ---------- labels, generic over kind ---------- */
function scoreKey(criteria, index) {
  const text = String(criteria[index] ?? '');
  return text.match(/^\s*(\d+)/)?.[1] || String(index);
}
function lvals(l) { return l.kind === 'noul' ? ['yes', 'no'] : l.kind === 'score' ? l.criteria.map((_, i) => scoreKey(l.criteria, i)) : Object.keys(l.criteria); }
function lval(t, name) {
  const a = t.l[name], l = LBY[name];
  if (!a || a[0] == null) return null;
  if (l.kind === 'noul') return a[0] === true ? 'yes' : 'no';
  if (l.kind === 'score') {
    const key = String(a[0]);
    return lvals(l).includes(key) ? key : null;
  }
  return String(a[0]);
}
const lnum = (_name, v) => +v;
function ldisp(name, v) {
  const l = LBY[name];
  if (l.kind === 'score') {
    const index = lvals(l).indexOf(v);
    return index < 0 ? v : String(l.criteria[index]).split(':')[0].replace(/^\s*\d+\s*/, '').trim() || v;
  }
  return (name === 'risky_action' && RISK_H[v]) || human(v);
}
function lcolor(name, v) {
  const l = LBY[name];
  if (SEM[name] && SEM[name][v]) return SEM[name][v];
  if (l.kind === 'noul') return BAD_YES.has(name) ? (v === 'yes' ? '#df5325' : '#9fcfc7') : (v === 'yes' ? '#025558' : OTHER);
  if (l.kind === 'score') return framp(Math.max(0, lvals(l).indexOf(v)) / (l.criteria.length - 1));
  return QUAL[lvals(l).indexOf(v) % QUAL.length];
}
function frus(t) { const v = FRUSL && lval(t, FRUSL.name); return v == null ? null : lnum(FRUSL.name, v); }
const frusColor = lv => lv == null ? 'var(--border-strong)' : framp((lv - 1) / 4);
const frusHtml = (lv, detail) => {
  if (lv == null) return NA;
  const color = frusColor(lv);
  const ink = lv < 2.5 ? '#025558' : lv < 3.5 ? '#765400' : '#b8341c';
  return `<span class="fr" style="color:${ink};background:${hexA(color, .16)}" title="${detail || `Frustration ${Number.isInteger(lv) ? lv : lv.toFixed(1)} of 5`}">${Number.isInteger(lv) ? lv : lv.toFixed(1)}</span>`;
};
const isYes = (t, name) => lval(t, name) === 'yes';
const notVerified = t => ['claimed_without_check', 'unverified'].includes(lval(t, 'verified'));
const corr3 = t => lval(t, 'user_corrections') === '3';
const chip = (name, t) => { const v = lval(t, name); return v == null ? NA : `<span class="chip" style="background:${hexA(lcolor(name, v), .16)}" title="${esc(human(name))}: ${esc(v)}">${esc(ldisp(name, v))}</span>`; };
const outcomeHtml = t => { const v = lval(t, 'outcome'); return v == null ? NA : `<span class="sdot" style="color:var(--text-strong)"><i style="background:${lcolor('outcome', v)}"></i>${esc(human(v))}</span>`; };
const FLAG_DEFS = [
  ['unfixed', 'Unfixed error: a failure the agent never corrected', t => isYes(t, 'unfixed_error'),
    '<path d="M12 8v5M12 16.5v.01"/><circle cx="12" cy="12" r="9"/>'],
  ['risky', 'Risky action the user did not ask for', t => isRiskyT(t),
    '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17v.01"/>'],
  ['unver', 'Not verified: no check run, or claimed to work without one', notVerified,
    '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><path d="M4 4l16 16"/>'],
  ['corr', 'Corrected three or more times by the user', corr3,
    '<path d="M9 14L4 9l5-5"/><path d="M4 9h10a6 6 0 010 12h-3"/>'],
];
const flagSvg = p => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${p}</svg>`;
const flagsText = t => FLAG_DEFS.filter(f => f[2](t)).map(f => `<span class="flagtxt"><span class="fl ${f[0]}">${flagSvg(f[3])}</span>${f[0] === 'risky' ? 'Risky action · ' + esc(riskyText(t)) : f[1]}</span>`).join('');
const flagCell = (t, flag) => { const yes = flag[2](t); return `<span class="bool ${yes ? 'yes' : 'no'}" title="${esc(yes ? (flag[0] === 'risky' ? riskyText(t) : flag[1]) : 'No ' + flag[0] + ' flag')}">${yes ? 'Yes' : '—'}</span>`; };
/* ---------- clusters ---------- */
const C = Object.create(null);
for (const dim of DIMS) {
  for (const c of D.dims[dim].clusters) {
    c.dim = dim; c.members = new Set();
    c.example_trace_ids = (c.example_trace_ids || []).flatMap(id => byTraceId.get(id) || []);
    C[c.id] = c;
  }
  const nd = D.dims[dim];
  C['none:' + dim] = {id: 'none:' + dim, dim, level: 'base', none: true, members: new Set(), color: OTHER,
    name: dim === 'failure' ? 'No assistant error found' : 'Not grouped',
    description: `Traces not placed in a ${dim} theme: ${nd.n_no_signal} had no ${dim} text to group, ${nd.n_noise} did not fit any theme.`};
}
for (const t of T) for (const dim of DIMS) {
  const c = C[t.a[dim]];
  if (c) { c.members.add(t.id); if (C[c.parent_id]) C[c.parent_id].members.add(t.id); }
  else C['none:' + dim].members.add(t.id);
}
const bySize = (a, b) => b.members.size - a.members.size || a.name.localeCompare(b.name);
function bases(dim, withNone) {
  const out = D.dims[dim].clusters.filter(c => c.level === 'base').sort(bySize);
  if (withNone && C['none:' + dim].members.size) out.push(C['none:' + dim]);
  return out;
}
function tops(dim) { return D.dims[dim].clusters.filter(c => c.level === 'top').sort(bySize); }
function kids(top) { return bases(top.dim).filter(k => k.parent_id === top.id); }
for (const dim of DIMS) {
  bases(dim).forEach((c, i) => c.color = QUAL[i % QUAL.length]);   // colour follows live size order, same order every view uses
  tops(dim).forEach(c => c.color = (kids(c)[0] || {color: OTHER}).color);
}
function clusterOf(t, dim) { return C[t.a[dim]] || C['none:' + dim]; }
function withOther(list) {
  if (list.length <= TOP_K + 1) return list;
  const rest = list.slice(TOP_K), members = new Set(rest.flatMap(c => [...c.members]));
  return [...list.slice(0, TOP_K), {id: 'other', other: true, dim: rest[0].dim, name: `Other (${rest.length} themes)`, color: OTHER, members}];
}

/* ---------- aggregates: unmeasured stays null, never 0 ---------- */
const PRI = Object.fromEntries((D.run.priority || []).map(p => [p.cluster_id, p]));
function stats(ids, clusterId) {
  const ts = [...ids].map(i => byId[i]), n = ts.length;
  const summ = ts.filter(t => t.has_summary);
  const share = (name, pred) => { const a = ts.filter(t => lval(t, name) != null); return a.length ? a.filter(pred).length / a.length : null; };
  const errShare = summ.length ? summ.filter(hasErr).length / summ.length : null;
  const fr = ts.map(frus).filter(v => v != null);
  const p = clusterId && PRI[clusterId];
  return {n, errShare: p ? p.error_share : errShare, errN: summ.filter(hasErr).length, summN: summ.length,
    doneShare: share('outcome', t => lval(t, 'outcome') === 'done'),
    riskyShare: share('risky_action', isRiskyT),
    unfixedN: ts.filter(t => isYes(t, 'unfixed_error')).length, unfixedShare: share('unfixed_error', t => isYes(t, 'unfixed_error')),
    frusMean: fr.length ? fr.reduce((a, b) => a + b, 0) / fr.length : null, frusMeasured: fr.length,
    frusN: fr.filter(v => v >= FRUS_HI).length, frusShare: fr.length ? fr.filter(v => v >= FRUS_HI).length / fr.length : null};
}

/* ---------- state, mirrored into the URL ---------- */
const S = {dim: PD, view: 'themes', colorBy: PD ? 'cluster' : 'err', cmp: OUTL ? 'l:outcome' : 'err', filters: [], sel: null, more: 20, q: '', shut: new Set(), activityKind: 'skills', activityQuery: '', activitySort: 'coverage', activityItem: null, activityHelpers: false, activityExpanded: new Set(), activityShowAll: false};
// filter spec: {t:'c', id} | {t:'l', name, v} | {t:'e', yes} | {t:'x', k} (named predicate) | {t:'b', ids:Set}
const XF = {frustrated: {label: 'Frustrated users', color: '#c23a1c', test: t => (frus(t) ?? 0) >= FRUS_HI}};
const fkey = f => f.t === 'c' ? 'c:' + (C[f.id]?.dim || '') : f.t === 'l' ? 'l:' + f.name : f.t === 'x' ? 'x:' + f.k : f.t;
function fids(f) {
  if (f.t === 'c') return C[f.id].members;
  if (f.t === 'l') return f._ids ||= new Set(T.filter(t => lval(t, f.name) === f.v).map(t => t.id));
  if (f.t === 'e') return f._ids ||= new Set(T.filter(t => t.has_summary && hasErr(t) === f.yes).map(t => t.id));
  if (f.t === 'x') return f._ids ||= new Set(T.filter(XF[f.k].test).map(t => t.id));
  return f.ids;
}
function flabel(f) {
  if (f.t === 'c') { const c = C[f.id]; return `${cap(c.dim)} · ${c.name}`; }
  if (f.t === 'l') return `${human(f.name)} · ${ldisp(f.name, f.v)}`;
  if (f.t === 'e') return f.yes ? 'Assistant made errors' : 'No assistant errors';
  if (f.t === 'x') return XF[f.k].label;
  return `Map selection · ${f.ids.size} traces`;
}
function fcolor(f) { return f.t === 'c' ? C[f.id].color : f.t === 'l' ? lcolor(f.name, f.v) : f.t === 'e' ? (f.yes ? ERR_YES : ERR_NO) : f.t === 'x' ? XF[f.k].color : '#25232e'; }
const same = (a, b) => fkey(a) === fkey(b) && a.id === b.id && a.v === b.v && a.yes === b.yes && a.k === b.k;
function isOn(f) { return S.filters.some(x => same(x, f)); }
function toggle(f) {
  const on = isOn(f);
  S.filters = S.filters.filter(x => fkey(x) !== fkey(f));
  if (!on) S.filters.push(f);
  S.more = 20; commit();
}
function baseVisible() {
  const q = S.q.toLowerCase();
  return T.filter(t => S.filters.every(f => fids(f).has(t.id)) && (!q || `${t.request} ${t.summary} ${t.topic || ''}`.toLowerCase().includes(q)));
}
function visible() {
  const ts = baseVisible(), item = S.view === 'activity' ? S.activityItem : null;
  return item ? ts.filter(t => t.has_tool_stats && (t[item.kind] || {})[item.name] > 0) : ts;
}
function encodeTraceKey(id) { return encodeURIComponent(id).replace(/[.!'()*]/g, char => '%' + char.charCodeAt(0).toString(16).toUpperCase()); }
function encF(f) { return f.t === 'c' ? 'c:' + encodeTraceKey(f.id) : f.t === 'l' ? `l:${encodeTraceKey(f.name)}:${encodeTraceKey(f.v)}` : f.t === 'e' ? 'e:' + (f.yes ? 1 : 0) : f.t === 'x' ? 'x:' + encodeTraceKey(f.k) : 'b:' + [...f.ids].map(encodeTraceKey).join('.'); }
function decodePart(value) { try { return decodeURIComponent(value); } catch { return ''; } }
function decF(s) {
  const [t, ...r] = s.split(':');
  if (t === 'c') { const id = decodePart(r.join(':')); if (C[id]) return {t: 'c', id}; }
  if (t === 'l') {
    const name = decodePart(r[0] || ''), value = decodePart(r.slice(1).join(':'));
    if (LBY[name] && lvals(LBY[name]).includes(value)) return {t: 'l', name, v: value};
  }
  if (t === 'e') return {t: 'e', yes: r[0] === '1'};
  if (t === 'x' && XF[r[0]]) return {t: 'x', k: r[0]};
  if (t === 'b') {
    const ids = new Set(r.join(':').split('.').map(decodePart));
    return ids.size && [...ids].every(id => byId[id]) ? {t: 'b', ids} : null;
  }
  return null;
}
function toUrl() {
  const p = new URLSearchParams();
  p.set('dim', S.dim); p.set('view', S.view);
  if (S.filters.length) p.set('f', S.filters.map(encF).join('|'));
  if (S.sel) p.set('sel', S.sel.kind[0] + ':' + S.sel.id);
  if (S.colorBy !== 'cluster') p.set('color', S.colorBy);
  if (S.view === 'compare') p.set('cmp', S.cmp);
  if (S.view === 'activity') {
    p.set('kind', S.activityKind);
    if (S.activityQuery) p.set('aq', S.activityQuery);
    if (S.activitySort !== 'coverage') p.set('as', S.activitySort);
    if (S.activityItem) p.set('item', S.activityItem.kind + ':' + S.activityItem.name);
    if (S.activityHelpers) p.set('helpers', '1');
    if (S.activityExpanded.size) p.set('ex', [...S.activityExpanded].join('|'));
  }
  if (S.q) p.set('q', S.q);
  return '#' + p.toString();
}
function fromUrl() {
  const p = new URLSearchParams(location.hash.slice(1));
  // Optional controls are omitted from the canonical URL when they have their
  // default value. Reset them before parsing so Back restores that default.
  S.colorBy = S.dim ? 'cluster' : 'err';
  S.cmp = OUTL ? 'l:outcome' : 'err';
  if (D.dims[p.get('dim')]) S.dim = p.get('dim');
  if (['themes', 'activity', 'map', 'compare'].includes(p.get('view'))) S.view = p.get('view');
  S.filters = (p.get('f') || '').split('|').filter(Boolean).map(decF).filter(Boolean);
  const sel = p.get('sel');
  S.sel = sel ? (sel[0] === 'c' && C[sel.slice(2)] ? {kind: 'cluster', id: sel.slice(2)} : sel[0] === 't' && byId[sel.slice(2)] ? {kind: 'trace', id: sel.slice(2)} : null) : null;
  const color = p.get('color') || 'cluster';
  if (color === 'cluster' || color === 'err' || color.startsWith('l:') && LBY[color.slice(2)]) S.colorBy = color;
  const cmp = p.get('cmp');
  if (cmp === 'err' || cmp?.startsWith('l:') && LBY[cmp.slice(2)] || cmp?.startsWith('d:') && D.dims[cmp.slice(2)] && cmp.slice(2) !== S.dim) S.cmp = cmp;
  S.activityKind = ['skills', 'tools', 'commands'].includes(p.get('kind')) ? p.get('kind') : 'skills';
  S.activityQuery = p.get('aq') || '';
  S.activitySort = p.get('as') === 'calls' ? 'calls' : 'coverage';
  if (S.activityKind === 'skills') S.activitySort = 'coverage';
  const item = p.get('item') || '', sep = item.indexOf(':');
  const activityKind = item.slice(0, sep), activityName = item.slice(sep + 1);
  S.activityItem = sep > 0 && ['skills', 'tools', 'commands'].includes(activityKind) && T.some(t => t.has_tool_stats && (t[activityKind] || {})[activityName] > 0) ? {kind: activityKind, name: activityName} : null;
  S.activityHelpers = p.get('helpers') === '1';
  S.activityExpanded = new Set((p.get('ex') || '').split('|').filter(Boolean));
  S.activityShowAll = false;
  S.q = p.get('q') || '';
}
function commit() { const h = toUrl(); if (h !== location.hash) history.pushState(null, '', h); render(); }
addEventListener('popstate', () => { fromUrl(); render(); });

/* ---------- header ---------- */
function renderHeader() {
  const r = D.run, p = r.population;
  const stageRows = progress ? (progress.stages || []) : [];
  const stageDone = stageRows.filter(stage => stage.status === 'completed').length;
  const activeStage = stageRows.find(stage => stage.name === progress?.stage);
  const stageLabel = progress?.stage_labels?.[progress.stage] || (progress?.stage || '').replace(/_/g, ' ');
  const progressText = progress?.status === 'running'
    ? `${stageLabel || 'Run'}${activeStage?.completed != null && activeStage?.total != null ? ` · ${activeStage.completed}/${activeStage.total}` : ''}`
    : stageRows.length ? `${stageDone} of ${stageRows.length} stages` : '';
  $('title').textContent = r.name;
  $('status').innerHTML = `<span class="dot">${esc(cap(r.status || "unknown"))}</span><span class="sep">·</span><span class="num">${N}</span> traces<span class="sep">·</span>${esc(r.source_label)}
    <span class="sep">·</span>${esc([...new Set(T.map(t => t.agent))].join(', '))}<span class="sep">·</span>${fmtTime(r.created)}<span class="sep">·</span><span class="num">${esc(costLabel(r))}</span>
    ${r.stage_failures && r.stage_failures.length ? `<span class="chip-warn" title="${esc(r.stage_failures.map(f => f.stage + ': ' + f.message).join('\n'))}">${r.stage_failures.length} failed stages</span>` : ''}
    ${r.warnings && r.warnings.length ? `<span class="chip-warn" title="${esc(r.warnings.join('\n'))}">${r.warnings.length} warnings</span>` : ''}
    ${LABELS.length === 0 ? '<span class="chip-warn">No usable label results</span>' : ''}
    ${T.some(t => !t.has_summary) ? `<span class="chip-warn">Summaries unavailable for ${T.filter(t => !t.has_summary).length} traces</span>` : ''}
    ${r.status === 'error' || r.status === 'cancelled' ? '<span class="chip-warn">Partial run</span>' : ''}
    ${progressText ? `<span class="sep">·</span><span>${esc(progressText)}</span>` : ''}
    ${progress?.error ? `<span class="chip-warn" title="${esc(progress.error)}">Run failed</span>` : ''}
    ${p.n_projection_truncated ? `<span class="chip-warn" title="Long traces were shortened before the models read them">${p.n_projection_truncated} traces shortened</span>` : ''}
    <button class="linkbtn" id="detBtn">Run details ▾</button>`;
  const cbs = r.cost_by_stage || {};
  const stageNames = progress ? [...new Set([...(progress.planned_stages || []), ...stageRows.map(stage => stage.name)])] : [];
  const stageHtml = stageNames.length ? stageNames.map(name => {
    const stage = stageRows.find(item => item.name === name);
    const status = stage?.status || (progress?.status === 'running' ? 'pending' : progress?.status === 'error' && name === progress?.stage ? 'error' : 'skipped');
    const label = progress?.stage_labels?.[name] || name.replace(/_/g, ' ');
    const count = stage?.completed != null && stage?.total != null ? `${stage.completed}/${stage.total}` : '—';
    const mark = status === 'completed' ? '✓' : status === 'error' ? '!' : status === 'running' ? '…' : '·';
    const stageCost = r.cost_by_stage_details?.[name];
    const costText = stageCost?.cost == null
      ? 'Cost unavailable'
      : `${stageCost.is_partial ? 'Partial cost: ' : ''}$${stageCost.cost.toFixed(4)}`;
    return `<span class="${status === 'completed' ? 'ok' : 'na'}">${mark}</span><span>${esc(label)} · ${esc(status)}</span><span class="num" style="color:var(--text-muted)">${count}</span><span class="num">${esc(costText)}</span>`;
  }).join('') : '<span class="na">Stage history unavailable</span>';
  const projectionKnown = p.classifier_state_chars != null && p.source_chars != null;
  const sourceChars = projectionKnown ? p.source_chars : 0;
  const classifierChars = projectionKnown ? p.classifier_state_chars : 0;
  const requestCoverageKnown = projectionKnown && p.classifier_question_chars != null && p.classifier_request_chars != null;
  const questionChars = requestCoverageKnown ? p.classifier_question_chars : 0;
  const requestChars = requestCoverageKnown ? p.classifier_request_chars : classifierChars;
  const summaryAverage = projectionKnown ? Math.round((p.summary_input_chars || 0) / Math.max(N, 1)) : 0;
  const classifierAverage = projectionKnown ? Math.round(classifierChars / Math.max(N, 1)) : 0;
  const questionAverage = requestCoverageKnown ? Math.round(questionChars / Math.max(N, 1)) : 0;
  const requestAverage = requestCoverageKnown ? Math.round(requestChars / Math.max(N, 1)) : 0;
  const projectionText = projectionKnown
    ? requestCoverageKnown
      ? `${p.n_projection_truncated || 0} of ${N} traces hit a summary or classifier character cap. Average summary prompt: ${summaryAverage.toLocaleString()} characters; classifier request: ${requestAverage.toLocaleString()} characters (state ${classifierAverage.toLocaleString()}, questions ${questionAverage.toLocaleString()}). The source traces are unchanged.`
      : `${p.n_projection_truncated || 0} of ${N} traces hit a summary or classifier character cap. Average summary prompt: ${summaryAverage.toLocaleString()} characters; classifier state: ${classifierAverage.toLocaleString()} characters. The source traces are unchanged.`
    : 'Character-based input coverage is unavailable for this run.';
  const reserveText = p.classifier_question_reserve === 'not_available'
    ? ' Selected classifier questions were not included in this preview; the final request may reserve more of the input cap.'
    : p.classifier_question_reserve === 'applied'
      ? ' Exact serialized question payloads share the configured cap with classifier state; Jev also applies its model-specific ceilings.'
      : '';
  const projectionPct = projectionKnown ? Math.min(100, pct(requestChars, sourceChars)) : 0;
  $('details').innerHTML = `<h4>Stages</h4><div class="stages">${stageHtml}</div>
    <h4>Models</h4><dl class="kv"><dt>Summary</dt><dd class="num">${esc(r.summary_model)}</dd><dt>Classifier</dt><dd class="num">${esc(r.classifier_model)}</dd><dt>Embeddings</dt><dd class="num">${esc(r.embedding_model)}</dd></dl>
    <h4>Questions asked</h4><dl class="kv">${LABELS.map(l => `<dt>${esc(human(l.name))}</dt><dd>${esc(l.instructions || '')}</dd>`).join('')}</dl>
    <h4>Classifier input coverage</h4><div class="proj"><i style="width:${projectionPct}%"></i></div>
    <p style="margin:0;color:var(--text-muted)">${esc(projectionText + reserveText)}</p>`;
  $('detBtn').onclick = e => {
    e.stopPropagation();
    const b = e.target.getBoundingClientRect(), pop = $('details');
    pop.style.left = (b.left - 60) + 'px'; pop.style.top = (b.bottom + 8) + 'px';
    pop.classList.toggle('open');
  };
}

/* ---------- headlines: each candidate names what it needs, and skips when the run lacks it ---------- */
const CANDIDATES = [
  {k: 'Biggest theme', needs: () => DIMS.length > 0, make() {
    const t = (tops(PD).length ? tops(PD) : bases(PD)).find(c => c.members.size >= MIN_N);
    return t && {number: pct(t.members.size, N) + '%', finding: `${esc(t.name)} is the biggest theme`, why: `${t.members.size} of ${N} traces`, action: 'Explore theme →', f: {t: 'c', id: t.id}};
  }},
  {k: 'Fix first', needs: () => !!LBY.unfixed_error, make() {
    const all = stats(new Set(T.map(t => t.id)));
    if (!all.unfixedN) return null;
    const c = DIMS.length ? bases(PD).map(c => ({c, s: stats(c.members)})).filter(d => d.s.n >= MIN_N && d.s.unfixedShare != null).sort((a, b) => b.s.unfixedShare - a.s.unfixedShare)[0] : null;
    return {number: `${all.unfixedN} ${all.unfixedN === 1 ? 'trace' : 'traces'}`, finding: 'Unfixed errors need review', why: c ? `Most common in ${esc(c.c.name)} (${Math.round(c.s.unfixedShare * 100)}% of that theme)` : 'Based on tool evidence', action: 'Review traces →', f: {t: 'l', name: 'unfixed_error', v: 'yes'}};
  }},
  {k: 'Frustration', needs: () => !!FRUSL, make() {
    const all = stats(new Set(T.map(t => t.id)));
    if (!all.frusN) return null;
    // theme = highest share of its own traces frustrated (themes of 5+ traces only)
    const c = DIMS.length ? bases(PD).map(c => ({c, s: stats(c.members)})).filter(d => d.s.n >= MIN_N && d.s.frusShare != null).sort((a, b) => b.s.frusShare - a.s.frusShare || b.s.frusN - a.s.frusN)[0] : null;
    return {number: `${all.frusN} ${all.frusN === 1 ? 'trace' : 'traces'}`, finding: 'High user frustration was recorded', why: c ? `Most often in ${esc(c.c.name)} (${c.s.frusN} of ${c.s.n} traces)` : 'Frustration level 4 or 5', action: 'Review traces →', f: {t: 'x', k: 'frustrated'}};
  }},
  {k: 'Done rate', needs: () => !!OUTL, make() {
    const all = stats(new Set(T.map(t => t.id)));
    if (all.doneShare == null) return null;
    const groups = (TASKL ? lvals(TASKL) : []).map(v => { const ids = new Set(T.filter(t => lval(t, 'task_type') === v).map(t => t.id)); return {v, s: stats(ids)}; })
      .filter(d => d.s.n >= 3 && d.s.doneShare != null).sort((a, b) => a.s.doneShare - b.s.doneShare);
    const w = TASKL && groups[0];
    return {number: Math.round(all.doneShare * 100) + '%', finding: 'Tasks were completed', why: w ? `${esc(human(w.v))} had the lowest done rate at ${Math.round(w.s.doneShare * 100)}%` : 'Share of answered outcome labels', action: w ? `Inspect ${human(w.v).toLowerCase()} traces →` : 'Review completed traces →',
      f: w ? {t: 'l', name: 'task_type', v: w.v} : {t: 'l', name: 'outcome', v: 'done'}};
  }},
];
let HEADS = [];
function renderHeadlines() {
  HEADS = CANDIDATES.filter(c => c.needs()).map(c => { const h = c.make(); return h && {...h, lead: c.k === 'Fix first'}; }).filter(Boolean);
  HEADS.sort((a, b) => Number(b.lead) - Number(a.lead));
  const el = $('heads');
  if (!HEADS.length) { el.innerHTML = `<div class="head empty">${DIMS.length ? `No theme is large enough (${MIN_N}+ traces) to highlight. Explore the themes below.` : 'This run has no themes to highlight. Explore its labels, traces and activity.'}</div>`; return; }
  el.innerHTML = HEADS.map((h, i) => `<button class="head ${h.lead ? 'lead ' : ''}${isOn(h.f) ? 'on' : ''}" data-h="${i}" aria-pressed="${isOn(h.f)}">
    <span class="number">${h.number}</span><span><span class="finding">${h.finding}</span><span class="why" style="display:block">${h.why}</span></span>
    <span class="go">${isOn(h.f) ? 'Showing these ✓' : esc(h.action)}</span></button>`).join('');
  el.querySelectorAll('.head[data-h]').forEach(b => b.onclick = () => {
    const f = HEADS[+b.dataset.h].f;
    if (f.t === 'c') { S.dim = C[f.id].dim; S.sel = isOn(f) ? null : {kind: 'cluster', id: f.id}; }
    toggle(f);
  });
}

/* ---------- filter bar ---------- */
function renderFilterbar() {
  const v = visible().length;
  const activity = S.view === 'activity' ? S.activityItem : null;
  $('fbar').innerHTML = `<span class="count">Showing <b class="num">${v}</b> of <b class="num">${N}</b> traces</span>` +
    (S.filters.length || activity
      ? S.filters.map((f, i) => `<span class="fchip"><i style="background:${fcolor(f)}"></i><span>${esc(flabel(f))}</span><button data-rm="${i}" aria-label="Remove filter">×</button></span>`).join('') +
        (activity ? `<span class="fchip"><span>${esc(cap(activity.kind))} · ${esc(activity.name)}</span><button id="clearActivity" aria-label="Remove activity selection">×</button></span>` : '') +
        `<button class="linkbtn" id="clearAll">Clear all</button>`
      : `<span class="hint">Click a theme, label value, or activity item to narrow the trace list.</span>`) +
    `<button class="btn sm right" id="copyLink" title="The address holds your filters, view and selection">Copy link to this view</button>`;
  $('fbar').querySelectorAll('[data-rm]').forEach(b => b.onclick = () => { S.filters.splice(+b.dataset.rm, 1); commit(); });
  const ci = $('clearActivity'); if (ci) ci.onclick = () => { S.activityItem = null; commit(); };
  const ca = $('clearAll'); if (ca) ca.onclick = () => { S.filters = []; S.activityItem = null; commit(); };
  $('copyLink').onclick = () => { navigator.clipboard?.writeText(location.href).catch(() => {}); toast('Link copied. Opening it restores these filters, this view and the open panel.'); };
}

/* ---------- theme tree ---------- */
function treeRow(c, kind, vis) {
  const s = stats(c.members, c.id), inView = [...c.members].filter(id => vis.has(id)).length;
  const on = S.sel && S.sel.id === c.id || isOn({t: 'c', id: c.id});
  const grp = kind.startsWith('grp'), open = !S.shut.has(c.id);
  const ic = c.none ? '<span class="ic"></span>' : !grp ? '<span class="ic" aria-hidden="true">↳</span>' : `<button class="chev ${open ? '' : 'shut'}" data-tog="${esc(c.id)}" aria-expanded="${open}" title="Show or hide its subthemes"><svg width="14" height="14" viewBox="0 0 12 12"><path d="M3 4.5l3 3 3-3" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></button>`;
  const count = S.filters.length ? `${inView}<small>/ ${s.n}</small>` : `${s.n}<small>· ${pct(s.n, N)}%</small>`;
  const errCell = c.dim === 'failure' && !c.none ? '<span class="na" title="Every trace here is a failure">·</span>' : fmtShare(s.errShare);
  return `<div class="row ${kind} ${on ? 'on' : ''} ${S.filters.length && !inView ? 'dim' : ''} ${c.none ? 'none' : ''}" data-c="${esc(c.id)}" style="--bc:${c.color}" role="button" tabindex="0" aria-pressed="${on}">
    ${ic}<span class="nm"><span class="t">${grp ? '' : `<i style="background:${c.color}"></i>`}<span title="${esc(c.name)}">${esc(c.name)}</span></span></span>
    <span class="n">${count}<div class="bar"><i style="width:${pct(s.n, N)}%"></i><b style="width:${pct(inView, N)}%"></b></div></span>
    <span class="e ${s.errShare > .6 ? 'bad' : ''}" title="${s.errShare == null ? 'No summaries' : `${s.errN} of ${s.summN} summarised traces`}">${errCell}</span>
    <span class="e">${fmtShare(s.doneShare)}</span>
    <span class="e">${fmtShare(s.riskyShare)}</span>
    <span class="e">${frusHtml(s.frusMean, `Mean of ${s.frusMeasured} measured traces; ${s.frusN} at level ${FRUS_HI} or 5`)}</span></div>`;
}
function renderThemes() {
  if (!S.dim) {
    $('canvas').innerHTML = `<div class="empty"><b>No themes in this run</b>This run has no grouping dimensions. Its labels and traces are still available in the other views.</div>`;
    return;
  }
  const vis = new Set(visible().map(t => t.id));
  let html = `<div class="tree"><div class="cols"><span></span><span>Theme</span><span>Traces</span><span title="from summary: share of summarised traces where the assistant made a mistake">Errors</span><span title="Share of traces where the outcome is done">Done %</span><span title="Share of traces where the agent took a risky action nobody asked for">Risky %</span><span title="Mean user frustration, 1 (calm) to 5 (angry)">Frustration</span></div>`;
  const ts = tops(S.dim);
  if (ts.length) for (const top of ts) {
    const ks = kids(top);
    html += treeRow(top, 'grp', vis);
    if (!S.shut.has(top.id)) html += ks.map(k => treeRow(k, 'kid', vis)).join('');
  } else html += bases(S.dim).map(c => treeRow(c, 'kid', vis)).join('');
  const none = C['none:' + S.dim];
  if (none.members.size) html += treeRow(none, 'grp none', vis);
  html += `</div>`;
  const cv = $('canvas');
  cv.innerHTML = html;
  cv.querySelectorAll('[data-tog]').forEach(b => b.onclick = e => {
    e.stopPropagation(); const id = b.dataset.tog;
    S.shut.has(id) ? S.shut.delete(id) : S.shut.add(id);
    history.replaceState(null, '', toUrl()); renderThemes();
  });
  cv.querySelectorAll('.row[data-c]').forEach(b => { b.onclick = () => pickCluster(b.dataset.c); b.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickCluster(b.dataset.c); } }; });
}
function pickCluster(id) {
  if (id === 'other') return toast('Pick a theme in the Themes tab to see one of the smaller themes.');
  const f = {t: 'c', id};
  S.sel = isOn(f) ? null : {kind: 'cluster', id};
  toggle(f);
}

/* ---------- centre views ---------- */
const ICON = {
  themes: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 6h16M8 12h12M8 18h12M4 12v6"/></svg>',
  activity: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 19h18M6 16V9M12 16V5M18 16v-4"/></svg>',
  map: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="5" cy="6" r="1.5"/><circle cx="12" cy="9" r="1.5"/><circle cx="8" cy="16" r="1.5"/><circle cx="18" cy="14" r="1.5"/><circle cx="17" cy="5" r="1.5"/><circle cx="14" cy="19" r="1.5"/></svg>',
  compare: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="8" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/><rect x="13" y="13" width="8" height="8" rx="1.5"/></svg>',
};
function colorOptions() {
  return [...(S.dim ? [['cluster', `${cap(S.dim)} theme`]] : []), ['err', 'Assistant errors'], ...LABELS.map(l => ['l:' + l.name, human(l.name)])];
}
function cmpOptions() {
  return [...LABELS.map(l => ['l:' + l.name, human(l.name)]), ['err', 'Assistant errors'], ...DIMS.filter(d => d !== S.dim).map(d => ['d:' + d, cap(d) + ' themes'])];
}
function renderStage() {
  $('tabs').innerHTML = [['themes', 'Themes'], ['activity', 'Activity'], ['map', 'Map'], ['compare', 'Compare']].map(([k, l]) => `<button class="${S.view === k ? 'on' : ''}" data-v="${k}">${ICON[k]}${l}</button>`).join('');
  $('tabs').querySelectorAll('[data-v]').forEach(b => b.onclick = () => { S.view = b.dataset.v; if (S.view === 'activity') S.sel = null; commit(); });
  $('dims').style.display = S.view === 'activity' || !DIMS.length ? 'none' : '';
  $('dims').innerHTML = `<span>Group by</span><div class="seg">${DIMS.map(d => `<button class="${S.dim === d ? 'on' : ''}" data-d="${d}">${cap(d)}</button>`).join('')}</div>`;
  $('dims').querySelectorAll('[data-d]').forEach(b => b.onclick = () => {
    S.dim = b.dataset.d;
    if (S.cmp === 'd:' + S.dim) S.cmp = 'err';
    commit();
  });
  const ctl = $('ctl');
  if (S.view === 'themes') { ctl.innerHTML = ''; renderThemes(); }
  else if (S.view === 'activity') {
    ctl.innerHTML = '';
    renderActivity();
  } else if (S.view === 'map') {
    const opts = colorOptions(); if (!opts.some(o => o[0] === S.colorBy)) S.colorBy = S.dim ? 'cluster' : 'err';
    ctl.innerHTML = `Colour <select id="colorBy">${opts.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join('')}</select><button class="btn sm" id="fullBtn">${$('stage').classList.contains('full') ? 'Exit full screen' : 'Full screen'}</button>`;
    $('colorBy').value = S.colorBy; $('colorBy').onchange = e => { S.colorBy = e.target.value; commit(); };
    $('fullBtn').onclick = () => { $('stage').classList.toggle('full'); renderStage(); };
    renderMap();
  } else {
    const opts = cmpOptions(); if (!opts.some(o => o[0] === S.cmp)) S.cmp = opts[0][0];
    ctl.innerHTML = `Against <select id="cmp">${opts.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join('')}</select>`;
    $('cmp').value = S.cmp; $('cmp').onchange = e => { S.cmp = e.target.value; commit(); };
    renderCompare();
  }
  if (S.view !== 'map') $('stage').classList.remove('full');
}

function canvasW() { return Math.max(520, $('canvas').clientWidth - 32); }

/* ---------- activity ---------- */
const ACTIVITY_KINDS = ['skills', 'tools', 'commands'];
const ACTIVITY_LABELS = {skills: 'Skills', tools: 'Tools', commands: 'Shell commands'};
const SHELL_TOOL_NAMES = new Set(D.activity?.shell_tools || []);
function activityEligible(traces) { return traces.filter(t => t.has_tool_stats); }
function activityRows(traces, kind) {
  const rows = new Map();
  for (const t of activityEligible(traces)) for (const [name, calls] of Object.entries(t[kind] || {})) {
    if (!(calls > 0)) continue;
    if (!rows.has(name)) rows.set(name, {name, traces: 0, calls: 0, ids: new Set()});
    const row = rows.get(name);
    row.traces++; row.calls += calls; row.ids.add(t.id);
  }
  return [...rows.values()].sort((a, b) => b.traces - a.traces || b.calls - a.calls || a.name.localeCompare(b.name));
}
function activityWeekKey(timestamp) {
  if (timestamp == null || timestamp === '') return null;
  const day = new Date(timestamp);
  if (Number.isNaN(day.getTime())) return null;
  day.setUTCHours(0, 0, 0, 0);
  day.setUTCDate(day.getUTCDate() - (day.getUTCDay() + 6) % 7);
  return day.toISOString().slice(0, 10);
}
function activityWeeks(traces, kind, name) {
  const weeks = new Map();
  for (const t of activityEligible(traces)) {
    const key = activityWeekKey(t.ts);
    if (!key) continue;
    if (!weeks.has(key)) weeks.set(key, {key, eligible: 0, using: 0});
    const week = weeks.get(key);
    week.eligible++;
    if ((t[kind] || {})[name] > 0) week.using++;
  }
  if (!weeks.size) return [];
  const keys = [...weeks.keys()].sort(), out = [];
  for (let date = new Date(keys[0] + 'T00:00:00Z'); date <= new Date(keys.at(-1) + 'T00:00:00Z'); date.setUTCDate(date.getUTCDate() + 7)) {
    const key = date.toISOString().slice(0, 10);
    out.push(weeks.get(key) || {key, eligible: 0, using: 0});
  }
  return out;
}
function activityPairs(traces, kind, name) {
  const using = activityEligible(traces).filter(t => (t[kind] || {})[name] > 0);
  const groups = Object.fromEntries(ACTIVITY_KINDS.map(k => [k, new Map()]));
  for (const t of using) for (const targetKind of ACTIVITY_KINDS) for (const [other, count] of Object.entries(t[targetKind] || {})) {
    if (!(count > 0) || targetKind === kind && other === name) continue;
    if (kind === 'commands' && targetKind === 'tools' && SHELL_TOOL_NAMES.has(other)) continue;
    if (kind === 'tools' && SHELL_TOOL_NAMES.has(name) && targetKind === 'commands') continue;
    groups[targetKind].set(other, (groups[targetKind].get(other) || 0) + 1);
  }
  return {using: using.length, groups: Object.fromEntries(ACTIVITY_KINDS.map(k => [k, [...groups[k]].map(([name, overlap]) => ({kind: k, name, overlap})).sort((a, b) => b.overlap - a.overlap || a.name.localeCompare(b.name))]))};
}
function activitySelect(kind, name, preserveScroll = false) {
  const scrollTop = preserveScroll ? document.querySelector('.activity-results')?.scrollTop : null;
  if (S.activityKind !== kind) { S.activityQuery = ''; S.activityShowAll = false; }
  S.activityKind = kind;
  S.activityItem = S.activityItem?.kind === kind && S.activityItem.name === name ? null : {kind, name};
  S.sel = null;
  commit();
  if (scrollTop != null) {
    document.querySelector('.activity-results').scrollTop = scrollTop;
    [...document.querySelectorAll('[data-activity-item]')].find(row => row.dataset.kind === kind && row.dataset.name === name)?.focus({preventScroll: true});
  }
}
function activityMetric(value, max, className, suffix, title) {
  const width = max ? (100 * value / max).toFixed(1) : '0';
  return `<span class="${className}" title="${esc(title)}"><span class="metric-label">${className === 'coverage' ? 'Traces using' : 'Total calls'}</span>${value.toLocaleString()}${suffix}<span class="bar" aria-hidden="true"><i style="width:${width}%"></i></span></span>`;
}
function activityRow(row, kind, eligible, maxCalls, child = false) {
  const on = S.activityItem?.kind === kind && S.activityItem.name === row.name;
  const risk = kind === 'commands' && isRisky(row.name) ? '<span class="activity-risk" title="Based on the command name, not a verdict about the trace">Potentially state-changing</span>' : '';
  const unit = kind === 'skills' ? 'loads' : 'calls';
  return `<button class="activity-row ${kind === 'skills' ? 'skills' : ''} ${on ? 'on' : ''} ${child ? 'activity-child' : ''}" data-activity-item data-kind="${kind}" data-name="${esc(row.name)}" aria-pressed="${!!on}">
    <span class="name" title="${esc(row.name)}">${esc(row.name)}${risk}</span>
    ${activityMetric(row.traces, eligible, 'coverage', `<small> / ${eligible} · ${pct(row.traces, eligible)}%</small>`, `${row.traces} of ${eligible} eligible traces (${pct(row.traces, eligible)}%)`)}
    ${kind === 'skills' ? '' : activityMetric(row.calls, maxCalls, 'calls', '', `${row.calls} ${unit}; bar relative to ${maxCalls} ${unit} for the most-used visible item`)}</button>`;
}
function renderActivity() {
  const base = baseVisible(), eligible = activityEligible(base), kind = S.activityKind;
  const all = activityRows(base, kind), total = all.reduce((n, row) => n + row.calls, 0);
  const used = new Set(all.flatMap(row => [...row.ids])).size;
  const query = S.activityQuery.trim().toLowerCase();
  const shown = all.filter(row => query ? row.name.toLowerCase().includes(query) : kind !== 'commands' || S.activityHelpers || !HELPERS.has(row.name.split(' ')[0]) || S.activityItem?.kind === kind && S.activityItem.name === row.name);
  shown.sort((a, b) => kind !== 'skills' && S.activitySort === 'calls' ? b.calls - a.calls || b.traces - a.traces || a.name.localeCompare(b.name) : b.traces - a.traces || b.calls - a.calls || a.name.localeCompare(b.name));
  const maxCalls = shown.reduce((max, row) => Math.max(max, row.calls), 0);
  const missing = base.length - eligible.length;
  const unit = kind === 'skills' ? 'loads' : 'calls';
  let list = '';
  if (kind === 'commands') {
    const groups = new Map();
    for (const row of shown) {
      const program = row.name.split(' ')[0];
      if (!groups.has(program)) groups.set(program, {program, rows: [], ids: new Set(), calls: 0});
      const group = groups.get(program);
      group.rows.push(row); group.calls += row.calls;
      row.ids.forEach(id => group.ids.add(id));
    }
    const programs = [...groups.values()].sort((a, b) => S.activitySort === 'calls' ? b.calls - a.calls : b.ids.size - a.ids.size || b.calls - a.calls);
    const maxProgramCalls = programs.reduce((max, group) => Math.max(max, group.calls), 0);
    list = programs.map(group => {
      const open = !!query || S.activityExpanded.has(group.program) || S.activityItem?.kind === 'commands' && S.activityItem.name.split(' ')[0] === group.program;
      return `<button class="activity-program" data-activity-program="${esc(group.program)}" aria-expanded="${open}"><span class="name"><span class="arrow">${open ? '▾' : '▸'}</span>${esc(group.program)}</span>
        ${activityMetric(group.ids.size, eligible.length, 'coverage', `<small> / ${eligible.length} · ${pct(group.ids.size, eligible.length)}%</small>`, `${group.ids.size} of ${eligible.length} eligible traces (${pct(group.ids.size, eligible.length)}%)`)}
        ${activityMetric(group.calls, maxProgramCalls, 'calls', '', `${group.calls} calls; bar relative to ${maxProgramCalls} calls for the most-used visible program`)}</button>${open ? group.rows.map(row => activityRow(row, kind, eligible.length, maxCalls, true)).join('') : ''}`;
    }).join('');
  } else list = shown.map(row => activityRow(row, kind, eligible.length, maxCalls)).join('');
  const empty = !base.length && (S.filters.length || S.q) ? 'No traces match the current run filters.' : !eligible.length ? 'Activity data was not recorded for these traces.' : !all.length && activityRows(T, kind).length && (S.filters.length || S.q) ? 'No items match the current run filters.' : !all.length ? `No ${ACTIVITY_LABELS[kind].toLowerCase()} were recorded.` : query ? 'No items match this search.' : 'No items are visible. Show common helpers to include them.';
  $('canvas').innerHTML = `<div class="activity-intro"><div><h2>Activity</h2><p>Usage across the current Insights run. Counts update with the run filters.</p></div><span class="note">${eligible.length} eligible traces${missing ? ` · ${missing} missing tool stats` : ''}</span></div>
    <div class="activity-cats">${ACTIVITY_KINDS.map(k => `<button data-activity-kind="${k}" class="${kind === k ? 'on' : ''}" aria-pressed="${kind === k}">${ACTIVITY_LABELS[k]}</button>`).join('')}</div>
    <div class="activity-summary ${kind === 'skills' ? 'skills' : ''}"><div><b>${all.length}</b><span>distinct ${ACTIVITY_LABELS[kind].toLowerCase()}</span></div><div><b>${used}</b><span>traces with any recorded use</span></div>${kind === 'skills' ? '' : `<div><b>${total.toLocaleString()}</b><span>recorded ${unit}</span></div>`}</div>
    <div class="activity-controls"><input id="activityQuery" aria-label="Search ${ACTIVITY_LABELS[kind].toLowerCase()}" placeholder="Search ${ACTIVITY_LABELS[kind].toLowerCase()}" value="${esc(S.activityQuery)}">
      ${kind === 'skills' ? '' : `<label>Sort <select id="activitySort"><option value="coverage" ${S.activitySort === 'coverage' ? 'selected' : ''}>Traces using</option><option value="calls" ${S.activitySort === 'calls' ? 'selected' : ''}>Total ${unit}</option></select></label>`}
      ${kind === 'commands' ? `<button class="linkbtn" id="activityHelpers">${S.activityHelpers ? 'Hide' : 'Show'} common helpers</button>` : ''}</div>
    <div class="activity-results" role="region" aria-label="${ACTIVITY_LABELS[kind]} ranking" tabindex="0"><div class="activity-head ${kind === 'skills' ? 'skills' : ''}"><span>${kind === 'skills' ? 'Skill' : kind === 'tools' ? 'Tool' : 'Command'}</span><span>Traces using</span>${kind === 'skills' ? '' : `<span>Total ${unit}</span>`}</div>
      <div id="activityList" class="activity-list ${kind === 'commands' ? 'grouped' : ''} ${S.activityShowAll ? 'show-all' : ''}">${list || `<div class="empty"><b>Nothing to show</b>${empty}</div>`}</div></div>
    ${kind !== 'commands' && shown.length > 10 ? `<button class="activity-more" id="activityMore" aria-controls="activityList" aria-expanded="${S.activityShowAll}">${S.activityShowAll ? 'Show fewer' : `Show all ${shown.length} ${ACTIVITY_LABELS[kind].toLowerCase()}`}</button>` : ''}
    <p class="note">${kind === 'skills' ? `Bar and percentage: share of ${eligible.length} eligible traces using each skill.` : `Bar length: traces using is relative to ${eligible.length} eligible traces; total ${unit} is relative to the largest visible ${kind === 'commands' ? 'command or program within its level' : 'item'}.`}</p>
    <p class="note">${kind === 'skills' ? 'A load is recorded when the Skill tool names this skill.' : kind === 'commands' ? 'Commands are normalized first programs or subcommands. Shell calls also appear in Tools.' : 'Shell tool calls also appear under Shell commands; category totals overlap.'}</p>`;
  $('canvas').querySelectorAll('[data-activity-kind]').forEach(b => b.onclick = () => { S.activityKind = b.dataset.activityKind; S.activityShowAll = false; if (S.activityKind === 'skills') S.activitySort = 'coverage'; S.activityItem = null; S.sel = null; commit(); });
  $('canvas').querySelectorAll('[data-activity-item]').forEach(b => b.onclick = () => activitySelect(b.dataset.kind, b.dataset.name, true));
  $('canvas').querySelectorAll('[data-activity-program]').forEach(b => b.onclick = () => {
    const name = b.dataset.activityProgram;
    S.activityExpanded.has(name) ? S.activityExpanded.delete(name) : S.activityExpanded.add(name);
    history.replaceState(null, '', toUrl()); renderActivity();
  });
  const activitySort = $('activitySort');
  if (kind !== 'skills') activitySort.onchange = e => { S.activitySort = e.target.value; S.activityShowAll = false; commit(); };
  const activityMore = $('activityMore');
  if (activityMore) activityMore.onclick = () => { S.activityShowAll = !S.activityShowAll; renderActivity(); };
  $('activityQuery').oninput = e => {
    S.activityQuery = e.target.value;
    S.activityShowAll = false;
    const pos = e.target.selectionStart;
    history.replaceState(null, '', toUrl()); renderActivity();
    $('activityQuery').focus(); $('activityQuery').setSelectionRange(pos, pos);
  };
  const helpers = $('activityHelpers');
  if (helpers) helpers.onclick = () => { S.activityHelpers = !S.activityHelpers; commit(); };
}
function activityWelcome() {
  return `<div class="inner activity-detail"><div class="kick">Activity explorer</div><h2>Select an item</h2>
    <p class="meta">Pick a skill, tool, or shell command to see its weekly coverage, what appears with it, and the matching traces.</p>
    <h5>What the counts mean</h5><p class="meta">A trace can use the same item more than once. Skill loads, tool calls, and shell command calls are counted separately, and shell calls overlap with tool calls.</p></div>`;
}
function activityDetail() {
  const {kind, name} = S.activityItem, base = baseVisible(), eligible = activityEligible(base);
  const row = activityRows(base, kind).find(r => r.name === name);
  const weeks = activityWeeks(base, kind, name), pairs = activityPairs(base, kind, name);
  const unit = kind === 'skills' ? 'loads' : 'calls';
  const weekHtml = weeks.map(w => `<div class="week"><span>${new Date(w.key + 'T00:00:00Z').toLocaleDateString('en-GB', {day: 'numeric', month: 'short', timeZone: 'UTC'})}</span><span class="track" title="${w.eligible ? `${w.using} of ${w.eligible} eligible traces` : 'No eligible traces this week'}"><i style="width:${pct(w.using, w.eligible)}%"></i></span><span class="num">${w.using}/${w.eligible}</span></div>`).join('');
  const pairHtml = ACTIVITY_KINDS.map(k => {
    const items = pairs.groups[k].slice(0, 5);
    return `<div class="pair-group">${k === kind ? 'Same category · ' : ''}${ACTIVITY_LABELS[k]}</div>${items.length ? items.map(p => `<button class="pair" data-activity-pair data-kind="${p.kind}" data-name="${esc(p.name)}"><span>${esc(p.name)}</span><small>${p.overlap} / ${pairs.using} traces · ${pct(p.overlap, pairs.using)}%</small></button>`).join('') : '<p class="meta">No co-occurrences recorded.</p>'}`;
  }).join('');
  return `<div class="inner activity-detail"><button class="back" data-close-activity>✕ Clear selection</button>
    <div class="kick">${ACTIVITY_LABELS[kind]}</div><h2>${esc(name)}</h2>
    <p class="meta">${row ? `${row.traces} of ${eligible.length} eligible traces (${pct(row.traces, eligible.length)}%)${kind === 'skills' ? '' : ` · ${row.calls.toLocaleString()} ${unit}`}` : 'No traces match this item under the current filters.'} The trace list shows the matches.</p>
    <h5>Weekly trace coverage</h5>${weeks.filter(w => w.eligible).length > 1 ? weekHtml : weeks.length ? `<p class="meta">${weeks[0].using} of ${weeks[0].eligible} eligible traces in this week.</p>` : '<p class="meta">No dated eligible traces.</p>'}
    <p class="meta">Each bar shows the share of eligible traces in that UTC week. The count at right shows the sample size.</p>
    <h5>Often together</h5><p class="meta">Items seen in the same trace; order and individual-call success are not recorded.</p>${pairHtml}</div>`;
}

function colorOf(t) {
  if (S.colorBy === 'cluster') return clusterOf(t, S.dim).color;
  if (S.colorBy === 'err') return !t.has_summary ? OTHER : hasErr(t) ? ERR_YES : ERR_NO;
  const name = S.colorBy.slice(2), v = lval(t, name);
  return v == null ? '#e5e3de' : lcolor(name, v);
}
function mapLegend() {
  if (S.colorBy === 'cluster') return withOther(bases(S.dim, true)).map(c => `<button data-c="${esc(c.id)}"><i style="background:${c.color}"></i><span>${esc(c.name)}</span></button>`).join('');
  if (S.colorBy === 'err') return `<button data-e="1"><i style="background:${ERR_YES}"></i><span>Assistant made errors</span></button><button data-e="0"><i style="background:${ERR_NO}"></i><span>No errors</span></button>${T.some(t => !t.has_summary) ? `<span class="legend-missing"><i></i>Summary unavailable</span>` : ''}`;
  const name = S.colorBy.slice(2);
  return lvals(LBY[name]).map(v => `<button data-l="${esc(name)}" data-v="${esc(v)}"><i style="background:${lcolor(name, v)}"></i><span>${esc(ldisp(name, v))}</span></button>`).join('') + (T.some(t => lval(t, name) == null) ? `<span class="legend-missing"><i></i>Not measured</span>` : '');
}
function bindLegend(root) {
  root.querySelectorAll('[data-c]').forEach(b => b.onclick = () => pickCluster(b.dataset.c));
  root.querySelectorAll('[data-e]').forEach(b => b.onclick = () => toggle({t: 'e', yes: b.dataset.e === '1'}));
  root.querySelectorAll('[data-l]').forEach(b => b.onclick = () => toggle({t: 'l', name: b.dataset.l, v: b.dataset.v}));
}
function renderMap() {
  if (!S.dim) {
    $('canvas').innerHTML = `<div class="empty map-empty"><b>No theme map for this run</b>This run has no grouping dimensions to place on a theme map. Labels and assistant errors can still colour traces when map coordinates are available.</div>`;
    return;
  }
  const pts = T.filter(t => {
    const xy = t.xy && t.xy[S.dim];
    return Array.isArray(xy) && xy.length >= 2 && Number.isFinite(xy[0]) && Number.isFinite(xy[1]);
  });
  const mapState = D.map_states && D.map_states[S.dim];
  if (pts.length < 5) {
    const reason = mapState?.reason || `Only ${pts.length} valid points are available; at least 5 are needed.`;
    $('canvas').innerHTML = `<div class="empty map-empty"><b>No map for ${esc(cap(S.dim))}</b>${esc(reason)}</div>`;
    return;
  }
  const vis = new Set(visible().map(t => t.id));
  const xs = pts.map(t => t.xy[S.dim][0]), ys = pts.map(t => t.xy[S.dim][1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const W = canvasW(), H = $('stage').classList.contains('full') ? Math.max(360, innerHeight - 190) : 440, P = 24;
  const sx = v => P + (W - 2 * P) * (v - x0) / ((x1 - x0) || 1), sy = v => P + (H - 2 * P) * (1 - (v - y0) / ((y1 - y0) || 1));
  const sel = S.sel && S.sel.kind === 'trace' ? S.sel.id : null;
  const dots = [...pts].sort((a, b) => vis.has(a.id) - vis.has(b.id)).map(t => {
    const c = clusterOf(t, S.dim);
    return `<circle class="pt" role="button" tabindex="0" aria-label="Open trace ${esc(t.request || t.id)}" data-t="${esc(t.id)}" cx="${sx(t.xy[S.dim][0]).toFixed(1)}" cy="${sy(t.xy[S.dim][1]).toFixed(1)}" r="${t.id === sel ? 8 : 5.5}" fill="${colorOf(t)}" stroke="${t.id === sel ? '#25232e' : '#fff'}" stroke-width="${t.id === sel ? 2.5 : 1.2}" style="opacity:${vis.has(t.id) ? .92 : .1}"
      data-tt="${esc((t.request || '').slice(0, 140))}" data-tt2="${esc(c.name)}${frus(t) != null ? ' · frustration ' + frus(t) : ''}${hasErr(t) ? ' · ' + t.errors.length + ' error' + (t.errors.length > 1 ? 's' : '') : ''}"/>`;
  }).join('');
  $('canvas').innerHTML = `<p class="caption">Each dot is a trace, placed by what its <b>${esc(S.dim)}</b> text says; similar traces sit close together. <b>Drag</b> to select an area, <b>click</b> a dot to read it.</p>
    <svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" id="msvg" style="background:var(--surface-app);border-radius:10px;cursor:crosshair">${dots}<rect class="brush" id="brush" width="0" height="0" style="display:none"/></svg>
    <div class="legend">${mapLegend()}</div>`;
  const svg = $('msvg');
  svg.querySelectorAll('.pt').forEach(p => {
    p.onclick = e => { e.stopPropagation(); select('trace', p.dataset.t); };
    p.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select('trace', p.dataset.t); } };
  });
  bindTips($('canvas')); bindLegend($('canvas').querySelector('.legend'));
  const toSvg = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
  let start = null; const br = $('brush');
  svg.onpointerdown = e => { if (e.target.classList.contains('pt')) return; start = toSvg(e); svg.setPointerCapture(e.pointerId); };
  svg.onpointermove = e => {
    if (!start) return; const p = toSvg(e);
    br.style.display = 'block';
    br.setAttribute('x', Math.min(p.x, start.x)); br.setAttribute('y', Math.min(p.y, start.y));
    br.setAttribute('width', Math.abs(p.x - start.x)); br.setAttribute('height', Math.abs(p.y - start.y));
  };
  svg.onpointerup = e => {
    if (!start) return; const p = toSvg(e), s = start; start = null; br.style.display = 'none';
    if (Math.abs(p.x - s.x) < 6 && Math.abs(p.y - s.y) < 6) return;
    const [ax, bx, ay, by] = [Math.min(p.x, s.x), Math.max(p.x, s.x), Math.min(p.y, s.y), Math.max(p.y, s.y)];
    const ids = new Set(pts.filter(t => { const X = sx(t.xy[S.dim][0]), Y = sy(t.xy[S.dim][1]); return X >= ax && X <= bx && Y >= ay && Y <= by; }).map(t => t.id));
    if (ids.size) { S.filters = S.filters.filter(f => f.t !== 'b'); S.filters.push({t: 'b', ids}); commit(); toast(`Selected ${ids.size} traces. Every view now shows only these.`); }
  };
}

function cmpCols() {
  if (S.cmp === 'err') return [true, false].map(yes => ({label: yes ? 'Errors' : 'No errors', color: yes ? ERR_YES : '#299D8F', test: t => t.has_summary && hasErr(t) === yes, f: {t: 'e', yes}}));
  if (S.cmp.startsWith('l:')) {
    const name = S.cmp.slice(2), vals = lvals(LBY[name]);
    return categoryColumns(vals.map(v => ({label: ldisp(name, v), color: lcolor(name, v), test: t => lval(t, name) === v, f: {t: 'l', name, v}})));
  }
  return categoryColumns(withOther(bases(S.cmp.slice(2), true)).map(c => ({label: c.name, color: c.color, other: c.other, test: t => c.members.has(t.id), f: c.other ? null : {t: 'c', id: c.id}})));
}
function categoryColumns(cols) {
  if (cols.length <= TOP_K + 1) return cols;
  const head = cols.slice(0, TOP_K), rest = cols.slice(TOP_K);
  return [...head, {label: `Other (${rest.length} categories)`, color: OTHER, other: true, test: t => rest.some(c => c.test(t)), f: null}];
}
function compareEligible(t) {
  if (S.cmp === 'err') return t.has_summary;
  if (S.cmp.startsWith('l:')) return lval(t, S.cmp.slice(2)) != null;
  return Boolean(clusterOf(t, S.cmp.slice(2)));
}
function hexA(col, a) {
  if (col.startsWith('hsl')) return col.replace(')', ` / ${a})`);
  const n = parseInt(col.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}
function renderCompare() {
  const hasDimensions = Object.keys(D.dims).length > 0;
  const vis = visible(), cols = cmpCols(), rows = hasDimensions ? withOther(bases(S.dim, true)) : [{id: 'all-traces', name: 'All traces', color: OTHER, members: new Set(T.map(t => t.id))}];
  const what = S.cmp === 'err' ? 'assistant errors' : S.cmp.startsWith('l:') ? human(S.cmp.slice(2)).toLowerCase() : S.cmp.slice(2) + ' themes';
  let html = `<p class="caption">${hasDimensions ? `How <b>${esc(S.dim)} themes</b> break down` : 'How all traces break down'} by <b>${esc(what)}</b>. Each percentage uses traces with both values in that row. Unmeasured values are excluded. Click a cell to filter${hasDimensions ? ' to both' : ''}.</p>
    ${vis.length ? '' : '<div class="empty compare-empty"><b>No traces match</b>Remove a filter or clear the search to compare traces.</div>'}
    <div style="overflow-x:auto"><table class="heat"><thead><tr><th></th>${cols.map(c => `<th title="${esc(c.label)}"><span style="display:inline-block;max-width:92px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:bottom">${esc(c.label)}</span></th>`).join('')}</tr></thead><tbody>`;
  rows.forEach(r => {
    const inRow = vis.filter(t => r.members.has(t.id)), eligible = inRow.filter(compareEligible);
    html += `<tr><th class="rh"><div><i style="background:${r.color}"></i><span>${esc(r.name)}</span></div><small>${eligible.length} eligible</small></th>`;
    cols.forEach((c, j) => {
      const n = eligible.filter(c.test).length, share = eligible.length ? n / eligible.length : 0;
      html += n ? `<td data-r="${esc(r.id)}" data-j="${j}" tabindex="0" role="button" style="background:${hexA(c.color, (.12 + share * .4).toFixed(2))}" aria-label="${n} of ${eligible.length} eligible traces, ${Math.round(share * 100)} percent, ${esc(r.name)} by ${esc(c.label)}" title="${n} of ${eligible.length} traces with both values">${n}<small>${Math.round(share * 100)}% · ${eligible.length}</small></td>` : `<td class="z" title="No traces with both values">·</td>`;
    });
    html += '</tr>';
  });
  if (!rows.length) html += `<tr><td class="compare-no-rows" colspan="${cols.length + 1}">No ${esc(S.dim)} themes or ungrouped traces are available.</td></tr>`;
  $('canvas').innerHTML = html + '</tbody></table></div>';
  const choose = td => {
    const row = rows.find(candidate => candidate.id === td.dataset.r), col = cols[+td.dataset.j];
    if (!row || !col) return;
    const rowFilter = row.other ? null : {t: 'c', id: row.id};
    const colFilter = col.f;
    const rowIds = row.other ? new Set([...row.members].filter(id => byId[id])) : null;
    const colIds = col.other ? new Set(T.filter(col.test).map(t => t.id)) : null;
    const rowAxis = hasDimensions ? `c:${S.dim}` : null;
    const colAxis = S.cmp.startsWith('d:') ? `c:${S.cmp.slice(2)}` : S.cmp.startsWith('l:') ? `l:${S.cmp.slice(2)}` : colFilter ? fkey(colFilter) : null;
    S.filters = S.filters.filter(f => fkey(f) !== rowAxis && (!colAxis || fkey(f) !== colAxis));
    if (rowFilter) S.filters.push(rowFilter);
    if (colFilter) S.filters.push(colFilter);
    if (rowIds || colIds) {
      const ids = rowIds && colIds ? new Set([...rowIds].filter(id => colIds.has(id))) : rowIds || colIds;
      S.filters = S.filters.filter(f => f.t !== 'b');
      S.filters.push({t: 'b', ids});
    }
    commit();
  };
  $('canvas').querySelectorAll('td[data-r]').forEach(td => {
    td.onclick = () => choose(td);
    td.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(td); } };
  });
}

/* ---------- trace list ---------- */
function renderTraces() {
  const vis = visible(), sel = S.sel && S.sel.kind === 'trace' ? S.sel.id : null;
  const rows = vis.slice(0, S.more).map(t => {
    const c = S.dim ? clusterOf(t, S.dim) : null;
    return `<button class="tr ${t.id === sel ? 'on' : ''}" data-t="${esc(t.id)}">
      <span class="when">${fmtDate(t.ts)}</span><span class="req" title="${esc(t.request)}">${esc(t.request || '(no summary)')}</span>
      <span class="cl">${c ? `<i style="background:${c.color}"></i><span>${esc(c.name)}</span>` : '<span>Not grouped</span>'}</span>
      <span>${chip('task_type', t)}</span><span class="oc">${outcomeHtml(t)}</span><span>${frusHtml(frus(t))}</span>${FLAG_DEFS.map(flag => flagCell(t, flag)).join('')}
      ${hasErr(t) ? `<span class="errb" title="Assistant errors">${t.errors.length}</span>` : '<span></span>'}</button>`;
  }).join('');
  $('tlist').innerHTML = `<div class="th"><h3>Traces</h3><span class="c num">${vis.length}</span><input id="q" placeholder="Search requests and summaries" value="${esc(S.q)}"></div>
    ${vis.length ? `<div class="tr trh"><span>Date</span><span>Request</span><span>Theme</span><span>Task</span><span class="oc">Outcome</span><span title="User frustration, 1 to 5">Frust.</span><span title="Unfixed error: a failure the agent never corrected">Unfixed</span><span title="Risky action the user did not ask for">Risky</span><span title="No check run, or claimed working without one">No check</span><span title="Corrected three or more times by the user">3+ corrections</span><span title="Number of assistant mistakes in the summary">Errors</span></div>` + rows : `<div class="empty"><b>No traces match</b>Remove a filter above or clear the search.</div>`}
    ${vis.length > S.more ? `<div class="tmore"><button class="btn" id="more">Show ${Math.min(20, vis.length - S.more)} more</button></div>` : ''}`;
  $('tlist').querySelectorAll('[data-t]').forEach(r => r.onclick = () => select('trace', r.dataset.t));
  $('q').oninput = e => {
    S.q = e.target.value; const pos = e.target.selectionStart;
    history.replaceState(null, '', toUrl());
    renderFilterbar(); renderStage(); renderTraces(); renderDrawer();
    const n = $('q'); n.focus(); n.setSelectionRange(pos, pos);
  };
  const m = $('more'); if (m) m.onclick = () => { S.more += 20; renderTraces(); };
}

/* ---------- right panel ---------- */
function select(kind, id) {
  S.sel = kind ? {kind, id, prev: S.sel && S.sel.kind === 'cluster' ? S.sel : null} : null;
  commit();
}
function renderDrawer() {
  const el = $('drawer');
  const selectedTrace = S.sel && S.sel.kind === 'trace' ? byId[S.sel.id] : null;
  const openSignalDetails = selectedTrace && el.dataset.signalTraceId === selectedTrace.id
    ? window.EvaluatorqSignals.captureOpen(el)
    : [];
  if (selectedTrace && selectedTrace.has_signal_details) loadSignalDetails(selectedTrace);
  else if (activeSignalDetailRequest) cancelSignalDetails();
  el.innerHTML = S.sel ? (S.sel.kind === 'cluster' ? clusterPanel(C[S.sel.id]) : tracePanel(byId[S.sel.id])) : S.view === 'activity' ? (S.activityItem ? activityDetail() : activityWelcome()) : glance();
  el.dataset.signalTraceId = selectedTrace?.id || '';
  window.EvaluatorqSignals.restoreOpen(el, openSignalDetails);
  window.EvaluatorqSignals.bind(el);
  el.querySelectorAll('[data-close-activity]').forEach(b => b.onclick = () => { S.activityItem = null; commit(); });
  el.querySelectorAll('[data-activity-pair]').forEach(b => b.onclick = () => activitySelect(b.dataset.kind, b.dataset.name));
  el.querySelectorAll('[data-close]').forEach(b => b.onclick = () => select(null));
  el.querySelectorAll('[data-backto]').forEach(b => b.onclick = () => select('cluster', b.dataset.backto));
  el.querySelectorAll('[data-t]').forEach(b => b.onclick = () => select('trace', b.dataset.t));
  el.querySelectorAll('[data-go]').forEach(b => b.onclick = () => { const c = C[b.dataset.go]; S.dim = c.dim; select('cluster', c.id); });
  el.querySelectorAll('[data-l]').forEach(b => b.onclick = () => toggle({t: 'l', name: b.dataset.l, v: b.dataset.v}));
  el.querySelectorAll('[data-e]').forEach(b => b.onclick = () => toggle({t: 'e', yes: b.dataset.e === '1'}));
  el.querySelectorAll('[data-x]').forEach(b => b.onclick = () => toggle({t: 'x', k: b.dataset.x}));
  el.querySelectorAll('[data-act="cluster-traces"]').forEach(b => b.onclick = () => viewClusterTraces(C[S.sel.id]));
  el.querySelectorAll('[data-act="cluster-question"]').forEach(b => b.onclick = () => turnClusterIntoQuestion(C[S.sel.id]));
  el.querySelectorAll('[data-helpers]').forEach(b => b.onclick = () => { S.helpers = !S.helpers; renderDrawer(); });
  el.querySelectorAll('[data-signal-retry]').forEach(b => b.onclick = () => {
    const trace = byId[b.dataset.signalRetry];
    if (trace) { loadSignalDetails(trace, true); renderDrawer(); }
  });
  bindMock(el);
}
function labelBlock(l, ts) {
  const counts = {}; let answered = 0;
  lvals(l).forEach(v => counts[v] = 0);
  ts.forEach(t => { const v = lval(t, l.name); if (v == null) return; counts[v]++; answered++; });
  const failed = ts.length - answered;
  const vs = lvals(l).filter(v => counts[v]);
  return `<div class="lab"><div class="lh"><b>${esc(human(l.name))}</b><span>${answered ? '' : 'not answered'}${failed ? `${answered ? '' : ' · '}${failed} failed` : ''}</span></div>
    ${answered ? `<div class="sbar">${vs.map(v => `<button data-l="${esc(l.name)}" data-v="${esc(v)}" class="${isOn({t: 'l', name: l.name, v}) ? 'on' : ''}" style="flex:${counts[v]};background:${lcolor(l.name, v)}" title="${esc(ldisp(l.name, v))}: ${counts[v]}">${counts[v]}</button>`).join('')}</div>
    <div class="slegend">${vs.map(v => `<span><i style="background:${lcolor(l.name, v)}"></i>${esc(ldisp(l.name, v))}</span>`).join('')}</div>` : `<div class="note">No answers for these traces.</div>`}</div>`;
}
function tagCounts(counts, empty, riskCheck, hideCommon) {
  let top = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
  let toggle = '';
  if (hideCommon) {
    const common = top.filter(([k]) => HELPERS.has(k.split(' ')[0]) && !isRisky(k));
    if (common.length) {
      toggle = `<button class="linkbtn" data-helpers style="font-size:12px;margin-top:6px">${S.helpers ? 'Hide common helpers' : `Show common helpers (${common.length})`}</button>`;
      if (!S.helpers) top = top.filter(e => !common.includes(e));
    }
  }
  const shown = top.slice(0, 12);
  if (riskCheck) top.slice(12).filter(([k]) => isRisky(k)).forEach(e => shown.push(e));   // risky commands are never folded away
  return `<div class="tags">${shown.map(([k, n]) => riskCheck && isRisky(k) ? `<span class="risk" title="Can change shared state or destroy data">⚠ ${esc(k)} <b class="num">${n}</b></span>` : `<span>${esc(k)} <b class="num">${n}</b></span>`).join('') || `<span>${empty}</span>`}</div>${toggle}`;
}
const Q_GROUPS = [['Conversation', ['user_frustration']], ['Coding work', ['task_type', 'outcome', 'verified', 'user_corrections']], ['Flags', ['unfixed_error', 'risky_action', 'scope_creep']]];
const Q_SKIP = new Set(['coding_agent']);   // the same answer for every trace in a coding run
function flagRow(name, ts) {
  if (name === 'risky_action') return riskBlock(ts);
  const answered = ts.filter(t => lval(t, name) != null), yes = answered.filter(t => isYes(t, name)).length, on = isOn({t: 'l', name, v: 'yes'});
  return `<button class="frow ${on ? 'on' : ''}" data-l="${esc(name)}" data-v="yes" title="${esc(LBY[name].instructions || '')}"><span class="fn"><i class="sdotx" style="width:8px;height:8px;border-radius:2px;background:${lcolor(name, 'yes')}"></i>${esc(human(name))}</span><span><b>${yes}</b> <small>of ${answered.length}</small></span></button>`;
}
// risky action is a choice: one stacked bar of the kinds, "none" left out of the bar and counted in the header
function riskBlock(ts) {
  const l = LBY.risky_action, counts = {}; let answered = 0;
  lvals(l).forEach(v => counts[v] = 0);
  ts.forEach(t => { const v = lval(t, 'risky_action'); if (v == null) return; counts[v]++; answered++; });
  const vs = lvals(l).filter(v => v !== 'none' && counts[v]), risky = vs.reduce((a, v) => a + counts[v], 0);
  const on = v => isOn({t: 'l', name: 'risky_action', v}) ? 'on' : '';
  return `<div class="lab" title="${esc(l.instructions || '')}"><div class="lh"><b>${esc(human(l.name))}</b><span><b class="num">${risky}</b> of ${answered} risky</span></div>
    ${vs.length ? `<div class="sbar">${vs.map(v => `<button data-l="risky_action" data-v="${esc(v)}" class="${on(v)}" style="flex:${counts[v]};background:${lcolor('risky_action', v)}" title="${esc(ldisp('risky_action', v))}: ${counts[v]}">${counts[v]}</button>`).join('')}</div>
    <div class="rlist">${vs.map(v => `<button data-l="risky_action" data-v="${esc(v)}" class="${on(v)}"><span><i style="background:${lcolor('risky_action', v)}"></i>${esc(ldisp('risky_action', v))}</span><b>${counts[v]}</b></button>`).join('')}</div>` : '<div class="note">No risky actions in these traces.</div>'}</div>`;
}
function questionGroups(ts) {
  const named = new Set(Q_GROUPS.flatMap(g => g[1]));
  const groups = [...Q_GROUPS, ['Other', LABELS.map(l => l.name).filter(n => !named.has(n) && !Q_SKIP.has(n))]];
  return groups.map(([title, names]) => {
    const ls = names.map(n => LBY[n]).filter(Boolean);
    if (!ls.length) return '';
    return `<div class="gh">${title}</div>` + ls.map(l => title === 'Flags' ? flagRow(l.name, ts) : labelBlock(l, ts)).join('');
  }).join('');
}
function glance() {
  const vis = visible(), s = stats(new Set(vis.map(t => t.id)));
  const tools = {}, skills = {}, cmds = {};
  vis.forEach(t => { for (const k in t.tools) tools[k] = (tools[k] || 0) + 1; for (const k in t.skills) skills[k] = (skills[k] || 0) + 1; for (const k in t.commands) cmds[k] = (cmds[k] || 0) + 1; });
  return `<div class="inner"><div class="kick">At a glance${S.filters.length ? ' · filtered' : ''}</div>
    <h2><span class="num">${vis.length}</span> traces</h2>
    <h5>Questions asked <span>click a segment to filter</span></h5>${LABELS.length ? questionGroups(vis) : '<div class="note">This run asked no questions.</div>'}
    <h5>Assistant errors <span>from summary</span></h5>
    <div class="stats"><button class="stat" data-e="1"><div class="l">Traces with errors</div><div class="v bad">${s.errN}</div></button><div class="stat"><div class="l">Of summarised</div><div class="v">${fmtShare(s.errShare)}</div></div></div>
    <h5>Tools <span>traces</span></h5>${tagCounts(tools, 'no tool calls')}
    <h5>Shell commands <span>traces</span></h5>${tagCounts(cmds, 'no shell commands', true, true)}<div class="riskkey">⚠ marks commands that can delete data, discard work or change remote state.</div>
    <h5>Skills <span>traces</span></h5>${tagCounts(skills, 'no skills loaded')}
    <div class="tip-card"><b>Tip:</b> drag across the Map to select a region, or click a Compare cell to cross two filters. <kbd>Esc</kbd> closes a panel. The address bar always holds this view.</div></div>`;
}
function errorBreakdown(ts) {
  const summarised = ts.filter(t => t.has_summary);
  const issues = summarised.filter(hasErr).length, clear = summarised.length - issues;
  const missing = ts.length - summarised.length;
  const missingText = missing ? ` · ${missing} without a summary` : '';
  const groups = [{yes: true, count: issues, label: 'With issues', color: ERR_YES}, {yes: false, count: clear, label: 'Without issues', color: ERR_NO}];
  const bar = groups.filter(g => g.count).map(g => `<button data-e="${g.yes ? 1 : 0}" class="${isOn({t: 'e', yes: g.yes}) ? 'on' : ''}" style="flex:${g.count};background:${g.color};${g.yes ? '' : 'color:#025558'}" title="${g.label}: ${g.count} of ${summarised.length} summarised traces">${g.count}</button>`).join('');
  const legend = groups.map(g => g.count ? `<button data-e="${g.yes ? 1 : 0}" class="${isOn({t: 'e', yes: g.yes}) ? 'on' : ''}"><i style="background:${g.color}"></i>${g.label} · ${g.count}</button>` : `<span><i style="background:${g.color}"></i>${g.label} · 0</span>`).join('');
  return `<h5>Assistant errors <span>${summarised.length} of ${ts.length} summarised${missingText}</span></h5>${summarised.length ? `<div class="sbar">${bar}</div><div class="slegend">${legend}</div>` : '<div class="note">No summaries were available for these traces.</div>'}`;
}
function clusterPanel(c) {
  const s = stats(c.members, c.id), ts = [...c.members].map(i => byId[i]);
  const ks = c.level === 'top' ? kids(c) : [];
  const parent = C[c.parent_id];
  const ex = (c.example_trace_ids || [...c.members]).filter(id => byId[id]).slice(0, 4);
  return `<div class="inner">${parent ? `<button class="back" data-go="${esc(parent.id)}">↑ ${esc(parent.name)}</button>` : `<button class="back" data-close>✕ Close</button>`}
    <div class="kick"><i style="background:${c.color}"></i>${cap(c.dim)} ${c.level === 'top' ? `group · ${ks.length} themes` : 'theme'}</div>
    <h2>${esc(c.name)}</h2><p>${esc(c.description || '')}</p>
    <div class="stats">
      <div class="stat"><div class="l">Traces</div><div class="v">${s.n} <span style="font-size:12px;color:var(--text-faint)">${pct(s.n, N)}%</span></div></div>
      <div class="stat"><div class="l">Assistant errors</div><div class="v ${s.errShare > .6 ? 'bad' : ''}">${fmtShare(s.errShare)}</div></div>
      <div class="stat"><div class="l">Done</div><div class="v">${fmtShare(s.doneShare)}</div></div>
      <div class="stat"><div class="l">Frustration</div><div class="v">${s.frusMean == null ? NA : s.frusMean.toFixed(1) + '<span style="font-size:12px;color:var(--text-faint)"> / 5</span>'}</div></div>
    </div>
    ${ks.length ? `<h5>Themes in this group</h5>${ks.map(k => { const ksd = stats(k.members, k.id); return `<button class="lrow" style="width:100%" data-go="${esc(k.id)}"><span class="sdot" style="color:var(--text-strong)"><i style="background:${k.color}"></i>${esc(k.name)}</span><span class="num">${ksd.n} · ${fmtShare(ksd.errShare)} err</span></button>`; }).join('')}` : ''}
    ${errorBreakdown(ts)}
    <h5>Questions asked</h5>${questionGroups(ts)}
    <h5>Example requests</h5>${ex.map(id => `<button class="ex" data-t="${esc(id)}">“${esc((byId[id].request || '').slice(0, 170))}${(byId[id].request || '').length > 170 ? '…' : ''}”</button>`).join('')}
    <div class="acts"><button class="btn sm" data-act="cluster-traces">View matching traces</button><button class="btn sm" data-act="cluster-question">Turn into a question</button></div></div>`;
}
function viewClusterTraces(c) {
  S.dim = c.dim;
  S.view = 'themes';
  S.filters = [{t: 'b', ids: new Set(c.members)}];
  S.q = '';
  S.more = 20;
  S.sel = null;
  S.activityItem = null;
  commit();
}
function turnClusterIntoQuestion(c) {
  const slug = c.name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  openSheet(NEW_RUN_URL, 'New Insights run', form => InsightsRunForm.draftQuestion(form, {
    name: `about_${slug || 'this_theme'}`,
    kind: 'noul',
    text: `Did the agent handle ${c.name} well?`,
  }));
}
function safeTraceUrl(value) {
  if (typeof value !== 'string' || !value.trim()) return '';
  try {
    const url = new URL(value, location.origin);
    return url.origin === location.origin && ['http:', 'https:'].includes(url.protocol) ? url.href : '';
  } catch (_) { return ''; }
}
function safeOrqUrl(value) {
  if (typeof value !== 'string' || !value.trim()) return '';
  try {
    const url = new URL(value);
    return url.protocol === 'https:' ? url.href : '';
  } catch (_) { return ''; }
}
function signalValue(value) {
  return window.EvaluatorqSignals.value(value);
}
const signalDetailCache = new Map();
const signalDetailFailures = new Map();
let activeSignalDetailRequest = null;
function cacheSignalDetails(traceId, detail) {
  signalDetailCache.delete(traceId);
  signalDetailCache.set(traceId, detail);
  while (signalDetailCache.size > 20) signalDetailCache.delete(signalDetailCache.keys().next().value);
  signalDetailFailures.delete(traceId);
}
function signalDetailsFor(traceId) {
  return signalDetailCache.get(traceId) || null;
}
function signalDetailsLoading(traceId) {
  return Boolean(activeSignalDetailRequest && activeSignalDetailRequest.traceId === traceId);
}
function cancelSignalDetails() {
  if (!activeSignalDetailRequest) return;
  activeSignalDetailRequest.controller.abort();
  activeSignalDetailRequest = null;
}
function loadSignalDetails(trace, retry = false) {
  if (!trace || !trace.has_signal_details || !trace.signal_detail_url) return;
  if (signalDetailCache.has(trace.id) || signalDetailsLoading(trace.id)) return;
  if (signalDetailFailures.has(trace.id) && !retry) return;
  if (activeSignalDetailRequest) cancelSignalDetails();
  if (retry) signalDetailFailures.delete(trace.id);
  const request = {traceId: trace.id, controller: new AbortController()};
  activeSignalDetailRequest = request;
  fetch(trace.signal_detail_url, {
    headers: {'Accept': 'application/json'},
    signal: request.controller.signal,
  }).then(async response => {
    if (!response.ok) throw new Error(response.status === 404 ? 'Saved signal details were not found.' : `Request failed (${response.status}).`);
    const detail = await response.json();
    if (detail.trace_id !== trace.trace_id || detail.span_id !== trace.span_id) throw new Error('Signal details did not match this trace.');
    if (!request.controller.signal.aborted) cacheSignalDetails(trace.id, detail);
  }).catch(error => {
    if (!request.controller.signal.aborted) {
      signalDetailFailures.delete(trace.id);
      signalDetailFailures.set(trace.id, error.message || 'Could not load signal details.');
      while (signalDetailFailures.size > 20) signalDetailFailures.delete(signalDetailFailures.keys().next().value);
    }
  }).finally(() => {
    if (activeSignalDetailRequest === request) activeSignalDetailRequest = null;
    if (S.sel && S.sel.kind === 'trace' && S.sel.id === trace.id) renderDrawer();
  });
}
function signalsPanel(t) {
  return window.EvaluatorqSignals.render({
    report: t.has_signals ? t.signals : null,
    detail: signalDetailsFor(t.id),
    loading: signalDetailsLoading(t.id),
    error: signalDetailFailures.get(t.id),
    retryId: t.id,
  });
}
function tracePanel(t) {
  const traceUrl = safeTraceUrl(t.trace_url);
  const orqUrl = safeOrqUrl(t.orq_url);
  const traceLink = orqUrl
    ? `<a class="linkbtn" href="${esc(orqUrl)}" target="_blank" rel="noopener noreferrer" style="float:right;font-size:12px">Open in Orq ↗</a>`
    : traceUrl ? `<a class="btn sm trace-open" href="${esc(traceUrl)}">Open full trace</a>` : '';
  return `<div class="inner">${S.sel.prev ? `<button class="back" data-backto="${esc(S.sel.prev.id)}">← ${esc(C[S.sel.prev.id].name.slice(0, 34))}</button>` : `<button class="back" data-close>✕ Close</button>`}${traceLink}
    <div class="tid">${esc(t.id.slice(0, 12))}… · ${fmtTime(t.ts)} · ${esc(t.agent)}</div>
    <h2>${esc(t.topic || 'Trace')}</h2><p>${esc(t.summary || 'No summary for this trace.')}</p>
    ${t.request ? `<h5>What the user asked</h5><div class="quote">${esc(t.request)}</div>` : ''}
    ${hasErr(t) ? `<h5>Assistant mistakes · ${t.errors.length}</h5><ul class="errs">${t.errors.map(e => `<li>${esc(e)}</li>`).join('')}</ul>` : ''}
    ${signalsPanel(t)}
    <h5>Coding work</h5>
    <div class="stats" style="margin-top:0"><div class="stat"><div class="l">Task</div><div class="v" style="font-size:13px">${chip('task_type', t)}</div></div><div class="stat"><div class="l">Outcome</div><div class="v" style="font-size:13px">${outcomeHtml(t)}</div></div>
      <div class="stat"><div class="l">Verified</div><div class="v" style="font-size:13px">${chip('verified', t)}</div></div><div class="stat"><div class="l">Frustration</div><div class="v">${frusHtml(frus(t))}</div></div>
      <div class="stat"><div class="l">User corrections</div><div class="v">${lval(t, 'user_corrections') == null ? NA : lnum('user_corrections', lval(t, 'user_corrections')) + (corr3(t) ? '+' : '')}</div></div></div>
    ${flagsText(t) ? `<h5>Flags</h5><div>${flagsText(t)}</div>` : ''}
    <h5>All answers</h5>${LABELS.map(l => { const v = lval(t, l.name);
      return `<div class="lrow"><span>${esc(human(l.name))}</span><span>${v == null ? '<span class="na">failed</span>' : `<span class="sdot" style="color:var(--text-strong)"><i style="background:${lcolor(l.name, v)}"></i>${l.name === 'user_frustration' ? lnum(l.name, v) : esc(ldisp(l.name, v))}</span>`}</span></div>`; }).join('')}
    <h5>Themes</h5><div class="tags">${DIMS.map(d => { const c = clusterOf(t, d); return `<button data-go="${esc(c.id)}"><i style="background:${c.color}"></i>${cap(d)}: ${esc(c.name.length > 40 ? c.name.slice(0, 39) + '…' : c.name)}</button>`; }).join('')}</div>
    <h5>Tools <span>calls</span></h5>${tagCounts(t.tools, 'no tool calls')}
    <h5>Shell commands <span>calls</span></h5>${tagCounts(t.commands, 'no shell commands', true)}
    <h5>Skills <span>loads</span></h5>${tagCounts(t.skills, 'no skills loaded')}</div>`;
}

/* ---------- shared bits ---------- */
function bindTips(root) {
  const tip = $('tip');
  root.querySelectorAll('[data-tt]').forEach(el => {
    el.addEventListener('mousemove', e => {
      const primary = document.createElement('div'); primary.textContent = el.dataset.tt;
      const secondary = document.createElement('div'); secondary.className = 't2'; secondary.textContent = el.dataset.tt2 || '';
      tip.replaceChildren(primary, secondary); tip.style.left = Math.min(e.clientX + 14, innerWidth - 320) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; tip.classList.add('on');
    });
    el.addEventListener('mouseleave', () => tip.classList.remove('on'));
  });
}
let toastT;
function toast(msg) { const t = $('toast'); t.textContent = msg; t.classList.add('on'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('on'), 2600); }
function bindMock(root) { root.querySelectorAll('[data-act="mock"]').forEach(b => b.onclick = e => { e.preventDefault(); toast('Not wired up in this mock.'); }); }
function render() { $('tip').classList.remove('on'); renderHeadlines(); renderFilterbar(); renderStage(); renderTraces(); renderDrawer(); }

/* ---------- run form dialog ---------- */
const NEW_RUN_URL = '/insights/new?mount=dialog';
let sheetOpener = null;
async function openSheet(url, title, ready) {
  sheetOpener = document.activeElement;
  const sheet = $('sheet');
  sheet.innerHTML = `<header><h2 id="sheetTitle">${esc(title)}</h2><button type="button" id="closeSheet" aria-label="Close">×</button></header><div class="body" id="sheetBody"><p class="insights-muted" role="status">Loading the run form…</p></div>`;
  sheet.setAttribute('role', 'dialog'); sheet.setAttribute('aria-modal', 'true'); sheet.setAttribute('aria-labelledby', 'sheetTitle');
  $('closeSheet').onclick = closeSheet;
  sheet.classList.add('on'); $('scrim').classList.add('on'); $('closeSheet').focus();
  const body = $('sheetBody');
  try {
    const response = await fetch(url, {headers: {Accept: 'text/html'}});
    if (!response.ok) throw new Error('The run form could not be loaded.');
    body.innerHTML = await response.text();
    const form = body.querySelector('#insights-new-form');
    InsightsRunForm.mount(form);
    if (ready) ready(form);
  } catch (err) {
    body.innerHTML = `<p class="insights-error" role="alert">${esc(err.message)} <button type="button" class="linkbtn" id="sheetRetry">Retry</button></p>`;
    $('sheetRetry').onclick = () => openSheet(url, title, ready);
  }
}
function closeSheet() { $('sheet').classList.remove('on'); $('scrim').classList.remove('on'); sheetOpener?.focus(); }
const openRerun = () => openSheet(`/insights/new?rerun=${encodeURIComponent(D.run.id)}&mount=dialog`, 'Re-run Insights');
$('newRun').onclick = () => openSheet(NEW_RUN_URL, 'New Insights run');
$('rerun')?.addEventListener('click', e => { e.preventDefault(); openRerun(); });
$('scrim').onclick = closeSheet;

document.addEventListener('keydown', e => {
  if ($('sheet').classList.contains('on') && e.key === 'Tab') {
    const focusable = Array.from($('sheet').querySelectorAll('button:not([disabled]),input:not([disabled]):not([type="file"]),textarea:not([disabled]),select:not([disabled]),a[href]')).filter(el => el.offsetParent !== null);
    if (!focusable.length) return;
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    return;
  }
  if (e.key !== 'Escape') return;
  if ($('sheet').classList.contains('on')) return closeSheet();
  if ($('details').classList.contains('open')) return $('details').classList.remove('open');
  if ($('stage').classList.contains('full')) { $('stage').classList.remove('full'); return renderStage(); }
  if (S.sel) select(null);
  else if (S.view === 'activity' && S.activityItem) { S.activityItem = null; commit(); }
});
document.addEventListener('click', e => { const d = $('details'); if (d.classList.contains('open') && !d.contains(e.target)) d.classList.remove('open'); });

fromUrl();
renderHeader();
bindMock(document);
render();
if (/[?&]rerun=1(?:&|$)/.test(location.search)) openRerun();

})();
