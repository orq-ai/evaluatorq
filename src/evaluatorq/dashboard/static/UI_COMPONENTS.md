# Dashboard components

Use `ui-components.js` and `ui-components.css` to reuse the foldout, result feedback, and precondition presentation from Signals. The components have no framework or Signals dependency and use the dashboard tokens in `theme.py`.

Start the dashboard with `uv run eq dashboard`, then open `/static/ui-components-example.html` on that server for a working example with fixed data. The example loads the components directly and never loads `insights-signals.js`.

## Load and render

In a Python dashboard view, pass `component_assets()` from `evaluatorq.dashboard.ui_components` before the consumer's script in `page(..., head_html=...)`. It supplies the stylesheet and deferred browser script. Deferred consumers run after the component script. Static HTML can load `/static/ui-components.css` and `/static/ui-components.js` directly, as the example does.

Wrap the host in `class="eq-components"` to get the shared typography and a container-query context. Add a host to the page:

```html
<div id="validation-host" class="eq-components"></div>
```

Once the script has loaded, use `window.EvaluatorqComponents`:

```javascript
const ui = window.EvaluatorqComponents;
const host = document.getElementById('validation-host');
const feedback = ui.resultFeedback({
  value: 0,
  outcome: 'Warnings',
  tone: 'positive',
  numeric: true,
  reason: 'No warnings were recorded.',
});
const checks = ui.preconditionList([
  {name: 'Source present', met: true, required: true, detail: 'Input is available.'},
]);
host.innerHTML = ui.foldout({
  key: 'validation',
  summaryHtml: '<span>Validation checks</span>',
  bodyHtml: feedback + checks,
});
```

## Contracts

- `escapeHtml(value)` escapes plain text or an attribute value. A missing value becomes an empty string.
- `foldout({key, summaryHtml, bodyHtml, className, data, open})` renders a native `<details>` with keyboard controls. It starts closed unless `open: true`. A stable, unique `key` enables state restoration. `className` adds feature styling. `data` is a map of lowercase hyphenated names without the `data-` prefix; invalid names and `ui-key` are rejected.
- `resultFeedback({value, outcome, tone, reason, bodyHtml, numeric, label})` puts the primary value beside the outcome. It preserves zero and false, escapes the value, outcome, reason and accessible `label`, and omits a duplicate outcome when it equals the value. `numeric: true` uses the monospaced measurement style. Tones are `info`, `positive`, `negative`, `warning`, and `neutral`; unsupported tones are rejected. The consumer decides what an outcome means.
- `preconditionList(items, {emptyText})` renders the checklist, met count and meter. Items contain `name`, `met`, `required`, and optional `detail`. `met` accepts `true`, `false`, or `'partial'`; other values remain not met. Text is escaped. An empty list shows `emptyText`, which defaults to “No preconditions recorded.”
- `captureOpen(host)` returns the keys of open `<details>` elements with `data-ui-key`. `restoreOpen(host, keys)` restores those keys after a render without touching unkeyed details or elements outside the host. Both accept an optional attribute name as the last argument for existing keyed views.

`summaryHtml` and `bodyHtml` are explicit trusted HTML slots. Escape dynamic text before putting it in a slot, for example `summaryHtml: '<span>' + ui.escapeHtml(title) + '</span>'`. Do not pass a dataset field or server response directly as HTML. Plain-text component fields do not need caller escaping.

Styling lives in `ui-components.css`; Signals keeps only its level layout, report mapping, badges, evidence formatting, and flagged-tag shortcuts. Keep stable keys within one host unique and scoped to that view.

## Validate

```sh
uv run pytest tests/dashboard/test_ui_components.py tests/dashboard/test_dashboard_js_runtime.py tests/dashboard/test_insights_signal_detail_route.py
```
