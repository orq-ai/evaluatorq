# Browser acceptance: trace shortcuts

Date: 2026-09-30. The dashboard ran at `http://127.0.0.1:8175/traces` with `uv run evaluatorq dashboard --port 8175 --no-browser`; browser control used the `agent-browser` CLI session `shortcuts-acceptance`. The page loaded 200 trace rows, and clicking the first row opened the real rendered conversation drawer. I did not submit Ask AI or make a paid model call.

## Reproducible actions and results

- `/`: fill the Ask AI textarea with `find the test trace`, click the page heading, then run `agent-browser press / --session shortcuts-acceptance`. The active element was `.finder-command-textarea`; its selection range was `0..19`, selecting the full query.
- `?`: with the drawer open and focus outside editable fields, dispatch `document.dispatchEvent(new KeyboardEvent("keydown", {key:"?", shiftKey:true, bubbles:true, cancelable:true}))` through `agent-browser eval`. The guide opened with `aria-labelledby="finder-shortcut-title"` and focus on its Close button. This uses the normalized `?` key because this headless browser's physical `Shift+/` reports `/`.
- `j` and `k`: `agent-browser press j` changed the open trace ID from `ac4e71771fa875d49cd2433ff31584a7` to `ce884fa70aae16401ac1ebc6758bde4d`; `agent-browser press k` changed it back. Each result was read from the rendered drawer's footer `button[data-trace-id]` after its HTMX swap.
- `o`: attach a capture listener to the rendered drawer's actual `Open in Orq` anchor that calls `preventDefault()` and increments a counter, then run `agent-browser press o`. The counter was `1`, proving the link activated while preventing external navigation.
- `c`: replace `navigator.clipboard.writeText` with a recording stub, then run `agent-browser press c`. The stub received `ac4e71771fa875d49cd2433ff31584a7`, equal to the current rendered drawer copy button's `data-trace-id`.
- Escape: with the guide and drawer open, `agent-browser press Escape` closed the guide and left the drawer open. A second `agent-browser press Escape` dismissed the drawer; an eval poll observed that `#finder-drawer [role=dialog]` was removed after the HTMX swap.

The page screenshots were captured with `agent-browser screenshot` and visually inspected: `.context/critics/after-hate-1-shortcuts-focus.png`, `.context/critics/after-hate-1-shortcuts-guide.png`, `.context/critics/after-hate-1-shortcuts-actions.png`, and `.context/critics/after-hate-1-shortcuts-drawer-guide.png`. The browser session and the owned dashboard server were closed after the walkthrough.
