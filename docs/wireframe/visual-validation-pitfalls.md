# Visual validation — how to not fool yourself

Hard-won checklist for comparing a **live rendered UI** against a **design mockup** (both are real DOM). Written after several false "it matches" verdicts. Share this with any agent doing pixel/parity work.

## The one rule

**Measure the DOM numerically. Do not eyeball scale.** Screenshots lie about size; `getBoundingClientRect` + `getComputedStyle` do not. Use screenshots only for gestalt (layout, color, overall feel), never to judge whether two things are the same size.

## The traps (each one caused a real miss)

1. **Zoom illusion.** A user-supplied crop is often zoomed (e.g. 1.7×). Eyeballing it makes elements look huge. NEVER size-match against a crop — measure the actual rendered element instead.

2. **Downscaled full-page compare.** Feeding two different-width full-page PNGs to a verifier normalizes scale in the reader's eye — a shrunk component reads as "same, just fewer cells." Always screenshot **element-scoped, at natural scale** (`screenshot "#selector"`), never a downscaled full page.

3. **Measured the content, missed the chrome.** The worst miss: I measured heatmap *cell* size/font/radius and declared a match — but the table had an inherited **border, white background, 10px radius, a header underline, a row-label divider, and a zebra stripe** that the mockup didn't have. **Measure the CONTAINER, not just the content:** border, background, border-radius, box-shadow, padding, margin, gap/border-spacing, per-side cell borders, dividers, and hover/zebra/nth-child states.

4. **Global CSS bleed.** Your component inherits from global `table` / `th` / `tr:nth-child(even)` / `:hover` rules you didn't write. Scoping your own `.component` CSS does NOT remove them. Explicitly query what the live element actually computes (border, bg, radius, borderBottom) and override every global rule that leaks in.

5. **Data-driven ≠ style gap — but don't hide behind it.** Different datasets (longer labels, fewer rows, different numbers) legitimately change size and layout. Note those as data-driven. But do NOT use "it's just data" to wave away a real scale or chrome difference — that rationalization is how the border miss survived.

6. **Specified vs rendered weight.** `getComputedStyle` returns the *specified* `font-weight` (e.g. 600) even when the font only ships 400/700 and actually renders 700. CSS weight 500–600 rounds UP to 700 when only 400/700 exist. So a "600 vs 700" diff may render identically. Know this before "fixing" it.

7. **Stale DOM after navigation.** After clicking an SPA tab, the `innerText` you slice may be the *previous* panel. I concluded "the tab didn't switch" when it had. Re-query fresh and confirm the switch via a content marker before measuring.

8. **Server-side CSS caching.** Editing the stylesheet source did NOT change the running app — CSS was baked at server startup. A browser hard-reload isn't enough; **restart the server** (or rely on real hot-reload) before re-measuring. Verify the new value landed, don't assume.

9. **Sloppy element selection.** A broad selector for "the panel title" grabbed a full-width *container* (height 297px) instead of the title span. Before trusting any metric, sanity-check the matched element's bounding box is tight/plausible. The mockup uses different class names — query **its** elements by CONTENT (e.g. the node whose text matches `/^\d{1,3}%$/`) and pick the smallest match.

10. **Screenshot hover artifacts.** The cursor sitting over an element adds a
    `:hover` tint to the screenshot that looks like a real style. Check computed
    background, not pixels — or move the cursor away first.

11. **Lazy / HTMX content.** Expanded or lazy-loaded fragments may not load via
    synthetic events (`htmx` global undefined, `hx-trigger="click once"` not firing
    on a programmatically-opened `<details>`). To measure them, fetch the fragment
    server-side or fire the exact trigger event — never measure an unloaded body
    (length 0) and assume it's empty by design.

12. **Font family name ≠ visual gap.** Mock "Avio Sans" vs live "ES Klarheit
    Kurrent" are both sans; "Kurrent" vs "ui-monospace" are both mono. Different
    named fonts of the same class are asset substitutions, not per-component gaps —
    unless the exact brand font is the deliverable.

13. **Biased verifier.** An agent that just saw the fix is primed to confirm it. Use
    a **fresh** verifier and give it the measured **numbers** (a diff table), not
    just two images and a vibe.

## Method

For each component:

1. **Measure both DOMs.** Extract for the signature element AND its container: dimensions, font-size/weight/family/text-transform/letter-spacing, color, background, border (all sides), border-radius, padding, gap/border-spacing, box-shadow. Query the mockup by content; query live by class.
2. **Diff the numbers** against explicit thresholds: font-size delta > ~2px, dimension delta > ~25% not explained by data, ANY mismatch in border/background/radius/shadow/case/family-class/alignment = gap.
3. **Element-scoped natural-scale screenshots** of both for layout/color/gestalt.
4. **Fix**, restart the server, **re-measure** (confirm the value changed), fresh verifier confirms. Loop until the numbers match.

## What to always inspect on a container

`border` (4 sides) · `background` · `border-radius` · `box-shadow` · `padding` · `margin` · `gap` / `border-spacing` · child-cell borders/dividers · `:hover` · `:nth-child` zebra · overflow/clip. Content typography is necessary but never sufficient.
