# evaluatorq — Color Design Language

How we use the Orq brand colors (teal + orange) in the docs and any UI.
One rule above all: **teal carries the interface, orange punctuates it.**

## Palette

| Token | Hex | Role |
|-------|-----|------|
| `--orq-teal` | `#025558` | Primary brand. Identity + structure. |
| `--orq-teal-light` | `#299d8f` | Teal on dark backgrounds. |
| `--orq-teal-dark` | `#013a3c` | Hover/pressed teal, deep fills. |
| `--orq-orange` | `#ff8f34` | Accent. Attention only. |
| `--orq-orange-deep` | `#df5325` | Orange **fills/buttons** (white text on top). |
| `--orq-orange-text` | `#c4421a` | Orange **text** on light surfaces (~4.6:1, AA). |
| `--orq-ink` | `#25232e` | Body text. |
| `--orq-sand` | `#f9f8f6` | Warm surfaces (code bg, cards). |

## Teal — the workhorse

Teal is the default. If something is interactive, structural, or "us," it's teal.

- Header, footer, nav, sidebar active state
- Headings (h1/h2)
- Links and their hover
- Primary buttons, focus rings, selection
- Logo disc background

Teal should feel calm and everywhere. You never count the teal on a page.

## Orange — the spotlight

Orange is a finger pointing at **one** thing. It is never decoration.

**Use it for:**
- The single primary call-to-action in a view (e.g. "Get Started")
- Hover/active accent on cards and the `:octicons-arrow-right:` links
- The logo mark inside the disc
- Code annotation dots, `!!! tip` admonition edge
- The one stat or word you want read first

**Never:**
- Body text or paragraphs
- Two orange focal points competing in the same viewport (max **one**)
- Large filled orange areas (it's an accent, not a surface)
- Orange text on sand/white — fails contrast; use `--orq-orange-deep` and only at ≥18px/bold

## The one-accent rule

Per screen, ask: *what is the single most important action here?* That gets orange.
Everything else stays teal or neutral. If you've used orange twice, one of them is wrong.

> Calm teal field, one orange spark. If everything is accented, nothing is.

## Dark mode

- Teal lightens to `--orq-teal-light` so it stays legible on slate.
- Orange is unchanged but used **even more sparingly** — it's louder on dark.
- Links use teal-light, never raw cyan (too harsh).

## Contrast quick-check

| Foreground | Background | OK? |
|-----------|-----------|-----|
| Orange `#ff8f34` | Teal `#025558` | ✅ marks/icons |
| Orange `#ff8f34` | Sand/white | ❌ text (2.3:1) |
| Orange `#df5325` | Sand/white | ❌ text (3.9:1) — fills only |
| Orange `#c4421a` | Sand/white | ✅ text (~4.6:1, use `--orq-orange-text`) |
| Teal `#025558` | Sand/white | ✅ text + headings |
| White | Teal `#025558` | ✅ header text |
