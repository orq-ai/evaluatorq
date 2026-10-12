---
name: dashboard-demo-video
description: Record a short video of an `eq dashboard` interaction and attach it to a PR. Use it when a PR changes how something in the browser behaves (a click, a filter, a tab, a modal, a flow across pages) and a screenshot of one state cannot show it, or when someone asks for a demo video, screen recording or clip of the dashboard. It covers the agent-browser recording traps (frame size, lost cursor script, the API-key notice), the visible-cursor script, the WebM encode and the `gh --attach` upload.
---

# dashboard-demo-video

A screenshot shows a state; a video shows a change between states. Record one only when the PR's point is an interaction. A static change gets screenshots, per CLAUDE.md "Show visible changes in the PR".

The recording is headless: `agent-browser record` captures the page only, with no OS cursor, browser chrome or sound. `cursor.js` in this directory draws a small white cursor and a click ring so a viewer can follow where each click lands. If the demo needs the real browser UI or narration, record the screen by hand instead.

## What the reader gets

A VP9 WebM at the page's own size (1440×900, 16:10, never scaled), about 1.5–3 MB for 30 s. GitHub plays `.webm` inline. Cap: 5 MB.

## Data

**Anyone with an attachment's URL can view it, without a GitHub login.** Record only synthetic or sanitized runs: no customer prompts, transcripts, keys or workspace names. Copy the one report you show into a scratch directory and serve only that directory, so nothing else appears in the run list. Grep it for keys first: `grep -E 'sk-[A-Za-z0-9]{10}|ORQ_API_KEY=' <report>.json` must print nothing.

## Steps

Start your own dashboard on a free port, as in any browser check: `lsof -nP -iTCP:<port> -sTCP:LISTEN` empty, then

```bash
env -u ORQ_API_KEY EVALUATORQ_DASHBOARD_SETTINGS=$S/settings.json EVALUATORQ_USER_SETTINGS=$S/user.json \
  uv run eq dashboard $S/demo-data --port $PORT --no-browser > $S/dash.log 2>&1 &
until curl -s -o /dev/null http://127.0.0.1:$PORT/; do sleep 1; done
```

Then record. These helpers drive a visible cursor; `$SKILL` is this directory.

```bash
ab(){ agent-browser --session demo "$@"; }
LAST="700 450"
# `record start` opens a fresh browser context that drops --init-script, so re-inject after every page load.
inject(){ ab eval "$(cat $SKILL/cursor.js)" >/dev/null; ab mouse move ${LAST% *} ${LAST#* }; }
# Glide to the first element matching CSS selector $1 whose text contains $2, then press.
# `ab click` fires no visible mouse movement, so the mouse is driven explicitly.
clk(){ local c=$(ab eval "(()=>{const e=[...document.querySelectorAll('$1')].find(e=>e.textContent.includes('$2')&&e.getBoundingClientRect().width>0);if(!e)return '';e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return Math.round(r.x+r.width/2)+' '+Math.round(r.y+r.height/2)})()" | tr -d '"')
  [ -z "$c" ] && { echo "NOT FOUND $1 $2"; return 1; }
  ab mouse move ${c% *} ${c#* }; ab wait 900; ab mouse down; ab mouse up; LAST=$c; ab wait 400; inject; }

URL="http://127.0.0.1:$PORT/?surface=sim"
ab open "$URL"
ab set viewport 1440 900          # BEFORE record start: the frame size is fixed when recording starts
ab record start $S/raw.webm "$URL"
ab wait "button[aria-label='Dismiss authentication notice']"
ab eval "document.querySelector('[aria-label*=Dismiss]')?.click()" >/dev/null
ab wait 2000; inject
clk a 'my-run-name'; ab wait 2500
clk label 'Breakdown'; ab wait 3000
# ... one clk + wait per step; 2-3 s per state so a reader can take it in
ab record stop; ab close
```

Encode, trimming the first 1.2 s where the notice is dismissed:

```bash
ffmpeg -y -ss 1.2 -i $S/raw.webm -an -c:v libvpx-vp9 -crf 20 -b:v 0 -row-mt 1 \
  -deadline good -cpu-used 2 -pix_fmt yuv420p $S/demo.webm
```

Check it before anyone else sees it: `ffprobe` should report 1440×900 with `display_aspect_ratio=8:5`, and a contact sheet should show every state in order:

```bash
ffmpeg -loglevel error -y -i $S/demo.webm -vf "fps=1,scale=480:-1,tile=5x6" -frames:v 1 $S/sheet.png
```

Kill your server by port (`lsof -tiTCP:$PORT -sTCP:LISTEN | xargs kill`), never with `pkill -f`.

## Attach it

Put `![demo](./demo.webm)` on its own line in the PR body file, then pass the file with `--attach`. `gh` uploads it and rewrites the path to the hosted URL:

```bash
gh pr edit <number> --body-file body.md --attach $S/demo.webm     # or gh pr create ... --attach
```

## Traps

- **Viewport after `record start`** leaves the video at the default 1280×634 while the page itself resizes. Set it first.
- **Device scale factor above 1** captures sharper frames but starves the screencast: 36 frames over 49 s, played back 13× too fast. Keep it at 1.
- **The API-key notice** reappears in the recording context even when dismissed before. Dismiss it inside the recording and trim.
- **Wall-clock trims drift.** The recorder emits frames only when the page changes, so video time is shorter than real time. Trim little; check the sheet.
- **Snapshot refs and `find text` miss** tabs and filters rendered as hidden radios behind `<label>`. Select by CSS selector plus text, as `clk` does.
- **A filter can reset the tab.** On a sim report, applying the goal-outcome filter returns the page to Overview. Apply filters first, then open the tab.
- **Re-encoding cannot add detail.** The raw capture is about 215 kbit/s VP8 from JPEG frames; a lower CRF only adds bytes. The raw file is uploadable as is if it is under 5 MB.
