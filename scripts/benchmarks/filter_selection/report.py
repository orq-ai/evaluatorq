#!/usr/bin/env python
"""Render one self-contained HTML report for a filter-selection benchmark run.

Usage:
    uv run python scripts/benchmarks/filter_selection/report.py <main-dir> \
        [--supplement <dir>] [--analysis analysis.md] [--out report.html]

Inputs (all produced by bench.py): raw.jsonl, meta.json, summary.json.
Every number in the page is computed from those files at render time; the
only prose that is static is the hypothesis, the decision and the caveats.
The output has inline CSS + SVG only: no JS, no external assets.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ARM_ORDER = ["llm+jev", "llm"]
BASELINE = "llm+jev"
CATEGORY_ORDER = [
    "facet",
    "multi_facet",
    "status",
    "numeric",
    "time",
    "combo",
    "semantic_only",
    "aggregate",
    "implied_trap",
    "paraphrase",
    "outside_catalogue",
]
ARM_CLASS = {"llm+jev": "a0", "llm": "a1"}
EXTRA_ARM_CLASSES = ["a2", "a3"]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_dataset(meta: dict[str, Any], results_dir: Path) -> dict[str, dict[str, Any]]:
    """Dataset rows by id, for the query text. Empty when the file is unreachable."""
    raw = meta.get("dataset_path")
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates += [p, Path.cwd() / p, HERE / p.name, results_dir / p.name]
    candidates.append(HERE / "dataset.jsonl")
    for cand in candidates:
        if cand.is_file():
            return {str(r["id"]): r for r in _load_jsonl(cand)}
    return {}


# --------------------------------------------------------------------------- #
# formatting helpers (None / NaN never leak into the page)
# --------------------------------------------------------------------------- #
DASH = "—"


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def lat_caveat(run: Run) -> str:
    conc = _num(run.meta.get("arm_concurrency"))
    if conc is None or conc == 1:
        return ""
    return (
        '<div class="banner" role="note"><strong>Not latency-grade.</strong> '
        f"This workload ran with arm concurrency {f_int(conc)}, so modes contended for the network and provider. Use it for accuracy only; latency and measured cache scenarios are not comparable to a sequential run.</div>"
    )


def f_ms(v: Any) -> str:
    n = _num(v)
    if n is None:
        return DASH
    if n >= 10_000:
        return f"{n / 1000:.1f} s"
    if n >= 1000:
        return f"{n / 1000:.2f} s"
    return f"{n:.0f} ms"


def f_pct(v: Any, digits: int = 0) -> str:
    n = _num(v)
    return DASH if n is None else f"{n * 100:.{digits}f}%"


def f_float(v: Any, digits: int = 1) -> str:
    n = _num(v)
    return DASH if n is None else f"{n:,.{digits}f}"


def f_int(v: Any) -> str:
    n = _num(v)
    return DASH if n is None else f"{n:,.0f}"


def f_usd(v: Any) -> str:
    n = _num(v)
    if n is None:
        return DASH
    if n == 0:
        return "$0"
    if abs(n) < 0.01:
        return f"${n:.5f}"
    if abs(n) < 1:
        return f"${n:.4f}"
    return f"${n:,.2f}"


def f_usd_k(v: Any) -> str:
    n = _num(v)
    return DASH if n is None else f"${n * 1000:,.2f}"


def arm_sort_key(arm: str) -> tuple[int, str]:
    base = arm.split("@", 1)[0]
    return (ARM_ORDER.index(base) if base in ARM_ORDER else len(ARM_ORDER), arm)


def arm_classes(arms: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    extra = iter(EXTRA_ARM_CLASSES)
    for arm in arms:
        out[arm] = ARM_CLASS.get(arm) or next(extra, "a5")
    return out


def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile on a pre-sorted list (q in 0..1)."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if raw <= step:
            break
    start = math.ceil(lo / step) * step
    ticks: list[float] = []
    t = start
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 12))
        t += step
    return ticks


# --------------------------------------------------------------------------- #
# markdown (small subset) for --analysis
# --------------------------------------------------------------------------- #
_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)([^*]+?)\*(?![\w*])")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _inline(text: str) -> str:
    codes: list[str] = []

    def stash(m: re.Match[str]) -> str:
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"

    text = _INLINE_CODE.sub(stash, text)
    text = html.escape(text, quote=False)
    text = _LINK.sub(lambda m: f'<a href="{m.group(2)}" rel="noreferrer">{m.group(1)}</a>', text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], text)


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_sep(line: str) -> bool:
    cells = _split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def markdown_to_html(md: str) -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    para: list[str] = []

    def flush_para() -> None:
        if para:
            out.append(f"<p>{_inline(' '.join(s.strip() for s in para))}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_para()
            i += 1
            buf: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(buf))}</code></pre>")
            continue
        if not stripped:
            flush_para()
            i += 1
            continue
        m = re.match(r"(#{1,6})\s+(.*?)\s*#*\s*$", stripped)
        if m:
            flush_para()
            level = min(len(m.group(1)) + 2, 6)  # md # -> h3 inside the verdict section
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if re.fullmatch(r"([-*_])\1{2,}", stripped):
            flush_para()
            out.append("<hr>")
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and _is_sep(lines[i + 1]):
            flush_para()
            head = _split_row(stripped)
            aligns = []
            for c in _split_row(lines[i + 1]):
                aligns.append("right" if c.endswith(":") and not c.startswith(":") else "left")
            i += 2
            body: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                body.append(_split_row(lines[i]))
                i += 1
            t = ['<div class="scroll"><table class="md"><thead><tr>']
            for j, c in enumerate(head):
                t.append(f'<th class="{aligns[j] if j < len(aligns) else "left"}">{_inline(c)}</th>')
            t.append("</tr></thead><tbody>")
            for row in body:
                t.append("<tr>")
                for j, c in enumerate(row):
                    t.append(f'<td class="{aligns[j] if j < len(aligns) else "left"}">{_inline(c)}</td>')
                t.append("</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            continue
        lm = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", line)
        if lm:
            flush_para()
            ordered = lm.group(2)[0].isdigit()
            items: list[str] = []
            while i < len(lines):
                im = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", lines[i])
                if im and im.group(2)[0].isdigit() == ordered:
                    items.append(im.group(3))
                    i += 1
                elif lines[i].startswith("  ") and lines[i].strip() and items:
                    items[-1] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(it)}</li>" for it in items) + f"</{tag}>")
            continue
        if stripped.startswith(">"):
            flush_para()
            buf2: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf2.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(f"<blockquote>{_inline(' '.join(buf2))}</blockquote>")
            continue
        para.append(line)
        i += 1
    flush_para()
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
def _resolve_catalogue_meta(meta: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    """Snapshot window (captured_at, window_start/end) from the catalogue file, without its payload."""
    raw = meta.get("catalogue_path")
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates += [p, Path.cwd() / p, HERE / p.name, results_dir / p.name]
    for cand in candidates:
        if cand.is_file():
            d = _load_json(cand)
            return {k: d.get(k) for k in ("profile", "captured_at", "window_start", "window_end") if isinstance(d, dict)}
    return {}


class Run:
    def __init__(self, results_dir: Path) -> None:
        self.dir = results_dir
        self.meta: dict[str, Any] = _load_json(results_dir / "meta.json")
        self.summary: dict[str, Any] = _load_json(results_dir / "summary.json")
        scored = results_dir / "scored.jsonl"
        self.raw = _load_jsonl(scored if scored.is_file() else results_dir / "raw.jsonl")
        self.queries = [r for r in self.raw if r.get("kind") == "query"]
        self.calls = [r for r in self.raw if r.get("kind") == "call"]
        self.catalogue_rows = [r for r in self.raw if r.get("kind") == "catalogue"]
        self.snapshot = _resolve_catalogue_meta(self.meta, results_dir)
        summary_arms = list((self.summary.get("arms") or {}).keys())
        raw_arms = sorted({str(r["arm"]) for r in self.queries})
        self.arms = sorted({*summary_arms, *raw_arms}, key=arm_sort_key)
        self.cls = arm_classes(self.arms)
        self.dataset = _resolve_dataset(self.meta, results_dir)

    def arm_summary(self, arm: str) -> dict[str, Any]:
        return (self.summary.get("arms") or {}).get(arm) or {}

    def queries_for(self, arm: str) -> list[dict[str, Any]]:
        return [q for q in self.queries if q.get("arm") == arm]

    @property
    def categories(self) -> list[str]:
        seen = {str(q.get("category")) for q in self.queries if q.get("category")}
        for arm in self.arms:
            seen |= set((self.arm_summary(arm).get("by_category") or {}).keys())
        ordered = [c for c in CATEGORY_ORDER if c in seen]
        return ordered + sorted(seen - set(ordered))


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def chip(arm: str, run: Run) -> str:
    return f'<span class="chip {run.cls.get(arm) or ARM_CLASS.get(arm, "a5")}"><i></i>{esc(arm)}</span>'


_ID_PREFIX = ""


def section(id_: str, title: str, lede: str, body: str) -> str:
    lede_html = f'<p class="lede">{lede}</p>' if lede else ""
    return f'<section id="{esc(_ID_PREFIX)}{id_}"><h2>{esc(title)}</h2>{lede_html}{body}</section>'


def sec_hypothesis(run: Run) -> str:
    body = f"""
<div class="two">
  <div>
    <h3>Hypothesis</h3>
    <p>Does production's configured <code>llm</code> filter-selection mode preserve
    quality versus <code>llm+jev</code>, with comparable datetime filters,
    latency, token usage and cost?</p>
  </div>
  <div>
    <h3>Decision it informs</h3>
    <p>Whether trace-finder should offer the configured fast-model filter path
    alongside the configured classifier path. Both arms use the same shared
    production query planner and source bounds.</p>
  </div>
</div>
<dl class="arms">
  <dt>{chip("llm+jev", run)}</dt><dd>Shared production planner; configured fast compiler role plus configured classifier task.</dd>
  <dt>{chip("llm", run)}</dt><dd>Shared production planner; configured fast compiler role and configured fast filter task. Both arms produce and are scored on datetime filters.</dd>
</dl>"""
    return section("hypothesis", "Hypothesis and decision", "", body)


def _kv(rows: list[tuple[str, str]]) -> str:
    return (
        '<dl class="kv">'
        + "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in rows)
        + "</dl>"
    )


def sec_setup(run: Run) -> str:
    m = run.meta
    s = run.summary
    cat_lat = s.get("catalogue_latency_ms") or {}
    sha = str(m.get("git_sha") or "")
    pk = m.get("packages") or {}
    pk_txt = ", ".join(f"{esc(k)} {esc(v)}" for k, v in pk.items())
    model_rows = []
    for key, label in (
        ("compiler_model", "Compiler model"),
        ("classifier_model", "Classifier task model"),
        ("fast_filter_model", "Fast filter task model"),
    ):
        if m.get(key):
            model_rows.append((label, f"<code>{esc(m[key])}</code>"))
    served: dict[tuple[str, str], set[str]] = defaultdict(set)
    for c in run.calls:
        if c.get("model_served"):
            served[(str(c["call"]), str(c.get("model_requested")))].add(str(c["model_served"]))
    served_txt = "; ".join(
        f"{esc(call)}: {esc(req)} → " + ", ".join(f"<code>{esc(x)}</code>" for x in sorted(v))
        for (call, req), v in sorted(served.items())
    )
    conc = _num(m.get("arm_concurrency"))
    if conc is None:
        conc_txt = DASH
    elif conc == 1:
        conc_txt = "1: arms ran one at a time; latency measured sequentially"
    else:
        conc_txt = f"{f_int(conc)}: arms ran concurrently, so latency is <strong>not comparable to a sequential run</strong> (accuracy and token figures still valid)"
    win = m.get("catalogue_window_days")
    win_txt = f"last {f_int(win)} days of traces (only for the catalogue load-latency measurement)" if _num(win) is not None else DASH
    sn = run.snapshot
    snap_txt = (
        f"<code>{esc(sn.get('window_start') or DASH)}</code> → <code>{esc(sn.get('window_end') or DASH)}</code>, captured <code>{esc(sn.get('captured_at') or DASH)}</code>"
        if sn
        else "snapshot file not found"
    )
    n_dp = m.get("n_datapoints")
    if n_dp is None:
        n_dp = len({str(q["id"]) for q in run.queries}) or None
    cat_n = _num(cat_lat.get("n")) if cat_lat else None
    cat_txt = (
        f"p50 {f_ms(cat_lat.get('p50'))}, p95 {f_ms(cat_lat.get('p95'))} over {f_int(cat_n)} loads"
        if cat_lat and cat_n
        else "not measured in this run"
    )
    rows: list[tuple[str, str]] = [
        ("Arms", " ".join(chip(a, run) for a in run.arms) or DASH),
        *model_rows,
        *[(f"Arm model: {a}", f"<code>{esc(mdl)}</code>") for a, mdl in (m.get("arm_models") or {}).items() if mdl],
    ]
    if served_txt:
        rows.append(("Models actually served", served_txt))
    rows += [
        ("Datapoints", f"{f_int(n_dp)} labelled queries" + (f" (limited to {f_int(m.get('limit_datapoints'))})" if m.get("limit_datapoints") else "")),
        ("Repeats / warmup", f"{f_int(m.get('repeats'))} repeats per datapoint; warmup queries discarded: {f_int(m.get('warmup'))}"),
        ("Scored query runs", f_int(s.get("n_queries")) + " (arm × datapoint × repeat)"),
        ("Facet values offered", f"all {f_int(m.get('catalogue_values_offered', 50))} available values plus confirmed named candidates"),
        ("Arm concurrency", conc_txt),
        ("Catalogue snapshot window", snap_txt),
        *([("Model sources", "; ".join(f"{esc(k)}: {esc(v)}" for k, v in (m.get("model_sources") or {}).items()))] if m.get("model_sources") else []),
        *([("Role models", "; ".join(f"{esc(k)}: <code>{esc((v or {}).get('role_model'))}</code> ({esc((v or {}).get('source'))})" for k, v in (m.get("role_models") or {}).items()))] if m.get("role_models") else []),
        *([("Settings snapshot", f"<code>{esc(m.get('settings_path'))}</code> (sha256 <code>{esc(str(m.get('settings_sha256') or DASH)[:16])}</code>)")] if m.get("settings_path") else []),
        ("Other bench processes", f"{f_int(m.get('other_bench_processes_at_start'))} at start, {f_int(m.get('other_bench_processes_at_end'))} at end" if m.get("other_bench_processes_at_start") is not None else DASH),
        ("Catalogue cache", "production cache hit/miss and elapsed time recorded per query" if m.get("cache_telemetry") else "not recorded"),
        ("Catalogue load latency", cat_txt),
        ("Workload clock (now)", f"<code>{esc(m.get('now') or DASH)}</code>"),
        ("Profile", f"<code>{esc(m.get('profile'))}</code>" if m.get("profile") else "none (no network access)" if m.get("dry_run") else DASH),
        ("Git SHA", f"<code>{esc(sha[:12] or DASH)}</code>"),
        ("Dataset sha256", f"<code>{esc(str(m.get('dataset_sha256') or DASH)[:16])}</code>"),
        ("Catalogue sha256", f"<code>{esc(str(m.get('catalogue_sha256') or DASH)[:16])}</code>"),
        ("Run window", f"{esc(m.get('started_at') or DASH)} → {esc(m.get('finished_at') or DASH)}"),
        ("Runtime", f"Python {esc(m.get('python') or DASH)}" + (f"; {pk_txt}" if pk_txt else "")),
    ]
    caveats = [
        "A fixed catalogue snapshot defines the scored available-value workload and is injected at the production cache-loader seam; planner/cache scenarios below are separate live-cache observations.",
        (
            "Both modes ran sequentially at concurrency one. "
            if (_num(m.get("arm_concurrency")) or 1) == 1
            else f"Arms ran with concurrency {f_int(m.get('arm_concurrency'))}; latency is not comparable. "
        )
        + "Mode order alternates deterministically by repeat. Network/provider load and co-tenant activity are uncontrolled.",
        "Cold/warm end-to-end and cache-loader timings are live production-cache scenarios, each with n=1; they are descriptive observations, not latency distributions and are not part of the fixed-snapshot scored workload.",
        "Requested and served models may differ. Usage/cost fields preserve missing and partial data rather than treating unavailable values as zero.",
        "Labels follow the filter-or-judgment rule. A query needing conversation judgment can make expected filters conservative.",
        "Both modes are scored against the same datetime labels and source bounds. Time bounds match within the configured tolerance.",
        "Rows are paired by labelled query; repeats are summarized within each row and are not independent samples. Main and supplement workloads remain separate.",
    ]
    if m.get("dry_run"):
        caveats.insert(0, "<strong>This is a dry run.</strong> It exercises the pipeline end to end and does not answer the hypothesis.")
    body = (
        '<div class="two wide-left">'
        f"<div>{_kv(rows)}</div>"
        '<div><h3>What is not controlled</h3><ul class="caveats">'
        + "".join(f"<li>{c}</li>" for c in caveats)
        + "</ul></div></div>"
    )
    return section("setup", "Setup", "", body)


# metric rows: (label, key-getter, formatter, better) better: 'low' | 'high' | None
def _g(path: str):
    def get(sm: dict[str, Any]) -> Any:
        cur: Any = sm
        for part in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    return get


def _per_k(path: str):
    g = _g(path)

    def get(sm: dict[str, Any]) -> Any:
        v = _num(g(sm))
        return None if v is None else v * 1000

    return get


HEADLINE: list[tuple[str, list[tuple[str, Any, Any, str | None, str]]]] = [
    (
        "Latency",
        [
            ("Planner critical-path p50", _g("latency_ms_p50"), f_ms, "low", "ms"),
            ("Planner critical-path p95", _g("latency_ms_p95"), f_ms, "low", "ms"),
            ("Catalogue cache events", _g("catalogue_cache_events"), f_int, None, "count"),
            ("Catalogue cache hits", _g("catalogue_cache_hit_events"), f_int, None, "count"),
            ("Catalogue cache misses", _g("catalogue_cache_miss_events"), f_int, None, "count"),
            ("Catalogue cache elapsed p95", _g("catalogue_cache_elapsed_ms_p95"), f_ms, None, "ms"),
            ("Cache-hit load p50", _g("catalogue_cache_hit_elapsed_ms_p50"), f_ms, "low", "ms"),
            ("Cache-hit load p95", _g("catalogue_cache_hit_elapsed_ms_p95"), f_ms, "low", "ms"),
            ("Cache-miss load p50", _g("catalogue_cache_miss_elapsed_ms_p50"), f_ms, "low", "ms"),
            ("Cache-miss load p95", _g("catalogue_cache_miss_elapsed_ms_p95"), f_ms, "low", "ms"),
            ("Measured cold-cache e2e p50", _g("cold_e2e_latency_ms_p50"), f_ms, "low", "ms"),
            ("Measured warm-cache e2e p50", _g("warm_e2e_latency_ms_p50"), f_ms, "low", "ms"),
        ],
    ),
    (
        "Tokens and cost",
        [
            ("Input tokens / query", _g("mean_tokens_per_query.input"), f_int, "low", "tok"),
            ("Output tokens / query", _g("mean_tokens_per_query.output"), f_int, "low", "tok"),
            ("Cached input tokens / query", _g("mean_tokens_per_query.cached"), f_int, "low", "tok"),
            ("Reported $ / query", _g("cost_per_query_reported_usd"), f_usd, "low", "usd"),
            ("Reported $ / 1k queries", _per_k("cost_per_query_reported_usd"), f_usd_k, "low", "usd1k"),
            ("Unpriced calls", _g("unpriced_calls"), f_int, "low", "count"),
            ("Partial-cost calls", _g("partial_cost_calls"), f_int, "low", "count"),
        ],
    ),
    (
        "Quality",
        [
            ("Full exact (facets + numeric + time)", _g("full_exact_rate"), f_pct, "high", "rate"),
            ("Facet precision", _g("facet_precision"), f_pct, "high", "rate"),
            ("Facet recall", _g("facet_recall"), f_pct, "high", "rate"),
            ("Facet exact", _g("facet_exact_rate"), f_pct, "high", "rate"),
            ("Numeric accuracy", _g("numeric_exact_rate"), f_pct, "high", "rate"),
            ("Time accuracy on time-bearing rows", _g("time_accuracy_on_time_rows"), f_pct, "high", "rate"),
            ("Judgment agreement", _g("judgment_agree_rate"), f_pct, "high", "rate"),
            ("False-filter rate (eligible rows)", _g("false_filter_rate"), f_pct, "low", "rate"),
            ("Hallucinated values / query", _g("hallucinated_per_query"), lambda v: f_float(v, 3), "low", "count"),
            ("Queries with a hallucinated value", _g("hallucinated_query_rate"), f_pct, "low", "rate"),
            ("Queries with a failed call", None, None, "low", "errors"),
        ],
    ),
]


def _delta(kind: str, v: float, base: float) -> tuple[str, str]:
    """Text for the difference to baseline, and a css hint of its sign."""
    if kind == "rate":
        d = (v - base) * 100
        txt = f"{d:+.1f} pp"
    elif base == 0:
        d = v - base
        txt = f"{d:+,.2f}" if d else "±0"
    else:
        d = (v - base) / abs(base) * 100
        txt = f"{d:+.0f}%"
    if round(d, 1 if kind == "rate" else (2 if base == 0 else 0)) == 0:
        return "±0", "z"
    return txt, "p" if d > 0 else "n"


def sec_headline(run: Run) -> str:
    arms = run.arms
    have_base = BASELINE in arms
    head = "".join(f'<th scope="col">{chip(a, run)}</th>' for a in arms)
    rows: list[str] = []
    for group, metrics in HEADLINE:
        rows.append(f'<tr class="grp"><th colspan="{len(arms) + 1}" scope="colgroup">{esc(group)}</th></tr>')
        for label, getter, fmt, better, kind in metrics:
            vals: dict[str, float | None] = {}
            texts: dict[str, str] = {}
            for a in arms:
                sm = run.arm_summary(a)
                if kind == "errors":
                    n_err, n = _num(sm.get("n_errors")), _num(sm.get("n"))
                    vals[a] = n_err
                    texts[a] = DASH if n_err is None else f"{f_int(n_err)} of {f_int(n)}"
                else:
                    v = _num(getter(sm))
                    vals[a] = v
                    texts[a] = fmt(v)
            present = [v for v in vals.values() if v is not None]
            best = None
            if present and better and len(present) > 1 and max(present) != min(present):
                best = min(present) if better == "low" else max(present)
            cells = []
            for a in arms:
                v = vals[a]
                cls = "best" if best is not None and v == best else ""
                delta_html = ""
                base_v = vals.get(BASELINE)
                if have_base and a != BASELINE and v is not None and base_v is not None:
                    dtxt, sign = _delta("rate" if kind == "rate" else "rel", v, base_v)
                    good = (
                        sign == "z"
                        or (better == "low" and sign == "n")
                        or (better == "high" and sign == "p")
                    )
                    tone = "flat" if sign == "z" else ("good" if good else "bad")
                    delta_html = "" if sign == "z" else f'<span class="delta {tone}">{esc(dtxt)}</span>'
                cells.append(f'<td class="num {cls}">{esc(texts[a])}{delta_html}</td>')
            rows.append(f'<tr><th scope="row">{esc(label)}</th>{"".join(cells)}</tr>')
    note = (
        f"Best value per row is outlined. Small figures under each non-baseline arm are the change versus <code>{esc(BASELINE)}</code>; green is better, red is worse."
        if have_base
        else "Best value per row is outlined. The baseline arm is not in this run, so no deltas are shown."
    )
    n_row = "".join(f'<td class="num">{f_int(run.arm_summary(a).get("n"))}</td>' for a in arms)
    body = (
        '<div class="scroll"><table class="metrics"><thead><tr><th scope="col">Metric</th>'
        f"{head}</tr></thead><tbody>"
        f'<tr class="grp"><th scope="row">Queries scored</th>{n_row}</tr>'
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return section("headline", "Headline per arm", note, lat_caveat(run) + body)


def _core_ok(q: dict[str, Any]) -> bool:
    sc = q.get("scores") or {}
    return bool(sc["core_exact"] if "core_exact" in sc else sc.get("full_exact"))


def _ok_badge(v: Any, yes: str = "pass", no: str = "fail") -> str:
    if v is True:
        return f'<span class="badge ok">{esc(yes)}</span>'
    if v is False:
        return f'<span class="badge no">{esc(no)}</span>'
    return '<span class="badge na">n/a</span>'


def f_p(v: Any) -> str:
    n = _num(v)
    if n is None:
        return DASH
    return "&lt;0.0001" if n < 0.0001 else f"{n:.4f}"


def _ids(ids: Any) -> str:
    if not isinstance(ids, list) or not ids:
        return DASH
    return '<span class="ids">' + "".join(f"<code>{esc(i)}</code>" for i in ids) + "</span>"


def _frac(a: Any, b: Any) -> str:
    x, y = _num(a), _num(b)
    return DASH if x is None or y is None else f"{x:,.0f} / {y:,.0f}"


def verdict_box(runs: list[tuple[str, Run]]) -> str:
    """Lead box: the pre-registered verdict per workload, straight from summary.json."""
    blocks = []
    for label, r in runs:
        comps = r.summary.get("comparisons") or {}
        base = r.summary.get("baseline_arm")
        items = []
        for a in sorted(comps, key=arm_sort_key):
            ni = comps[a].get("non_inferiority") or {}
            ha = ni.get("headline_a") or {}
            items.append(
                f"<li>{chip(a, r)} {_ok_badge(ni.get('non_inferior'), 'non-inferior', 'not non-inferior')} "
                f"<span class=\"mono\">Headline A {esc(_frac(ha.get('arm_pass_rows'), ha.get('rows')))} vs {esc(_frac(ha.get('baseline_pass_rows'), ha.get('rows')))} rows</span></li>"
            )
        body = (
            f"<ul class=\"vlist\">{''.join(items)}</ul>"
            if items and base
            else f"<p class=\"empty-note\">No comparison against <code>{esc(BASELINE)}</code> in this run.</p>"
        )
        n_dp = r.meta.get("n_datapoints") or len({str(q["id"]) for q in r.queries})
        blocks.append(f"<div><h3>{esc(label)} ({f_int(n_dp)} datapoints)</h3>{body}</div>")
    rule = {
        str((c.get("non_inferiority") or {}).get("rule") or "")
        for _, r in runs
        for c in (r.summary.get("comparisons") or {}).values()
    } - {""}
    rule_html = "".join(f'<p class="rule"><strong>Pre-registered rule.</strong> {esc(x)}</p>' for x in sorted(rule))
    return f'<section id="verdict-box" class="vbox"><h2>Pre-registered verdict</h2><div class="two even">{"".join(blocks)}</div>{rule_html}</section>'


def sec_verdict(run: Run) -> str:
    comps = run.summary.get("comparisons") or {}
    base = run.summary.get("baseline_arm")
    arms = sorted(comps, key=arm_sort_key)
    if not arms or not base:
        return section(
            "verdict",
            "Pre-registered non-inferiority verdict",
            "",
            f'<p class="empty-note">No comparison against <code>{esc(BASELINE)}</code> in this run (baseline arm absent), so no verdict can be computed.</p>',
        )
    rules = {str((comps[a].get("non_inferiority") or {}).get("rule") or "") for a in arms} - {""}
    rule_html = "".join(f'<p class="rule"><strong>Rule.</strong> {esc(r)}</p>' for r in sorted(rules))
    vrows = []
    srows: list[str] = []
    for a in arms:
        ni = comps[a].get("non_inferiority") or {}
        ha, ff, hl = ni.get("headline_a") or {}, ni.get("false_filter") or {}, ni.get("hallucinated") or {}
        vrows.append(
            f'<tr><th scope="row">{chip(a, run)}</th><td class="txt">{_ok_badge(ni.get("non_inferior"), "non-inferior", "not non-inferior")}</td>'
            f'<td class="num">{esc(_frac(ha.get("arm_pass_rows"), ha.get("rows")))} vs {esc(_frac(ha.get("baseline_pass_rows"), ha.get("rows")))} {_ok_badge(ha.get("ok"), "ok", "fail")}</td>'
            f'<td class="num">{esc(_frac(ff.get("arm_events"), ff.get("arm_eligible")))} vs {esc(_frac(ff.get("baseline_events"), ff.get("baseline_eligible")))} events ({f_pct(ff.get("arm_rate"), 1)} vs {f_pct(ff.get("baseline_rate"), 1)}; errored: {f_int(ff.get("arm_errored"))}) {_ok_badge(ff.get("ok"), "ok", "fail")}</td>'
            f'<td class="num">{f_int(hl.get("arm_total"))} {_ok_badge(hl.get("ok"), "ok", "fail")}</td></tr>'
        )
        se = comps[a].get("sensitivity_without_prompt_examples")
        if isinstance(se, dict):
            srows.append(
                f'<tr><th scope="row">{chip(a, run)}</th><td class="num">{f_int(se.get("rows"))}</td><td class="num">{f_int(len(se.get("excluded") or []))}</td>'
                f'<td class="num">{esc(_frac(se.get("arm_pass_rows"), se.get("rows")))} vs {esc(_frac(se.get("baseline_pass_rows"), se.get("rows")))}</td>'
                f'<td class="txt">{_ok_badge(se.get("a_ok"), "ok", "fail")}</td></tr>'
            )
    verdict = (
        '<div class="scroll"><table class="metrics"><thead><tr><th scope="col">Arm</th><th scope="col">Verdict</th>'
        f'<th scope="col">Headline A pass rows, majority vote (arm vs <code>{esc(base)}</code>)</th>'
        '<th scope="col">False filters on eligible rows (arm vs baseline)</th>'
        '<th scope="col">Hallucinated values (arm total)</th></tr></thead><tbody>'
        + "".join(vrows)
        + "</tbody></table></div>"
    )
    if srows:
        verdict += (
            "<h3>Sensitivity: Headline A without prompt-example rows</h3>"
            '<div class="scroll"><table class="metrics"><thead><tr><th scope="col">Arm</th><th scope="col">Rows</th><th scope="col">Excluded</th>'
            f'<th scope="col">Pass rows (arm vs <code>{esc(base)}</code>)</th><th scope="col">Headline A check</th></tr></thead><tbody>'
            + "".join(srows)
            + "</tbody></table></div>"
        )
    mrows, drows = [], []
    for a in arms:
        mc = comps[a].get("mcnemar_headline_a") or {}
        d = mc.get("discordant_rows") or {}
        mrows.append(
            f'<tr><th scope="row">{chip(a, run)}</th><td class="num">{f_int(mc.get("n_rows"))}</td>'
            f'<td class="num">{f_int(mc.get("baseline_only_pass"))}</td><td class="num">{f_int(mc.get("arm_only_pass"))}</td>'
            f'<td class="num">{f_int(mc.get("both_pass"))}</td><td class="num">{f_int(mc.get("both_fail"))}</td>'
            f'<td class="num">{f_p(mc.get("p_value"))}</td></tr>'
        )
        drows.append(
            f'<tr><th scope="row">{chip(a, run)}</th><td class="txt">{_ids(d.get("baseline_only"))}</td><td class="txt">{_ids(d.get("arm_only"))}</td></tr>'
        )
    mcn = (
        f"<h3>Exact McNemar vs {esc(base)} (Headline A, majority vote, paired by row)</h3>"
        '<div class="scroll"><table class="metrics"><thead><tr><th scope="col">Arm</th><th scope="col">Rows</th>'
        '<th scope="col">Baseline only passes</th><th scope="col">Arm only passes</th><th scope="col">Both pass</th><th scope="col">Both fail</th>'
        '<th scope="col">Exact p</th></tr></thead><tbody>'
        + "".join(mrows)
        + "</tbody></table></div>"
        '<div class="scroll"><table class="metrics"><thead><tr><th scope="col">Discordant rows</th>'
        '<th scope="col">Only baseline passes</th><th scope="col">Only arm passes</th></tr></thead><tbody>'
        + "".join(drows)
        + "</tbody></table></div>"
    )
    nf = run.summary.get("compiler_noise_floor")
    if isinstance(nf, dict):
        nf_html = (
            "<h3>Compiler noise floor</h3>"
            f"<p>The same compiler call decides <code>needs_judgment</code> in every arm that runs it. Across {f_int(nf.get('pairs'))} paired runs "
            f"({esc(', '.join(str(x) for x in (nf.get('arms') or [])) or DASH)}) it disagreed with itself {f_int(nf.get('disagreements'))} times "
            f"(<strong>{f_pct(nf.get('rate'), 1)}</strong>). Judgment disagreement between arms up to this rate is run-to-run noise, not an arm effect.</p>"
        )
    else:
        nf_html = '<h3>Compiler noise floor</h3><p class="empty-note">Not available: it needs both llm+jev and llm in the run.</p>'
    return section(
        "verdict",
        "Pre-registered non-inferiority verdict",
        "Written before the full run. Latency and cost are reported, not gated.",
        rule_html + verdict + mcn + nf_html,
    )


def _headline_table(run: Run, key: str) -> str:
    rows = []
    for arm in run.arms:
        headline = run.arm_summary(arm).get(key)
        if not isinstance(headline, dict):
            rows.append(
                f'<tr><th scope="row">{chip(arm, run)}</th><td class="txt" colspan="4">not measured for this run</td></tr>'
            )
            continue
        rows.append(
            f'<tr><th scope="row">{chip(arm, run)}</th><td class="num">{f_int(headline.get("rows"))}</td>'
            f'<td class="num">{esc(_frac(headline.get("pass_rows"), headline.get("rows")))}</td>'
            f'<td class="num">{f_pct(headline.get("pass_rate_majority"), 1)}</td>'
            f'<td class="num">{f_pct(headline.get("mean_pass_rate"), 1)}</td></tr>'
        )
    return (
        '<div class="scroll"><table class="metrics"><thead><tr><th scope="col">Arm</th><th scope="col">Rows</th>'
        '<th scope="col">Pass rows (majority)</th><th scope="col">Majority-vote rate</th><th scope="col">Mean pass rate</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def sec_headlines(run: Run) -> str:
    body = (
        "<h3>Headline A: facets and numeric exact</h3>"
        "<p>Over all rows except category <code>paraphrase</code>. Each query row's score aggregates its three repeats; paired mode comparisons operate on rows, not repeats as independent samples.</p>"
        + _headline_table(run, "headline_a")
        + "<h3>Headline B: facets, numeric and datetime exact (both modes)</h3>"
        "<p>Same rows, with datetime required to match for both <code>llm+jev</code> and <code>llm</code>.</p>"
        + _headline_table(run, "headline_b")
        + '<p class="legend"><span>Coverage diagnostics are relative to the catalogue actually offered to the production planner; they are not a top-k structural ceiling.</span></p>'
    )
    return section("headlines", "Headline pass rates", "", body)


DIAG: list[tuple[str, list[tuple[str, Any, Any]]]] = [
    (
        "Flexibility bucket (paraphrase rows)",
        [
            ("Rows", _g("flexibility.n"), f_int),
            ("Facet precision", _g("flexibility.facet_precision"), f_pct),
            ("Facet recall", _g("flexibility.facet_recall"), f_pct),
            ("Facet exact", _g("flexibility.facet_exact_rate"), f_pct),
        ],
    ),
    (
        "Expected values not offered by the catalogue",
        [
            ("Query runs", _g("outside_catalogue.n"), f_int),
            ("Correct", _g("outside_catalogue.correct"), f_int),
            ("Miss (no filter)", _g("outside_catalogue.miss"), f_int),
            ("Wrong value", _g("outside_catalogue.wrong_value"), f_int),
            ("Fallback to another facet", _g("outside_catalogue.fallback"), f_int),
            ("Correct, with extra filters", _g("outside_catalogue.correct_with_extra"), f_int),
        ],
    ),
    (
        "False filters (rows whose expected facets, numeric and time are all empty)",
        [
            ("Eligible query runs", _g("false_filter_eligible"), f_int),
            ("False-filter count", _g("false_filter_count"), f_int),
            ("False-filter rate", _g("false_filter_rate"), f_pct),
        ],
    ),
    (
        "Time",
        [
            ("Time accuracy on time-bearing rows", _g("time_accuracy_on_time_rows"), f_pct),
            ("Spurious time rate on other rows", _g("spurious_time_rate"), f_pct),
        ],
    ),
    (
        "Hallucinated values",
        [
            ("Total across runs", _g("hallucinated_total"), f_int),
        ],
    ),
]


def sec_diagnostics(run: Run) -> str:
    head = "".join(f'<th scope="col">{chip(a, run)}</th>' for a in run.arms)
    rows = []
    for group, metrics in DIAG:
        rows.append(f'<tr class="grp"><th colspan="{len(run.arms) + 1}" scope="colgroup">{esc(group)}</th></tr>')
        for label, getter, fmt in metrics:
            cells = "".join(f'<td class="num">{esc(fmt(getter(run.arm_summary(a))))}</td>' for a in run.arms)
            rows.append(f'<tr><th scope="row">{esc(label)}</th>{cells}</tr>')
    call_names = sorted({c for a in run.arms for c in (run.arm_summary(a).get("mean_tokens_by_call") or {})})
    crow = []
    for a in run.arms:
        for c in call_names:
            v = (run.arm_summary(a).get("mean_tokens_by_call") or {}).get(c)
            if not isinstance(v, dict):
                continue
            crow.append(
                f'<tr><th scope="row">{chip(a, run)}</th><td class="txt">{esc(c)}</td><td class="num">{f_int(v.get("n"))}</td>'
                f'<td class="num">{f_float(v.get("input"), 0)}</td><td class="num">{f_float(v.get("output"), 0)}</td>'
                f'<td class="num">{f_float(v.get("reasoning"), 1)}</td><td class="num">{f_float(v.get("cached"), 0)}</td>'
                f'<td class="num">{f_ms(v.get("latency_ms_p50"))}</td><td class="num">{f_usd(v.get("cost_usd_mean"))}</td></tr>'
            )
    per_call = (
        lat_caveat(run) + "<h3>Per call: mean tokens, p50 latency, mean cost</h3>"
        '<div class="scroll"><table class="metrics"><thead><tr><th scope="col">Arm</th><th scope="col">Call</th><th scope="col">Calls</th>'
        '<th scope="col">Input tok</th><th scope="col">Output tok</th><th scope="col">Reasoning tok</th><th scope="col">Cached tok</th>'
        '<th scope="col">p50 latency</th><th scope="col">Mean $ / call</th></tr></thead><tbody>'
        + "".join(crow)
        + "</tbody></table></div>"
        if crow
        else ""
    )
    body = (
        '<div class="scroll"><table class="metrics"><thead><tr><th scope="col">Metric</th>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>" + per_call
    )
    return section(
        "diagnostics",
        "Diagnostics",
        "Reported separately from the headlines. A dash means the arm does not produce that output or no row qualifies (for the flexibility bucket, a dash precision means no facet was emitted).",
        body,
    )


def sec_glance(run: Run) -> str:
    """Summary above the detail: change vs baseline for the three decision metrics."""
    if BASELINE not in run.arms:
        return ""
    base = run.arm_summary(BASELINE)
    cards: list[str] = []
    for arm in run.arms:
        if arm == BASELINE:
            continue
        sm = run.arm_summary(arm)
        parts = []
        for label, key, kind, better, fmt in (
            ("Headline A mean pass rate", "headline_a.mean_pass_rate", "rate", "high", f_pct),
            ("Measured warm-cache e2e p50", "warm_e2e_latency_ms_p50", "rel", "low", f_ms),
            ("Reported $ / query", "cost_per_query_reported_usd", "rel", "low", f_usd),
        ):
            v, b = _num(_g(key)(sm)), _num(_g(key)(base))
            if v is None or b is None:
                parts.append(
                    f'<div class="stat"><span class="k">{esc(label)}</span><span class="v">{esc(fmt(v))}</span><span class="d flat">no baseline value</span></div>'
                )
                continue
            dtxt, sign = _delta(kind, v, b)
            tone = "flat" if sign == "z" else ("good" if (better == "low") == (sign == "n") else "bad")
            parts.append(
                f'<div class="stat"><span class="k">{esc(label)}</span><span class="v">{esc(fmt(v))}</span>'
                f'<span class="d {tone}">{esc(dtxt)} vs {esc(BASELINE)} ({esc(fmt(b))})</span></div>'
            )
        cards.append(f'<article class="card">{chip(arm, run)}<div class="stats">{"".join(parts)}</div></article>')
    if not cards:
        return ""
    return lat_caveat(run) + f'<div class="glance" aria-label="Change versus baseline">{"".join(cards)}</div>'


def _heat_cell(v: Any, n: Any, invert: bool = False) -> str:
    f = _num(v)
    if f is None:
        return '<td class="hm empty">—</td>'
    shade = (1 - f) if invert else f
    shade = min(max(shade, 0.0), 1.0)
    n_txt = f'<small>n={f_int(n)}</small>' if _num(n) is not None else ""
    return f'<td class="hm" style="--v:{shade:.3f}">{f_pct(f)}{n_txt}</td>'


    return sum(1 for q in run.queries_for(arm) if str(q.get("category")) == cat)


def _matrix(run: Run, key: str, invert: bool = False, nkey: str | None = None) -> str:
    cats = run.categories
    head = "".join(f'<th scope="col"><span>{esc(c)}</span></th>' for c in cats)
    rows = []
    for arm in run.arms:
        bc = run.arm_summary(arm).get("by_category") or {}
        cells = []
        for c in cats:
            cm = bc.get(c) or {}
            cells.append(_heat_cell(_g(key)(cm), (_g(nkey)(cm) if nkey else cm.get("n")) if cm else _category_n(run, arm, c) or None, invert))
        overall = _heat_cell(_g(key)(run.arm_summary(arm)), (_g(nkey)(run.arm_summary(arm)) if nkey else run.arm_summary(arm).get("n")), invert)
        rows.append(f'<tr><th scope="row">{chip(arm, run)}</th>{overall}{"".join(cells)}</tr>')
    return (
        '<div class="scroll"><table class="heat"><thead><tr><th></th><th scope="col"><span>all</span></th>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def sec_matrix(run: Run) -> str:
    extras = [
        ("Facet precision", "facet_precision", False),
        ("Facet recall", "facet_recall", False),
        ("Numeric accuracy", "numeric_exact_rate", False),
        ("Time accuracy on time-bearing rows", "time_accuracy_on_time_rows", False),
        ("False-filter rate (lower is better)", "false_filter_rate", True),
    ]
    more = "".join(
        f'<details class="more"><summary>{esc(lbl)}</summary>{_matrix(run, key, inv)}</details>'
        for lbl, key, inv in extras
    )
    body = _matrix(run, "headline_a.mean_pass_rate", nkey="headline_a.rows") + more
    return section(
        "matrix",
        "Accuracy by category",
        "Headline A mean pass rate (facets and numeric correct; time excluded) per arm and category, over repeats. Paraphrase rows are the flexibility bucket and sit outside the headline. Each cell shows the rate and the number of rows behind it; empty cells mean no headline rows. The folded views below use the per-query metrics over all rows.",
        body,
    )


# ---------------------------------- SVG ------------------------------------ #
def _svg_open(w: int, h: int, label: str) -> str:
    return f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" preserveAspectRatio="xMidYMid meet">'


def chart_latency(run: Run) -> str:
    lanes: list[tuple[str, list[float]]] = []
    for arm in run.arms:
        vals = sorted(v for v in (_num(q.get("wall_ms")) for q in run.queries_for(arm)) if v is not None)
        if vals:
            lanes.append((arm, vals))
    if not lanes:
        return '<p class="empty-note">No latency samples in this run.</p>'
    w, left, right, top, lane_h, bottom = 820, 110, 24, 14, 64, 36
    h = top + lane_h * len(lanes) + bottom
    xmax = max(v[-1] for _, v in lanes) * 1.04
    xmax = xmax if xmax > 0 else 1.0
    px = lambda x: left + (x / xmax) * (w - left - right)  # noqa: E731
    parts = [_svg_open(w, h, "Critical-path latency per arm")]
    for t in nice_ticks(0, xmax, 7):
        x = px(t)
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{h - bottom}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{h - bottom + 16}" text-anchor="middle">{esc(f_ms(t))}</text>')
    for idx, (arm, vals) in enumerate(lanes):
        cy = top + lane_h * idx + lane_h / 2
        cls = run.cls[arm]
        sm = run.arm_summary(arm)
        q1, q3 = percentile(vals, 0.25), percentile(vals, 0.75)
        p50 = _num(sm.get("latency_ms_p50"))
        p95 = _num(sm.get("latency_ms_p95"))
        if p50 is None:
            p50 = percentile(vals, 0.5)
        if p95 is None:
            p95 = percentile(vals, 0.95)
        parts.append(f'<text class="lane-label {cls}" x="{left - 12}" y="{cy + 4:.1f}" text-anchor="end">{esc(arm)}</text>')
        parts.append(f'<line class="whisker {cls}" x1="{px(vals[0]):.1f}" x2="{px(vals[-1]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>')
        parts.append(f'<rect class="box {cls}" x="{px(q1):.1f}" y="{cy - 11:.1f}" width="{max(px(q3) - px(q1), 1.5):.1f}" height="22" rx="2"/>')
        # deterministic jitter so the picture is reproducible
        for i, v in enumerate(vals):
            jitter = (((i * 2654435761) % 1000) / 1000 - 0.5) * 30
            parts.append(f'<circle class="dot {cls}" cx="{px(v):.1f}" cy="{cy + jitter:.1f}" r="2.4"><title>{esc(arm)} {esc(f_ms(v))}</title></circle>')
        parts.append(f'<line class="median" x1="{px(p50):.1f}" x2="{px(p50):.1f}" y1="{cy - 16:.1f}" y2="{cy + 16:.1f}"/>')
        parts.append(f'<path class="p95 {cls}" d="M{px(p95):.1f} {cy - 8:.1f} l6 8 l-6 8 l-6 -8 z"><title>p95 {esc(f_ms(p95))}</title></path>')
        parts.append(f'<text class="value" x="{px(p50):.1f}" y="{cy - 20:.1f}" text-anchor="middle">p50 {esc(f_ms(p50))}</text>')
    parts.append("</svg>")
    legend = (
        '<p class="legend"><span><i class="lg dot"></i>one scored query run</span>'
        '<span><i class="lg box"></i>interquartile range</span>'
        '<span><i class="lg med"></i>median (p50)</span>'
        '<span><i class="lg dia"></i>p95</span></p>'
    )
    return "".join(parts) + legend


def _scatter(run: Run, xkey: str, xfmt, xlabel: str, title: str) -> str:
    pts = []
    for arm in run.arms:
        sm = run.arm_summary(arm)
        x, y = _num(sm.get(xkey)), _num((sm.get("headline_a") or {}).get("mean_pass_rate"))
        if x is not None and y is not None:
            pts.append((arm, x, y))
    if not pts:
        return f'<div class="panel"><h3>{esc(title)}</h3><p class="empty-note">No arm has both values.</p></div>'
    w, h, left, right, top, bottom = 400, 280, 52, 20, 16, 44
    xs = [p[1] for p in pts]
    xmax = max(xs) * 1.2 if max(xs) > 0 else 1.0
    px = lambda x: left + (x / xmax) * (w - left - right)  # noqa: E731
    py = lambda y: top + (1 - y) * (h - top - bottom)  # noqa: E731
    parts = [_svg_open(w, h, title)]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{py(t):.1f}" y2="{py(t):.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{py(t) + 4:.1f}" text-anchor="end">{t * 100:.0f}%</text>')
    for t in nice_ticks(0, xmax, 4):
        parts.append(f'<line class="grid v" x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{top}" y2="{h - bottom}"/>')
        parts.append(f'<text class="tick" x="{px(t):.1f}" y="{h - bottom + 16}" text-anchor="middle">{esc(xfmt(t))}</text>')
    parts.append(f'<text class="axis" x="{(left + w - right) / 2:.1f}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    parts.append(f'<text class="axis" transform="translate(12 {(top + h - bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">full exact match</text>')
    for arm, x, y in pts:
        cls = run.cls[arm]
        cx, cy = px(x), py(y)
        anchor = "end" if cx > w - 90 else "start"
        dx = -10 if anchor == "end" else 10
        parts.append(f'<circle class="pt {cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="7"><title>{esc(arm)}: {esc(xfmt(x))}, {esc(f_pct(y))}</title></circle>')
        parts.append(f'<text class="pt-label {cls}" x="{cx + dx:.1f}" y="{cy - 12:.1f}" text-anchor="{anchor}">{esc(arm)}</text>')
    parts.append("</svg>")
    return f'<div class="panel"><h3>{esc(title)}</h3>{"".join(parts)}</div>'


def sec_latency(run: Run) -> str:
    return section(
        "latency",
        "Latency distribution",
        "Critical-path latency is measured from each scored query against an injected fixed catalogue snapshot. Cold/warm end-to-end scenarios and cache hit/miss timings use the live production cache; each cold/warm scenario has n=1 and is not a latency distribution or part of scored-workload latency.",
        lat_caveat(run) + f'<div class="panel">{chart_latency(run)}</div>',
    )


def sec_scatter(run: Run) -> str:
    cost = _scatter(run, "cost_per_query_reported_usd", f_usd, "Reported $ per query", "Cost vs quality")
    return section(
        "tradeoff",
        "Cost vs quality",
        "Measured per-query cost only; no modeled end-to-end latency is inferred from a catalogue sample.",
        lat_caveat(run) + f'<div class="panel">{cost}</div>',
    )


# ------------------------------ per datapoint ------------------------------ #
def _fmt_facets_chips(actual: dict[str, list[str]], expected: dict[str, list[str]], hallucinated: set[tuple[str, str]]) -> str:
    out: list[str] = []
    for facet in sorted({*actual, *expected}):
        got = set(actual.get(facet, []))
        want = set(expected.get(facet, []))
        for v in sorted(got | want):
            if v in got and v in want:
                cls, mark = "ok", ""
            elif v in got:
                cls, mark = ("hall" if (facet, v) in hallucinated else "extra"), "+"
            else:
                cls, mark = "miss", "−"
            out.append(f'<span class="fv {cls}"><b>{esc(facet)}</b>{esc(v)}{mark and f"<sup>{mark}</sup>"}</span>')
    return "".join(out)


def _fmt_numeric(num: dict[str, Any]) -> str:
    return ", ".join(f"{esc(k)} = {esc(f_int(v) if _num(v) is not None else v)}" for k, v in sorted(num.items()))


def _fmt_time(t: Any) -> str:
    if not t:
        return "none"
    return f"{esc(t.get('start') or 'open')} → {esc(t.get('end') or 'now')}"


def _expected_html(exp: dict[str, Any]) -> str:
    parts = []
    facets = exp.get("facets") or {}
    if facets:
        parts.append("".join(f'<span class="fv"><b>{esc(f)}</b>{esc(v)}</span>' for f, vs in sorted(facets.items()) for v in vs))
    if exp.get("numeric"):
        parts.append(f'<div class="kvline">numeric: {_fmt_numeric(exp["numeric"])}</div>')
    parts.append(f'<div class="kvline">time: {_fmt_time(exp.get("time"))}</div>')
    flags = []
    if exp.get("needs_judgment"):
        flags.append("needs judgment")
    if exp.get("aggregate"):
        flags.append("aggregate")
    if flags:
        parts.append(f'<div class="kvline">{esc(", ".join(flags))}</div>')
    if not facets and not exp.get("numeric"):
        parts.insert(0, '<span class="muted">no facet or numeric filters</span>')
    return "".join(parts)


def _actual_html(q: dict[str, Any]) -> str:
    a = q.get("actual") or {}
    sc = q.get("scores") or {}
    exp = q.get("expected") or {}
    hall = {(str(f), str(v)) for f, v in (a.get("hallucinated") or [])}
    parts = []
    chips = _fmt_facets_chips(a.get("facets") or {}, exp.get("facets") or {}, hall)
    parts.append(chips or '<span class="muted">no facets</span>')
    num = a.get("numeric") or {}
    ncls = "ok" if sc.get("numeric_exact") else "bad"
    parts.append(f'<div class="kvline {ncls}">numeric: {_fmt_numeric(num) if num else "none"}</div>')
    tcls = "ok" if sc.get("time_match") else "bad"
    time_txt = _fmt_time(a.get("time"))
    calendar_note = ""
    parts.append(f'<div class="kvline {tcls}">time: {time_txt}{calendar_note}</div>')
    if a.get("dimensions") is not None:
        parts.append(f'<div class="kvline muted">dimensions: {esc(f_int(a.get("dimensions")))}</div>')
    if q.get("errors"):
        parts.append(f'<div class="kvline bad">error: {esc("; ".join(str(e) for e in q["errors"])[:300])}</div>')
    return "".join(parts)
def _call_attempts_html(run: Run, query: dict[str, Any]) -> str:
    attempts = [c for c in run.calls if c.get("arm") == query.get("arm") and c.get("id") == query.get("id") and c.get("repeat") == query.get("repeat")]
    if not attempts:
        return ""
    items = []
    for c in sorted(attempts, key=lambda row: (str(row.get("call")), row.get("call_attempt", 0))):
        structured = c.get("structured_output")
        calendar_day = structured.get("calendar_day") if isinstance(structured, dict) else None
        items.append(
            f'<div class="kvline"><b>{esc(c.get("call") or "call")}</b> attempt {f_int(c.get("call_attempt"))}'
            f'{" · retry" if c.get("is_retry") else ""}; {esc(c.get("model_served") or c.get("model_requested") or DASH)}'
            f'; {f_ms(c.get("latency_ms"))}; {f_int(c.get("input_tokens"))}/{f_int(c.get("output_tokens"))} input/output tokens'
            f'; {f_usd(c.get("total_cost"))} ({esc(c.get("cost_state") or DASH)})'
            f'; calendar_day {esc(calendar_day if calendar_day is not None else DASH)}'
            + (f'; error {esc(c["error"])}' if c.get("error") else "") + "</div>"
        )
    return '<details class="more attempts"><summary>Recorded call attempts (including retries/errors)</summary>' + "".join(items) + "</details>"


def sec_datapoints(run: Run) -> str:
    by_id: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    meta_by_id: dict[str, dict[str, Any]] = {}
    for q in run.queries:
        i = str(q["id"])
        by_id[i][str(q["arm"])].append(q)
        meta_by_id.setdefault(i, q)
    if not by_id:
        return section("datapoints", "Per-datapoint results", "", '<p class="empty-note">No query rows in raw.jsonl.</p>')

    def key(i: str):
        m = re.match(r"([A-Za-z]*)(\d+)$", i)
        return (m.group(1), int(m.group(2))) if m else (i, 0)

    items = []
    for i in sorted(by_id, key=key):
        first = meta_by_id[i]
        ds = run.dataset.get(i) or {}
        query = ds.get("query") or first.get("query") or "(query text unavailable)"
        cat = str(first.get("category") or ds.get("category") or "")
        pills = []
        outcomes = []
        for arm in run.arms:
            qs = by_id[i].get(arm, [])
            if not qs:
                pills.append(f'<span class="pill na" title="{esc(arm)}: not run">{DASH}</span>')
                continue
            ok = sum(1 for q in qs if _core_ok(q))
            outcomes.append(ok == len(qs))
            tone = "pass" if ok == len(qs) else ("fail" if ok == 0 else "mixed")
            word = "pass" if tone == "pass" else ("fail" if tone == "fail" else "mixed")
            pills.append(f'<span class="pill {tone} {run.cls[arm]}" title="{esc(arm)}: {ok} of {len(qs)} repeats exact"><i></i>{word} {ok}/{len(qs)}</span>')
        split = len(set(outcomes)) > 1
        exp = first.get("expected") or ds.get("expected") or {}
        detail_arms = []
        for arm in run.arms:
            qs = by_id[i].get(arm, [])
            if not qs:
                continue
            groups: list[tuple[dict[str, Any], int, bool]] = []
            seen: dict[str, int] = {}
            for q in sorted(qs, key=lambda r: r.get("repeat", 0)):
                sig = json.dumps([q.get("actual"), q.get("errors")], sort_keys=True)
                if sig in seen:
                    g = groups[seen[sig]]
                    groups[seen[sig]] = (g[0], g[1] + 1, g[2])
                else:
                    seen[sig] = len(groups)
                    groups.append((q, 1, _core_ok(q)))
            blocks = []
            for q, cnt, ok in groups:
                rep = f'<span class="rep">×{cnt}</span>' if cnt > 1 else ""
                blocks.append(
                    f'<div class="out {"ok" if ok else "bad"}"><div class="out-h">{"exact" if ok else "not exact"} {rep}'
                    f'<span class="lat">{esc(f_ms(q.get("wall_ms")))}</span></div>{_actual_html(q)}{_call_attempts_html(run, q)}</div>'
                )

            detail_arms.append(f'<div class="arm-col"><h4>{chip(arm, run)}</h4>{"".join(blocks)}</div>')
        notes = ds.get("notes")
        items.append(
            f'<details class="dp{" split" if split else ""}"><summary>'
            f'<span class="dp-id">{esc(i)}</span>'
            f'<span class="dp-q">{esc(query)}</span>'
            f'<span class="dp-cat">{esc(cat)}</span>'
            f'<span class="dp-pills">{"".join(pills)}</span></summary>'
            f'<div class="dp-body"><div class="exp"><h4>Expected</h4>{_expected_html(exp)}'
            + (f'<p class="notes">{esc(notes)}</p>' if notes else "")
            + f'</div><div class="arm-cols">{"".join(detail_arms)}</div></div></details>'
        )
    n_split = sum(1 for it in items if 'class="dp split"' in it)
    legend = (
        '<p class="legend"><span><i class="fv-lg ok"></i>expected and returned</span>'
        '<span><i class="fv-lg extra"></i>returned, not expected (+)</span>'
        '<span><i class="fv-lg hall"></i>not in catalogue (+)</span>'
        '<span><i class="fv-lg miss"></i>expected, not returned (−)</span></p>'
    )
    lede = (
        f"{len(items)} datapoints. A pill is the share of repeats where facets and numeric are exact (Headline A); datetime is also compared for both modes under Headline B. Rows with a dashed edge are ones where arms disagree "
        f"({n_split} here). Open a row to compare expected and returned filters."
    )
    return section("datapoints", "Per-datapoint results", lede, legend + f'<div class="dps">{"".join(items)}</div>')


def sec_analysis(run: Run, analysis_path: Path | None) -> str:
    if analysis_path is not None and analysis_path.is_file():
        body = f'<div class="prose">{markdown_to_html(analysis_path.read_text(encoding="utf-8"))}</div>'
    else:
        want = f" Pass <code>--analysis {esc(analysis_path)}</code> pointing to an existing file." if analysis_path else " Re-render with <code>--analysis analysis.md</code>."
        body = (
            '<div class="placeholder" role="note"><strong>Analysis not written yet.</strong>'
            f" This section renders the markdown verdict that interprets the numbers above.{want}</div>"
        )
    return section("analysis", "Analysis & verdict", "", body)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
CSS = """
:root{
  --ground:#f3f5f8; --surface:#ffffff; --surface-2:#eef1f6; --ink:#12161d; --ink-2:#4a5565; --ink-3:#7a8597;
  --rule:#dbe0e8; --rule-2:#c6cdd8; --focus:#1f5fd1;
  --a0:#566a85; --a1:#0b8a8f; --a2:#c0561b; --a3:#7a4ec2; --a4:#9a7b12; --a5:#a33a68;
  --good:#17804f; --bad:#c23a2e; --warn:#a8710a;
  --good-bg:#dcf1e5; --bad-bg:#f8dfdb; --warn-bg:#f6ead0;
  --hm-lo:#f6d9d2; --hm-hi:#bde3cb;
  --shadow:0 1px 0 rgba(18,22,29,.04);
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme: dark){
  :root{
    --ground:#0f131a; --surface:#171c25; --surface-2:#1e2430; --ink:#e7ebf2; --ink-2:#a9b3c4; --ink-3:#7e8a9e;
    --rule:#2a3241; --rule-2:#3a4457; --focus:#7aa7ff;
    --a0:#8da2c0; --a1:#35c0c5; --a2:#ee8a50; --a3:#b794f0; --a4:#d8b54a; --a5:#e07aa6;
    --good:#4cc58a; --bad:#f07b6f; --warn:#e0b04a;
    --good-bg:#16382a; --bad-bg:#432220; --warn-bg:#3d3017;
    --hm-lo:#4a2a27; --hm-hi:#1f5a40;
    --shadow:none;
  }
}
:root[data-theme="dark"]{
  --ground:#0f131a; --surface:#171c25; --surface-2:#1e2430; --ink:#e7ebf2; --ink-2:#a9b3c4; --ink-3:#7e8a9e;
  --rule:#2a3241; --rule-2:#3a4457; --focus:#7aa7ff;
  --a0:#8da2c0; --a1:#35c0c5; --a2:#ee8a50; --a3:#b794f0; --a4:#d8b54a; --a5:#e07aa6;
  --good:#4cc58a; --bad:#f07b6f; --warn:#e0b04a;
  --good-bg:#16382a; --bad-bg:#432220; --warn-bg:#3d3017;
  --hm-lo:#4a2a27; --hm-hi:#1f5a40; --shadow:none;
}
:root[data-theme="light"]{
  --ground:#f3f5f8; --surface:#ffffff; --surface-2:#eef1f6; --ink:#12161d; --ink-2:#4a5565; --ink-3:#7a8597;
  --rule:#dbe0e8; --rule-2:#c6cdd8; --focus:#1f5fd1;
  --a0:#566a85; --a1:#0b8a8f; --a2:#c0561b; --a3:#7a4ec2; --a4:#9a7b12; --a5:#a33a68;
  --good:#17804f; --bad:#c23a2e; --warn:#a8710a;
  --good-bg:#dcf1e5; --bad-bg:#f8dfdb; --warn-bg:#f6ead0;
  --hm-lo:#f6d9d2; --hm-hi:#bde3cb; --shadow:0 1px 0 rgba(18,22,29,.04);
}
*{box-sizing:border-box}
html{color-scheme:light dark;background:var(--ground)}
:root[data-theme="dark"]{color-scheme:dark}
:root[data-theme="light"]{color-scheme:light}
body{margin:0;background:var(--ground);color:var(--ink);font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--focus)}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px;border-radius:3px}
code,.mono{font-family:var(--mono);font-size:.88em;background:var(--surface-2);padding:.08em .35em;border-radius:3px}
pre{background:var(--surface-2);padding:12px 14px;border-radius:6px;overflow-x:auto}
pre code{background:none;padding:0}
.page{max-width:1160px;margin:0 auto;padding:40px 28px 96px;display:flex;flex-direction:column;gap:56px}
header.top{display:flex;flex-direction:column;gap:14px;border-bottom:1px solid var(--rule);padding-bottom:28px}
.eyebrow{font:600 12px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
h1{font-size:clamp(26px,4vw,38px);line-height:1.12;margin:0;letter-spacing:-.02em;font-weight:700;text-wrap:balance;max-width:24ch}
.sub{margin:0;color:var(--ink-2);max-width:68ch}
nav.toc{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px}
nav.toc a{color:var(--ink-2);text-decoration:none;border-bottom:1px solid var(--rule-2)}
nav.toc a:hover{color:var(--ink);border-color:var(--ink)}
.banner{background:var(--warn-bg);color:var(--ink);border:1px solid var(--warn);border-radius:6px;padding:10px 14px;font-size:14px;max-width:80ch}
section{display:flex;flex-direction:column;gap:18px;scroll-margin-top:16px}
h2{font-size:22px;margin:0;letter-spacing:-.01em;text-wrap:balance}
h3{font-size:12px;margin:0 0 6px;font:600 12px/1.2 var(--mono);letter-spacing:.07em;text-transform:uppercase;color:var(--ink-3)}
h4{margin:0 0 8px;font-size:13px}
p{margin:0}
.lede{color:var(--ink-2);max-width:72ch}
.two{display:grid;grid-template-columns:1fr 1fr;gap:28px}
.two.wide-left{grid-template-columns:minmax(0,1.15fr) minmax(0,1fr)}
.two.even{grid-template-columns:1fr 1fr}
@media (max-width:860px){.two,.two.wide-left,.two.even{grid-template-columns:1fr}}
.two p{max-width:62ch}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--surface);box-shadow:var(--shadow)}
.panel{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:16px 18px;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:10px;min-width:0}
.panel h3{margin:0}
/* chips + arms */
.chip{display:inline-flex;align-items:center;gap:7px;font:600 13px/1 var(--mono);white-space:nowrap}
.chip i{width:10px;height:10px;border-radius:50%;background:var(--c,var(--a5));display:inline-block}
.a0{--c:var(--a0)}.a1{--c:var(--a1)}.a2{--c:var(--a2)}.a3{--c:var(--a3)}.a4{--c:var(--a4)}.a5{--c:var(--a5)}
.wl-head{display:flex;flex-direction:column;gap:6px;border-top:3px solid var(--ink);padding-top:18px}
.wl-title{font-size:30px;letter-spacing:-.02em}
.vbox{border:1px solid var(--rule-2);border-radius:8px;background:var(--surface-2);padding:18px 20px}
.vlist{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.vlist li{display:flex;flex-wrap:wrap;align-items:center;gap:10px}
.workload{display:flex;flex-direction:column;gap:56px}
.badge{display:inline-block;font:700 12px/1 var(--mono);padding:4px 8px;border-radius:999px}
.badge.ok{background:var(--good-bg);color:var(--good)}
.badge.no{background:var(--bad-bg);color:var(--bad)}
.badge.na{background:var(--surface-2);color:var(--ink-3)}
.rule{margin:0;padding:10px 14px;border-left:3px solid var(--rule-2);background:var(--surface-2);color:var(--ink-2);max-width:96ch}
.ids{display:flex;flex-wrap:wrap;gap:4px;margin:2px 0}
.ids code{font-size:12px}
table.metrics td.txt{text-align:left;white-space:normal;font-family:inherit}
dl.arms{display:grid;grid-template-columns:max-content 1fr;gap:10px 18px;margin:0;padding-top:4px}
dl.arms dt{padding-top:2px}
dl.arms dd{margin:0;color:var(--ink-2);max-width:78ch}
dl.kv{display:grid;grid-template-columns:max-content 1fr;gap:7px 20px;margin:0;font-size:14px}
dl.kv dt{color:var(--ink-3)}
dl.kv dd{margin:0;word-break:break-word}
ul.caveats{margin:0;padding-left:18px;display:flex;flex-direction:column;gap:7px;color:var(--ink-2);font-size:14px}
/* at a glance */
.glance{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:16px 18px;display:flex;flex-direction:column;gap:12px;box-shadow:var(--shadow)}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.stat{display:flex;flex-direction:column;gap:2px;min-width:0}
.stat .k{font-size:12px;color:var(--ink-3)}
.stat .v{font:600 20px/1.2 var(--mono);font-variant-numeric:tabular-nums}
.d{font-size:12px}
.d.good,.delta.good{color:var(--good)}.d.bad,.delta.bad{color:var(--bad)}.d.flat,.delta.flat{color:var(--ink-3)}
/* tables */
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
table.metrics th,table.metrics td{padding:8px 14px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
table.metrics thead th{position:sticky;top:0;background:var(--surface-2);font-size:13px}
table.metrics tbody th[scope=row]{font-weight:500;color:var(--ink-2);font-size:14px}
table.metrics td.num{text-align:right;font-family:var(--mono);font-size:14px;white-space:nowrap}
table.metrics thead th:not(:first-child){text-align:right}
table.metrics tr.grp th{background:var(--surface-2);font:600 11px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);padding:9px 14px}
table.metrics tr.grp:first-child th{border-top:0}
td.best{font-weight:700;box-shadow:inset 0 0 0 1.5px var(--good);border-radius:3px}
.delta{display:block;font:500 11px/1.3 var(--mono)}
table.md{font-size:14px}
table.md th,table.md td{padding:7px 12px;border-bottom:1px solid var(--rule)}
table.md th{background:var(--surface-2);text-align:left}
table.md .right{text-align:right}
/* heatmap */
table.heat{border-collapse:separate;border-spacing:3px;padding:8px;width:auto;min-width:100%}
table.heat th{font-weight:500;font-size:12px;color:var(--ink-2);padding:4px 6px}
table.heat thead th{vertical-align:bottom;text-align:center}
table.heat thead th span{display:inline-block;writing-mode:vertical-rl;transform:rotate(180deg);white-space:nowrap;font-family:var(--mono);height:7.5em}
table.heat tbody th{text-align:left;padding-right:12px;white-space:nowrap}
td.hm{text-align:center;font:600 13px/1.25 var(--mono);padding:9px 6px;min-width:62px;border-radius:4px;color:var(--ink);
  background:color-mix(in srgb,var(--hm-hi) calc(var(--v) * 100%),var(--hm-lo))}
td.hm small{display:block;font:400 10px/1.2 var(--mono);color:var(--ink-2);margin-top:2px}
td.hm.empty{background:var(--surface-2);color:var(--ink-3);font-weight:400}
details.more{border:1px solid var(--rule);border-radius:8px;background:var(--surface)}
details.more>summary{cursor:pointer;padding:9px 14px;font-size:13px;color:var(--ink-2)}
details.more[open]>summary{border-bottom:1px solid var(--rule)}
details.more .scroll{border:0;box-shadow:none}
/* svg */
svg.chart{width:100%;height:auto;display:block;font-family:var(--mono)}
svg .grid{stroke:var(--rule);stroke-width:1}
svg .grid.v{stroke-dasharray:2 4}
svg .tick{fill:var(--ink-3);font-size:11px}
svg .axis{fill:var(--ink-2);font-size:11px}
svg .lane-label{font-size:13px;font-weight:600;fill:var(--c)}
svg .whisker{stroke:var(--c);stroke-width:1.5;opacity:.5}
svg .box{fill:var(--c);fill-opacity:.16;stroke:var(--c);stroke-width:1.5}
svg .dot{fill:var(--c);fill-opacity:.55}
svg .median{stroke:var(--ink);stroke-width:2.5}
svg .p95{fill:var(--surface);stroke:var(--c);stroke-width:2}
svg .value{fill:var(--ink-2);font-size:11px}
svg .pt{fill:var(--c);fill-opacity:.85;stroke:var(--surface);stroke-width:2}
svg .pt-label{fill:var(--c);font-size:12px;font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.lg{display:inline-block}
.lg.dot{width:7px;height:7px;border-radius:50%;background:var(--ink-3)}
.lg.box{width:16px;height:9px;border:1.5px solid var(--ink-3);background:color-mix(in srgb,var(--ink-3) 18%,transparent)}
.lg.med{width:3px;height:14px;background:var(--ink)}
.lg.dia{width:9px;height:9px;border:2px solid var(--ink-3);transform:rotate(45deg)}
.empty-note,.muted{color:var(--ink-3)}
/* datapoints */
.dps{display:flex;flex-direction:column;border:1px solid var(--rule);border-radius:8px;background:var(--surface);overflow:hidden}
details.dp{border-bottom:1px solid var(--rule)}
details.dp:last-child{border-bottom:0}
details.dp.split>summary{box-shadow:inset 3px 0 0 var(--warn)}
details.dp>summary{display:grid;grid-template-columns:44px minmax(0,1fr) 110px auto;gap:14px;align-items:center;padding:10px 14px;cursor:pointer;list-style:none}
details.dp>summary::-webkit-details-marker{display:none}
details.dp>summary:hover{background:var(--surface-2)}
details.dp[open]>summary{background:var(--surface-2)}
.dp-id{font:600 12px/1 var(--mono);color:var(--ink-3)}
.dp-q{font-size:14px;overflow-wrap:anywhere}
.dp-cat{font:12px/1.2 var(--mono);color:var(--ink-3)}
.dp-pills{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
@media (max-width:760px){details.dp>summary{grid-template-columns:36px 1fr}.dp-cat,.dp-pills{grid-column:2;justify-content:flex-start}}
.pill{display:inline-flex;align-items:center;gap:5px;font:600 11px/1 var(--mono);padding:4px 7px;border-radius:999px;white-space:nowrap}
.pill i{width:7px;height:7px;border-radius:50%;background:var(--c)}
.pill.pass{background:var(--good-bg);color:var(--good)}
.pill.fail{background:var(--bad-bg);color:var(--bad)}
.pill.mixed{background:var(--warn-bg);color:var(--warn)}
.pill.na{background:var(--surface-2);color:var(--ink-3)}
.dp-body{display:grid;grid-template-columns:minmax(220px,.8fr) minmax(0,2.2fr);gap:22px;padding:14px 18px 18px;border-top:1px dashed var(--rule-2)}
@media (max-width:860px){.dp-body{grid-template-columns:1fr}}
.arm-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}
.out{border-left:3px solid var(--rule-2);padding:2px 0 2px 10px;margin-bottom:10px;display:flex;flex-direction:column;gap:4px}
.out.ok{border-color:var(--good)}.out.bad{border-color:var(--bad)}
.out-h{font:600 11px/1.2 var(--mono);color:var(--ink-2);display:flex;gap:8px;align-items:center}
.out-h .lat{margin-left:auto;font-weight:400;color:var(--ink-3)}
.rep{color:var(--ink-3)}
.fv{display:inline-flex;gap:6px;align-items:baseline;font:12px/1.3 var(--mono);border:1px solid var(--rule-2);border-radius:4px;padding:2px 6px;margin:0 5px 5px 0;background:var(--surface)}
.fv b{font-weight:500;color:var(--ink-3)}
.fv.ok{border-color:var(--good);background:var(--good-bg)}
.fv.extra{border-color:var(--bad);background:var(--bad-bg)}
.fv.hall{border:1px solid var(--bad);background:var(--bad-bg);text-decoration:underline wavy var(--bad)}
.fv.miss{border-style:dashed;border-color:var(--warn);color:var(--ink-2);text-decoration:line-through}
.fv sup{font-weight:700}
.fv-lg{display:inline-block;width:16px;height:10px;border-radius:3px;border:1.5px solid var(--rule-2)}
.fv-lg.ok{border-color:var(--good);background:var(--good-bg)}.fv-lg.extra,.fv-lg.hall{border-color:var(--bad);background:var(--bad-bg)}
.fv-lg.miss{border-style:dashed;border-color:var(--warn)}
.kvline{font:12px/1.4 var(--mono);color:var(--ink-2);overflow-wrap:anywhere}
.kvline.ok{color:var(--good)}.kvline.bad{color:var(--bad)}
.notes{font-size:12px;color:var(--ink-3);margin-top:8px;max-width:48ch}
/* analysis */
.prose{max-width:78ch;display:flex;flex-direction:column;gap:12px}
.prose h3{font:700 17px/1.3 var(--sans);letter-spacing:0;text-transform:none;color:var(--ink);margin:10px 0 0}
.prose h4,.prose h5,.prose h6{margin:6px 0 0;font-size:15px}
.prose ul,.prose ol{margin:0;padding-left:22px;display:flex;flex-direction:column;gap:4px}
.prose blockquote{margin:0;padding:2px 14px;border-left:3px solid var(--rule-2);color:var(--ink-2)}
.prose hr{border:0;border-top:1px solid var(--rule);width:100%}
.placeholder{border:1.5px dashed var(--rule-2);border-radius:8px;padding:18px 20px;color:var(--ink-2);background:var(--surface);max-width:78ch}
footer{color:var(--ink-3);font-size:12px;border-top:1px solid var(--rule);padding-top:16px}
@media (prefers-reduced-motion:no-preference){details.dp>summary{transition:background .12s}}
@media print{body{background:#fff}details.dp:not([open])>.dp-body{display:none}}
"""

NAV = [
    ("hypothesis", "Hypothesis"),
    ("setup", "Setup"),
    ("verdict", "Non-inferiority"),
    ("headlines", "Headlines A/B"),
    ("diagnostics", "Diagnostics"),
    ("headline", "Latency / cost"),
    ("matrix", "By category"),
    ("latency", "Latency"),
    ("tradeoff", "Cost vs quality"),
    ("datapoints", "Datapoints"),
]


def render_workload(run: Run, prefix: str, title: str | None) -> str:
    """All sections for one results dir. Ids are prefixed so two workloads can share a page."""
    global _ID_PREFIX
    _ID_PREFIX = prefix
    try:
        parts = [
            sec_hypothesis(run),
            sec_setup(run),
            sec_verdict(run),
            sec_headlines(run),
            sec_diagnostics(run),
            sec_glance(run),
            sec_headline(run),
            sec_matrix(run),
            sec_latency(run),
            sec_scatter(run),
            sec_datapoints(run),
        ]
    finally:
        _ID_PREFIX = ""
    head = ""
    if title:
        m = run.meta
        head = (
            f'<div class="wl-head" id="{esc(prefix)}workload"><h2 class="wl-title">{esc(title)}</h2>'
            f'<p class="sub mono">{esc(run.dir.name)} · {f_int(m.get("n_datapoints") or len({str(q["id"]) for q in run.queries}))} datapoints · '
            f'{f_int(m.get("repeats"))} repeats · {esc(m.get("profile") or "unknown profile")}</p>'
            "<p class=\"sub\">Reported separately; numbers are never pooled with the other workload."
            + (" <strong>Accuracy-only workload: not latency-grade (arm concurrency above 1).</strong>" if (_num(m.get("arm_concurrency")) or 1) > 1 else "")
            + "</p></div>"
        )
    toc = "".join(f'<a href="#{esc(prefix)}{i}">{t}</a>' for i, t in NAV)
    return f'<div class="workload">{head}<nav class="toc" aria-label="Sections">{toc}</nav>{"".join(p for p in parts if p)}</div>'


def render(runs: list[tuple[str, Run]], analysis_path: Path | None) -> str:
    first = runs[0][1]
    names = " + ".join(r.dir.name for _, r in runs)
    sub = " · ".join(
        f"{esc(label)}: {len(r.arms)} arm{'s' if len(r.arms) != 1 else ''}, {f_int(r.meta.get('n_datapoints') or len({str(q['id']) for q in r.queries}))} datapoints, "
        f"{f_int(r.meta.get('repeats'))} repeats"
        for label, r in runs
    )
    sha = str(first.meta.get("git_sha") or "unknown")[:9]
    banner = (
        '<div class="banner" role="note"><strong>Dry run.</strong> Everything below shows the pipeline works; the figures are not evidence for or against the hypothesis.</div>'
        if any(r.meta.get("dry_run") for _, r in runs)
        else ""
    )
    multi = len(runs) > 1
    bodies = []
    for idx, (label, r) in enumerate(runs):
        bodies.append(render_workload(r, f"w{idx}-" if multi else "", label if multi else None))
    global _ID_PREFIX
    _ID_PREFIX = ""
    top_toc = (
        "".join(f'<a href="#w{i}-workload">{esc(label)}</a>' for i, (label, _) in enumerate(runs)) + '<a href="#analysis">Analysis &amp; verdict</a>'
    )
    parts = [
        '<header class="top">',
        '<div class="eyebrow">trace-finder · filter selection benchmark</div>',
        "<h1>Does one LLM call pick filters as well as llm+jev?</h1>",
        f'<p class="sub">{sub} · commit <span class="mono">{esc(sha)}</span></p>',
        f'<p class="sub mono">{esc(names)}</p>',
        banner,
        f'<nav class="toc" aria-label="Workloads">{top_toc}</nav>' if multi else "",
        "</header>",
        verdict_box(runs),
        *bodies,
        sec_analysis(first, analysis_path),
        f"<footer>Generated from raw.jsonl, meta.json and summary.json in <span class=\"mono\">{esc(names)}</span> at render time by report.py.</footer>",
    ]
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Filter selection benchmark · {esc(names)}</title>"
        f"<style>{CSS}</style></head><body><main class=\"page\">{''.join(p for p in parts if p)}</main></body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--supplement", type=Path, default=None, help="second results dir, rendered as a separate workload section")
    ap.add_argument("--analysis", type=Path, default=None, help="markdown file rendered into 'Analysis & verdict'")
    ap.add_argument("--out", type=Path, default=None, help="default: <results-dir>/report.html")
    args = ap.parse_args(argv)

    dirs: list[tuple[str, Path]] = [("Main workload", args.results_dir)]
    if args.supplement is not None:
        dirs.append(("Supplement workload", args.supplement))
    for _, d in dirs:
        missing = [n for n in ("raw.jsonl", "meta.json", "summary.json") if not (d / n).is_file()]
        if missing:
            print(f"error: {d} is missing {', '.join(missing)}", file=sys.stderr)
            return 2
    doc = render([(label, Run(d)) for label, d in dirs], args.analysis)
    out = args.out or (args.results_dir / "report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(str(out.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
