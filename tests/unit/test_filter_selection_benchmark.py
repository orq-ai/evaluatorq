import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = ROOT / "scripts/benchmarks/filter_selection/bench.py"
REPORT_PATH = ROOT / "scripts/benchmarks/filter_selection/report.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_module("filter_selection_bench", BENCH_PATH)
report = _load_module("filter_selection_report", REPORT_PATH)


def _catalogue_file(path: Path) -> Path:
    catalogue = {
        "status": ["error"],
        "project": ["api"],
        "product": ["prod"],
        "model": ["example"],
        "provider": ["example"],
        "trace_type": ["timeout"],
        "agent_name": ["example"],
        "tool_name": ["example"],
    }
    target = path / "catalogue.json"
    target.write_text(json.dumps({"catalogue": catalogue}), encoding="utf-8")
    return target


def _dp(*, id="one", now="2025-01-01T00:00:00Z", facets=None):
    return {
        "id": id,
        "now": now,
        "query": "synthetic query",
        "category": "facet",
        "expected": {"facets": facets or {}},
    }


def _load(tmp_path: Path, rows, *, ids=None):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return bench.load_inputs(
        Namespace(dataset=dataset, catalogue=_catalogue_file(tmp_path), ids=ids, limit_datapoints=None)
    )


def test_sparse_category_matrix_renders_missing_category_as_empty_cell():
    run = report.Run.__new__(report.Run)
    run.arms = ["llm"]
    run.cls = {"llm": "a1"}
    run.queries = [{"arm": "llm", "category": "facet"}, {"arm": "llm", "category": "numeric"}]
    run.summary = {"arms": {"llm": {"by_category": {"facet": {"headline_a": {"mean_pass_rate": 1}}}}}}
    run.arm_summary = lambda arm: run.summary["arms"][arm]
    run.queries_for = lambda arm: [q for q in run.queries if q["arm"] == arm]
    matrix = report._matrix(run, "headline_a.mean_pass_rate")
    assert "—" in matrix


def test_report_consumes_benchmark_summarizer_cache_schema(tmp_path: Path):
    raw_rows = [
        {"kind": "query", "arm": mode, "id": "one", "category": "facet", "repeat": 0, "actual": {"dimensions": 0},
         "wall_ms": 10.0, "error": None, "scores": {"facet_tp": 0, "facet_fp": 0, "facet_fn": 0,
         "hallucinated": 0, "time_bearing": False, "core_exact": True, "full_exact": True,
         "outside_kind": None, "false_filter": False, "coverage_limited": False, "prompt_example": False}}
        for mode in bench.ALL_ARMS
    ]
    raw_rows += [
        {"kind": "cache_scenario", "mode": mode, "id": "one", "cache_state": state,
         "wall_ms": {"llm+jev": {"cold": 219.0, "warm": 137.0}, "llm": {"cold": 271.0, "warm": 153.0}}[mode][state],
         "catalogue_cache": {"hit": state == "warm", "elapsed_seconds": 0.002}, "error": None}
        for mode in bench.ALL_ARMS for state in ("cold", "warm")
    ]
    summary = bench.summarize(raw_rows)
    results_dir = tmp_path / "run"
    results_dir.mkdir()
    (results_dir / "meta.json").write_text(json.dumps({"arm_concurrency": 1, "n_datapoints": 1}), encoding="utf-8")
    (results_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (results_dir / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8")
    run = report.Run(results_dir)
    page = report.render([("synthetic", run)], analysis_path=None)
    from html.parser import HTMLParser

    class Text(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []

        def handle_data(self, data):
            self.parts.append(data)

    parsed = Text()
    parsed.feed(page)
    rendered = ''.join(parsed.parts)
    for value in (219, 137, 271, 153):
        assert f'{value} ms' in rendered


@pytest.mark.parametrize(
    "rows,ids,error",
    [
        ([], None, SystemExit),
        ([_dp(facets={"not_a_facet": ["x"]})], None, SystemExit),
        ([_dp(now="2025-01-01T00:00:00Z"), _dp(id="two", now="2025-01-02T00:00:00Z")], None, SystemExit),
        ([_dp(now="not-a-date")], None, ValueError),
        ([_dp(now=None)], None, SystemExit),
        ([_dp()], "missing", SystemExit),
    ],
)
def test_load_inputs_rejects_invalid_files(tmp_path, rows, ids, error):
    with pytest.raises(error):
        _load(tmp_path, rows, ids=ids)


def test_load_inputs_validation_survives_python_optimization(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("", encoding="utf-8")
    catalogue = _catalogue_file(tmp_path)
    code = (
        "import importlib.util, sys; from argparse import Namespace; "
        f"s=importlib.util.spec_from_file_location('bench', {str(BENCH_PATH)!r}); "
        "m=importlib.util.module_from_spec(s); sys.modules['bench']=m; s.loader.exec_module(m); "
        "\ntry:\n"
        f" m.load_inputs(Namespace(dataset={str(dataset)!r}, catalogue={str(catalogue)!r}, ids=None, limit_datapoints=None))\n"
        "except SystemExit:\n"
        " raise SystemExit(42)\n"
    )
    result = subprocess.run([sys.executable, "-O", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 42, result.stderr


@pytest.mark.parametrize("target", [
    'https://example.test/"onmouseover="alert',
    'https://example.test/?x=1&y=2',
    'https://example.test/&#34;onmouseover=&#34;alert',
])
def test_analysis_links_keep_untrusted_targets_inside_href(target: str):
    from html.parser import HTMLParser

    links = []

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                links.append(dict(attrs))

    parser = Links()
    parser.feed(report.markdown_to_html(f"[open]({target})"))
    assert links == [{"href": target, "rel": "noreferrer"}]
