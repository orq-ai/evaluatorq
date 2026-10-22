"""Portable HTML retains dashboard information without its running server."""
from __future__ import annotations

import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evaluatorq.contracts import Message, TokenUsage
from evaluatorq.dashboard.report_tabs import redteam_report_tabs, sim_report_tabs
from evaluatorq.simulation.cli import app as sim_app
from evaluatorq.simulation.reports.export_html import export_html as sim_html
from evaluatorq.redteam.reports.export_html import export_html as rt_html
from tests.dashboard.test_sim_transcript import _make_sim_run_with_transcript


class Elements(HTMLParser):
    def __init__(self, document: str):
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_export_section_order_matches_dashboard_tabs(rt_report_multi):
    for obj, export, tabs in [
        (_make_sim_run_with_transcript(), sim_html, sim_report_tabs),
        (rt_report_multi, rt_html, redteam_report_tabs),
    ]:
        dashboard = Elements(tabs('test', obj))
        expected = [a['for'] for tag, a in dashboard.tags if tag == 'label' and a.get('class') == 'tab-label']
        exported = Elements(export(obj))
        sections = [a for tag, a in exported.tags if tag == 'section' and a.get('class') == 'export-section']
        assert len(sections) == len(expected)
        assert [a['id'].split('-', 1)[1] for a in sections] == (
            ['overview', 'breakdown', 'transcripts', 'config'] if export is sim_html
            else ['overview', 'agents', 'focus-areas', 'breakdowns', 'attacks', 'usage', 'config']
        )


def test_exports_are_offline_and_have_no_server_controls(rt_report_multi):
    for document in [sim_html(_make_sim_run_with_transcript()), rt_html(rt_report_multi)]:
        elements = Elements(document)
        assert '@import' not in document
        assert 'data:font/woff2;base64,' in document
        assert not any(tag in {'script', 'iframe', 'dialog', 'template', 'input', 'button'} for tag, _ in elements.tags)
        assert not any(key.startswith('hx-') for _, attrs in elements.tags for key in attrs)
        assert not any(a.get('href', '').startswith('/r/') for _, a in elements.tags)
        assert 'Collapsed conversations and attacks are omitted from print' in document
        assert all('open' not in a if 'export-evidence' in a.get('class', '').split() else 'open' in a for tag, a in elements.tags if tag == 'details')
        ids = [a['id'] for _, a in elements.tags if 'id' in a]
        assert len(ids) == len(set(ids)), 'A section or chart id is duplicated.'
        for tag, attrs in elements.tags:
            href = attrs.get('href')
            if tag == 'a' and href and href.startswith('#'):
                assert href[1:] in ids, href


def test_saved_sim_run_keeps_config_usage_and_experiment():
    run = _make_sim_run_with_transcript()
    run.run_name = 'Original saved name'
    run.target = 'saved-target'
    run.target_model = 'saved-model'
    run.max_turns = 17
    run.token_usage_total = TokenUsage(total_tokens=12345)
    run.experiment_url = 'https://my.orq.ai/example/experiments/synthetic'
    document = sim_html(run)
    body = document.split('</style>', 1)[1]
    for text in ['Original saved name', 'saved-target', 'saved-model', '>17<', '12,345', run.experiment_url]:
        assert text in body


def test_sim_export_cli_preserves_saved_run_wrapper(tmp_path):
    run = _make_sim_run_with_transcript()
    run.run_name = 'Saved identity'
    run.target_model = 'saved-model'
    run.max_turns = 17
    source, target = tmp_path/'run.json', tmp_path/'report.html'
    source.write_text(run.model_dump_json())
    result = CliRunner().invoke(sim_app, ['export', '-i', str(source), '-o', str(target), '--format', 'html'])
    assert result.exit_code == 0, result.output
    body = target.read_text().split('</style>', 1)[1]
    assert 'Saved identity' in body and 'saved-model' in body and '>17<' in body


def test_sim_export_keeps_partial_transcript_on_error():
    run = _make_sim_run_with_transcript()
    from evaluatorq.simulation.contracts import TerminatedBy
    run.results[0].terminated_by = TerminatedBy.error
    run.results[0].metadata['error'] = 'Synthetic target timeout'
    run.results[0].messages = [Message(role='user', content='Partial transcript evidence')]
    document = sim_html(run)
    assert 'Partial transcript evidence' in document
    assert 'Synthetic target timeout' in document
    cohort = document.split('href="#conv-1"', 1)[1].split('</a>', 1)[0]
    assert '>Error<' in cohort
    assert '>Failed<' not in cohort


def test_attack_export_keeps_full_text_and_unknown_outcome(rt_report_multi):
    result = rt_report_multi.results[0]
    result.vulnerable = None
    result.evaluation = None
    long_text = 'Evidence ' * 1000 + 'FINAL_EVIDENCE_MARKER'
    result.messages = [Message(role='user', content=long_text)]
    document = rt_html(rt_report_multi)
    assert long_text in document
    assert 'Not evaluated' in document
    assert 'Expand an attack to read its verdict and full transcript.' in document
    assert sum(tag == 'details' and 'export-attack' in a.get('class', '').split() for tag, a in Elements(document).tags) == len(rt_report_multi.results)


def test_sim_export_preserves_tool_call_evidence():
    run = _make_sim_run_with_transcript()
    run.results[0].messages = [Message.model_validate({
        'role': 'assistant', 'content': '',
        'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {'name': 'lookup_order', 'arguments': '{"order":"EXAMPLE-42"}'}}],
    }), Message(role='tool', content='Synthetic tool result', tool_call_id='call-1')]
    body = sim_html(run).split('</style>', 1)[1]
    assert 'lookup_order' in body and 'EXAMPLE-42' in body and 'Synthetic tool result' in body


@pytest.mark.parametrize('value', ['javascript:alert(1)', 'data:text/html,test', 'file:///tmp/test'])
def test_export_rejects_unsafe_saved_links(value):
    run = _make_sim_run_with_transcript()
    run.experiment_url = value
    run.agent_info = {'key': 'example', 'id': 'example', 'workspace_key': 'demo', 'base_url': value}
    links = [a.get('href', '') for tag, a in Elements(sim_html(run)).tags if tag == 'a']
    assert not any(url.startswith(('javascript:', 'data:', 'file:')) for url in links)


def test_export_imports_no_dashboard_runtime(tmp_path, rt_report_multi):
    sim = tmp_path/'sim.json'
    rt = tmp_path/'rt.json'
    sim.write_text(_make_sim_run_with_transcript().model_dump_json())
    rt.write_text(rt_report_multi.model_dump_json())
    code = '''
import sys
from pathlib import Path
class BlockRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fasthtml', 'starlette', 'uvicorn', 'cryptography', 'dotenv'}:
            raise AssertionError('Offline export imported ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
from evaluatorq.simulation.contracts import SimulationRun
from evaluatorq.redteam.contracts import RedTeamReport
from evaluatorq.simulation.reports.export_html import export_html as sim_html
from evaluatorq.redteam.reports.export_html import export_html as rt_html
assert sim_html(SimulationRun.model_validate_json(Path(sys.argv[1]).read_text())).startswith('<!DOCTYPE html>')
assert rt_html(RedTeamReport.model_validate_json(Path(sys.argv[2]).read_text())).startswith('<!DOCTYPE html>')
'''
    result = subprocess.run([sys.executable, '-c', code, str(sim), str(rt)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_attack_export_preserves_response_without_messages(rt_report_multi):
    result = rt_report_multi.results[0]
    result.messages = []
    result.response = 'Older saved final response ' + 'evidence ' * 700
    body = rt_html(rt_report_multi).split('</style>', 1)[1]
    assert result.response in body
    assert 'Recorded final response' in body


def test_redteam_keeps_narrative_and_full_judge_failure_reasons(rt_report_multi):
    from evaluatorq.redteam.contracts import JuryHealth
    rt_report_multi.executive_summary = 'Saved assessment narrative with <safe> escaping.'
    reason = 'Provider rejected the structured judge reply: ' + 'x' * 200
    rt_report_multi.summary.jury_health = JuryHealth(samples=2, samples_with_judge_failure=1, replacements_used=1, judge_failure_reasons=[reason])
    body = rt_html(rt_report_multi).split('</style>', 1)[1]
    assert 'Saved assessment narrative with &lt;safe&gt; escaping.' in body
    assert reason in body
    assert 'Replacement judges' in body


@pytest.mark.parametrize('total,evaluated', [(0, 0), (2, 0)])
def test_redteam_retains_saved_narrative_without_a_verdict(total, evaluated):
    from evaluatorq.dashboard.report_tabs import _rt_exec_summary
    body = _rt_exec_summary({'total_attacks': total, 'evaluated_attacks': evaluated, 'narrative': 'Saved narrative evidence'}, {})
    assert 'Saved narrative evidence' in body
    if total:
        assert '2</strong> had no verdict' in body


@pytest.mark.parametrize('label', ['agent', 'Custom export target'])
def test_sim_saved_run_export_honors_explicit_target_label(tmp_path, label):
    run = _make_sim_run_with_transcript()
    run.target = 'Original target'
    source, target = tmp_path/'run.json', tmp_path/'report.html'
    source.write_text(run.model_dump_json())
    result = CliRunner().invoke(sim_app, ['export', '-i', str(source), '-o', str(target), '--format', 'html', '--target-label', label])
    assert result.exit_code == 0, result.output
    assert f'target {label}</p>' in target.read_text()


def test_conversation_evidence_is_native_and_closed_by_default(rt_report_multi):
    for document, count in [(sim_html(_make_sim_run_with_transcript()), 2), (rt_html(rt_report_multi), len(rt_report_multi.results))]:
        evidence = [(tag, attrs) for tag, attrs in Elements(document).tags if 'export-evidence' in attrs.get('class', '').split()]
        assert len(evidence) == count
        assert all(tag == 'details' and 'open' not in attrs for tag, attrs in evidence)
        assert document.count('<summary class="export-conversation-summary">') == count or 'rt-attack-row-summary' in document
