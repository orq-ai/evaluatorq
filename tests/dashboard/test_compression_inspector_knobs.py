"""Compression knobs: parsing and validation, and live re-rendering of a loaded trace."""

from __future__ import annotations

import html
import re
from dataclasses import replace
from typing import Any

import pytest
from starlette.testclient import TestClient

from evaluatorq.common.model_input import EdgeLimit, JevOptions
from evaluatorq.dashboard import compression_inspector, compression_inspector_data as inspector_data
from evaluatorq.dashboard.compression_inspector_data import CompressionKnobs, KnobError
from evaluatorq.dashboard.app import build_app
from evaluatorq.insights.transcript import DEFAULT_MESSAGE_KINDS
from evaluatorq.trace_finder.settings import DashboardSettings, effective_settings, save_settings
from evaluatorq.local_sessions import SessionRef, load_session_document
from tests.dashboard.test_compression_inspector_routes import _claude_session

DEFAULTS = CompressionKnobs.defaults(500_000)


def _posted(knobs: CompressionKnobs) -> dict[str, str]:
    """The form a browser posts for these knobs: unchecked boxes are absent."""
    return {name: '1' if value is True else str(value) for name, value in knobs.as_form().items() if value is not False}


def test_default_knobs_reproduce_the_whole_readable_view_beside_the_classifier_jev_state() -> None:
    assert DEFAULTS.budget == DEFAULTS.global_char_cap == 500_000
    assert DEFAULTS.include == DEFAULT_MESSAGE_KINDS
    assert DEFAULTS.tools == 'per_call'
    assert {DEFAULTS.system_chars, DEFAULTS.user_chars, DEFAULTS.assistant_edge} == {-1}
    assert {DEFAULTS.reasoning_chars, DEFAULTS.tool_input_chars, DEFAULTS.tool_max_chars} == {-1}
    assert DEFAULTS.jev == JevOptions()


def test_valid_form_values_round_trip() -> None:
    knobs = CompressionKnobs(
        global_char_cap=40_000,
        jev=JevOptions(
            state_chars=30_000,
            user=EdgeLimit(900, 300),
            tool_result=EdgeLimit(50, 25),
            include_developer=False,
            include_tool_results=False,
        ),
        budget=12_000,
        include=frozenset({'user', 'reasoning'}),
        system_chars=2_000,
        user_chars=1_500,
        assistant_edge=250,
        reasoning_chars=-1,
        tools='collapsed',
        tool_input_chars=80,
        tool_max_chars=0,
        original_view='readable',
        wrap=True,
    )

    assert CompressionKnobs.from_form(_posted(knobs)) == knobs
    assert CompressionKnobs.from_form(_posted(DEFAULTS)) == DEFAULTS


def test_absent_checkboxes_read_as_unchecked() -> None:
    form = {name: value for name, value in _posted(DEFAULTS).items() if '_include_' not in name}

    knobs = CompressionKnobs.from_form(form)

    assert knobs.include == frozenset()
    assert not any(
        (
            knobs.jev.include_system,
            knobs.jev.include_developer,
            knobs.jev.include_user,
            knobs.jev.include_assistant,
            knobs.jev.include_tool_calls,
            knobs.jev.include_tool_results,
        )
    )


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        pytest.param('global_char_cap', 'abc', id='not-a-number'),
        pytest.param('global_char_cap', '1.5', id='fraction'),
        pytest.param('global_char_cap', '', id='blank'),
        pytest.param('global_char_cap', '0', id='cap-below-one'),
        pytest.param('global_char_cap', '-1', id='cap-cannot-be-whole'),
        pytest.param('global_char_cap', '5000001', id='cap-above-maximum'),
        pytest.param('jev_state_chars', '0', id='state-limit-below-one'),
        pytest.param('view_budget', '-1', id='budget-cannot-be-whole'),
        pytest.param('view_user_chars', '0', id='user-limit-zero'),
        pytest.param('view_user_chars', '-2', id='user-limit-below-whole'),
        pytest.param('view_tool_max_chars', '-2', id='result-limit-below-whole'),
        pytest.param('jev_user_edge', '0', id='edge-below-one'),
        pytest.param('jev_user_edge', '1001', id='edge-over-half-the-total'),
        pytest.param('jev_user_total', '1999', id='total-under-twice-the-edge'),
        pytest.param('view_tools', 'everything', id='unknown-tool-mode'),
        pytest.param('original_view', 'pdf', id='unknown-original-view'),
        pytest.param('original_view', '', id='blank-original-view'),
    ],
)
def test_invalid_form_values_are_rejected(field: str, value: str) -> None:
    with pytest.raises(KnobError):
        CompressionKnobs.from_form({**_posted(DEFAULTS), field: value})


def test_the_error_names_the_field_and_never_echoes_the_input() -> None:
    with pytest.raises(KnobError) as error:
        CompressionKnobs.from_form({**_posted(DEFAULTS), 'global_char_cap': '<b>secret</b>'})

    assert 'Global character cap' in str(error.value)
    assert 'secret' not in str(error.value)


def test_a_missing_number_is_rejected_instead_of_defaulted() -> None:
    form = _posted(DEFAULTS)
    del form['view_assistant_edge']

    with pytest.raises(KnobError, match='assistant edge'):
        CompressionKnobs.from_form(form)


def test_defaults_show_the_atif_original_without_wrapping() -> None:
    assert DEFAULTS.original_view == 'atif'
    assert DEFAULTS.wrap is False
    assert CompressionKnobs.from_form({k: v for k, v in _posted(DEFAULTS).items() if k != 'wrap'}).wrap is False


def test_zero_keeps_tool_results_out_and_whole_is_allowed_for_readable_caps() -> None:
    form = {**_posted(DEFAULTS), 'view_tool_max_chars': '0', 'view_user_chars': '-1'}

    knobs = CompressionKnobs.from_form(form)

    assert knobs.tool_max_chars == 0
    assert knobs.user_chars == -1


# ---- live preview route -------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    save_settings(DashboardSettings.model_validate({'compression_inspector_enabled': True}))
    return TestClient(build_app())


def _csrf(client: TestClient) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', client.get('/compression-inspector').text)
    assert match is not None
    return match.group(1)


def _load(client: TestClient, *, session_id: str = 'knob-session') -> str:
    """Load a long conversation through the real Claude reader and return its cache token."""
    path = _claude_session(session_id, prompt='P' * 3_000, answer='ANSWER-TEXT ' + 'q' * 40)
    response = client.post(
        '/compression-inspector/sessions/load',
        data={'csrf': _csrf(client), 'session': f'claude-code:{path}'},
    )
    match = re.search(r'name="token" value="([^"]+)"', response.text)
    assert match is not None
    return match.group(1)


def _token_of(response: Any) -> str:
    match = re.search(r'name="token" value="([^"]+)"', response.text)
    assert match is not None
    return match.group(1)


def _preview(client: TestClient, token: str, *, htmx: bool = True, **changes: Any) -> Any:
    knobs = replace(CompressionKnobs.defaults(effective_settings().trace_input_chars), **changes)
    return client.post(
        '/compression-inspector/preview',
        data={**_posted(knobs), 'csrf': _csrf(client), 'token': token},
        headers={'HX-Request': 'true'} if htmx else {},
    )


def _pane(text: str, title_id: str) -> str:
    match = re.search(rf'aria-labelledby="{title_id}">.*?</section>', text, re.DOTALL)
    assert match is not None
    return match.group(0)


def test_excluding_a_message_kind_removes_it_from_the_readable_view_only(client: TestClient) -> None:
    token = _load(client)

    response = _preview(client, token, original_view='readable', include=DEFAULT_MESSAGE_KINDS - {'assistant'})

    assert response.status_code == 200
    assert 'ANSWER-TEXT' not in _pane(response.text, 'compression-original-title')
    assert 'ANSWER-TEXT' in _pane(response.text, 'compression-jev-title')


def test_a_user_limit_shortens_a_long_user_message_in_the_readable_view(client: TestClient) -> None:
    token = _load(client)

    whole = _pane(_preview(client, token, original_view='readable').text, 'compression-original-title')
    capped = _pane(
        _preview(client, token, original_view='readable', user_chars=200).text, 'compression-original-title'
    )

    assert 'P' * 3_000 in whole
    assert 'P' * 3_000 not in capped
    assert len(capped) < len(whole) - 2_500


def test_a_jev_edge_limit_changes_the_trimmed_user_text(client: TestClient) -> None:
    token = _load(client)

    default = _pane(_preview(client, token).text, 'compression-jev-title')
    narrowed = _pane(
        _preview(client, token, jev=JevOptions(user=EdgeLimit(400, 100))).text, 'compression-jev-title'
    )

    assert 'P' * 1_000 in default
    assert 'P' * 1_000 not in narrowed
    assert 'P' * 100 in narrowed
    assert '2800 chars left out' in narrowed


def test_invalid_knobs_show_an_inline_error_without_a_preview(client: TestClient) -> None:
    token = _load(client)
    knobs = {**_posted(CompressionKnobs.defaults(effective_settings().trace_input_chars)), 'view_budget': 'abc'}

    response = client.post(
        '/compression-inspector/preview',
        data={**knobs, 'csrf': _csrf(client), 'token': token},
        headers={'HX-Request': 'true'},
    )

    assert response.status_code == 200
    assert 'role="alert"' in response.text
    assert 'Readable view budget' in response.text
    assert 'compression-panels' not in response.text


def test_limits_too_small_for_the_jev_state_are_shown_as_a_notice(client: TestClient) -> None:
    token = _load(client)

    response = _preview(client, token, global_char_cap=5)

    assert response.status_code == 200
    assert 'role="alert"' in response.text
    assert 'compression-panels' not in response.text


def test_an_unknown_token_asks_to_load_the_trace_again(client: TestClient) -> None:
    response = _preview(client, 'not-a-token')

    assert response.status_code == 200
    assert 'Load the trace again to adjust it.' in response.text
    assert 'compression-panels' not in response.text


def test_the_oldest_loaded_trace_expires_after_eight_newer_ones(client: TestClient) -> None:
    first = _load(client)
    for _ in range(8):
        _load(client)

    assert 'Load the trace again' in _preview(client, first).text


def test_htmx_gets_only_the_live_region_and_a_plain_post_gets_the_page(client: TestClient) -> None:
    token = _load(client)

    fragment = _preview(client, token).text
    page = _preview(client, token, htmx=False).text

    assert fragment.startswith('<div id="compression-output"')
    assert 'Inspect trace compression' not in fragment
    assert 'Inspect trace compression' in page
    assert 'id="compression-output"' in page
    assert 'Reset to defaults' in page


def test_preview_post_without_the_csrf_token_is_rejected(client: TestClient) -> None:
    token = _load(client)

    response = client.post('/compression-inspector/preview', data={'token': token}, headers={'HX-Request': 'true'})

    assert response.status_code == 403


def _pane_text(response: Any, title_id: str = 'compression-original-title') -> str:
    return html.unescape(_pane(response.text, title_id))


def test_the_atif_original_is_the_default_and_shows_the_trajectory_not_the_plaintext(client: TestClient) -> None:
    token = _load(client)

    pane = _pane_text(_preview(client, token))

    assert 'Original, ATIF trajectory' in pane
    assert '"steps"' in pane
    assert 'ANSWER-TEXT' in pane
    assert '0 user:' not in pane


def test_the_readable_original_is_the_plaintext_render(client: TestClient) -> None:
    token = _load(client)

    pane = _pane_text(_preview(client, token, original_view='readable'))

    assert 'Original, readable text' in pane
    assert '0 user: PPP' in pane
    assert '"steps"' not in pane


def test_the_source_original_is_the_raw_session_file(client: TestClient) -> None:
    token = _load(client)

    source = _pane_text(_preview(client, token, original_view='source'))
    atif = _pane_text(_preview(client, token, original_view='atif'))

    assert 'Original, raw source file' in source
    assert '"parentUuid"' in source and '"uuid": "user-1"' in source
    assert 'ANSWER-TEXT' in source
    assert '"parentUuid"' not in atif


def test_a_raw_source_longer_than_the_byte_limit_is_cut_with_a_note(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = _load(client)
    monkeypatch.setattr(inspector_data, 'SOURCE_BYTES', 300)

    response = _preview(client, token, original_view='source')

    assert 'Showing the first 300 of ' in html.unescape(response.text)
    assert 'ANSWER-TEXT' not in _pane_text(response)


def test_an_original_longer_than_the_character_limit_is_cut_with_a_note(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = _load(client)
    monkeypatch.setattr(inspector_data, 'ORIGINAL_CHARS', 500)

    response = _preview(client, token)

    assert 'Cut to 500 characters' in response.text
    assert 'chars left out' in response.text
    assert '"steps"' in _pane_text(response) or '"schema_version"' in _pane_text(response)


def test_every_original_is_scrubbed_of_known_secrets(client: TestClient) -> None:
    path = _claude_session('secret-session', prompt='API_TOKEN=gXeRk29vLq0Pz8Wd please', answer='ok')
    loaded = client.post(
        '/compression-inspector/sessions/load',
        data={'csrf': _csrf(client), 'session': f'claude-code:{path}'},
    )
    token = _token_of(loaded)
    assert 'gXeRk29vLq0Pz8Wd' in path.read_text()

    for view in ('atif', 'source', 'readable'):
        assert 'gXeRk29vLq0Pz8Wd' not in _preview(client, token, original_view=view).text


def test_an_orq_trace_has_no_raw_source_option_and_the_view_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _claude_session('orq-like', prompt='hello', answer='world')
    document = load_session_document(SessionRef(source='claude-code', path=path))

    async def fetch(*_args: Any, **_kwargs: Any) -> Any:
        return document

    monkeypatch.setattr(compression_inspector, 'fetch_orq_document', fetch)
    loaded = client.post('/compression-inspector/source/orq', data={'csrf': _csrf(client), 'trace_id': 'trace-9'})
    token = _token_of(loaded)

    assert 'value="atif"' in loaded.text and 'value="readable"' in loaded.text
    assert 'value="source"' not in loaded.text
    refused = _preview(client, token, original_view='source')
    assert refused.status_code == 200
    assert 'role="alert"' in refused.text
    assert 'only available for local sessions' in refused.text
    assert 'compression-panels' not in refused.text


def test_a_local_session_offers_the_raw_source_option(client: TestClient) -> None:
    path = _claude_session('offers-source', prompt='hello', answer='world')
    loaded = client.post(
        '/compression-inspector/sessions/load', data={'csrf': _csrf(client), 'session': f'claude-code:{path}'}
    )

    assert 'value="source"' in loaded.text


def test_the_wrap_checkbox_sets_the_wrapped_class_on_the_output(client: TestClient) -> None:
    token = _load(client)

    off = _preview(client, token).text
    on = _preview(client, token, wrap=True).text

    assert 'is-wrapped' not in off.split('<dl')[0]
    assert 'is-wrapped' in on.split('<dl')[0]
    assert re.search(r'name="wrap"[^>]* checked', _preview(client, token, htmx=False, wrap=True).text)
    assert not re.search(r'name="wrap"[^>]* checked', _preview(client, token, htmx=False).text)


def test_an_unknown_original_view_shows_an_inline_error(client: TestClient) -> None:
    token = _load(client)
    knobs = {**_posted(CompressionKnobs.defaults(effective_settings().trace_input_chars)), 'original_view': 'pdf'}

    response = client.post(
        '/compression-inspector/preview',
        data={**knobs, 'csrf': _csrf(client), 'token': token},
        headers={'HX-Request': 'true'},
    )

    assert 'role="alert"' in response.text
    assert 'Original shows' in response.text
    assert 'compression-panels' not in response.text


def test_coverage_defaults_on_and_round_trips_through_the_form() -> None:
    assert DEFAULTS.coverage is True
    assert CompressionKnobs.from_form(_posted(DEFAULTS)).coverage is True
    off = replace(DEFAULTS, coverage=False)
    assert 'coverage' not in _posted(off)
    assert CompressionKnobs.from_form(_posted(off)).coverage is False


def test_coverage_checkbox_is_in_the_form_and_restored_by_reset(client: TestClient) -> None:
    token = _load(client)
    page = _preview(client, token, htmx=False).text

    assert 'name="coverage"' in page
    assert re.search(r'name="coverage"[^>]*checked[^>]*data-default="1"', page)


def test_coverage_adds_a_header_to_the_readable_text_and_a_context_to_the_jev_state(client: TestClient) -> None:
    token = _load(client)

    on = _preview(client, token, original_view='readable')
    off = _preview(client, token, original_view='readable', coverage=False)

    assert '[context: includes' in _pane(on.text, 'compression-original-title')
    assert '&quot;context&quot;' in _pane(on.text, 'compression-jev-title') or '"context"' in _pane(
        on.text, 'compression-jev-title'
    )
    assert '[context:' not in off.text
    assert 'context' not in _pane(off.text, 'compression-jev-title')


def test_unchecking_a_message_kind_changes_the_coverage_in_both_panes(client: TestClient) -> None:
    token = _load(client)

    response = _preview(
        client,
        token,
        original_view='readable',
        include=DEFAULT_MESSAGE_KINDS - {'assistant'},
        jev=replace(DEFAULTS.jev, include_assistant=False),
    )

    original = _pane(response.text, 'compression-original-title')
    assert 'left out: assistant, reasoning' in original
    assert 'assistant' in _pane(response.text, 'compression-jev-title').split('left_out')[1][:80]


def test_the_atif_original_carries_no_coverage_header(client: TestClient) -> None:
    token = _load(client)

    response = _preview(client, token)

    assert '[context:' not in _pane(response.text, 'compression-original-title')
