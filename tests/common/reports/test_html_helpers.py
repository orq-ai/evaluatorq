# tests/common/reports/test_html_helpers.py
from evaluatorq.common.reports import html_helpers as h
from evaluatorq.common.reports.palette import ORQ_SCALE_GOOD_BAD


def test_scale_color_endpoints_and_clamp():
	assert h.scale_color(0.0, ORQ_SCALE_GOOD_BAD).lower() == '#2ebd85'
	assert h.scale_color(1.0, ORQ_SCALE_GOOD_BAD).lower() == '#d92d20'
	# clamps out-of-range
	assert h.scale_color(-5, ORQ_SCALE_GOOD_BAD).lower() == '#2ebd85'
	assert h.scale_color(9, ORQ_SCALE_GOOD_BAD).lower() == '#d92d20'


def test_scale_color_midpoint_is_between():
	mid = h.scale_color(0.5, ORQ_SCALE_GOOD_BAD).lstrip('#')
	r = int(mid[0:2], 16)
	# midpoint is the yellow stop (#f2b600) -> red channel high
	assert r > 200


def test_svg_donut_renders_slices_and_center():
	svg = h.svg_donut(
		labels=['Achieved', 'Failed'],
		values=[1, 3],
		colors=['#2ebd85', '#d92d20'],
		center_label='25%',
		title='Goal Outcomes',
	)
	assert svg.startswith('<figure')
	assert '<svg' in svg and '</svg>' in svg
	assert svg.count('<path') == 2  # one arc per non-zero slice
	assert '25%' in svg
	assert 'Goal Outcomes' in svg


def test_svg_donut_empty_when_all_zero():
	assert h.svg_donut(labels=['a'], values=[0], colors=['#000'], center_label='', title='t') == ''


def test_svg_bar_renders_labeled_bars():
	svg = h.svg_bar(
		rows=[('1 turn', 3), ('2 turns', 1)],
		title='Conversations by turn count',
	)
	assert svg.startswith('<figure')
	assert svg.count('<rect') == 2
	assert '1 turn' in svg and '3' in svg
	assert 'Conversations by turn count' in svg


def test_svg_bar_empty_when_no_rows():
	assert h.svg_bar(rows=[], title='t') == ''


def test_render_heatmap_cells_and_labels():
	html = h.render_heatmap(
		x_labels=['c1', 'c2'],
		y_labels=['must explain charge', 'must not be rude'],
		cells=[[1.0, 0.0], [1.0, 1.0]],  # row-major: y by x
		scale=ORQ_SCALE_GOOD_BAD,
		title='Criteria pass/fail',
		value_fmt=lambda v: 'PASS' if v >= 0.5 else 'FAIL',
	)
	assert 'class="heatmap-table"' in html
	assert html.count('<td') >= 4
	assert 'must explain charge' in html
	assert 'PASS' in html and 'FAIL' in html
	assert 'Criteria pass/fail' in html


def test_render_heatmap_safety_flag_uses_hot_class():
	html = h.render_heatmap(
		x_labels=['c1'], y_labels=['no PII leak'], cells=[[0.0]],
		scale=ORQ_SCALE_GOOD_BAD, title='t',
		value_fmt=lambda v: 'FAIL', safety_mask=[[True]],
	)
	assert 'heatmap-cell--safety' in html


def test_render_heatmap_absent_cell_is_neutral():
	html = h.render_heatmap(
		x_labels=['c1'], y_labels=['r1'], cells=[[-1.0]],
		scale=ORQ_SCALE_GOOD_BAD, title='t', value_fmt=lambda v: '—' if v < 0 else 'x',
	)
	assert 'background:#e4e2df' in html
	assert '—' in html


def test_render_histogram_bins():
	html = h.render_histogram(values=[0.0, 0.1, 0.9, 1.0], bins=2, title='Score distribution')
	assert html.startswith('<figure')
	assert html.count('<rect') == 2
	assert 'Score distribution' in html


def test_render_line_chart_series():
	html = h.render_line_chart(
		x_labels=['1', '2', '3'],
		series=[('response_quality', [0.5, 0.7, 0.9])],
		title='Turn quality',
	)
	assert html.startswith('<figure')
	assert '<polyline' in html
	assert 'response_quality' in html
	assert 'Turn quality' in html


def test_render_sparkline_minibars():
	svg = h.render_sparkline([1, 3, 2])
	assert svg.startswith('<svg') and svg.endswith('</svg>')
	assert svg.count('<rect') == 3


def test_render_sparkline_empty():
	assert h.render_sparkline([]) == ''


def test_kpi_cards_renders_each_card_with_status():
	html = h.kpi_cards([
		{'label': 'Success Rate', 'value': '25%', 'status': 'fail'},
		{'label': 'Conversations', 'value': '4', 'status': 'neutral'},
	])
	assert 'class="kpi-band"' in html
	assert html.count('kpi-card') >= 2
	assert 'Success Rate' in html and '25%' in html
	assert 'kpi-card--fail' in html


def test_status_badge_classes():
	assert 'status-badge--pass' in h.status_badge('ACHIEVED', 'pass')
	assert 'status-badge--fail' in h.status_badge('NOT ACHIEVED', 'fail')
	assert 'NOT ACHIEVED' in h.status_badge('NOT ACHIEVED', 'fail')


from evaluatorq.common.reports.html_helpers import load_css


def test_report_css_has_new_design_tokens():
	css = load_css()
	for token in [
		'.hero', '.kpi-band', '.kpi-card', '.report-card', '.chart-card',
		'.status-badge--pass', '.status-badge--fail', '.heatmap-table',
		'.heatmap-cell', '.sparkline', '@media',
	]:
		assert token in css, f'missing {token}'
