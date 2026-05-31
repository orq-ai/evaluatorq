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
