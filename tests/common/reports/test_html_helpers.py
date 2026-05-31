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
