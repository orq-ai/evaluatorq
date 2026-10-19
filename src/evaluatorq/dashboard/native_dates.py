"""Shared native date controls used by Traces and Insights local sessions."""

from __future__ import annotations

from fasthtml.common import B, Input, Label, NotStr, Span, to_xml


def date_input(
    *,
    control_id: str,
    name: str,
    label: str,
    value: str = '',
    form: str | None = None,
    css_class: str | None = None,
    required: bool = False,
    preserve: bool = False,
    utc: str | None = None,
) -> str:
    """Render one accessible native date input with the requested form attributes."""
    attrs: dict[str, object] = {
        'id': control_id,
        'name': name,
        'type': 'date',
        'value': value,
        'aria_label': label,
        'required': required,
    }
    if form is not None:
        attrs['form'] = form
    if css_class is not None:
        attrs['cls'] = css_class
    if preserve:
        attrs['hx_preserve'] = True
    if utc is not None:
        attrs['data_utc'] = utc
    return to_xml(Input(**attrs))


def date_row(label: str, control: str, *, css_class: str, label_tag: str = 'b') -> str:
    """Render a labelled native date control in the row vocabulary of its surface."""
    if label_tag == 'label':
        return to_xml(Label(label, NotStr(control), cls=css_class))
    return to_xml(Span(B(label), NotStr(control), cls=css_class))
