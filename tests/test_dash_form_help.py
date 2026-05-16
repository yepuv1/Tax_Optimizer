"""Tests for per-field help hints in the scenario form.

Every :class:`dash_app.forms.FormField` declares an optional
``help`` string. The dashboard renders it via two surfaces:

1. A small ⓘ icon (``span.form-hint-icon``) inserted into the
   label children. The icon is the discoverable affordance — its
   ``aria-label`` attribute carries the help text for assistive
   technology (we deliberately skip the native HTML ``title``
   attribute, which would race the Bootstrap popover and produce
   a duplicate / immediate tooltip on hover).
2. A :class:`dash_bootstrap_components.Tooltip` whose ``target``
   matches the icon's ``id``. Provides the styled, delayed
   popover with the help text — the single visible tooltip
   surface.

These tests pin the contract without spinning up a Dash dev
server:

* **Schema completeness** — every field in ``FIELD_SCHEMA`` has a
  non-empty ``help``. New fields can't be merged without a hint.
* **Rendering** — exercising ``_field_row`` / ``_help_components``
  directly produces the icon + tooltip pair with matching IDs and
  no native ``title`` attributes anywhere in the tree.
* **Negative case** — a field built without ``help`` does NOT
  render the icon (defensive guard so we don't ship broken
  tooltip targets if a future field is added without a hint).
"""

from __future__ import annotations

from typing import Any, Iterable

import pytest

pytest.importorskip("dash")

from dash_app.forms import FIELD_SCHEMA, FormField  # noqa: E402
from dash_app.layout import (  # noqa: E402
    _field_row,
    _help_components,
    _hint_id,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _to_json(component: Any) -> Any:
    """Recursively serialize a Dash component tree to plain JSON.

    ``component.to_plotly_json()`` only converts the *top* node;
    ``children`` come back as live Dash components. We walk the
    tree so the assertions can match on serialized dicts.
    """
    if hasattr(component, "to_plotly_json"):
        return _to_json(component.to_plotly_json())
    if isinstance(component, dict):
        return {k: _to_json(v) for k, v in component.items()}
    if isinstance(component, list):
        return [_to_json(v) for v in component]
    return component


def _walk(tree: Any) -> Iterable[dict]:
    """Yield every dict node in a serialized Dash component tree."""
    if isinstance(tree, dict):
        yield tree
        for v in tree.get("props", {}).values():
            yield from _walk(v)
    elif isinstance(tree, list):
        for v in tree:
            yield from _walk(v)


def _find(tree: Any, predicate) -> list[dict]:
    return [n for n in _walk(tree) if predicate(n)]


# ---------------------------------------------------------------------
# Schema-level coverage
# ---------------------------------------------------------------------


class TestSchemaHelpCoverage:
    def test_every_field_has_non_empty_help(self) -> None:
        """Regression guard: a new field can't be merged without a
        help string. The Dash form is the user's primary entry into
        the model, so missing tooltips are a UX defect."""
        missing = [f.path for f in FIELD_SCHEMA if not (f.help or "").strip()]
        assert not missing, (
            f"{len(missing)} field(s) missing `help` text: {missing[:5]}..."
        )

    def test_help_text_is_concise(self) -> None:
        """Soft cap on hint length so they fit a Bootstrap tooltip
        without truncation. 360 chars is the rendered ``max-width``
        budget × 2 lines at the current font-size; anything longer
        starts wrapping into a 3rd line and reads as a paragraph,
        not a hint.
        """
        too_long = [
            (f.path, len(f.help)) for f in FIELD_SCHEMA
            if f.help and len(f.help) > 360
        ]
        assert not too_long, (
            f"hints longer than 360 chars (will wrap awkwardly): "
            f"{too_long[:5]}"
        )

    def test_help_text_does_not_repeat_label(self) -> None:
        """A hint that just restates the label is no hint at all.
        We check that the help string isn't a strict prefix of the
        label (the usual anti-pattern, e.g. label='Spouse A current
        age' / help='Spouse A current age').
        """
        offenders = []
        for f in FIELD_SCHEMA:
            if not f.help:
                continue
            if f.help.lower().rstrip(".") == f.label.lower().rstrip(".") :
                offenders.append(f.path)
        assert not offenders, (
            f"help text identical to label (no info added): {offenders}"
        )


# ---------------------------------------------------------------------
# _help_components / _field_row rendering
# ---------------------------------------------------------------------


class TestHelpComponentRendering:
    def test_field_with_help_renders_icon_and_tooltip(self) -> None:
        fld = next(
            f for f in FIELD_SCHEMA if f.path == "inputs.spouse_a_age_start"
        )
        components = _help_components(fld)
        # Three pieces: NBSP separator span, icon span, tooltip.
        assert len(components) == 3, components

        tree = _to_json(components)
        icons = _find(
            tree,
            lambda n: n.get("type") == "Span"
            and "form-hint-icon" in (n.get("props", {}).get("className") or ""),
        )
        tooltips = _find(tree, lambda n: n.get("type") == "Tooltip")

        assert len(icons) == 1, "expected exactly one ⓘ icon"
        assert len(tooltips) == 1, "expected exactly one Bootstrap tooltip"

        # Icon and tooltip must share the same target id, otherwise
        # Bootstrap silently ignores the tooltip (no popover on hover).
        icon_id = icons[0]["props"]["id"]
        tooltip_target = tooltips[0]["props"]["target"]
        assert icon_id == tooltip_target == _hint_id(fld.path)

        # The icon must NOT carry a native ``title`` attribute —
        # that would race the Bootstrap popover and produce a
        # duplicate immediate-tooltip on hover. Accessibility is
        # provided via ``aria-label`` instead.
        assert "title" not in icons[0]["props"], (
            "Icon must not set a native `title` attribute "
            "(would produce a duplicate browser tooltip)"
        )
        assert icons[0]["props"]["aria-label"] == fld.help
        # Tooltip carries the same help text as its first child.
        assert tooltips[0]["props"]["children"] == fld.help

    def test_help_id_strips_dotted_path(self) -> None:
        """Bootstrap's tooltip target uses the id as a CSS selector
        in some code paths; dots are CSS class separators and must
        be escaped. We dodge the issue by replacing dots with double
        underscores on the icon's id.
        """
        assert _hint_id("inputs.spouse_a_age_start") == "hint-inputs__spouse_a_age_start"
        assert _hint_id("config.market.equity_mu") == "hint-config__market__equity_mu"
        # Plain (no dots) id is just prefixed.
        assert _hint_id("foo") == "hint-foo"

    def test_field_without_help_skips_icon(self) -> None:
        """Defensive guard: a future field added without a hint must
        NOT render an icon (otherwise we'd ship a tooltip target
        with no associated tooltip, which throws a console error).
        """
        fld = FormField(
            path="config.scratch", label="Scratch", help=None
        )
        components = _help_components(fld)
        assert components == []

        # The full row also stays icon-free.
        tree = _to_json(_field_row(fld, 0))
        icons = _find(
            tree,
            lambda n: n.get("type") == "Span"
            and "form-hint-icon" in (n.get("props", {}).get("className") or ""),
        )
        assert icons == []

    def test_field_row_does_not_set_native_title(self) -> None:
        """The rendered field row must not carry a native HTML
        ``title`` attribute on any node — that was the source of
        the duplicate-tooltip bug (browser fires native tooltip
        immediately, then the delayed ``dbc.Tooltip`` fires too).
        Help is delivered exclusively via the icon's
        ``dbc.Tooltip`` + ``aria-label``.
        """
        fld = next(
            f for f in FIELD_SCHEMA if f.path == "config.start_year"
        )
        tree = _to_json(_field_row(fld, 2024))
        with_title = [
            n for n in _walk(tree)
            if n.get("props", {}).get("title")
        ]
        assert not with_title, (
            "no node in the field row may set a native `title` "
            f"attribute, but found {len(with_title)}: "
            f"{[n.get('type') for n in with_title]}"
        )


# ---------------------------------------------------------------------
# Smoke: ids stay unique across the whole schema
# ---------------------------------------------------------------------


class TestHintIdsAreUnique:
    def test_every_hint_id_is_unique(self) -> None:
        """Two fields with the same hint id would mean Bootstrap
        wires the second tooltip to the first icon (or fails
        silently). We rely on the field path being globally unique
        in the schema; this test pins that contract for hint ids.
        """
        ids = [_hint_id(f.path) for f in FIELD_SCHEMA if f.help]
        dupes = {i for i in ids if ids.count(i) > 1}
        assert not dupes, f"duplicate hint ids: {dupes}"

    def test_every_field_in_schema_renders_a_row_without_error(self) -> None:
        """Smoke: ``_field_row`` doesn't blow up on any schema entry.
        Catches regressions where the help-rendering path adds a
        kwarg the underlying ``html.Span`` / ``dbc.Tooltip`` doesn't
        accept (the kind of error that surfaces only at form
        render-time in the live app).
        """
        for fld in FIELD_SCHEMA:
            row = _field_row(fld, None)
            assert row is not None
