"""Top-level Dash layout.

The page is a two-column dashboard:

  * Left: a sidebar with the scenario form (Simple / Advanced sub-tabs)
    and a top bar of Load/Save/Run controls.
  * Right: the results panel (Overview / Taxes / Strategies / Monte
    Carlo / Year-by-year).

The form is rendered from `dash_app.forms.FIELD_SCHEMA`, so adding a
new scenario field is a one-line schema entry.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from . import figures
from .forms import FormField, fields_by_group, fields_by_tier


# ---------------------------------------------------------------------
# Form rendering
# ---------------------------------------------------------------------


def _input_for(fld: FormField, value: Any) -> Any:
    """Render the right control for a field's `kind`."""
    base_id = fld.component_id

    if fld.kind == "select":
        options = [
            {"label": label, "value": "" if v is None else v}
            for v, label in fld.options
        ]
        return dcc.Dropdown(
            id=base_id,
            options=options,
            value="" if value is None else value,
            clearable=False,
            style={"fontSize": "0.85rem"},
        )

    if fld.kind == "bool":
        return dbc.Switch(id=base_id, value=bool(value), persistence=False)

    if fld.kind == "percent":
        # Percent stays a float in storage; we just hint the unit on the
        # label rather than rescaling on input. That keeps round-trips
        # exact (0.07 in JSON, 0.07 in the box) and avoids a "did the
        # user enter 7 or 0.07?" guessing game.
        #
        # `step="any"` is critical: HTML5 step validation is implemented
        # as `(value - min) % step == 0`, which fails on legitimate
        # decimals because of float-point representation
        # (`0.07 / 0.005 == 14.000000000000002`, not 14). Browsers then
        # render the box as `:invalid` -> red text. Using "any" disables
        # the step constraint while keeping `min`/`max` bounds intact.
        return dcc.Input(
            id=base_id,
            type="number",
            value=value,
            step="any",
            min=fld.min,
            max=fld.max,
            debounce=True,
            className="form-control form-control-sm",
        )

    if fld.kind == "int":
        return dcc.Input(
            id=base_id,
            type="number",
            value=value,
            step=fld.step or 1,
            min=fld.min,
            max=fld.max,
            debounce=True,
            className="form-control form-control-sm",
        )

    # Generic numeric / text inputs.
    #
    # `<input type="number">` defaults to `step=1` when the attribute is
    # missing, and even an explicit `step=1000` flags real-world values
    # like `taxable_brokerage=460270.90` (or `pension_balance=416741.12`)
    # as `:invalid` because they are not exact multiples — Chrome
    # paints the text red. The schema's per-field `fld.step` is
    # primarily a *spinner increment hint* (how far the up/down arrows
    # should jump), not a validation rule, so we always render with
    # `step="any"` here to disable HTML5 step validation while still
    # honoring `min` / `max` bounds. Spinner increments default to 1,
    # which is acceptable for currency entry where most users type the
    # value rather than spinning.
    if fld.kind == "number":
        return dcc.Input(
            id=base_id,
            type="number",
            value=value,
            step="any",
            min=fld.min,
            max=fld.max,
            debounce=True,
            className="form-control form-control-sm",
        )

    return dcc.Input(
        id=base_id,
        type="text",
        value=value,
        debounce=True,
        className="form-control form-control-sm",
    )


def _label_text(fld: FormField) -> str:
    if fld.kind == "percent":
        return f"{fld.label} (decimal, e.g. 0.07)"
    return fld.label


def _hint_id(path: str) -> str:
    """Stable DOM id for a field's help-tooltip target.

    Dotted paths (e.g. ``inputs.spouse_a_age_start``) aren't valid in
    a plain HTML id selector context — Dash itself accepts them, but
    Bootstrap's tooltip target uses the id as a CSS selector via
    ``getElementById`` and can stumble on the dot in some browsers.
    Replacing dots with double underscores keeps the id readable
    while staying CSS-safe.
    """
    return "hint-" + path.replace(".", "__")


def _help_components(fld: FormField) -> list[Any]:
    """Render the ⓘ icon + Bootstrap tooltip for a field's help text.

    Returns an empty list when the field has no ``help`` so callers
    can splat the result into a label's children unconditionally.

    The single tooltip surface is the ``dbc.Tooltip`` (delayed
    Bootstrap popover with our richer styling). We deliberately do
    NOT set a native ``title`` attribute on the icon — the browser
    would render its own immediate native tooltip in addition to
    the Bootstrap one, producing a double-popover effect on hover.

    Screen-reader / accessibility support is preserved via the
    ⓘ icon's ``aria-label`` attribute, which assistive technology
    prefers over ``title`` anyway (``title`` is famously
    inconsistently surfaced by AT software).
    """
    if not fld.help:
        return []
    icon_id = _hint_id(fld.path)
    return [
        # Non-breaking space so the icon clings to the label text and
        # never wraps onto its own line on narrow sidebars.
        html.Span("\u00a0"),
        html.Span(
            "ⓘ",
            id=icon_id,
            className="form-hint-icon",
            **{"aria-label": fld.help},
        ),
        # Tooltip placement="left" so the popover never gets clipped
        # by the right edge of the (relatively narrow) sidebar.
        dbc.Tooltip(
            fld.help,
            target=icon_id,
            placement="left",
            delay={"show": 150, "hide": 50},
            className="form-hint-tooltip",
        ),
    ]


def _field_row(fld: FormField, value: Any) -> dbc.Row:
    # Help is delivered exclusively via the ⓘ icon's ``dbc.Tooltip``
    # inside the label's children — no native ``title`` on the
    # label itself, otherwise the browser fires an immediate
    # tooltip in addition to the delayed Bootstrap one.
    return dbc.Row(
        [
            dbc.Col(
                html.Label(
                    [_label_text(fld), *_help_components(fld)],
                    className="text-end small text-muted py-1 d-block w-100",
                ),
                width=7,
            ),
            dbc.Col(_input_for(fld, value), width=5, className="py-1"),
        ],
        className="g-1",
    )


def _section(group: str, fields: list[FormField], values: dict[str, Any]) -> dbc.AccordionItem:
    rows = [_field_row(f, values.get(f.path)) for f in fields]
    return dbc.AccordionItem(rows, title=group)


def render_form(values: dict[str, Any], tier: str) -> dbc.Accordion:
    flds = list(fields_by_tier(tier))
    grouped = fields_by_group(flds)
    items = [_section(g, fs, values) for g, fs in grouped.items()]
    return dbc.Accordion(
        items,
        always_open=False,
        start_collapsed=(tier == "advanced"),
        className="scenario-form",
    )


# ---------------------------------------------------------------------
# Sidebar (form + top bar)
# ---------------------------------------------------------------------


def top_bar() -> dbc.Card:
    """Run-control card pinned above the main tabs.

    Three vertically-stacked sections, each in its own row so the
    columns stay consistent regardless of viewport width:

    1. **Scenario I/O** — upload + download controls plus a
       persistent "currently loaded" badge so the filename stays
       visible even after the user kicks off a run (the run-status
       area below would otherwise overwrite it).
    2. **Run mode** — a horizontal radio strip across the full
       width. Horizontal layout keeps the three options on a
       single line without wrapping the longest label
       ("Four + Monte Carlo (~10-30s)") onto a second row, which
       was the source of the inconsistent heights in the previous
       layout.
    3. **Run options + execute** — the MC-only knobs (paths /
       seed) sit on the left, with a generous Run button on the
       right. All three controls share the same row height so the
       baseline stays clean.

    The ``run-status`` div at the bottom is intentionally
    *separate* from the "scenario loaded" badge — runs can
    overwrite it freely without touching the filename indicator.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                # ---- Section 1: scenario I/O ---------------------
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label(
                                    "Load scenario JSON",
                                    className="form-label small",
                                ),
                                dcc.Upload(
                                    id="scenario-upload",
                                    children=html.Div(
                                        [
                                            html.I(
                                                className="me-2",
                                                style={"fontSize": "1.1rem"},
                                                children="\U0001F4C1",  # folder
                                            ),
                                            "Drag & drop a scenario file or ",
                                            html.A(
                                                "browse...",
                                                className="text-primary",
                                                style={"cursor": "pointer"},
                                            ),
                                        ],
                                        className="text-muted small text-center",
                                    ),
                                    multiple=False,
                                    accept=".json,application/json",
                                    className="upload-box",
                                    style={
                                        "border": "1px dashed #94a3b8",
                                        "borderRadius": "6px",
                                        "padding": "10px",
                                        "minHeight": "44px",
                                        "cursor": "pointer",
                                    },
                                ),
                                # Persistent indicator of the
                                # currently-loaded scenario. Sits
                                # right below the upload box so the
                                # filename never gets clobbered by
                                # later run-status writes.
                                html.Div(
                                    id="scenario-loaded-name",
                                    className="small text-muted mt-1 scenario-loaded-name",
                                ),
                            ],
                            md=8,
                        ),
                        dbc.Col(
                            [
                                html.Label(
                                    "Save scenario",
                                    className="form-label small",
                                ),
                                dbc.Button(
                                    "Download JSON",
                                    id="scenario-save-btn",
                                    color="secondary",
                                    size="sm",
                                    className="w-100",
                                ),
                                dcc.Download(id="scenario-download"),
                            ],
                            md=4,
                            className="d-flex flex-column",
                        ),
                    ],
                    className="g-2",
                ),
                html.Hr(className="my-2"),
                # ---- Section 2: run mode -------------------------
                html.Div(
                    [
                        html.Label(
                            "Run mode",
                            className="form-label small mb-1",
                        ),
                        dcc.RadioItems(
                            id="run-mode",
                            options=[
                                {"label": " Single sim (~1s)",
                                 "value": "single"},
                                {"label": " Four strategies (~5-10s)",
                                 "value": "four_strategies"},
                                {"label": " Four + Monte Carlo (~10-30s)",
                                 "value": "four_plus_mc"},
                            ],
                            value="four_strategies",
                            # Horizontal layout — `inline-block` per
                            # label keeps the three on one line and
                            # prevents the longest label from
                            # wrapping into a second visual row.
                            labelStyle={
                                "display": "inline-block",
                                "marginRight": "1.25rem",
                                "fontSize": "0.85rem",
                            },
                            inputStyle={"marginRight": "6px"},
                            className="run-mode-radios",
                        ),
                    ],
                    className="mb-2",
                ),
                # ---- Section 3: run options + execute ------------
                #
                # Layout note: row 1 holds the optimizer controls
                # (objective + maxiter + popsize), row 2 holds the MC
                # paths + seed + Run button. Pre-fix the optimizer
                # objective and DE knobs were hard-coded at the
                # ``runner._build_four`` level (terminal / 20 / 10);
                # exposing them here lets advanced users target CVaR
                # or probability-of-success and trade optimizer cost
                # against solution quality.
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label(
                                    "Optimizer objective",
                                    className="form-label small",
                                ),
                                dcc.Dropdown(
                                    id="opt-objective",
                                    options=[
                                        {
                                            "label": "Terminal NW (default)",
                                            "value": "terminal",
                                        },
                                        {
                                            "label": "CVaR (downside risk)",
                                            "value": "cvar",
                                        },
                                        {
                                            "label": "Probability of success",
                                            "value": "p_success",
                                        },
                                    ],
                                    value="terminal",
                                    clearable=False,
                                    className="form-control-sm",
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.Label(
                                    "Optimizer maxiter",
                                    className="form-label small",
                                ),
                                dcc.Input(
                                    id="opt-maxiter",
                                    type="number",
                                    value=20,
                                    min=1,
                                    max=200,
                                    step=1,
                                    className="form-control form-control-sm",
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label(
                                    "Optimizer popsize",
                                    className="form-label small",
                                ),
                                dcc.Input(
                                    id="opt-popsize",
                                    type="number",
                                    value=10,
                                    min=4,
                                    max=60,
                                    step=1,
                                    className="form-control form-control-sm",
                                ),
                            ],
                            md=3,
                        ),
                    ],
                    className="g-2 align-items-end mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label(
                                    "MC paths",
                                    className="form-label small",
                                ),
                                dcc.Input(
                                    id="mc-paths",
                                    type="number",
                                    value=200,
                                    min=10,
                                    max=2000,
                                    step=10,
                                    className="form-control form-control-sm",
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label(
                                    "Seed",
                                    className="form-label small",
                                ),
                                dcc.Input(
                                    id="mc-seed",
                                    type="number",
                                    value=0,
                                    step=1,
                                    className="form-control form-control-sm",
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                # Empty label spacer — keeps the
                                # button baseline aligned with the
                                # two inputs to its left.
                                html.Label(
                                    html.Span("\u00a0"),
                                    className="form-label small",
                                ),
                                dbc.Button(
                                    "Run",
                                    id="run-btn",
                                    color="primary",
                                    size="sm",
                                    className="w-100",
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-2 align-items-end",
                ),
                html.Div(
                    id="run-status",
                    className="small text-muted mt-2",
                ),
            ]
        ),
        className="mb-3",
    )


def sidebar(values: dict[str, Any]) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Scenario", className="card-title"),
                dbc.Tabs(
                    [
                        dbc.Tab(
                            render_form(values, "simple"),
                            label="Simple",
                            tab_id="form-tab-simple",
                        ),
                        dbc.Tab(
                            render_form(values, "advanced"),
                            label="Advanced",
                            tab_id="form-tab-advanced",
                        ),
                    ],
                    id="form-tabs",
                    active_tab="form-tab-simple",
                ),
            ]
        ),
        className="sidebar-card",
    )


# ---------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------


def kpi_row_placeholder() -> html.Div:
    return html.Div(id="overview-kpis", className="kpi-row")


def overview_tab() -> dbc.Tab:
    return dbc.Tab(
        [
            html.Div(id="overview-kpis", className="my-3"),
            dcc.Graph(
                id="fig-balance-stack",
                className="tax-figure my-3",
                config={"displaylogo": False},
            ),
            dcc.Graph(
                id="fig-overview-growth",
                className="tax-figure my-3",
                config={"displaylogo": False},
            ),
        ],
        label="Overview", tab_id="tab-overview",
    )


def taxes_tab() -> dbc.Tab:
    # Each chart gets a `tax-figure` className so the CSS in
    # `assets/fira-code.css` can wrap it in a light slate-200 border
    # — matches the card aesthetic used by the KPI tiles and the
    # report iframe so the Taxes tab doesn't read as "two charts
    # floating on the page background". `my-3` gives the cards a
    # bit of vertical breathing room from each other and from the
    # tab nav.
    #
    # The `html.Hr` between the two cards is a deliberate visual
    # break: the top card answers "what tax did each strategy
    # pay?" while the bottom card answers "what conversion /
    # RMD activity drove that?". They tell two related but
    # distinct stories, so a thin divider helps the eye treat
    # them as a sequence rather than a single tall scroll
    # region. Styling lives in `.tax-figure-divider` (CSS).
    return dbc.Tab(
        [
            dcc.Graph(
                id="fig-taxes-panel",
                config={"displaylogo": False},
                className="tax-figure my-3",
            ),
            html.Hr(className="tax-figure-divider"),
            dcc.Graph(
                id="fig-conversion-panel",
                config={"displaylogo": False},
                className="tax-figure my-3",
            ),
        ],
        label="Taxes", tab_id="tab-taxes",
    )


def strategies_tab() -> dbc.Tab:
    """Strategies-tab layout.

    Two stacked sections:

    1. ``strategies-comparison`` — a per-knob × per-strategy table
       (or, in single-strategy runs, the optimizer-picks callout).
       Surfaces *what each strategy changed* relative to the
       baseline so the user can see at a glance which decision
       axes the optimizer overrode.
    2. ``fig-strategy-compare`` — bar chart of the headline outcome
       metrics (terminal NW, lifetime tax, lifetime IRMAA) so the
       user can visually correlate parameter overrides above with
       outcome deltas below.
    """
    return dbc.Tab(
        [
            html.Div(id="strategies-comparison", className="my-3"),
            dcc.Graph(id="fig-strategy-compare", config={"displaylogo": False}),
        ],
        label="Strategies", tab_id="tab-strategies",
    )


def mc_tab() -> dbc.Tab:
    return dbc.Tab(
        [
            dcc.Graph(id="fig-mc-histogram", config={"displaylogo": False}),
            dcc.Graph(id="fig-mc-fan", config={"displaylogo": False}),
        ],
        label="Monte Carlo", tab_id="tab-mc",
    )


def yearly_tab() -> dbc.Tab:
    # Conditional styles per column group are computed once at
    # layout-build time from `_YEARLY_COLUMN_GROUPS` (the single
    # source of truth for column order + per-group color). Dash
    # gracefully ignores rules whose `column_id` isn't actually
    # in the table data, so it's safe to declare every rule
    # here even though `_populate_yearly_table` may filter to a
    # subset of the columns at runtime.
    yearly_styles = figures.yearly_table_styles()
    return dbc.Tab(
        [
            html.Div(
                [
                    html.Label("Strategy", className="form-label small"),
                    dcc.Dropdown(
                        id="yearly-strategy",
                        clearable=False,
                        style={"width": "260px", "fontSize": "0.85rem"},
                    ),
                ],
                className="mb-2",
            ),
            dash_table.DataTable(
                id="yearly-table",
                page_size=100,
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "0.85rem", "padding": "4px"},
                style_header={"backgroundColor": "#f1f5f9", "fontWeight": "600"},
                style_header_conditional=yearly_styles[
                    "style_header_conditional"
                ],
                style_data_conditional=yearly_styles[
                    "style_data_conditional"
                ],
                style_cell_conditional=yearly_styles[
                    "style_cell_conditional"
                ],
                sort_action="native",
                filter_action="native",
            ),
        ],
        label="Year-by-year", tab_id="tab-yearly",
    )


def report_tab() -> dbc.Tab:
    """Action-plan report tab.

    The iframe auto-renders whenever the user activates this tab and a
    completed run is in the cache (callback: ``_render_report_tab``).
    The "Download HTML" button is purely for saving a copy of what the
    user already sees rendered above it; it does not drive the iframe
    state.

    The iframe lets the report's own ``<style>`` block paint the
    document the same way it would in a standalone browser tab, with
    no CSS bleed from the dashboard's Bootstrap theme.
    """
    return dbc.Tab(
        [
            html.Div(
                [
                    html.Span(
                        "The action plan renders automatically below "
                        "when you switch to this tab after a Run.",
                        className="text-muted small me-3",
                    ),
                    dbc.Button(
                        "Download HTML",
                        id="report-download-btn",
                        color="primary",
                        outline=True,
                        size="sm",
                        title=(
                            "Save the report as a self-contained HTML "
                            "file. Print to PDF from your browser if "
                            "you want a PDF."
                        ),
                    ),
                    dcc.Download(id="report-download"),
                ],
                className="my-3 d-flex align-items-center",
            ),
            dcc.Loading(
                html.Iframe(
                    id="report-iframe",
                    srcDoc=(
                        # Match the placeholder font stack from
                        # `_placeholder_srcdoc` in `dash_app/app.py`
                        # so the cold-state message inside the iframe
                        # uses the same Fira Code typography as the
                        # surrounding dashboard. The iframe is isolated
                        # from the parent page's CSS so we have to
                        # request the font directly here.
                        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                        "<link rel='preconnect' href='https://fonts."
                        "googleapis.com'>"
                        "<link rel='preconnect' href='https://fonts."
                        "gstatic.com' crossorigin>"
                        "<link href='https://fonts.googleapis.com/css2?"
                        "family=Fira+Code:wght@400;500&display=swap' "
                        "rel='stylesheet'>"
                        "<style>body{font-family:'Fira Code','Fira Mono',"
                        "ui-monospace,Menlo,Monaco,Consolas,monospace;"
                        "color:#475569;padding:32px;font-size:0.9rem;"
                        "line-height:1.5;font-variant-ligatures:none;}"
                        "</style></head>"
                        "<body><div>Run a scenario to populate the "
                        "action-plan report.</div></body></html>"
                    ),
                    style={
                        "width": "100%",
                        "height": "900px",
                        "border": "1px solid #e2e8f0",
                        "borderRadius": "6px",
                        "background": "white",
                    },
                ),
                type="default",
                color="#0d6efd",
            ),
        ],
        label="Report", tab_id="tab-report",
    )


def results_panel() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            dbc.Tabs(
                [
                    overview_tab(), taxes_tab(), strategies_tab(),
                    mc_tab(), yearly_tab(), report_tab(),
                ],
                id="results-tabs",
                active_tab="tab-overview",
            )
        ),
        className="results-card",
    )


# ---------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------


def build_layout(default_values: dict[str, Any]) -> html.Div:
    return dbc.Container(
        [
            dcc.Store(id="form-values", data=default_values),
            dcc.Store(id="run-result", storage_type="memory"),
            html.H3("Tax Optimizer - Interactive Planner", className="my-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [top_bar(), sidebar(default_values)],
                        lg=4, md=12,
                    ),
                    dbc.Col(
                        dcc.Loading(results_panel(), type="default"),
                        lg=8, md=12,
                    ),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
        className="pb-5",
    )
