"""Dash app entry point + callbacks.

`make_app()` builds and returns a configured `Dash` instance so it can
be embedded in a WSGI server (`app.server`) or run via
`python -m dash_app`. All callbacks live in this module to keep the
import tree shallow.
"""

from __future__ import annotations

import base64
import html as html_module
import json
import warnings
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, html, no_update


# ``html.escape`` from the stdlib clashes with the ``dash.html`` import
# above (we use the latter for component construction). Keep an alias
# under a distinct name so the report-tab callback can sanitize error
# strings before injecting them into the iframe srcDoc.
def html_module_escape(text: str) -> str:
    return html_module.escape(text, quote=True)


def _placeholder_srcdoc(message_html: str) -> str:
    """Wrap an inline HTML message in a styled iframe srcDoc.

    Used when there's nothing meaningful to render yet (no run, cache
    miss, or build error). Keeps the iframe self-contained so its
    styling doesn't leak from the parent dashboard's Bootstrap and
    vice-versa, matching the rendered-report behavior.

    The font stack mirrors the dashboard's ``fira-code.css`` so cold-
    state messages feel continuous with the surrounding UI even
    though the iframe document is isolated. We import Fira Code from
    Google Fonts inline because the iframe's document scope doesn't
    inherit the parent page's ``external_stylesheets``.
    """
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link rel='preconnect' href='https://fonts.gstatic.com' "
        "crossorigin>"
        "<link href='https://fonts.googleapis.com/css2?family=Fira+Code:"
        "wght@400;500&display=swap' rel='stylesheet'>"
        "<style>"
        "body{font-family:'Fira Code','Fira Mono',ui-monospace,Menlo,"
        "Monaco,Consolas,monospace;color:#475569;padding:32px;"
        "font-size:0.9rem;line-height:1.5;font-variant-ligatures:none;}"
        "</style></head><body>"
        f"<div>{message_html}</div></body></html>"
    )

from tax_optimizer.config import Config
from tax_optimizer.inputs import Inputs
from tax_optimizer.scenario import (
    ScenarioError,
    apply_scenario,
)

from . import figures
from .forms import get_field
from .layout import build_layout
from .report_builder import (
    build_html_payload,
    build_inline_html,
    cache_run,
    get_cached,
)
from .runner import (
    deserialize_strategy_df,
    run_scenario,
    serialize_run_result,
)
from .state import (
    apply_form_values,
    cfg_inputs_to_form_values,
    default_form_values,
    form_values_to_scenario,
)


def _build_kpi_tiles(tiles: list[tuple[str, str]]) -> list[Any]:
    return [
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(label, className="text-muted small"),
                    html.Div(value, className="fs-5 fw-semibold"),
                ]
            ),
            className="kpi-tile flex-grow-1",
            style={"minWidth": "180px"},
        )
        for label, value in tiles
    ]


def _strategy_callout(strategies: dict[str, Any], winner_name: str | None) -> Any:
    if not strategies or not winner_name or winner_name not in strategies:
        return html.Div("Run with 'Four strategies' or 'Four + MC' to see optimizer picks.",
                        className="text-muted small")
    w = strategies[winner_name]
    cs = w.get("cfg_summary", {})
    items = [
        ("Winner", winner_name),
        ("Roth-401(k) % (A / B)",
         f"{cs.get('spouse_a_roth_401k_pct', 0) * 100:.0f}%"
         f" / {cs.get('spouse_b_roth_401k_pct', 0) * 100:.0f}%"),
        ("Roth conversion target",
         f"{cs.get('roth_conversion_target_bracket', 0) * 100:.0f}%"),
        ("After-tax 401(k) % (A / B)",
         f"{cs.get('spouse_a_after_tax_401k_pct', 0) * 100:.0f}%"
         f" / {cs.get('spouse_b_after_tax_401k_pct', 0) * 100:.0f}%"),
        ("SS claim age (A / B)",
         f"{cs.get('ss_start_age_a', '-')}"
         f" / {cs.get('ss_start_age_b', '-')}"),
        ("Terminal NW", f"${w['summary'].get('terminal_after_tax', 0):,.0f}"),
    ]
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Optimizer picks", className="text-muted small mb-2"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(label + ": ", className="text-muted"),
                                html.Strong(value),
                            ],
                            className="me-3 d-inline-block",
                        )
                        for label, value in items
                    ]
                ),
            ]
        ),
        className="callout-card",
    )


# ---------------------------------------------------------------------
# Strategy comparison table
# ---------------------------------------------------------------------

# Display labels for the canonical four strategies. Anything not in
# this map renders with the raw key (typically the user's own
# strategy name from a custom run). Keeping this map small and
# explicit avoids surprising abbreviations on third-party strategy
# names.
_STRATEGY_HEADERS: dict[str, str] = {
    "S0_baseline": "Baseline",
    "S1_all_roth_401k": "All Roth",
    "S2_bracket_fill_22": "Fill to 22%",
    "S3_optimized": "Optimizer",
}

# Strategies whose values define the "no override" reference column
# for diff highlighting. The first match wins so users running with a
# subset (e.g. only S0 + S3) still get a sensible baseline.
_BASELINE_PREFERENCE: tuple[str, ...] = (
    "S0_baseline",
    "S1_all_roth_401k",
    "S2_bracket_fill_22",
)


def _fmt_value(value: Any, kind: str) -> str:
    """Render a strategy parameter / outcome value as a human string."""
    if value is None:
        return "—"
    try:
        if kind == "pct":
            return f"{float(value) * 100:.0f}%"
        if kind == "money":
            return f"${float(value):,.0f}"
        if kind == "int":
            return f"{int(value)}"
    except (TypeError, ValueError):
        return str(value)
    return str(value)


def _approx_equal(a: Any, b: Any, *, tol: float = 1e-9) -> bool:
    """Robust equality for numeric / int / None values across strategies.

    The cfg_summary dict round-trips through JSON so we can't rely on
    object identity. Floats can also wobble at the 1e-12 level when
    the same `inputs` dataclass is fed through different
    `dataclasses.replace()` chains. ``tol=1e-9`` is well below any
    user-visible delta (we round to 0% / $0 in the table) but well
    above float-equality wobble.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def _pick_baseline(names: list[str]) -> str:
    """Return the column name to diff every other column against."""
    for candidate in _BASELINE_PREFERENCE:
        if candidate in names:
            return candidate
    # Fallback: the first column the runner emitted. ``RunResult``
    # preserves insertion order so this is deterministic.
    return names[0]


def _strategy_compare_table(
    strategies: dict[str, Any], winner_name: str | None
) -> Any:
    """Build the per-knob × per-strategy comparison table.

    Rows are split into two visually distinct groups:

    1. **Decision parameters** — the seven knobs the optimizer can
       turn (Roth-401(k) % per spouse, conversion bracket, after-tax
       401(k) % per spouse, SS claim age per spouse). Cells whose
       value differs from the baseline column get a yellow tint;
       cells in the optimizer column that differ from baseline get
       a stronger green tint to make "what the optimizer overrode"
       jump out at a glance.

    2. **Outcomes** — terminal after-tax NW, lifetime federal tax
       NPV, lifetime IRMAA NPV, peak marginal rate. Same diff
       coloring rules so users can visually correlate which knob
       move drove which outcome delta.

    Falls back to :func:`_strategy_callout` for single-strategy
    runs, where there's nothing to compare *against*. The callout
    is the right shape for that case (Winner / picks summary).
    """
    if not strategies:
        return html.Div(
            "Click 'Run' to populate the dashboard.",
            className="text-muted small",
        )

    names = list(strategies.keys())
    if len(names) < 2:
        # Nothing to diff against — defer to the single-row callout.
        return _strategy_callout(strategies, winner_name)

    baseline_name = _pick_baseline(names)

    # (display label, payload key, value kind, source bucket)
    param_rows: list[tuple[str, str, str, str]] = [
        ("Roth-401(k) % (Spouse A)", "spouse_a_roth_401k_pct", "pct", "cfg_summary"),
        ("Roth-401(k) % (Spouse B)", "spouse_b_roth_401k_pct", "pct", "cfg_summary"),
        ("Roth conversion target bracket", "roth_conversion_target_bracket",
         "pct", "cfg_summary"),
        ("After-tax 401(k) % (Spouse A)", "spouse_a_after_tax_401k_pct",
         "pct", "cfg_summary"),
        ("After-tax 401(k) % (Spouse B)", "spouse_b_after_tax_401k_pct",
         "pct", "cfg_summary"),
        ("SS claim age (Spouse A)", "ss_start_age_a", "int", "cfg_summary"),
        ("SS claim age (Spouse B)", "ss_start_age_b", "int", "cfg_summary"),
    ]
    outcome_rows: list[tuple[str, str, str, str]] = [
        ("Terminal after-tax NW", "terminal_after_tax", "money", "summary"),
        ("Lifetime federal tax (NPV)", "lifetime_tax_npv", "money", "summary"),
        ("Lifetime IRMAA (NPV)", "lifetime_irmaa_npv", "money", "summary"),
        ("Peak marginal rate", "peak_marginal", "pct", "summary"),
    ]

    def _value_for(strat_name: str, key: str, source: str) -> Any:
        return strategies[strat_name].get(source, {}).get(key)

    def _cell(strat_name: str, key: str, kind: str, source: str) -> Any:
        val = _value_for(strat_name, key, source)
        baseline_val = _value_for(baseline_name, key, source)
        differs = strat_name != baseline_name and not _approx_equal(
            val, baseline_val
        )
        # Outcome rows always differ in $ terms — coloring those by
        # diff would just paint every cell. Restrict the diff
        # highlight to the parameter rows where the column-vs-column
        # comparison actually says something about the strategy's
        # *inputs* rather than its outputs.
        is_param_row = source == "cfg_summary"
        css_classes = ["text-end"]
        if is_param_row and differs:
            if strat_name == winner_name:
                css_classes.append("strategy-cell-optimized")
            else:
                css_classes.append("strategy-cell-changed")
        return html.Td(
            _fmt_value(val, kind), className=" ".join(css_classes)
        )

    def _row(label: str, key: str, kind: str, source: str) -> Any:
        return html.Tr(
            [html.Td(label, className="text-muted small")]
            + [_cell(name, key, kind, source) for name in names]
        )

    def _section_header(text: str) -> Any:
        return html.Tr(
            html.Td(
                text,
                colSpan=len(names) + 1,
                className=(
                    "fw-semibold text-muted small "
                    "strategy-compare-section bg-light"
                ),
            )
        )

    def _column_header(strat_name: str) -> Any:
        label = _STRATEGY_HEADERS.get(strat_name, strat_name)
        is_winner = strat_name == winner_name
        return html.Th(
            [
                html.Span(label),
                html.Span(
                    " ★", className="text-success ms-1",
                    title="Optimizer's chosen strategy",
                ) if is_winner else "",
                html.Div(
                    strat_name,
                    className="text-muted small fw-normal",
                    style={"fontSize": "0.7rem"},
                ),
            ],
            className="text-end",
        )

    legend = html.Div(
        [
            html.Span("Cell shading: ", className="text-muted small me-2"),
            html.Span(
                "differs from baseline",
                className="strategy-cell-changed px-2 me-2 small",
            ),
            html.Span(
                "optimizer's override",
                className="strategy-cell-optimized px-2 me-2 small",
            ),
            html.Span(
                f"(baseline column: {_STRATEGY_HEADERS.get(baseline_name, baseline_name)})",
                className="text-muted small",
            ),
        ],
        className="mb-2",
    )

    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [html.Th("Parameter", className="text-muted small")]
                    + [_column_header(n) for n in names]
                )
            ),
            html.Tbody(
                [_section_header("Decision parameters")]
                + [_row(*r) for r in param_rows]
                + [_section_header("Outcomes")]
                + [_row(*r) for r in outcome_rows]
            ),
        ],
        bordered=True,
        hover=True,
        size="sm",
        responsive=True,
        className="strategy-compare-table mb-3",
    )

    return html.Div(
        [
            html.Div(
                "Strategy comparison — what each strategy changed",
                className="fw-semibold mb-2",
            ),
            legend,
            table,
        ]
    )


# ---------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------


def make_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            # Fira Code (Google Fonts). The CSS rules that *apply* the
            # font live in `dash_app/assets/fira-code.css`, which Dash
            # auto-loads from the `assets/` folder on every page. We
            # only request the weights actually used in the UI to keep
            # the font payload trim. ``display=swap`` lets the browser
            # paint with the system fallback first and swap when Fira
            # Code arrives — no flash of invisible text.
            "https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&display=swap",
        ],
        suppress_callback_exceptions=True,
        title="Tax Optimizer",
    )
    # Fast handshake to fonts.googleapis.com / fonts.gstatic.com so the
    # font CSS + woff2 files arrive in parallel with the rest of the
    # bundle. Without these `<link rel="preconnect">` tags the browser
    # only opens TCP+TLS to gstatic *after* parsing the stylesheet,
    # adding ~200ms to first paint on a cold load.
    app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""
    app.layout = build_layout(default_form_values())

    # ---- Form values store -----------------------------------------
    # Whenever any form input changes, we rebuild the flat `{path: value}`
    # store. We use pattern-matching IDs so this single callback handles
    # every field.

    @app.callback(
        Output("form-values", "data"),
        Input({"type": "form-input", "path": ALL}, "value"),
        State({"type": "form-input", "path": ALL}, "id"),
        State("form-values", "data"),
        prevent_initial_call=True,
    )
    def _sync_form_values(values, ids, current):
        current = dict(current or {})
        for v, ident in zip(values, ids):
            path = ident["path"]
            fld = get_field(path)
            if fld is None:
                continue
            # Empty-string from a Dropdown maps to None.
            if v == "":
                current[path] = None
            else:
                current[path] = v
        return current

    # ---- Scenario load (upload + quick-load) -----------------------

    @app.callback(
        Output({"type": "form-input", "path": ALL}, "value"),
        Output("run-status", "children", allow_duplicate=True),
        Input("scenario-upload", "contents"),
        State("scenario-upload", "filename"),
        State({"type": "form-input", "path": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _load_scenario(upload_contents, upload_filename, ids):
        if not upload_contents:
            return [no_update] * len(ids), no_update
        try:
            _header, _, b64 = upload_contents.partition(",")
            raw = base64.b64decode(b64).decode("utf-8")
            payload = json.loads(raw)
            source = upload_filename or "uploaded JSON"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                cfg, inputs = apply_scenario(Config(), Inputs(), payload)
            values = cfg_inputs_to_form_values(cfg, inputs)
        except (ScenarioError, json.JSONDecodeError, ValueError) as e:
            return [no_update] * len(ids), html.Span(
                f"Failed to load scenario: {e}", className="text-danger"
            )

        new_values = []
        for ident in ids:
            v = values.get(ident["path"])
            new_values.append("" if v is None else v)
        return new_values, html.Span(
            f"Loaded scenario from {source}.", className="text-success"
        )

    # ---- Save scenario --------------------------------------------

    @app.callback(
        Output("scenario-download", "data"),
        Input("scenario-save-btn", "n_clicks"),
        State("form-values", "data"),
        prevent_initial_call=True,
    )
    def _save_scenario(_clicks, values):
        """Download the current form state as a scenario JSON file.

        We emit the *form-shaped* dict (built by
        :func:`form_values_to_scenario`) rather than the canonical
        ``scenario_to_dict(cfg, inputs)`` output. The two differ in
        three places that would otherwise corrupt the user's saved
        file:

        1. ``config.spending`` — `_spending_to_dict` always emits
           ``kind="custom"`` with an explicit phases array, even when
           the underlying profile was built via
           ``SpendingProfile.retirement_smile(...)``. The form only
           understands ``flat`` / ``smile``, so a saved-then-reloaded
           file would render as a "custom" profile the user never
           authored. ``form_values_to_scenario`` round-trips the
           original ``smile`` / ``flat`` discriminator faithfully.
        2. Deprecated paths in ``_HIDDEN_PATHS`` (``inputs.annual_expenses``,
           ``inputs.ss.start_age``) get injected by
           ``_inputs_to_dict`` because it walks every dataclass field;
           the form schema explicitly hides them.
        3. Default-valued config fields the user never set
           (``roth_conversion_amount=0.0``,
           ``section125_reduces_fica_wages=True``, ...) are emitted
           by ``scenario_to_dict`` as if the user had specified them.

        We still call :func:`apply_form_values` first as a validation
        step so any malformed form input fails loudly with a clear
        ``ScenarioError`` instead of producing a half-baked JSON file
        the user would only notice on re-load.
        """
        if not values:
            return no_update
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                # Validation: round-trip the form through the decoder
                # so any malformed value surfaces here, not on re-load.
                apply_form_values(values)
                payload = form_values_to_scenario(values)
        except ScenarioError as e:
            return dict(content=f"# Error: {e}\n", filename="scenario_error.txt")

        return dict(
            content=json.dumps(payload, indent=2, default=str),
            filename="scenario.json",
        )

    # ---- Run --------------------------------------------------------

    @app.callback(
        Output("run-result", "data"),
        Output("run-status", "children", allow_duplicate=True),
        Input("run-btn", "n_clicks"),
        State("form-values", "data"),
        State("run-mode", "value"),
        State("mc-paths", "value"),
        State("mc-seed", "value"),
        prevent_initial_call=True,
    )
    def _run(_clicks, values, mode, n_paths, seed):
        if not values:
            return no_update, html.Span(
                "No scenario loaded.", className="text-warning"
            )
        try:
            cfg, inputs = apply_form_values(values)
        except ScenarioError as e:
            return no_update, html.Span(
                f"Scenario error: {e}", className="text-danger"
            )

        try:
            rr = run_scenario(
                cfg, inputs,
                mode=mode or "single",
                n_paths=int(n_paths or 200),
                seed=int(seed or 0),
            )
        except Exception as e:  # noqa: BLE001 - surface any sim error to the UI
            return no_update, html.Span(
                f"Run failed: {e}", className="text-danger"
            )
        # Cache the full Python objects so the report-download callback
        # can re-render the action plan without re-running the
        # simulator. The Store carries only the run_id (UUID) plus the
        # JSON-safe figure data; the actual `RunResult` (with its
        # embedded `Config`, `Inputs`, and per-year DataFrames) lives
        # in `report_builder._RUN_CACHE` keyed by `run_id`.
        run_id = cache_run(cfg, inputs, rr)
        payload = serialize_run_result(rr)
        payload["run_id"] = run_id
        msg = (
            f"Ran '{rr.mode}' in {rr.elapsed_s:.1f}s. "
            f"Winner: {rr.winner_name}."
        )
        return payload, html.Span(msg, className="text-success")

    # ---- Download HTML action-plan report ---------------------------

    @app.callback(
        Output("report-download", "data"),
        Output("run-status", "children", allow_duplicate=True),
        Input("report-download-btn", "n_clicks"),
        State("run-result", "data"),
        prevent_initial_call=True,
    )
    def _download_report(_clicks, run_data):
        """Save the action-plan report as a self-contained HTML file.

        Iframe rendering is owned by ``_render_report_tab`` (which fires
        on tab activation), so this callback now only emits the
        download payload — the user can save the file regardless of
        whether the Report tab is currently active. Both callbacks
        share the same memoized ``CachedRun._html_cache`` so the work
        of building the markdown + tornado sweep happens at most once
        per run.
        """
        if not run_data or not run_data.get("strategies"):
            return no_update, html.Span(
                "Run a scenario first - the report needs simulation results.",
                className="text-warning",
            )
        run_id = run_data.get("run_id")
        cached = get_cached(run_id)
        if cached is None:
            return no_update, html.Span(
                "Run results have expired from the cache. Click 'Run' again.",
                className="text-warning",
            )
        try:
            payload = build_html_payload(cached)
        except Exception as e:  # noqa: BLE001 - surface any render error
            return no_update, html.Span(
                f"Report build failed: {e}", className="text-danger"
            )
        return payload, html.Span(
            f"Report ready: {payload['filename']}.", className="text-success"
        )

    # ---- Render the Report tab (auto on tab activation) -------------

    @app.callback(
        Output("report-iframe", "srcDoc"),
        Input("results-tabs", "active_tab"),
        Input("run-result", "data"),
        prevent_initial_call=True,
    )
    def _render_report_tab(active_tab, run_data):
        """Build the action-plan HTML when the Report tab is selected.

        Triggers on both the active-tab change *and* the ``run-result``
        Store data so:

        * Switching to the Report tab right after a Run renders the
          fresh report immediately.
        * Re-running while the Report tab is already active swaps the
          iframe to the new run's report.
        * Switching away to a different tab and back doesn't pay for
          a rebuild — :func:`build_inline_html` is memoized per
          :class:`CachedRun`, so the second call short-circuits to
          the cached HTML.

        We surface a friendly placeholder srcDoc when there is no
        cached run yet (cold start) or when the cache has been evicted
        (long-lived session that ran more than ``_MAX_CACHED_RUNS``
        scenarios since this one).
        """
        if active_tab != "tab-report":
            # User isn't looking at the report; don't burn cycles on
            # the tornado sweep until they actually need it.
            return no_update
        if not run_data or not run_data.get("strategies"):
            return _placeholder_srcdoc(
                "Run a scenario first to populate the action-plan report."
            )
        run_id = run_data.get("run_id")
        cached = get_cached(run_id)
        if cached is None:
            return _placeholder_srcdoc(
                "Run results have expired from the cache. "
                "Click <strong>Run</strong> again to refresh the report."
            )
        try:
            return build_inline_html(cached)
        except Exception as e:  # noqa: BLE001 - surface any render error
            return _placeholder_srcdoc(
                f"<span style='color:#dc3545'>Report build failed: "
                f"{html_module_escape(str(e))}</span>"
            )

    # ---- Overview tab -----------------------------------------------

    @app.callback(
        Output("overview-kpis", "children"),
        Output("fig-balance-stack", "figure"),
        Input("run-result", "data"),
    )
    def _render_overview(run_data):
        if not run_data or not run_data.get("strategies"):
            return [], figures.empty_figure("Click 'Run' to populate the dashboard")
        winner_name = run_data.get("winner_name")
        strategies = run_data["strategies"]
        winner = strategies.get(winner_name) or next(iter(strategies.values()))
        df = deserialize_strategy_df(winner["df"])
        tiles = figures.overview_kpis(winner["summary"], run_data.get("mc"))
        return _build_kpi_tiles(tiles), figures.balance_stack(
            df, title=f"Account balances - {winner_name}"
        )

    # ---- Taxes tab --------------------------------------------------

    @app.callback(
        Output("fig-taxes-panel", "figure"),
        Output("fig-conversion-panel", "figure"),
        Input("run-result", "data"),
    )
    def _render_taxes(run_data):
        if not run_data or not run_data.get("strategies"):
            placeholder = figures.empty_figure("Click 'Run' to populate the dashboard")
            return placeholder, placeholder
        winner_name = run_data.get("winner_name")
        strategies = run_data["strategies"]
        winner = strategies.get(winner_name) or next(iter(strategies.values()))
        df = deserialize_strategy_df(winner["df"])
        return (
            figures.taxes_panel(df, title=f"Taxes - {winner_name}"),
            figures.conversion_panel(df, title=f"Conversions & RMDs - {winner_name}"),
        )

    # ---- Strategies tab --------------------------------------------

    @app.callback(
        Output("strategies-comparison", "children"),
        Output("fig-strategy-compare", "figure"),
        Input("run-result", "data"),
    )
    def _render_strategies(run_data):
        if not run_data or not run_data.get("strategies"):
            return (
                html.Div("Click 'Run' to populate the dashboard",
                         className="text-muted"),
                figures.empty_figure("Run first"),
            )
        strategies = run_data["strategies"]
        winner_name = run_data.get("winner_name")
        return (
            _strategy_compare_table(strategies, winner_name),
            figures.strategy_compare_panel(strategies),
        )

    # ---- Monte Carlo tab -------------------------------------------

    @app.callback(
        Output("fig-mc-histogram", "figure"),
        Output("fig-mc-fan", "figure"),
        Input("run-result", "data"),
    )
    def _render_mc(run_data):
        mc = (run_data or {}).get("mc") if run_data else None
        if not mc:
            placeholder = figures.empty_figure(
                "Run with mode 'Four + Monte Carlo' to populate this tab"
            )
            return placeholder, placeholder
        return figures.mc_terminal_histogram(mc), figures.mc_fan_chart(mc)

    # ---- Year-by-year tab ------------------------------------------

    @app.callback(
        Output("yearly-strategy", "options"),
        Output("yearly-strategy", "value"),
        Input("run-result", "data"),
    )
    def _populate_yearly_dropdown(run_data):
        if not run_data or not run_data.get("strategies"):
            return [], None
        names = list(run_data["strategies"].keys())
        winner = run_data.get("winner_name") or names[0]
        return [{"label": n, "value": n} for n in names], winner

    @app.callback(
        Output("yearly-table", "data"),
        Output("yearly-table", "columns"),
        Input("yearly-strategy", "value"),
        State("run-result", "data"),
    )
    def _populate_yearly_table(strategy_name, run_data):
        if not run_data or not strategy_name:
            return [], []
        s = run_data["strategies"].get(strategy_name)
        if s is None:
            return [], []
        df = deserialize_strategy_df(s["df"])
        df = figures.filter_to_detail_cols(df)
        columns = [{"name": c, "id": c} for c in df.columns]
        # Cast everything JSON-safe.
        records = df.where(df.notna(), None).to_dict("records")
        return records, columns

    return app


__all__ = ["make_app"]
