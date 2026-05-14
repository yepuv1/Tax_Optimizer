"""Dash app entry point + callbacks.

`make_app()` builds and returns a configured `Dash` instance so it can
be embedded in a WSGI server (`app.server`) or run via
`python -m dash_app`. All callbacks live in this module to keep the
import tree shallow.
"""

from __future__ import annotations

import base64
import json
import warnings
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, html, no_update

from tax_optimizer.config import Config
from tax_optimizer.inputs import Inputs
from tax_optimizer.scenario import (
    ScenarioError,
    apply_scenario,
    scenario_to_dict,
)

from . import figures
from .forms import get_field
from .layout import build_layout
from .report_builder import build_html_payload, cache_run, get_cached
from .runner import (
    deserialize_strategy_df,
    run_scenario,
    serialize_run_result,
)
from .state import (
    apply_form_values,
    cfg_inputs_to_form_values,
    default_form_values,
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
# App factory
# ---------------------------------------------------------------------


def make_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="Tax Optimizer",
    )
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
        if not values:
            return no_update
        try:
            cfg, inputs = apply_form_values(values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                payload = scenario_to_dict(cfg, inputs)
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
        cfg, inputs, rr = cached
        try:
            payload = build_html_payload(cfg, inputs, rr)
        except Exception as e:  # noqa: BLE001 - surface any render error
            return no_update, html.Span(
                f"Report build failed: {e}", className="text-danger"
            )
        return payload, html.Span(
            f"Report ready: {payload['filename']}.", className="text-success"
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
        Output("strategies-callout", "children"),
        Output("fig-strategy-compare", "figure"),
        Input("run-result", "data"),
    )
    def _render_strategies(run_data):
        if not run_data or not run_data.get("strategies"):
            return (
                html.Div("Click 'Run' to populate the dashboard", className="text-muted"),
                figures.empty_figure("Run first"),
            )
        strategies = run_data["strategies"]
        winner_name = run_data.get("winner_name")
        return (
            _strategy_callout(strategies, winner_name),
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
