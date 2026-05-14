"""Form values <-> scenario JSON round-trip.

Wraps `tax_optimizer.scenario.apply_scenario` and
`tax_optimizer.scenario.scenario_to_dict` so the Dash form never has to
care about how the simulator decodes blocks like `market` or `spending`.

Functions:

  * `form_values_to_scenario(values)`: take a flat `{path: value}` dict
    (the form state) and produce a nested scenario JSON dict suitable
    for `apply_scenario`.

  * `scenario_to_form_values(scenario_dict)`: take a nested scenario
    dict (e.g. from `scenarios/example02.json`) and produce a flat
    `{path: value}` dict for the form.

  * `default_form_values()`: form values for the bare `Config()` /
    `Inputs()` defaults; used as the initial form state.

  * `apply_form_values(values)`: convenience that goes
    values -> scenario dict -> `(Config, Inputs)`.
"""

from __future__ import annotations

import warnings
from typing import Any

from tax_optimizer.config import Config
from tax_optimizer.inputs import Inputs
from tax_optimizer.scenario import apply_scenario, scenario_to_dict

from .forms import FIELD_SCHEMA, FormField, get_field


# Fields that have to ride along with a `kind` discriminator so the
# scenario decoder accepts them. We strip these from the emitted
# scenario when their parent kind doesn't match.
_MARKET_KIND_FIELDS: dict[str, set[str]] = {
    "lognormal": {
        "equity_mu", "equity_sigma", "bond_mu", "bond_sigma",
        "equity_bond_corr", "cape_today", "cape_long_run",
    },
    "deterministic": {"equity", "bond"},
    "bootstrap": {"block_size"},
    "historical_sequence": set(),
}

# Spending block fields the form sends for each `kind`. "flat" and "smile"
# match the canned profiles; "custom" is intentionally limited to the
# universal fields (base_spending / inflation) since the form doesn't
# expose `phases` / `lump_events` / `ltc_shock` and the scenario decoder
# rejects any extra keys via `_check_keys("spending(custom)", ...)`.
_SPENDING_KIND_FIELDS: dict[str, set[str]] = {
    "flat": {"base_spending", "inflation"},
    "smile": {"base_spending", "inflation", "ltc_years", "ltc_annual_today"},
    "custom": {"base_spending", "inflation"},
}

# Fields the scenario encoder emits that we don't want to surface in the
# form (deprecation noise, computed properties, etc.).
_HIDDEN_PATHS: frozenset[str] = frozenset({
    "inputs.annual_expenses",  # deprecated; replaced by config.annual_expenses_today.
    "inputs.ss.start_age",     # legacy fallback; we expose start_age_a / b instead.
})


def _normalize_spending_block(block: Any) -> dict[str, Any] | None:
    """Re-shape a `_spending_to_dict`-style payload into a flat/smile form input.

    `tax_optimizer.scenario._spending_to_dict` always serializes a non-None
    `SpendingProfile` as ``{"kind": "custom", "phases": [...], "lump_events":
    [...], "ltc_shock": {...} | None, ...}``. Our form only renders fields
    for ``flat`` and ``smile``, so we demote a `custom` block back to one of
    those by inspecting `ltc_shock`:

      * ltc_shock present with years > 0  ->  "smile"
      * otherwise                          ->  "flat"

    The original `phases` / `lump_events` / `ltc_shock` keys are dropped so
    the form's emit path matches the schema the decoder validates.
    """
    if not isinstance(block, dict):
        return None
    kind = str(block.get("kind", "")).lower()
    base = block.get("base_spending")
    inflation = block.get("inflation")
    if kind in {"flat", "smile"}:
        out = {"kind": kind}
        if base is not None:
            out["base_spending"] = base
        if inflation is not None:
            out["inflation"] = inflation
        if kind == "smile":
            for k in ("ltc_years", "ltc_annual_today"):
                if block.get(k) is not None:
                    out[k] = block[k]
        return out

    # kind in {"custom", ""} or anything else: detect smile by ltc_shock.
    ltc = block.get("ltc_shock") if isinstance(block.get("ltc_shock"), dict) else None
    if ltc and float(ltc.get("years") or 0) > 0:
        return {
            "kind": "smile",
            "base_spending": base,
            "inflation": inflation,
            "ltc_years": int(ltc.get("years") or 0),
            "ltc_annual_today": float(ltc.get("annual_cost_today") or 0.0),
        }
    return {
        "kind": "flat",
        "base_spending": base,
        "inflation": inflation,
    }


# --- Helpers ---------------------------------------------------------

def _set_nested(d: dict[str, Any], parts: list[str], value: Any) -> None:
    cur = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _get_nested(d: dict[str, Any], parts: list[str]) -> Any:
    cur: Any = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _coerce_for_form(field: FormField, value: Any) -> Any:
    """Light coercion for display in `dcc.Input` / `dcc.Dropdown`.

    Numbers stay numbers; bools stay bools; "select" + None pass through.
    """
    if value is None:
        return None
    if field.kind == "bool":
        return bool(value)
    if field.kind in ("number", "percent"):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if field.kind == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return value


def _coerce_for_scenario(field: FormField, value: Any) -> Any:
    """Coerce a form value back to the scenario JSON's expected type.

    Empty strings / `None` map to `None` so optional fields stay
    optional. The scenario decoder then drops `None` for fields where
    that's meaningful (e.g. `regime_change_target`) or keeps it (e.g.
    `ss_cola_rate=None` => follow inflation).
    """
    if value is None or value == "":
        return None
    if field.kind == "bool":
        return bool(value)
    if field.kind == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if field.kind in ("number", "percent"):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return value


# --- Form -> scenario -----------------------------------------------

def form_values_to_scenario(values: dict[str, Any]) -> dict[str, Any]:
    """Build a scenario JSON dict from a flat `{path: value}` map.

    Unknown paths are ignored (forward-compat). Discriminated unions
    (`market`, `spending`) are pruned so only fields belonging to the
    selected `kind` are emitted.
    """
    scn: dict[str, Any] = {"config": {}, "inputs": {}}

    market_kind = values.get("config.market.kind") or "lognormal"
    raw_spending_kind = (values.get("config.spending.kind") or "flat").lower()
    # Defensive: a foreign / typo'd kind still gets a usable scenario.
    spending_kind = raw_spending_kind if raw_spending_kind in _SPENDING_KIND_FIELDS else "flat"
    # If the form doesn't have a base_spending value at all, drop the
    # whole `config.spending` block and let the simulator fall back to
    # `cfg.annual_expenses_today` via `Config.resolved_spending()`.
    skip_spending = values.get("config.spending.base_spending") in (None, "")

    for fld in FIELD_SCHEMA:
        if fld.path in _HIDDEN_PATHS:
            continue
        raw = values.get(fld.path)
        coerced = _coerce_for_scenario(fld, raw)

        if fld.path.startswith("config.market."):
            tail = fld.path.split(".", 2)[2]
            if tail == "kind":
                _set_nested(scn, fld.path.split("."), market_kind)
                continue
            allowed = _MARKET_KIND_FIELDS.get(market_kind, set())
            if tail not in allowed:
                continue
            if coerced is None:
                # cape_today is allowed to stay None (= disable scaling)
                if tail == "cape_today":
                    continue
                # Drop missing optional fields so the decoder uses defaults.
                continue
            _set_nested(scn, fld.path.split("."), coerced)
            continue

        if fld.path.startswith("config.spending."):
            if skip_spending:
                continue
            tail = fld.path.split(".", 2)[2]
            if tail == "kind":
                _set_nested(scn, fld.path.split("."), spending_kind)
                continue
            allowed = _SPENDING_KIND_FIELDS.get(spending_kind, set())
            if tail not in allowed:
                continue
            if coerced is None:
                continue
            _set_nested(scn, fld.path.split("."), coerced)
            continue

        # Generic path. We always emit the value, even None, so users
        # can intentionally clear a field (e.g. set ss_cola_rate=None).
        # The scenario decoder ignores keys with `None` values for
        # optional regime targets via _REGIMES_BY_NAME's None handling.
        if coerced is None:
            # Drop None values: omitting the key falls back to dataclass
            # defaults, which is what the user almost always wants.
            continue
        _set_nested(scn, fld.path.split("."), coerced)

    return scn


# --- Scenario -> form -----------------------------------------------

def scenario_to_form_values(scenario_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested scenario dict into `{path: value}` for the form.

    Missing paths are returned as `None` so the form can show empty
    inputs. Any leaf that the schema doesn't know about is silently
    skipped.

    The `config.spending` block is normalized in-place: a round-tripped
    ``kind="custom"`` from `_spending_to_dict` is demoted to the
    `flat`/`smile` form variants the dropdown actually offers, so the
    form-to-scenario emit path stays valid.
    """
    scn = dict(scenario_dict) if isinstance(scenario_dict, dict) else {}
    cfg_block = dict(scn.get("config") or {})
    if "spending" in cfg_block and cfg_block["spending"] is not None:
        normalized = _normalize_spending_block(cfg_block["spending"])
        if normalized is not None:
            cfg_block["spending"] = normalized
            scn["config"] = cfg_block

    out: dict[str, Any] = {}
    for fld in FIELD_SCHEMA:
        parts = fld.path.split(".")
        raw = _get_nested(scn, parts)
        out[fld.path] = _coerce_for_form(fld, raw)
    return out


# --- Defaults -------------------------------------------------------

def default_form_values() -> dict[str, Any]:
    """Form values matching `Config()` / `Inputs()` defaults."""
    cfg = Config()
    inputs = Inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        scn = scenario_to_dict(cfg, inputs)
    # The default Config has `market=None` / `spending=None`; surface
    # sensible defaults for the form's discriminator widgets.
    if scn.get("config", {}).get("market") is None:
        scn.setdefault("config", {})["market"] = {"kind": "lognormal"}
    if scn.get("config", {}).get("spending") is None:
        scn.setdefault("config", {})["spending"] = {
            "kind": "flat",
            "base_spending": cfg.annual_expenses_today,
            "inflation": cfg.inflation,
        }
    return scenario_to_form_values(scn)


# --- Apply ----------------------------------------------------------

def apply_form_values(values: dict[str, Any]) -> tuple[Config, Inputs]:
    """Convenience: form values -> (Config, Inputs)."""
    scn = form_values_to_scenario(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return apply_scenario(Config(), Inputs(), scn)


def cfg_inputs_to_form_values(cfg: Config, inputs: Inputs) -> dict[str, Any]:
    """Round-trip a (Config, Inputs) pair into form values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        scn = scenario_to_dict(cfg, inputs)
    return scenario_to_form_values(scn)
