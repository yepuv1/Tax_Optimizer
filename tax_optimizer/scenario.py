"""Scenario file loading and ad-hoc override parsing.

A *scenario* is a JSON document that overrides the built-in `Config` and
`Inputs` defaults. The shape mirrors the dataclasses one-to-one (any
field omitted keeps its default); complex objects use a small
tagged-dict form.

    {
      "config": {
        "horizon_age": 95,
        "tax_regime": "sunset",
        "regime_change_year_offset": 5,
        "regime_change_target": "tcja",
        "market":   { "kind": "lognormal",
                      "equity_mu": 0.07, "equity_sigma": 0.18 },
        "spending": { "kind": "smile", "base_spending": 95000,
                      "ltc_years": 4, "ltc_annual_today": 100000 },
        "mortality": { "year_of_death_a": 25, "pension_survivor_pct": 0.5 },
        "asset_location": { "uniform_equity_pct": 0.6 }
      },
      "inputs": {
        "spouse_a_age_start": 52,
        "spouse_b_age_start": 50,
        "spouse_a_retire_age": 65,
        "spouse_b_retire_age": 67,
        "spouse_a_total_contrib_pct": 0.10,
        "spouse_b_total_contrib_pct": 0.08,
        "spouse_a_roth_401k_pct": 0.0,
        "spouse_b_roth_401k_pct": 0.0,
        "starting": { "spouse_a_pretax_401k": 400000, "hsa": 25000 },
        "income":   { "spouse_a_gross": 140000 },
        "contrib":  { "hsa_family": 9000 },
        "pension":  { "monthly_at_nrd": 1200, "start_age": 65 },
        "ss":       { "monthly_spouse_a": 3100, "start_age": 70 }
      }
    }

Setting expected expenses is done on the `config` side, via the
`spending` block (recommended) or the scalar `annual_expenses_today`
fallback. The `inputs.annual_expenses` field is retained for backward
compatibility but is ignored by the simulator and emits a
`DeprecationWarning` when set to a non-default value.

Unknown keys raise `ScenarioError` so typos don't silently no-op. If a
legacy scenario file puts one of the spouse_* fields under `config`,
the error message points at the new location on `inputs`.

This module is also responsible for `--set DOTTED.PATH=VALUE` parsing
and for serializing the current effective scenario back to JSON for
`--print-defaults`.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from .config import Config
from .inputs import Inputs
from .market import (
    CMA_PRESETS,
    AssetLocation,
    AssetMix,
    BootstrapModel,
    DeterministicModel,
    HistoricalSequenceModel,
    LognormalModel,
    MarketModel,
    lognormal_from_cma,
)
from .mortality import Mortality
from .spending import LongTermCareShock, LumpEvent, SpendingPhase, SpendingProfile
from .tax.regimes import PRE_TCJA_2017, SUNSET_2026, TCJA_EXTENDED, TaxRegime
from .tax.state import StateTaxRegime, lookup as lookup_state_regime


class ScenarioError(ValueError):
    """Raised on malformed scenario files or override expressions."""


_REGIMES_BY_NAME: dict[str, TaxRegime] = {
    "tcja": TCJA_EXTENDED,
    "tcja_extended": TCJA_EXTENDED,
    "pre_tcja": PRE_TCJA_2017,
    "pre_tcja_2017": PRE_TCJA_2017,
    "sunset": SUNSET_2026,
    "sunset_2026": SUNSET_2026,
}


# Old layout: these spouse_* fields used to live on `Config` and were
# accepted under the JSON `config` section. They now live on `Inputs`.
# We keep the set around purely so that legacy scenario files / --set
# expressions get a targeted migration hint instead of a generic
# "Unknown config field" error.
_LEGACY_CONFIG_SPOUSE_FIELDS: frozenset[str] = frozenset(
    {
        "spouse_a_age_start",
        "spouse_b_age_start",
        "spouse_a_retire_age",
        "spouse_b_retire_age",
        "spouse_a_total_contrib_pct",
        "spouse_b_total_contrib_pct",
        "spouse_a_roth_401k_pct",
        "spouse_b_roth_401k_pct",
    }
)


# Old layout: SS claim age and pension NRD used to live on `Config`.
# They now live on the nested Inputs blocks.
#   config.ss_start_age      -> inputs.ss.start_age
#   config.pension_start_age -> inputs.pension.start_age
_LEGACY_CONFIG_AGE_GATES: dict[str, str] = {
    "ss_start_age": "inputs.ss.start_age",
    "pension_start_age": "inputs.pension.start_age",
}


# ---------------------------------------------------------------------------
# Loading from JSON
# ---------------------------------------------------------------------------


def load_scenario_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON scenario file. Returns a raw dict (no validation yet)."""
    p = Path(path)
    if not p.exists():
        raise ScenarioError(f"Scenario file not found: {p}")
    try:
        with p.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ScenarioError(f"Invalid JSON in {p}: {e}") from e
    if not isinstance(data, dict):
        raise ScenarioError(
            f"Scenario root must be a JSON object, got {type(data).__name__}"
        )
    return data


def apply_scenario(
    cfg: Config, inputs: Inputs, scenario: dict[str, Any]
) -> tuple[Config, Inputs]:
    """Apply a parsed scenario dict on top of the given cfg / inputs."""
    extras = set(scenario) - {"config", "inputs"}
    if extras:
        raise ScenarioError(
            f"Unknown top-level scenario keys: {sorted(extras)}. "
            f"Expected only 'config' and/or 'inputs'."
        )

    config_patch = dict(scenario.get("config") or {})
    inputs_patch = dict(scenario.get("inputs") or {})

    legacy = _LEGACY_CONFIG_SPOUSE_FIELDS & set(config_patch)
    if legacy:
        names = ", ".join(f"inputs.{f}" for f in sorted(legacy))
        raise ScenarioError(
            f"These spouse_* fields moved to `inputs`: {sorted(legacy)}. "
            f"Move them under the `inputs` section of your scenario JSON "
            f"(e.g. {names})."
        )

    legacy_ages = set(_LEGACY_CONFIG_AGE_GATES) & set(config_patch)
    if legacy_ages:
        moves = ", ".join(
            f"{old} -> {_LEGACY_CONFIG_AGE_GATES[old]}"
            for old in sorted(legacy_ages)
        )
        raise ScenarioError(
            f"These age-gate fields moved out of `config`: {sorted(legacy_ages)}. "
            f"Update your scenario JSON: {moves}."
        )

    # Wrap dataclass `__post_init__` ValueError raises into
    # ScenarioError so downstream callers (CLI, Dash app) can catch
    # both shapes uniformly. Without this, a typo like
    # `inputs.household_kind: "married"` raised a bare ValueError that
    # the Dash run-banner couldn't easily distinguish from an internal
    # exception.
    try:
        if config_patch:
            cfg = _apply_config_dict(cfg, config_patch)
        if inputs_patch:
            inputs = _apply_inputs_dict(inputs, inputs_patch)
    except ScenarioError:
        raise
    except ValueError as exc:
        raise ScenarioError(f"Scenario validation failed: {exc}") from exc
    return cfg, inputs


# ---------------------------------------------------------------------------
# --set DOTTED.PATH=VALUE parsing
# ---------------------------------------------------------------------------


def apply_set_overrides(
    cfg: Config, inputs: Inputs, overrides: list[str]
) -> tuple[Config, Inputs]:
    """Apply repeatable `--set key.path=value` style overrides.

    `value` is parsed as a JSON literal (so quote strings, use true/false/
    null, etc.). Bare identifiers fall back to plain strings for ergonomics
    so `--set config.tax_regime=sunset` works without quoting.
    """
    if not overrides:
        return cfg, inputs

    cfg_patch: dict[str, Any] = {}
    inputs_patch: dict[str, Any] = {}

    for raw in overrides:
        if "=" not in raw:
            raise ScenarioError(
                f"--set expects KEY=VALUE, got {raw!r} (missing '=')."
            )
        key, _, val_str = raw.partition("=")
        key = key.strip()
        val_str = val_str.strip()
        if not key:
            raise ScenarioError(f"--set has empty key in {raw!r}.")

        try:
            value: Any = json.loads(val_str)
        except json.JSONDecodeError:
            value = val_str  # bare identifier fallback

        parts = key.split(".")
        if not parts or parts[0] not in {"config", "inputs"}:
            raise ScenarioError(
                f"--set key {key!r} must start with 'config.' or 'inputs.'."
            )
        root, rest = parts[0], parts[1:]
        if not rest:
            raise ScenarioError(
                f"--set key {key!r} is missing a field after the root."
            )

        # Reject the legacy `config.spouse_*` form with a targeted hint.
        if (
            root == "config"
            and len(rest) == 1
            and rest[0] in _LEGACY_CONFIG_SPOUSE_FIELDS
        ):
            raise ScenarioError(
                f"--set key {key!r} moved: use inputs.{rest[0]} instead "
                f"(spouse_* fields now live on Inputs)."
            )

        # Reject the legacy `config.{ss,pension}_start_age` form too.
        if (
            root == "config"
            and len(rest) == 1
            and rest[0] in _LEGACY_CONFIG_AGE_GATES
        ):
            new_path = _LEGACY_CONFIG_AGE_GATES[rest[0]]
            raise ScenarioError(
                f"--set key {key!r} moved: use {new_path} instead "
                f"(age-gate fields now live on Inputs)."
            )

        target = cfg_patch if root == "config" else inputs_patch
        _nest_set(target, rest, value)

    try:
        if cfg_patch:
            cfg = _apply_config_dict(cfg, cfg_patch)
        if inputs_patch:
            inputs = _apply_inputs_dict(inputs, inputs_patch)
    except ScenarioError:
        raise
    except ValueError as exc:
        raise ScenarioError(f"--set validation failed: {exc}") from exc
    return cfg, inputs


def _nest_set(d: dict[str, Any], parts: list[str], value: Any) -> None:
    cur = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


# ---------------------------------------------------------------------------
# Serialization (--print-defaults)
# ---------------------------------------------------------------------------


def scenario_to_dict(cfg: Config, inputs: Inputs) -> dict[str, Any]:
    """Serialize the current cfg / inputs back to a scenario-shaped dict.

    Round-trip property: `apply_scenario(Config(), Inputs(),
    scenario_to_dict(cfg, inputs))` reproduces (cfg, inputs) for every
    field this module knows how to encode.
    """
    return {
        "config": _config_to_dict(cfg),
        "inputs": _inputs_to_dict(inputs),
    }


def _config_to_dict(cfg: Config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if f.name == "tax_regime":
            out[f.name] = _regime_name(v)
        elif f.name == "regime_change_target":
            out[f.name] = _regime_name(v) if v is not None else None
        elif f.name in ("state_regime", "state_regime_change_target"):
            out[f.name] = v.name if v is not None else None
        elif f.name == "mortality":
            out[f.name] = asdict(v)
        elif f.name == "market":
            out[f.name] = _market_to_dict(v) if v is not None else None
        elif f.name == "asset_location":
            out[f.name] = _asset_location_to_dict(v)
        elif f.name == "spending":
            out[f.name] = _spending_to_dict(v) if v is not None else None
        else:
            out[f.name] = v
    return out


def _inputs_to_dict(inp: Inputs) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(inp):
        v = getattr(inp, f.name)
        if is_dataclass(v):
            out[f.name] = asdict(v)
        else:
            out[f.name] = v
    return out


def _regime_name(r: TaxRegime) -> str:
    for k, v in _REGIMES_BY_NAME.items():
        if v is r:
            return k
    # Custom regime: fall back to a placeholder string
    return getattr(r, "name", "custom")


def _market_to_dict(m: MarketModel) -> dict[str, Any]:
    if isinstance(m, DeterministicModel):
        return {"kind": "deterministic", "equity": m.equity, "bond": m.bond}
    if isinstance(m, LognormalModel):
        out: dict[str, Any] = {
            "kind": "lognormal",
            "equity_mu": m.equity_mu,
            "equity_sigma": m.equity_sigma,
            "bond_mu": m.bond_mu,
            "bond_sigma": m.bond_sigma,
            "equity_bond_corr": m.equity_bond_corr,
        }
        # Always emit `cape_long_run` so the round-trip preserves a
        # user-set value even when `cape_today` is unset (no scaling
        # applied this run, but the constant may still be the user's
        # chosen long-run mean for future sensitivity sweeps).
        # `cape_today` stays gated on being non-None because emitting
        # `null` there is the documented "no scaling" sentinel.
        if m.cape_today is not None:
            out["cape_today"] = m.cape_today
        out["cape_long_run"] = m.cape_long_run
        return out
    if isinstance(m, BootstrapModel):
        return {"kind": "bootstrap", "block_size": m.block_size}
    if isinstance(m, HistoricalSequenceModel):
        return {"kind": "historical_sequence"}
    return {"kind": "custom", "repr": repr(m)}


def _asset_location_to_dict(a: AssetLocation) -> dict[str, Any]:
    return {
        "pretax_equity_pct": a.pretax.equity_pct,
        "roth_equity_pct": a.roth.equity_pct,
        "taxable_equity_pct": a.taxable.equity_pct,
        "hsa_equity_pct": a.hsa.equity_pct,
    }


def _spending_to_dict(s: SpendingProfile) -> dict[str, Any]:
    return {
        "kind": "custom",
        "base_spending": s.base_spending,
        "inflation": s.inflation,
        "phases": [asdict(p) for p in s.phases],
        "lump_events": [asdict(e) for e in s.lump_events],
        "ltc_shock": asdict(s.ltc_shock) if s.ltc_shock else None,
    }


# ---------------------------------------------------------------------------
# Internal: nested apply helpers
# ---------------------------------------------------------------------------


def _apply_config_dict(cfg: Config, patch: dict[str, Any]) -> Config:
    if not isinstance(patch, dict):
        raise ScenarioError(f"'config' must be an object, got {type(patch).__name__}.")

    valid = {f.name for f in fields(cfg)}
    unknown = set(patch) - valid
    if unknown:
        legacy = unknown & _LEGACY_CONFIG_SPOUSE_FIELDS
        if legacy:
            names = ", ".join(f"inputs.{f}" for f in sorted(legacy))
            raise ScenarioError(
                f"These spouse_* fields moved to `inputs`: {sorted(legacy)}. "
                f"Move them under the `inputs` section (e.g. {names})."
            )
        legacy_ages = unknown & set(_LEGACY_CONFIG_AGE_GATES)
        if legacy_ages:
            moves = ", ".join(
                f"{old} -> {_LEGACY_CONFIG_AGE_GATES[old]}"
                for old in sorted(legacy_ages)
            )
            raise ScenarioError(
                f"These age-gate fields moved out of `config`: "
                f"{sorted(legacy_ages)}. Update your JSON: {moves}."
            )
        raise ScenarioError(
            f"Unknown config field(s): {sorted(unknown)}. "
            f"Valid fields include: {sorted(valid)}."
        )

    _warn_spending_inconsistency(patch)

    kwargs: dict[str, Any] = {}
    for k, v in patch.items():
        kwargs[k] = _coerce_config_field(k, v, current=getattr(cfg, k))
    return replace(cfg, **kwargs)


def _warn_spending_inconsistency(patch: dict[str, Any]) -> None:
    """Warn when both `annual_expenses_today` and `spending.base_spending`
    are set in the same scenario but disagree.

    Only `spending.base_spending` reaches the simulator in that case
    (`Config.resolved_spending()` picks `spending` first), so the
    `annual_expenses_today` value is silently ignored. A scenario where
    the two values match is left silent — it's redundant but not
    misleading.
    """
    if "annual_expenses_today" not in patch:
        return
    spending = patch.get("spending")
    if not isinstance(spending, dict):
        return  # `spending: null` or no spending block → no conflict.
    if "base_spending" not in spending:
        return
    try:
        scalar = float(patch["annual_expenses_today"])
        profile = float(spending["base_spending"])
    except (TypeError, ValueError):
        return  # leave coercion errors to the regular path
    if scalar == profile:
        return
    warnings.warn(
        f"Scenario sets both config.annual_expenses_today (${scalar:,.0f}) "
        f"and config.spending.base_spending (${profile:,.0f}). The simulator "
        f"uses config.spending.base_spending when a spending block is "
        f"present; config.annual_expenses_today is ignored in this case. "
        f"Remove one of the two to silence this warning.",
        UserWarning,
        stacklevel=4,
    )


def _coerce_config_field(name: str, v: Any, *, current: Any) -> Any:
    if name == "tax_regime":
        return _coerce_regime(v, field_label="tax_regime")
    if name == "regime_change_target":
        if v is None:
            return None
        return _coerce_regime(v, field_label="regime_change_target")
    if name == "state_regime":
        return _coerce_state_regime(v, field_label="state_regime")
    if name == "state_regime_change_target":
        if v is None:
            return None
        return _coerce_state_regime(v, field_label="state_regime_change_target")
    if name == "mortality":
        return _coerce_mortality(v, current=current)
    if name == "market":
        return _coerce_market(v)
    if name == "asset_location":
        return _coerce_asset_location(v, current=current)
    if name == "spending":
        return _coerce_spending(v)
    return v


def _coerce_state_regime(v: Any, *, field_label: str) -> StateTaxRegime:
    if isinstance(v, StateTaxRegime):
        return v
    if isinstance(v, str):
        try:
            return lookup_state_regime(v)
        except KeyError as exc:
            raise ScenarioError(str(exc)) from exc
    raise ScenarioError(
        f"{field_label} must be a string abbreviation "
        f"(e.g. 'CA', 'NY', 'IL', 'MA', 'stateless'); "
        f"got {type(v).__name__}."
    )


def _coerce_regime(v: Any, *, field_label: str) -> TaxRegime:
    if isinstance(v, TaxRegime):
        return v
    if isinstance(v, str):
        key = v.strip().lower()
        if key not in _REGIMES_BY_NAME:
            raise ScenarioError(
                f"Unknown {field_label} {v!r}. "
                f"Valid: {sorted({'tcja', 'pre_tcja', 'sunset'})}."
            )
        return _REGIMES_BY_NAME[key]
    raise ScenarioError(
        f"{field_label} must be a string name; got {type(v).__name__}."
    )


def _coerce_mortality(v: Any, *, current: Mortality) -> Mortality:
    if v is None:
        return Mortality()
    if not isinstance(v, dict):
        raise ScenarioError(f"'mortality' must be an object; got {type(v).__name__}.")
    valid = {f.name for f in fields(Mortality)}
    unknown = set(v) - valid
    if unknown:
        raise ScenarioError(
            f"Unknown mortality field(s): {sorted(unknown)}. Valid: {sorted(valid)}."
        )
    return replace(current, **v)


_LOGNORMAL_FIELDS = {
    "equity_mu",
    "equity_sigma",
    "bond_mu",
    "bond_sigma",
    "equity_bond_corr",
    "cape_today",
    "cape_long_run",
}


def _coerce_market(v: Any) -> MarketModel | None:
    if v is None:
        return None
    if isinstance(v, MarketModel):
        return v
    if not isinstance(v, dict):
        raise ScenarioError(f"'market' must be an object; got {type(v).__name__}.")
    kind = str(v.get("kind", "")).lower()
    params = {k: val for k, val in v.items() if k != "kind"}

    # CMA preset shortcut: "cma": "vanguard_2025" picks a canned profile,
    # any other lognormal-shaped keys override it. Implies kind="lognormal"
    # if kind is unset.
    cma = params.pop("cma", None)
    if cma is not None:
        if not isinstance(cma, str):
            raise ScenarioError(
                f"market.cma must be a string preset name; got {type(cma).__name__}."
            )
        if cma not in CMA_PRESETS:
            raise ScenarioError(
                f"Unknown CMA preset {cma!r}. "
                f"Available: {sorted(CMA_PRESETS)}"
            )
        if kind in ("", "lognormal"):
            _check_keys("market(lognormal+cma)", params, _LOGNORMAL_FIELDS)
            return lognormal_from_cma(cma, **params)
        raise ScenarioError(
            f"market.cma is only valid with kind='lognormal' "
            f"(or unset); got kind={kind!r}."
        )

    if kind in ("", "deterministic"):
        _check_keys("market(deterministic)", params, {"equity", "bond"})
        return DeterministicModel(**params)
    if kind == "lognormal":
        _check_keys("market(lognormal)", params, _LOGNORMAL_FIELDS)
        return LognormalModel(**params)
    if kind == "bootstrap":
        _check_keys(
            "market(bootstrap)",
            params,
            {"block_size", "equity_history", "bond_history"},
        )
        if "equity_history" in params:
            params["equity_history"] = tuple(params["equity_history"])
        if "bond_history" in params:
            params["bond_history"] = tuple(params["bond_history"])
        return BootstrapModel(**params)
    if kind in ("historical_sequence", "historical"):
        _check_keys(
            "market(historical_sequence)",
            params,
            {"equity_history", "bond_history"},
        )
        if "equity_history" in params:
            params["equity_history"] = tuple(params["equity_history"])
        if "bond_history" in params:
            params["bond_history"] = tuple(params["bond_history"])
        return HistoricalSequenceModel(**params)
    raise ScenarioError(
        f"Unknown market kind {kind!r}. "
        f"Valid: 'deterministic', 'lognormal', 'bootstrap', 'historical_sequence'."
    )


def _coerce_asset_location(v: Any, *, current: AssetLocation) -> AssetLocation:
    if isinstance(v, AssetLocation):
        return v
    if not isinstance(v, dict):
        raise ScenarioError(
            f"'asset_location' must be an object; got {type(v).__name__}."
        )

    if "uniform_equity_pct" in v:
        if len(v) != 1:
            raise ScenarioError(
                "asset_location.uniform_equity_pct cannot be combined with "
                "per-bucket fields; use one form or the other."
            )
        return AssetLocation.uniform(equity_pct=float(v["uniform_equity_pct"]))

    valid = {"pretax_equity_pct", "roth_equity_pct", "taxable_equity_pct", "hsa_equity_pct"}
    _check_keys("asset_location", v, valid)
    return AssetLocation(
        pretax=AssetMix(
            float(v.get("pretax_equity_pct", current.pretax.equity_pct)),
            current.pretax.label,
        ),
        roth=AssetMix(
            float(v.get("roth_equity_pct", current.roth.equity_pct)),
            current.roth.label,
        ),
        taxable=AssetMix(
            float(v.get("taxable_equity_pct", current.taxable.equity_pct)),
            current.taxable.label,
        ),
        hsa=AssetMix(
            float(v.get("hsa_equity_pct", current.hsa.equity_pct)),
            current.hsa.label,
        ),
    )


def _coerce_spending(v: Any) -> SpendingProfile | None:
    if v is None:
        return None
    if isinstance(v, SpendingProfile):
        return v
    if not isinstance(v, dict):
        raise ScenarioError(f"'spending' must be an object; got {type(v).__name__}.")

    kind = str(v.get("kind", "custom")).lower()
    if kind == "flat":
        _check_keys("spending(flat)", v, {"kind", "base_spending", "inflation"})
        return SpendingProfile.flat(
            base_spending=float(v["base_spending"]),
            inflation=float(v.get("inflation", 0.025)),
        )
    if kind == "smile":
        _check_keys(
            "spending(smile)",
            v,
            {"kind", "base_spending", "inflation", "ltc_years", "ltc_annual_today"},
        )
        return SpendingProfile.retirement_smile(
            base_spending=float(v["base_spending"]),
            inflation=float(v.get("inflation", 0.025)),
            ltc_years=int(v.get("ltc_years", 3)),
            ltc_annual_today=float(v.get("ltc_annual_today", 80_000.0)),
        )
    if kind == "custom":
        _check_keys(
            "spending(custom)",
            v,
            {"kind", "base_spending", "inflation", "phases", "lump_events", "ltc_shock"},
        )
        phases = [SpendingPhase(**p) for p in v.get("phases", [])] or None
        lump_events = [LumpEvent(**e) for e in v.get("lump_events", [])]
        ltc = v.get("ltc_shock")
        ltc_obj = LongTermCareShock(**ltc) if isinstance(ltc, dict) else None
        kw: dict[str, Any] = {
            "base_spending": float(v["base_spending"]),
            "inflation": float(v.get("inflation", 0.025)),
        }
        if phases is not None:
            kw["phases"] = phases
        kw["lump_events"] = lump_events
        kw["ltc_shock"] = ltc_obj
        return SpendingProfile(**kw)
    raise ScenarioError(
        f"Unknown spending kind {kind!r}. Valid: 'flat', 'smile', 'custom'."
    )


def _apply_inputs_dict(inputs: Inputs, patch: dict[str, Any]) -> Inputs:
    if not isinstance(patch, dict):
        raise ScenarioError(f"'inputs' must be an object; got {type(patch).__name__}.")
    valid = {f.name for f in fields(inputs)}
    unknown = set(patch) - valid
    if unknown:
        raise ScenarioError(
            f"Unknown inputs field(s): {sorted(unknown)}. "
            f"Valid: {sorted(valid)}."
        )

    kwargs: dict[str, Any] = {}
    for k, v in patch.items():
        cur = getattr(inputs, k)
        if is_dataclass(cur) and isinstance(v, dict):
            sub_valid = {f.name for f in fields(cur)}
            sub_unknown = set(v) - sub_valid
            if sub_unknown:
                raise ScenarioError(
                    f"Unknown inputs.{k} field(s): {sorted(sub_unknown)}. "
                    f"Valid: {sorted(sub_valid)}."
                )
            kwargs[k] = replace(cur, **v)
        else:
            kwargs[k] = v
    return replace(inputs, **kwargs)


def _check_keys(label: str, mapping: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ScenarioError(
            f"Unknown {label} key(s): {sorted(unknown)}. "
            f"Valid: {sorted(allowed)}."
        )
