"""Tornado sensitivity + plain-English action / takeaway renderers."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pandas as pd

from .config import Config
from .inputs import Inputs
from .metrics import terminal_after_tax_nw
from .results import StrategyResult
from .simulator import simulate


def _ranked_strategies(results: dict[str, StrategyResult]) -> list[tuple[str, StrategyResult]]:
    return sorted(
        results.items(),
        key=lambda kv: kv[1].summary["terminal_after_tax"],
        reverse=True,
    )


def winning_strategy(
    results: dict[str, StrategyResult], default_cfg: Config, default_inputs: Inputs
) -> tuple[str, Config, Inputs]:
    if not results:
        return "S0_baseline", default_cfg, default_inputs
    name, r = _ranked_strategies(results)[0]
    return name, r.cfg, r.inputs


# Backward-compat shim for any external caller. Returns (name, cfg) only.
def winning_cfg(results: dict[str, StrategyResult], default_cfg: Config) -> tuple[str, Config]:
    if not results:
        return "S0_baseline", default_cfg
    name, r = _ranked_strategies(results)[0]
    return name, r.cfg


# Tornado knobs that live on a nested Inputs sub-dataclass. Map each
# flat tornado label to a (sub_attr, leaf_attr) pair on Inputs so the
# sensitivity loop can route the perturbation without leaking the
# nested path into the displayed param name.
_NESTED_INPUTS_FIELDS: dict[str, tuple[str, str]] = {
    "ss_start_age": ("ss", "start_age"),
    "pension_start_age": ("pension", "start_age"),
}


def _resolve_owner(field_name: str) -> str:
    """Return 'cfg', 'inputs', or 'inputs_nested' depending on owner."""
    if field_name in _NESTED_INPUTS_FIELDS:
        return "inputs_nested"
    if field_name in {f.name for f in fields(Config)}:
        return "cfg"
    if field_name in {f.name for f in fields(Inputs)}:
        return "inputs"
    raise KeyError(f"Field {field_name!r} not found on Config or Inputs.")


def tornado_sensitivity(
    base_cfg: Config, inputs: Inputs | None = None
) -> tuple[pd.DataFrame, float]:
    """Per-knob tornado around `(base_cfg, inputs)`. Returns (df, base_terminal).

    Each perturbation auto-routes to whichever dataclass actually owns
    the field, so adding a new tornado knob doesn't require touching
    this function's plumbing -- just append the (name, lo, hi) tuple.
    """
    if inputs is None:
        inputs = Inputs()
    bequest_rate = base_cfg.heir_marginal_rate
    base_terminal = terminal_after_tax_nw(
        simulate(base_cfg, inputs), heir_marginal_rate=bequest_rate
    )

    def _terminal(param: str, value) -> float:
        owner = _resolve_owner(param)
        if owner == "cfg":
            new_cfg = replace(base_cfg, **{param: value})
            return terminal_after_tax_nw(
                simulate(new_cfg, inputs),
                heir_marginal_rate=new_cfg.heir_marginal_rate,
            )
        if owner == "inputs_nested":
            sub_attr, leaf_attr = _NESTED_INPUTS_FIELDS[param]
            sub_obj = getattr(inputs, sub_attr)
            new_sub = replace(sub_obj, **{leaf_attr: value})
            return terminal_after_tax_nw(
                simulate(base_cfg, replace(inputs, **{sub_attr: new_sub})),
                heir_marginal_rate=bequest_rate,
            )
        return terminal_after_tax_nw(
            simulate(base_cfg, replace(inputs, **{param: value})),
            heir_marginal_rate=bequest_rate,
        )

    def int_clamp(v: int, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, v)))

    perturbations = [
        ("spouse_a_roth_401k_pct", 0.0, 1.0),
        ("spouse_b_roth_401k_pct", 0.0, 1.0),
        ("roth_conversion_target_bracket", 0.0, 0.32),
        ("spouse_a_total_contrib_pct",
            max(0.0, inputs.spouse_a_total_contrib_pct - 0.05),
            min(1.0, inputs.spouse_a_total_contrib_pct + 0.05)),
        ("spouse_b_total_contrib_pct",
            max(0.0, inputs.spouse_b_total_contrib_pct - 0.05),
            min(1.0, inputs.spouse_b_total_contrib_pct + 0.05)),
        ("spouse_a_retire_age",
            int_clamp(inputs.spouse_a_retire_age - 2, 50, 75),
            int_clamp(inputs.spouse_a_retire_age + 2, 50, 75)),
        ("spouse_b_retire_age",
            int_clamp(inputs.spouse_b_retire_age - 2, 50, 75),
            int_clamp(inputs.spouse_b_retire_age + 2, 50, 75)),
        ("ss_start_age",
            int_clamp(inputs.ss.start_age - 3, 62, 70),
            int_clamp(inputs.ss.start_age + 3, 62, 70)),
        ("nominal_growth_rate",
            max(0.0, base_cfg.nominal_growth_rate - 0.01),
            base_cfg.nominal_growth_rate + 0.01),
        ("inflation",
            max(0.0, base_cfg.inflation - 0.01),
            base_cfg.inflation + 0.01),
        ("annual_expenses_today",
            base_cfg.annual_expenses_today * 0.90,
            base_cfg.annual_expenses_today * 1.10),
    ]

    rows: list[dict] = []
    for param, lo, hi in perturbations:
        if lo == hi:
            continue
        delta_lo = _terminal(param, lo) - base_terminal
        delta_hi = _terminal(param, hi) - base_terminal
        rows.append(
            {
                "param": param,
                "low_value": lo,
                "high_value": hi,
                "delta_low": delta_lo,
                "delta_high": delta_hi,
                "swing": max(abs(delta_lo), abs(delta_hi)),
            }
        )
    df = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    return df, base_terminal


def _action_for_param(
    param: str,
    direction: str,
    lo: float,
    hi: float,
    base_cfg: Config,
    base_inputs: Inputs,
) -> str:
    # Auto-route the lookup to whichever object owns the field.
    try:
        owner = _resolve_owner(param)
    except KeyError:
        owner = "cfg"
    if owner == "inputs_nested":
        sub_attr, leaf_attr = _NESTED_INPUTS_FIELDS[param]
        base_val = getattr(getattr(base_inputs, sub_attr), leaf_attr, None)
    else:
        base_val = getattr(
            base_cfg if owner == "cfg" else base_inputs, param, None
        )
    if param.endswith("_roth_401k_pct"):
        spouse = "Spouse A" if param.startswith("spouse_a") else "Spouse B"
        target = hi if direction == "higher" else lo
        return (
            f"Move {spouse}'s 401(k) deferrals to {target:.0%} Roth "
            f"(currently {base_val:.0%})."
        )
    if param.endswith("_total_contrib_pct"):
        spouse = "Spouse A" if param.startswith("spouse_a") else "Spouse B"
        target = hi if direction == "higher" else lo
        return (
            f"Adjust {spouse}'s total 401(k) contribution rate to {target:.0%} "
            f"(currently {base_val:.0%})."
        )
    if param == "roth_conversion_target_bracket":
        target = hi if direction == "higher" else lo
        return f"Set Roth-conversion bracket target to {target:.0%} (currently {base_val:.0%})."
    if param.endswith("_retire_age"):
        spouse = "Spouse A" if param.startswith("spouse_a") else "Spouse B"
        target = hi if direction == "higher" else lo
        verb = "delay" if target > base_val else "move up"
        return (
            f"{verb.capitalize()} {spouse}'s retirement to age {int(target)} "
            f"(currently {int(base_val)})."
        )
    if param == "ss_start_age":
        target = int(hi if direction == "higher" else lo)
        return f"Begin Social Security at age {target} (currently {int(base_val)})."
    if param == "annual_expenses_today":
        target = hi if direction == "higher" else lo
        verb = "trim" if target < base_val else "budget"
        return f"{verb.capitalize()} annual spending toward ${target:,.0f}."
    if param == "nominal_growth_rate":
        return "Market assumption (not an action)."
    if param == "inflation":
        return "Macro assumption only; useful as a stress-test."
    return f"Move toward {direction} end of the tested range."


def render_actions(
    results: dict[str, StrategyResult],
    sens_df: pd.DataFrame,
    base_cfg: Config,
    base_terminal: float,
    base_inputs: Inputs | None = None,
) -> str:
    if base_inputs is None:
        base_inputs = Inputs()
    winner_name, winner_cfg, winner_inputs = winning_strategy(
        results, base_cfg, base_inputs
    )
    baseline_sum = (
        results["S0_baseline"].summary if "S0_baseline" in results else {}
    )
    winner_sum = results[winner_name].summary

    lines: list[str] = ["### Recommended actions", ""]
    if winner_name == "S0_baseline":
        lines.append("1. Stay the course. Baseline already maximizes terminal after-tax NW.")
    elif winner_name == "S1_all_roth_401k":
        a_pct = winner_inputs.spouse_a_total_contrib_pct
        b_pct = winner_inputs.spouse_b_total_contrib_pct
        lines.append(
            f"1. Switch contributions to Roth 401(k). Direct 100% of Spouse A's "
            f"{a_pct:.0%} and Spouse B's {b_pct:.0%} salary deferrals into the Roth bucket."
        )
    elif winner_name == "S2_bracket_fill_22":
        lines.append(
            "1. Plan gap-year Roth conversions. Keep current pretax contributions "
            f"while working, then convert pretax -> Roth up to "
            f"{winner_cfg.roth_conversion_target_bracket:.0%} federal bracket each gap year."
        )
    elif winner_name == "S3_optimized":
        a_roth = winner_inputs.spouse_a_roth_401k_pct
        b_roth = winner_inputs.spouse_b_roth_401k_pct
        conv = winner_cfg.roth_conversion_target_bracket
        bits = []
        if a_roth > 0.05 or b_roth > 0.05:
            bits.append(
                f"set Spouse A Roth-401(k) split to {a_roth:.0%} and Spouse B to {b_roth:.0%}"
            )
        if conv > 0:
            bits.append(f"target Roth conversions up to the {conv:.0%} bracket in gap years")
        if not bits:
            bits.append("keep current contribution mix")
        lines.append("1. Hybrid plan (optimizer-chosen): " + "; ".join(bits) + ".")
    else:
        lines.append(f"1. Adopt `{winner_name}` allocation.")

    if baseline_sum:
        lift = winner_sum["terminal_after_tax"] - baseline_sum["terminal_after_tax"]
        if lift > 0:
            lines.append(f"   - Expected lift vs S0_baseline: ${lift:,.0f} terminal after-tax NW.")

    top = sens_df.head(3)
    lines.append("")
    lines.append("2. Highest-leverage knobs (top 3 by tornado swing):")
    for _, row in top.iterrows():
        param = row["param"]
        lo, hi = row["low_value"], row["high_value"]
        d_lo, d_hi = row["delta_low"], row["delta_high"]
        better_dir = "higher" if d_hi > d_lo else "lower"
        better_delta = max(d_hi, d_lo)
        action = _action_for_param(param, better_dir, lo, hi, base_cfg, base_inputs)
        lines.append(
            f"   - {param} - swing ${row['swing']:,.0f}; pushing it {better_dir} "
            f"adds ~${better_delta:,.0f}. {action}"
        )

    lines.append("")
    lines.append("3. Always-good hygiene:")
    lines.append("   - Max out the HSA family contribution (triple-tax-advantaged).")
    lines.append("   - Hold an emergency / IRMAA-buffer of 1-2 years of expenses in taxable.")
    lines.append("   - Re-run this script annually with updated balances and salaries.")
    return "\n".join(lines)


def render_takeaways(
    results: dict[str, StrategyResult], cfg: Config, inputs: Inputs | None = None
) -> str:
    if not results:
        return "(No strategies have been simulated yet.)"
    if inputs is None:
        inputs = Inputs()
    ranked = sorted(
        results.items(),
        key=lambda kv: kv[1].summary["terminal_after_tax"],
        reverse=True,
    )
    winner_name, winner_r = ranked[0]
    winner_sum = winner_r.summary
    baseline_name = "S0_baseline" if "S0_baseline" in results else ranked[-1][0]
    baseline_sum = results[baseline_name].summary

    def fmt(v: float) -> str:
        return f"${v:,.0f}"

    lines: list[str] = []
    if winner_name == baseline_name:
        lines.append(
            f"- Winning strategy: {baseline_name} is already optimal "
            f"(terminal after-tax NW {fmt(winner_sum['terminal_after_tax'])})."
        )
    else:
        lines.append(
            f"- Winning strategy: {winner_name} - terminal after-tax NW "
            f"{fmt(winner_sum['terminal_after_tax'])} "
            f"(+{fmt(winner_sum['terminal_after_tax'] - baseline_sum['terminal_after_tax'])} "
            f"vs {baseline_name})."
        )

    lines.append(
        f"- IRMAA exposure ({winner_name}): {winner_sum['years_irmaa']} year(s), "
        f"peak tier {winner_sum['peak_irmaa_tier']}."
    )
    lines.append(
        f"- Peak federal marginal rate ({winner_name}): {winner_sum['peak_marginal']:.0%}."
    )

    horizon_years = cfg.horizon_age - inputs.spouse_a_age_start + 1
    header = (
        f"### Run summary ({horizon_years}-year horizon, ages "
        f"{inputs.spouse_a_age_start}-{cfg.horizon_age})\n"
    )
    return header + "\n".join(lines)
