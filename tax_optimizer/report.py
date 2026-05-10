"""Markdown action-plan report.

This is the same report the standalone notebook renders in section 9
("Recommended actions") -- ported here so the CLI can produce it by
default and so anyone using the Python API can call
`build_action_report(cfg, inputs, results, sens_df, base_terminal, mc=...)`
to get the same markdown string the notebook displays.

The report is intentionally self-contained markdown with no plots: it
prints fine to a terminal, renders cleanly on GitHub, and survives
copy-paste into email or Slack.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import Config
from .inputs import Inputs
from .market import LognormalModel
from .monte_carlo import MonteCarloResult
from .results import StrategyResult


def _winning_strategy(results: dict[str, StrategyResult]) -> str:
    return max(results, key=lambda n: results[n].summary["terminal_after_tax"])


def _peak_marginal_year(df: pd.DataFrame) -> dict[str, Any]:
    """Age and AGI of the year with the highest marginal rate."""
    idx = df["marginal"].idxmax()
    row = df.loc[idx]
    return {
        "age": int(row["spouse_a_age"]),
        "agi": float(row["agi"]),
        "marginal": float(row["marginal"]),
    }


def _conversion_window(inputs: Inputs, cfg: Config) -> tuple[int, int]:
    """Age range where Roth conversions are most useful: from the later-
    retiring spouse's retire age until RMDs hit."""
    start_age = max(inputs.spouse_a_retire_age, inputs.spouse_b_retire_age)
    end_age = cfg.rmd_start_age - 1
    return start_age, end_age


def _lever_changes(
    base_cfg: Config,
    base_inputs: Inputs,
    w_cfg: Config,
    w_inputs: Inputs,
) -> list[dict]:
    """Return one dict per decision-vector axis whose value differs
    between the user-supplied inputs and the optimizer-chosen plan.

    Used by the TL;DR section so the reader can immediately see *what
    the optimizer wants you to change*. Axes that match the baseline
    are omitted entirely (no noise).
    """
    out: list[dict] = []
    pct_pairs = [
        ("Spouse A Roth-401(k) share", "spouse_a_roth_401k_pct", base_inputs, w_inputs),
        ("Spouse B Roth-401(k) share", "spouse_b_roth_401k_pct", base_inputs, w_inputs),
        ("Spouse A mega-backdoor share", "spouse_a_after_tax_401k_pct", base_inputs, w_inputs),
        ("Spouse B mega-backdoor share", "spouse_b_after_tax_401k_pct", base_inputs, w_inputs),
        ("Roth conversion target bracket", "roth_conversion_target_bracket", base_cfg, w_cfg),
    ]
    for label, attr, base_obj, w_obj in pct_pairs:
        b = getattr(base_obj, attr, None)
        a = getattr(w_obj, attr, None)
        if b is None or a is None:
            continue
        if abs(float(a) - float(b)) < 1e-6:
            continue
        out.append({"label": label, "before": float(b), "after": float(a), "kind": "pct"})
    age_pairs = [
        ("Spouse A SS claim age", "start_age_a"),
        ("Spouse B SS claim age", "start_age_b"),
    ]
    for label, attr in age_pairs:
        b = getattr(base_inputs.ss, attr, None)
        a = getattr(w_inputs.ss, attr, None)
        if b is None or a is None or b == a:
            continue
        out.append({"label": label, "before": int(b), "after": int(a), "kind": "age"})
    return out


def _market_summary(cfg: Config) -> str:
    """Short human description of `cfg.market`. For the parametric
    lognormal model we also surface the equity/bond mu, sigma and
    correlation so an advisor can sanity-check the assumptions."""
    name = type(cfg.market).__name__
    if isinstance(cfg.market, LognormalModel):
        m = cfg.market
        return (
            f"{name} "
            f"(equity μ={m.equity_mu:.1%}/σ={m.equity_sigma:.1%}, "
            f"bond μ={m.bond_mu:.1%}/σ={m.bond_sigma:.1%}, "
            f"ρ={m.equity_bond_corr:.2f})"
        )
    return name


def _assumptions_block(cfg: Config, inputs: Inputs) -> list[str]:
    """Return the markdown lines for the "Assumptions driving this
    plan" subsection. Always renders the same shape so users learn
    where to look for each knob."""
    horizon = cfg.horizon_age - inputs.spouse_a_age_start + 1
    out = ["### Assumptions driving this plan", ""]
    out.append("| Assumption | Value |")
    out.append("|---|---|")
    out.append(
        f"| Heir marginal tax rate (pretax / HSA bequest haircut) "
        f"| {cfg.heir_marginal_rate:.0%} |"
    )
    out.append(f"| Inflation | {cfg.inflation:.1%}/yr |")
    out.append(
        f"| Nominal growth (deterministic baseline) | {cfg.nominal_growth_rate:.1%}/yr |"
    )
    out.append(f"| Market model | {_market_summary(cfg)} |")
    if cfg.regime_change_year_offset is not None and cfg.regime_change_target is not None:
        regime = (
            f"`{cfg.tax_regime.name}` → `{cfg.regime_change_target.name}` "
            f"in year {cfg.regime_change_year_offset}"
        )
    else:
        regime = f"`{cfg.tax_regime.name}` (no scheduled change)"
    out.append(f"| Federal tax regime | {regime} |")
    out.append(f"| State tax regime | `{cfg.state_regime.name}` |")
    out.append(
        f"| Mortality | Spouse A: year {cfg.mortality.year_of_death_a}, "
        f"Spouse B: year {cfg.mortality.year_of_death_b} (horizon = year {horizon}) |"
    )
    out.append(
        f"| Step-up in basis at first death | "
        f"{'on' if cfg.stepup_at_first_death else 'off'} |"
    )
    out.append(
        f"| ACA premium tax credit | "
        f"{'enabled' if cfg.aca_enabled else 'disabled'} |"
    )
    out.append(
        f"| Pre-65 healthcare (today's $) | ${cfg.health_pre65_today:,.0f}/yr |"
    )
    out.append(
        f"| Medicare base premium (today's $) | "
        f"${cfg.medicare_base_b_d_premium:,.0f}/yr |"
    )
    out.append(f"| IRMAA MAGI lookback | {cfg.irmaa_lookback_years} year(s) |")
    out.append("")
    return out


def _tldr_section(
    winner: str,
    results: dict[str, StrategyResult],
    base_cfg: Config,
    base_inputs: Inputs,
    mc: MonteCarloResult | None,
) -> list[str]:
    """Render the headline verdict, lever-change summary, and key risk
    metrics. The intent is that a reader who only reads the first 15
    lines of the report still leaves with the right takeaways."""
    w = results[winner]
    w_sum = w.summary
    base_sum = results["S0_baseline"].summary if "S0_baseline" in results else {}
    out = ["## TL;DR", ""]

    if base_sum:
        delta = w_sum["terminal_after_tax"] - base_sum["terminal_after_tax"]
        pct = (
            (delta / base_sum["terminal_after_tax"] * 100)
            if base_sum["terminal_after_tax"]
            else 0.0
        )
        if delta > 0:
            verdict = (
                f"**Verdict:** `{winner}` beats `S0_baseline` by "
                f"**${delta:,.0f}** ({pct:+.1f}%) in terminal after-tax NW."
            )
        elif delta < 0:
            verdict = (
                f"**Verdict:** `{winner}` lags `S0_baseline` by "
                f"${abs(delta):,.0f} ({pct:+.1f}%) — your current setup "
                f"already looks optimal."
            )
        else:
            verdict = (
                f"**Verdict:** `{winner}` matches `S0_baseline`; no "
                f"actionable change found by the optimizer."
            )
        out.append(verdict)
    else:
        out.append(
            f"**Winning strategy:** `{winner}` — terminal after-tax NW "
            f"${w_sum['terminal_after_tax']:,.0f}."
        )
    out.append("")

    changes = _lever_changes(base_cfg, base_inputs, w.cfg, w.inputs)
    if changes:
        out.append("**Levers the optimizer wants you to change:**")
        for ch in changes:
            if ch["kind"] == "pct":
                out.append(
                    f"- {ch['label']}: {ch['before']:.0%} → **{ch['after']:.0%}**"
                )
            else:
                out.append(
                    f"- {ch['label']}: {ch['before']} → **{ch['after']}**"
                )
    else:
        out.append(
            "_No lever changes recommended: your current inputs already "
            "match the optimizer's choice on every decision axis._"
        )
    out.append("")

    out.append("**Key risk readings:**")
    if mc is not None:
        s = mc.summary()
        tag = "safe" if s["prob_success"] >= 0.9 else "**watch — below 90%**"
        out.append(
            f"- P(success) under {s['n_paths']} stochastic paths: "
            f"**{s['prob_success']:.1%}** ({tag})"
        )
        out.append(f"- CVaR(10%) terminal NW: ${s['cvar_terminal_p10']:,.0f}")
    out.append(f"- Peak federal marginal rate: **{w_sum['peak_marginal']:.0%}**")
    tax_line = f"- Lifetime federal tax NPV: ${w_sum['lifetime_tax_npv']:,.0f}"
    if base_sum:
        d = w_sum["lifetime_tax_npv"] - base_sum["lifetime_tax_npv"]
        if abs(d) >= 1:
            tax_line += f" ({d:+,.0f} vs S0)"
    out.append(tax_line)
    irmaa_line = f"- Lifetime IRMAA NPV: ${w_sum['lifetime_irmaa_npv']:,.0f}"
    if base_sum:
        d = w_sum["lifetime_irmaa_npv"] - base_sum["lifetime_irmaa_npv"]
        if abs(d) >= 1:
            irmaa_line += f" ({d:+,.0f} vs S0)"
    out.append(irmaa_line)
    out.append("")
    return out


def build_action_report(
    cfg: Config,
    inputs: Inputs,
    results: dict[str, StrategyResult],
    sens_df: pd.DataFrame,
    base_terminal: float,
    mc: MonteCarloResult | None = None,
    *,
    scenario_path: str | None = None,
) -> str:
    """Render the deterministic + (optional) Monte Carlo action plan as markdown.

    Parameters mirror the notebook's `build_action_report`. Returns a
    single markdown string (no trailing newline).
    """
    winner = _winning_strategy(results)
    w = results[winner]
    w_cfg, w_inputs, w_df, w_sum = w.cfg, w.inputs, w.df, w.summary
    base_sum = results["S0_baseline"].summary if "S0_baseline" in results else {}
    horizon = cfg.horizon_age - inputs.spouse_a_age_start + 1
    starting_total = inputs.starting.total_excl_real_estate
    household_wages = (
        inputs.income.spouse_a_gross
        + inputs.income.spouse_b_gross
        + inputs.income.spouse_a_bonus
    )
    peak = _peak_marginal_year(w_df)
    conv_start_age, conv_end_age = _conversion_window(inputs, cfg)

    md: list[str] = []
    md.append("# Retirement Tax Optimization — Action Plan")
    md.append("")
    md.append(
        f"_{horizon}-year horizon, ages {inputs.spouse_a_age_start}"
        f"/{inputs.spouse_b_age_start} to {cfg.horizon_age}, "
        f"tax regime `{cfg.tax_regime.name}`._"
    )
    if scenario_path:
        md.append("")
        md.append(f"_Scenario file: `{scenario_path}`._")
    md.append("")

    md.extend(_tldr_section(winner, results, cfg, inputs, mc))

    # ---- 1. Household snapshot --------------------------------------------
    md.append("## 1. Household snapshot")
    md.append("")
    md.append("| Item | Value |")
    md.append("|---|---:|")
    md.append(
        f"| Spouse A age (retire / SS) | {inputs.spouse_a_age_start} "
        f"({inputs.spouse_a_retire_age} / {inputs.ss.effective_start_age_a}) |"
    )
    md.append(
        f"| Spouse B age (retire / SS) | {inputs.spouse_b_age_start} "
        f"({inputs.spouse_b_retire_age} / {inputs.ss.effective_start_age_b}) |"
    )
    md.append(f"| Combined gross W-2 income | ${household_wages:,.0f} |")
    # Use the resolved spending profile's base, not `inputs.annual_expenses`.
    # The simulator drives spending off `cfg.resolved_spending()`, so a
    # scenario that sets `cfg.spending.base_spending` but leaves
    # `inputs.annual_expenses` at the default would otherwise mislabel
    # this row.
    resolved_spending = cfg.resolved_spending()
    md.append(
        f"| Annual expenses (today's $) | ${resolved_spending.base_spending:,.0f} |"
    )
    md.append(f"| Total liquid + retirement assets | ${starting_total:,.0f} |")
    md.append(
        f"| &nbsp;&nbsp;&nbsp; Spouse A pretax (401k + IRA) | "
        f"${inputs.starting.spouse_a_pretax_401k + inputs.starting.spouse_a_pretax_ira:,.0f} |"
    )
    md.append(
        f"| &nbsp;&nbsp;&nbsp; Spouse B pretax (401k + IRA) | "
        f"${inputs.starting.spouse_b_pretax_401k + inputs.starting.spouse_b_pretax_ira:,.0f} |"
    )
    md.append(
        f"| &nbsp;&nbsp;&nbsp; Roth (both spouses, pooled) | "
        f"${inputs.starting.spouse_a_roth_ira + inputs.starting.spouse_b_roth_ira:,.0f} |"
    )
    md.append(
        f"| &nbsp;&nbsp;&nbsp; Taxable brokerage | "
        f"${inputs.starting.taxable_brokerage:,.0f} |"
    )
    md.append(
        f"| &nbsp;&nbsp;&nbsp; HSA / pension cash-balance | "
        f"${inputs.starting.hsa:,.0f} / ${inputs.starting.pension_balance:,.0f} |"
    )
    if (
        inputs.spouse_a_employer_match_rate > 0
        or inputs.spouse_b_employer_match_rate > 0
    ):
        a_match = (
            f"{inputs.spouse_a_employer_match_rate:.0%} on first "
            f"{inputs.spouse_a_employer_match_max_pct:.0%}"
            if inputs.spouse_a_employer_match_rate > 0
            else "—"
        )
        b_match = (
            f"{inputs.spouse_b_employer_match_rate:.0%} on first "
            f"{inputs.spouse_b_employer_match_max_pct:.0%}"
            if inputs.spouse_b_employer_match_rate > 0
            else "—"
        )
        md.append(f"| Employer 401(k) match (A / B) | {a_match} / {b_match} |")
    md.append("")

    md.extend(_assumptions_block(cfg, inputs))

    # ---- 2. Recommended plan ----------------------------------------------
    md.append("## 2. Recommended plan")
    md.append("")
    md.append(f"**Winning strategy:** `{winner}`")
    md.append("")
    md.append("| Lever | Recommended | Currently |")
    md.append("|---|---:|---:|")
    # The optimizer's three decision variables are the two Roth-401(k)
    # splits and the conversion target bracket. The total deferral
    # percentages and withdrawal strategy aren't searched over today, so
    # we only show them when they actually differ from the baseline
    # (e.g. a hand-tuned scenario or a future expanded decision space).
    if w_inputs.spouse_a_total_contrib_pct != inputs.spouse_a_total_contrib_pct:
        md.append(
            f"| Spouse A 401(k) deferral | {w_inputs.spouse_a_total_contrib_pct:.0%} of salary "
            f"| {inputs.spouse_a_total_contrib_pct:.0%} |"
        )
    md.append(
        f"| Spouse A Roth share of deferral | {w_inputs.spouse_a_roth_401k_pct:.0%} "
        f"| {inputs.spouse_a_roth_401k_pct:.0%} |"
    )
    if w_inputs.spouse_b_total_contrib_pct != inputs.spouse_b_total_contrib_pct:
        md.append(
            f"| Spouse B 401(k) deferral | {w_inputs.spouse_b_total_contrib_pct:.0%} of salary "
            f"| {inputs.spouse_b_total_contrib_pct:.0%} |"
        )
    md.append(
        f"| Spouse B Roth share of deferral | {w_inputs.spouse_b_roth_401k_pct:.0%} "
        f"| {inputs.spouse_b_roth_401k_pct:.0%} |"
    )
    md.append(
        f"| Roth conversion target bracket (gap years) "
        f"| {w_cfg.roth_conversion_target_bracket:.0%} "
        f"| {cfg.roth_conversion_target_bracket:.0%} |"
    )
    if w_cfg.withdrawal_strategy != cfg.withdrawal_strategy:
        md.append(
            f"| Withdrawal strategy in retirement | `{w_cfg.withdrawal_strategy}` "
            f"| `{cfg.withdrawal_strategy}` |"
        )
    md.append("")

    # ---- 3. Expected outcomes ---------------------------------------------
    md.append("## 3. Expected outcomes (deterministic, point-estimate)")
    md.append("")
    strategy_order = ["S0_baseline", "S1_all_roth", "S2_bracket_fill_22", "S3_optimizer"]
    cols = [s for s in strategy_order if s in results]
    extra = [s for s in results if s not in strategy_order]
    cols.extend(extra)

    if len(cols) > 1:
        md.append(
            "_Best value in each row is **bolded**. Higher is better for "
            "terminal NW; lower is better for tax, IRMAA, peak marginal._"
        )
        md.append("")
        header_cells = [c.replace("_", " ") for c in cols]
        md.append("| Metric | " + " | ".join(header_cells) + " |")
        md.append("|---|" + "---:|" * len(cols))

        metric_specs = [
            ("Terminal after-tax NW", "terminal_after_tax", "max", "${v:,.0f}"),
            ("Lifetime federal tax (NPV)", "lifetime_tax_npv", "min", "${v:,.0f}"),
            ("Lifetime IRMAA (NPV)", "lifetime_irmaa_npv", "min", "${v:,.0f}"),
            ("Peak federal marginal rate", "peak_marginal", "min", "{v:.0%}"),
            ("Years with IRMAA", "years_irmaa", "min", "{v:.0f}"),
            ("Peak IRMAA tier", "peak_irmaa_tier", "min", "{v:.0f}"),
        ]
        for label, key, mode, fmt in metric_specs:
            vals = [results[c].summary[key] for c in cols]
            best_idx = (
                max(range(len(vals)), key=lambda i: vals[i])
                if mode == "max"
                else min(range(len(vals)), key=lambda i: vals[i])
            )
            tied = [i for i, v in enumerate(vals) if v == vals[best_idx]]
            cells = []
            for i, v in enumerate(vals):
                text = fmt.format(v=v)
                if i in tied and len(tied) < len(vals):
                    text = f"**{text}**"
                cells.append(text)
            md.append(f"| {label} | " + " | ".join(cells) + " |")
    else:
        md.append(
            f"| Terminal after-tax NW | ${w_sum['terminal_after_tax']:,.0f} |"
        )
    md.append("")
    md.append(
        f"_Peak federal marginal year (for `{winner}`): age **{peak['age']}**, "
        f"AGI ~${peak['agi']:,.0f}, marginal **{peak['marginal']:.0%}**._"
    )
    md.append("")

    # ---- 4. Risk picture (Monte Carlo) ------------------------------------
    if mc is not None:
        s = mc.summary()
        md.append("## 4. Risk picture (Monte Carlo)")
        md.append("")
        md.append(
            f'_Based on {s["n_paths"]} stochastic paths under the '
            f"`{type(mc.cfg.market).__name__}` market model._"
        )
        md.append("")
        md.append("| Metric | Value | Reading |")
        md.append("|---|---:|---|")
        md.append(
            f'| Probability of success | {s["prob_success"]:.1%} | '
            f'{"safe" if s["prob_success"] >= 0.9 else "watch — below 90%"} |'
        )
        md.append(
            f"| Terminal NW p5 / p50 / p95 | "
            f'${s["terminal_p5"]:,.0f} / ${s["terminal_p50"]:,.0f} '
            f'/ ${s["terminal_p95"]:,.0f} | spread of bad-to-good outcomes |'
        )
        md.append(
            f'| CVaR(10%) terminal | ${s["cvar_terminal_p10"]:,.0f} | '
            f"expected NW in the worst 10% of paths |"
        )
        md.append(
            f'| Median lifetime tax (NPV) | ${s["lifetime_tax_p50"]:,.0f} | — |'
        )
        if s["median_ruin_year_offset"] >= 0:
            md.append(
                f'| Median ruin year | {s["median_ruin_year_offset"]:.0f} | '
                f"**at least half of failure paths run out by year "
                f'{s["median_ruin_year_offset"]:.0f}** |'
            )
        else:
            md.append("| Median ruin year | — | no path ran out of money |")
        md.append("")

    # ---- 5. Top sensitivity levers ----------------------------------------
    md.append("## 5. Highest-leverage levers (top 3 by tornado swing)")
    md.append("")
    md.append("| Knob | Range tested | Best direction | $ swing |")
    md.append("|---|---|---|---:|")
    for _, row in sens_df.head(3).iterrows():
        param = row["param"]
        lo, hi = row["low_value"], row["high_value"]
        d_lo, d_hi = row["delta_low"], row["delta_high"]
        # Honest direction labelling: if neither side of the tested range
        # improves on the baseline, say so. Otherwise pick the direction
        # that yields a positive delta. (Previously this always said
        # "higher" when delta_high == delta_low, producing misleading
        # "+$0" recommendations for knobs already at their boundary.)
        if d_hi <= 0 and d_lo <= 0:
            direction_cell = "at boundary in tested range (—)"
        elif d_hi >= d_lo:
            direction_cell = f"higher (+${d_hi:,.0f})"
        else:
            direction_cell = f"lower (+${d_lo:,.0f})"
        if param.endswith("_pct") or "rate" in param or param == "inflation":
            rng = f"{lo:.0%} → {hi:.0%}"
        elif param.endswith("_age"):
            rng = f"{int(lo)} → {int(hi)}"
        else:
            rng = f"${lo:,.0f} → ${hi:,.0f}"
        md.append(
            f"| `{param}` | {rng} | {direction_cell} "
            f'| ${row["swing"]:,.0f} |'
        )
    md.append("")

    # ---- 6. Year-by-year action timeline ----------------------------------
    md.append("## 6. Year-by-year action timeline")
    md.append("")
    md.append("| Phase | Ages (Spouse A) | What to do |")
    md.append("|---|---|---|")
    md.append(
        f"| **Accumulation** | {inputs.spouse_a_age_start}–{inputs.spouse_a_retire_age - 1} | "
        f"Defer {w_inputs.spouse_a_total_contrib_pct:.0%} of A's salary "
        f"({w_inputs.spouse_a_roth_401k_pct:.0%} Roth / "
        f"{1 - w_inputs.spouse_a_roth_401k_pct:.0%} Traditional) and "
        f"{w_inputs.spouse_b_total_contrib_pct:.0%} of B's "
        f"({w_inputs.spouse_b_roth_401k_pct:.0%} Roth / "
        f"{1 - w_inputs.spouse_b_roth_401k_pct:.0%} Traditional). "
        f"Max HSA. Build 1–2 years' expenses in taxable as IRMAA buffer. |"
    )
    if conv_start_age <= conv_end_age and w_cfg.roth_conversion_target_bracket > 0:
        md.append(
            f"| **Gap years (conversion window)** | {conv_start_age}–{conv_end_age} | "
            f"No wages and SS not yet started → fill the "
            f"{w_cfg.roth_conversion_target_bracket:.0%} bracket each year via "
            f"pretax → Roth conversions. Pay conversion tax from taxable. |"
        )
    else:
        md.append(
            f"| **Gap years** | {conv_start_age}–{conv_end_age} | "
            f"Optimizer chose 0% conversion target — keep gap-year income low and "
            f"rely on taxable + Roth withdrawals. |"
        )
    md.append(
        "| **Medicare / IRMAA** | 65+ | "
        "IRMAA tiers depend on AGI from **2 years prior**. "
        "Watch the 22% / 24% bracket and the IRMAA cliffs at 65, 67. |"
    )
    md.append(
        f"| **SS claim** | {inputs.ss.start_age} | "
        f"Both spouses begin Social Security "
        f"(${(inputs.ss.monthly_spouse_a + inputs.ss.monthly_spouse_b) * 12:,.0f}/yr today). "
        f"After this, conversion-window income includes SS. |"
    )
    md.append(
        f"| **RMDs begin** | {cfg.rmd_start_age} | "
        f"Required minimum distributions hit Spouse A's pretax balance. "
        f"Spouse B's RMDs follow on B's schedule. Reduce or stop conversions. |"
    )
    md.append(
        f"| **Drawdown** | {cfg.rmd_start_age + 1}–{cfg.horizon_age} | "
        f"Withdrawal sequence: `{w_cfg.withdrawal_strategy}`. "
        f"Take RMDs first, then top up from taxable / Roth as needed. |"
    )
    md.append("")

    # ---- 7. Year-by-year withdrawal & conversion plan ---------------------
    md.append("## 7. Year-by-year withdrawal & conversion plan")
    md.append("")
    md.append(
        "_Retirement years only. Pre-retirement years are pure "
        "accumulation (contributions, no withdrawals or conversions). "
        "Cells with no activity render as `—`._"
    )
    md.append("")
    md.append(
        "| Age (A) | Pretax W/D | Roth conv. | Roth W/D | Taxable W/D | RMD | AGI | Fed tax | IRMAA |"
    )
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def _money(v: float | None) -> str:
        # Use `round(...) == 0` instead of `== 0.0` so floats that are
        # numerically near zero (e.g. 1e-9 from float arithmetic in the
        # cascade) also render as `—`. Without this, the table shows
        # confusing `$0` cells next to `—` cells for what are logically
        # the same "nothing happened" outcome.
        if v is None:
            return "—"
        f = float(v)
        if round(f) == 0:
            return "—"
        return f"${f:,.0f}"

    retire_age = min(inputs.spouse_a_retire_age, inputs.spouse_b_retire_age)
    retire_rows = w_df[w_df["spouse_a_age"] >= retire_age]
    for _, r in retire_rows.iterrows():
        md.append(
            f'| {int(r["spouse_a_age"])} '
            f'| {_money(r["pretax_withdrawal"])} '
            f'| {_money(r["roth_conversion"])} '
            f'| {_money(r["roth_withdrawal"])} '
            f'| {_money(r["taxable_withdrawal"])} '
            f'| {_money(r["rmd"])} '
            f'| {_money(r["agi"])} '
            f'| {_money(r["federal_tax"])} '
            f'| {_money(r["irmaa"])} |'
        )
    md.append("")
    md.append(
        "_Pretax W/D = combined Spouse A + B pretax withdrawals; RMD = the "
        "IRS-required floor (A and B combined). Roth conversions show up as "
        "AGI in the same row but produce no cash to the household — they "
        "shift dollars from pretax → Roth and the conversion tax is paid "
        "from taxable._"
    )
    md.append("")

    # ---- 8. Always-good hygiene -------------------------------------------
    md.append("## 8. Always-good hygiene")
    md.append("")
    md.append(
        "- **Max the HSA each year** (triple tax-advantaged: deductible, tax-free "
        "growth, tax-free qualified withdrawals)."
    )
    md.append(
        "- **Hold 1–2 years of expenses in taxable** as an IRMAA / sequence-of-returns "
        "buffer; it lets you avoid IRA withdrawals in down-market years and to "
        "manage AGI around IRMAA cliffs."
    )
    md.append(
        "- **Re-run this annually** with updated balances, salaries, and "
        "tax-regime assumptions. Roth-conversion targets in particular are very "
        "sensitive to bracket changes."
    )
    md.append(
        "- **Beneficiary review:** primary + contingent on every account, including "
        "the HSA (which becomes ordinary income to a non-spouse beneficiary)."
    )
    md.append("")

    # ---- 9. Caveats -------------------------------------------------------
    md.append("## 9. Caveats — what this plan does NOT model")
    md.append("")
    md.append(
        "- **State income tax** outside the bundled `STATELESS / CA / NY / IL / MA` "
        "presets. Add a custom `StateTaxRegime` for your state if it's not in this list."
    )
    md.append(
        "- **Estate / inherited-IRA dynamics** beyond the flat `heir_marginal_rate` "
        "haircut on terminal pretax + HSA balances. SECURE Act 10-year drawdown "
        "timing for non-spouse heirs is not modeled."
    )
    md.append("- **Health-care shocks** beyond a single optional LTC stress test.")
    md.append(
        "- **Tax-law revisions** beyond the bundled regimes (`TCJA_EXTENDED`, "
        "`PRE_TCJA_2017`, `SUNSET_2026`). Re-run with `cfg.tax_regime = SUNSET_2026` "
        "to stress-test current-law sunset."
    )
    md.append("")
    md.append(
        "_This is decision-support output, not tax/legal/investment advice. "
        "Consult a qualified professional before acting._"
    )

    return "\n".join(md)
