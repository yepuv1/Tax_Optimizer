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

    # ---- 1. Household snapshot --------------------------------------------
    md.append("## 1. Household snapshot")
    md.append("")
    md.append("| Item | Value |")
    md.append("|---|---:|")
    md.append(
        f"| Spouse A age (retire / SS) | {inputs.spouse_a_age_start} "
        f"({inputs.spouse_a_retire_age} / {inputs.ss.start_age}) |"
    )
    md.append(
        f"| Spouse B age (retire / SS) | {inputs.spouse_b_age_start} "
        f"({inputs.spouse_b_retire_age} / {inputs.ss.start_age}) |"
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
    md.append("")

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
    md.append(
        "_For terminal NW higher is better (↑); for tax / IRMAA lower is better (↓)._"
    )
    md.append("")
    md.append("| Metric | Optimized | Baseline (S0) | Δ vs S0 | Direction |")
    md.append("|---|---:|---:|---:|:---:|")
    if base_sum:
        for label, key, kind in [
            ("Terminal after-tax NW", "terminal_after_tax", "money"),
            ("Lifetime federal tax (NPV)", "lifetime_tax_npv", "money_neg"),
            ("Lifetime IRMAA (NPV)", "lifetime_irmaa_npv", "money_neg"),
        ]:
            opt = w_sum[key]
            bas = base_sum[key]
            delta = opt - bas
            sign = "+" if delta > 0 else ("−" if delta < 0 else "")
            is_good = (delta > 0 and kind == "money") or (
                delta < 0 and kind == "money_neg"
            )
            arrow = "✅" if is_good else ("⚠️" if delta != 0 else "—")
            md.append(
                f"| {label} | ${opt:,.0f} | ${bas:,.0f} "
                f"| {sign}${abs(delta):,.0f} | {arrow} |"
            )
        for label, key, fmt in [
            ("Peak federal marginal rate", "peak_marginal", "{:.0%}"),
            ("Years with IRMAA surcharge", "years_irmaa", "{:.0f}"),
            ("Peak IRMAA tier", "peak_irmaa_tier", "{:.0f}"),
        ]:
            opt_v = w_sum[key]
            bas_v = base_sum[key]
            tag = "✅" if opt_v < bas_v else ("⚠️" if opt_v > bas_v else "—")
            md.append(
                f"| {label} | {fmt.format(opt_v)} | {fmt.format(bas_v)} | — | {tag} |"
            )
    else:
        md.append(
            f"| Terminal after-tax NW | ${w_sum['terminal_after_tax']:,.0f} | — | — | — |"
        )
    md.append("")
    md.append(
        f"_Peak federal marginal year: age **{peak['age']}**, "
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
        better_dir = "higher" if d_hi > d_lo else "lower"
        better_delta = max(d_hi, d_lo)
        if param.endswith("_pct") or "rate" in param or param == "inflation":
            rng = f"{lo:.0%} → {hi:.0%}"
        elif param.endswith("_age"):
            rng = f"{int(lo)} → {int(hi)}"
        else:
            rng = f"${lo:,.0f} → ${hi:,.0f}"
        md.append(
            f"| `{param}` | {rng} | {better_dir} "
            f'(+${better_delta:,.0f}) | ${row["swing"]:,.0f} |'
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
        return "—" if v is None or float(v) == 0.0 else f"${float(v):,.0f}"

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
        "- **State income tax.** Federal-only today. Add a state-tax overlay if your "
        "state taxes retirement income (CA, NY, MA, etc.)."
    )
    md.append(
        "- **ACA premium-tax-credit cliffs** in the 60–64 pre-Medicare years "
        "(can dwarf federal-bracket optimization for some households)."
    )
    md.append(
        "- **Estate / step-up basis** beyond the simple 22% pretax haircut in "
        "`terminal_after_tax_nw`. Inherited-IRA 10-year-rule treatment is not modeled."
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
