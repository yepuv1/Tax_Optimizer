"""Year-by-year simulator.

This is a single-path simulator. The Monte Carlo wrapper in
`monte_carlo.py` runs this many times with different RNG seeds.

Compared to the v1 simulator, this one threads:
  * `cfg.regime_for_year(year_offset)`  - active TaxRegime per year
  * `cfg.mortality.filing_status(...)`  - 'mfj' or 'single' per year
  * `cfg.resolved_market()`             - per-year (equity, bond) returns
  * `cfg.asset_location`                - account-level equity %
  * `cfg.resolved_spending()`           - smile + lump events + LTC
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .conversion import planned_roth_conversion
from .inputs import Inputs
from .pension import (
    PENSION_INTEREST,
    pension_annual_credit,
    project_pension_balance,
)
from .rmd import rmd_amount
from .state import State, initial_state
from .tax.federal import federal_tax
from .tax.irmaa import MEDICARE_ELIGIBLE_AGE, irmaa_annual_surcharge
from .withdrawals import withdraw_for_need


def simulate(
    cfg: Config,
    inputs: Inputs | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Run one path of the simulator.

    `rng` is only used when `cfg.market` is stochastic. For deterministic
    runs, a default RNG is created but its draws are ignored.
    """
    if inputs is None:
        inputs = Inputs()
    if rng is None:
        rng = np.random.default_rng(0)

    state = initial_state(cfg, inputs)
    rows: list[dict] = []

    spouse_a_salary = inputs.income.spouse_a_gross
    spouse_b_salary = inputs.income.spouse_b_gross
    spouse_a_bonus = inputs.income.spouse_a_bonus

    n_years = cfg.horizon_age - inputs.spouse_a_age_start + 1

    market = cfg.resolved_market()
    market.begin_path(n_years, rng)

    spending = cfg.resolved_spending()
    asset_loc = cfg.asset_location

    # Reference projected pension balance at NRD; used to scale annuity
    # if the actual cash-balance grew faster (or slower) than expected.
    test_balance = project_pension_balance(
        inputs.pension.balance_today,
        inputs.income.spouse_a_gross,
        max(0, inputs.pension.start_age - inputs.spouse_a_age_start),
        wage_growth=cfg.wage_growth,
    )

    for year_offset in range(n_years):
        year = cfg.start_year + year_offset
        a_age = inputs.spouse_a_age_start + year_offset
        b_age = inputs.spouse_b_age_start + year_offset
        state.spouse_a_age = a_age
        state.spouse_b_age = b_age
        state.year = year

        # Mortality + filing status for this year.
        alive_a = cfg.mortality.alive_a(year_offset)
        alive_b = cfg.mortality.alive_b(year_offset)
        filing_status = cfg.mortality.filing_status(year_offset)

        regime = cfg.regime_for_year(year_offset)

        a_working = alive_a and a_age < inputs.spouse_a_retire_age
        b_working = alive_b and b_age < inputs.spouse_b_retire_age

        # ---- Contributions (only while alive AND working) ----
        a_total_contrib = (
            spouse_a_salary * inputs.spouse_a_total_contrib_pct if a_working else 0.0
        )
        b_total_contrib = (
            spouse_b_salary * inputs.spouse_b_total_contrib_pct if b_working else 0.0
        )
        a_pretax_contrib = a_total_contrib * (1 - inputs.spouse_a_roth_401k_pct)
        a_roth_contrib = a_total_contrib * inputs.spouse_a_roth_401k_pct
        b_pretax_contrib = b_total_contrib * (1 - inputs.spouse_b_roth_401k_pct)
        b_roth_contrib = b_total_contrib * inputs.spouse_b_roth_401k_pct

        state.spouse_a_pretax += a_pretax_contrib
        state.spouse_b_pretax += b_pretax_contrib
        state.roth += a_roth_contrib + b_roth_contrib

        # ---- Income items ----
        wages = (spouse_a_salary + spouse_a_bonus if a_working else 0.0) + (
            spouse_b_salary if b_working else 0.0
        )
        wages_box1 = wages - a_pretax_contrib - b_pretax_contrib

        if a_age == inputs.pension.start_age and state.pension_annuity == 0.0 and alive_a:
            scale = state.pension_balance / max(test_balance, 1.0)
            state.pension_annuity = inputs.pension.annual_at_nrd * scale
        # Surviving spouse pension election applied if A has died.
        if a_age >= inputs.pension.start_age and alive_a:
            pension_income = state.pension_annuity
        elif a_age >= inputs.pension.start_age and not alive_a and alive_b:
            pension_income = state.pension_annuity * cfg.mortality.pension_survivor_pct
        else:
            pension_income = 0.0

        ssn_income = 0.0
        a_ss_eligible = alive_a and a_age >= inputs.ss.start_age
        b_ss_eligible = alive_b and b_age >= inputs.ss.start_age
        if alive_a and alive_b:
            if a_ss_eligible:
                ssn_income += inputs.ss.monthly_spouse_a * 12
            if b_ss_eligible:
                ssn_income += inputs.ss.monthly_spouse_b * 12
        else:
            # Survivor benefit: keep the larger of the two monthly amounts.
            if a_ss_eligible or b_ss_eligible:
                bigger = max(inputs.ss.monthly_spouse_a, inputs.ss.monthly_spouse_b)
                ssn_income = bigger * 12

        interest_inc = inputs.income.interest if a_working else 0.0
        ltcg_inc = inputs.income.capital_gains if a_working else 0.0
        qdiv_inc = inputs.income.dividends if a_working else 0.0

        base_kwargs = dict(
            wages=wages_box1,
            interest=interest_inc,
            qualified_div=qdiv_inc,
            ltcg=ltcg_inc,
            pension=pension_income,
            social_security=ssn_income,
        )

        # ---- Roth conversion (gap years, regime-aware) ----
        conv_a, conv_b = planned_roth_conversion(
            cfg, inputs, state, base_kwargs, regime=regime, filing_status=filing_status
        )
        # If a spouse is dead, zero out their conversion.
        if not alive_a:
            conv_a = 0.0
        if not alive_b:
            conv_b = 0.0
        conv = conv_a + conv_b
        if conv > 0:
            base_kwargs["roth_conversion"] = conv
            state.spouse_a_pretax = max(0.0, state.spouse_a_pretax - conv_a)
            state.spouse_b_pretax = max(0.0, state.spouse_b_pretax - conv_b)
            state.roth += conv

        # ---- RMDs (only required for living spouses) ----
        a_rmd = rmd_amount(state.spouse_a_pretax, a_age, cfg.rmd_start_age) if alive_a else 0.0
        b_rmd = rmd_amount(state.spouse_b_pretax, b_age, cfg.rmd_start_age) if alive_b else 0.0
        rmd_total = a_rmd + b_rmd

        # ---- Spending need (smile + lump events + LTC) ----
        # Lump events always add to net_need; the strategy picks where
        # the cash comes from. (Future enhancement: honor
        # `preferred_source` inside the strategy solvers.)
        years_until_horizon = (n_years - 1) - year_offset
        net_need, lump_events = spending.amount_for(
            year_offset, a_age, years_until_horizon=years_until_horizon
        )
        lump_total = 0.0
        for event in lump_events:
            inflated = event.amount_today * (1 + spending.inflation) ** year_offset
            net_need += inflated
            lump_total += inflated

        # ---- Live cost-basis fraction ----
        current_basis_frac = (
            min(1.0, max(0.0, state.cumulative_basis / state.taxable))
            if state.taxable > 0 else 1.0
        )

        if a_working or b_working:
            withdraws = {
                "pretax_a": a_rmd,
                "pretax_b": b_rmd,
                "pretax": rmd_total,
                "roth": 0.0,
                "taxable": 0.0,
            }
        else:
            withdraws = withdraw_for_need(
                net_need, state, cfg, base_kwargs, a_rmd, b_rmd,
                cfg.withdrawal_strategy, current_basis_frac,
                regime=regime, filing_status=filing_status,
            )

        # ---- Final tax kwargs ----
        final_kwargs = dict(base_kwargs)
        final_kwargs["pretax_withdrawal"] = (
            final_kwargs.get("pretax_withdrawal", 0.0) + withdraws["pretax"]
        )
        final_kwargs["ltcg"] = (
            final_kwargs.get("ltcg", 0.0)
            + withdraws["taxable"] * (1 - current_basis_frac)
        )

        tax_result = federal_tax(regime=regime, filing_status=filing_status, **final_kwargs)
        federal = tax_result["tax"]

        # ---- IRMAA (Medicare premium surcharge) ----
        n_medicare = (
            int(alive_a and a_age >= MEDICARE_ELIGIBLE_AGE)
            + int(alive_b and b_age >= MEDICARE_ELIGIBLE_AGE)
        )
        irmaa = irmaa_annual_surcharge(
            tax_result["agi"], n_medicare, regime=regime, filing_status=filing_status
        )
        irmaa_cost = irmaa["total"]
        irmaa_tier = irmaa["tier"]

        # ---- Apply withdrawals to balances (single coherent pass) ----
        state.spouse_a_pretax = max(0.0, state.spouse_a_pretax - withdraws["pretax_a"])
        state.spouse_b_pretax = max(0.0, state.spouse_b_pretax - withdraws["pretax_b"])
        state.roth -= withdraws["roth"]
        state.taxable -= withdraws["taxable"]
        state.cumulative_basis = max(
            state.cumulative_basis - withdraws["taxable"] * current_basis_frac, 0.0
        )

        gross_cash_in = withdraws["pretax"] + withdraws["roth"] + withdraws["taxable"]
        if not (a_working or b_working):
            total_after_tax_cash = (
                pension_income + ssn_income + gross_cash_in - federal - irmaa_cost
            )
            surplus = total_after_tax_cash - net_need
            if surplus > 0:
                state.taxable += surplus
                state.cumulative_basis += surplus
            elif surplus < 0 and state.taxable > 0:
                draw = min(-surplus, state.taxable)
                state.taxable -= draw
                state.cumulative_basis = max(
                    state.cumulative_basis - draw * current_basis_frac, 0.0
                )
        else:
            after_tax_wages = (
                wages_box1 + interest_inc + qdiv_inc + ltcg_inc - federal - irmaa_cost
            )
            net_savings = after_tax_wages - net_need
            if net_savings > 0:
                state.taxable += net_savings
                state.cumulative_basis += net_savings

        # ---- Growth via MarketModel + AssetLocation ----
        eq_r, bd_r = market.returns(year_offset)
        state.spouse_a_pretax *= 1 + asset_loc.pretax.annual_return(eq_r, bd_r)
        state.spouse_b_pretax *= 1 + asset_loc.pretax.annual_return(eq_r, bd_r)
        state.roth *= 1 + asset_loc.roth.annual_return(eq_r, bd_r)
        state.taxable *= 1 + asset_loc.taxable.annual_return(eq_r, bd_r) - cfg.taxable_drag
        state.hsa *= 1 + asset_loc.hsa.annual_return(eq_r, bd_r)
        if a_age < inputs.pension.start_age:
            state.pension_balance *= 1 + PENSION_INTEREST
            state.pension_balance += pension_annual_credit(
                spouse_a_salary if a_working else 0.0
            )

        # Salary growth only for living working spouses.
        if alive_a:
            spouse_a_salary *= 1 + cfg.wage_growth
            spouse_a_bonus *= 1 + cfg.wage_growth
        if alive_b:
            spouse_b_salary *= 1 + cfg.wage_growth

        rows.append(
            {
                "year": year,
                "spouse_a_age": a_age,
                "spouse_b_age": b_age,
                "alive_a": alive_a,
                "alive_b": alive_b,
                "filing_status": filing_status,
                "regime": regime.name,
                "wages": wages,
                "pension": pension_income,
                "ssn": ssn_income,
                "rmd": rmd_total,
                "rmd_a": a_rmd,
                "rmd_b": b_rmd,
                "roth_conversion": conv,
                "roth_conversion_a": conv_a,
                "roth_conversion_b": conv_b,
                "pretax_withdrawal": withdraws["pretax"],
                "pretax_withdrawal_a": withdraws["pretax_a"],
                "pretax_withdrawal_b": withdraws["pretax_b"],
                "roth_withdrawal": withdraws["roth"],
                "taxable_withdrawal": withdraws["taxable"],
                "agi": tax_result["agi"],
                "taxable_income": tax_result["taxable_income"],
                "federal_tax": federal,
                "marginal": tax_result["marginal"],
                "irmaa": irmaa_cost,
                "irmaa_tier": irmaa_tier,
                "spending_need": net_need,
                "equity_return": eq_r,
                "bond_return": bd_r,
                "pretax_balance": state.spouse_a_pretax + state.spouse_b_pretax,
                "pretax_a_balance": state.spouse_a_pretax,
                "pretax_b_balance": state.spouse_b_pretax,
                "roth_balance": state.roth,
                "taxable_balance": state.taxable,
                "hsa_balance": state.hsa,
                "pension_balance": state.pension_balance,
            }
        )
    return pd.DataFrame(rows)
