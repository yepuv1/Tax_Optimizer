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
from .limits import elective_deferral_cap, hsa_family_cap
from .pension import (
    PENSION_INTEREST,
    pension_annual_credit,
    project_pension_balance,
)
from .rmd import rmd_amount
from .state import State, initial_state
from .tax.federal import federal_tax
from .tax.irmaa import MEDICARE_ELIGIBLE_AGE, irmaa_annual_surcharge
from .withdrawals import cover_deficit, withdraw_for_need


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
        # IRS-cap each spouse's elective deferral (with age-50+ catch-up).
        # Without this guard, a user setting `spouse_a_total_contrib_pct
        # = 0.30` on a $300k salary would silently route $90k into a
        # vehicle whose real-world cap is ~$31k.
        a_target = (
            spouse_a_salary * inputs.spouse_a_total_contrib_pct if a_working else 0.0
        )
        b_target = (
            spouse_b_salary * inputs.spouse_b_total_contrib_pct if b_working else 0.0
        )
        a_total_contrib = min(a_target, elective_deferral_cap(a_age)) if a_working else 0.0
        b_total_contrib = min(b_target, elective_deferral_cap(b_age)) if b_working else 0.0
        a_pretax_contrib = a_total_contrib * (1 - inputs.spouse_a_roth_401k_pct)
        a_roth_contrib = a_total_contrib * inputs.spouse_a_roth_401k_pct
        b_pretax_contrib = b_total_contrib * (1 - inputs.spouse_b_roth_401k_pct)
        b_roth_contrib = b_total_contrib * inputs.spouse_b_roth_401k_pct

        # Employer 401(k) match. Pays in pre-tax (per IRS rules) on top
        # of the employee's elective deferral, regardless of whether
        # the employee chose Roth. Sized on the actually-deferred
        # percentage (already capped by elective_deferral_cap above)
        # rather than the user's pre-cap target.
        a_effective_pct = (
            (a_total_contrib / spouse_a_salary) if (a_working and spouse_a_salary > 0) else 0.0
        )
        b_effective_pct = (
            (b_total_contrib / spouse_b_salary) if (b_working and spouse_b_salary > 0) else 0.0
        )
        a_employer_match = (
            spouse_a_salary
            * min(a_effective_pct, inputs.spouse_a_employer_match_max_pct)
            * inputs.spouse_a_employer_match_rate
            if a_working else 0.0
        )
        b_employer_match = (
            spouse_b_salary
            * min(b_effective_pct, inputs.spouse_b_employer_match_max_pct)
            * inputs.spouse_b_employer_match_rate
            if b_working else 0.0
        )

        state.spouse_a_pretax += a_pretax_contrib + a_employer_match
        state.spouse_b_pretax += b_pretax_contrib + b_employer_match
        state.roth += a_roth_contrib + b_roth_contrib

        # HSA family contribution: capped at IRS family limit + 55+
        # catch-up, only valid while at least one spouse works AND is
        # Medicare-ineligible. Pre-tax: reduces wages_box1 below.
        hsa_cap = hsa_family_cap(a_age, b_age, either_working=(a_working or b_working))
        hsa_contrib = min(max(0.0, inputs.contrib.hsa_family), hsa_cap)
        state.hsa += hsa_contrib

        # ---- Income items ----
        wages = (spouse_a_salary + spouse_a_bonus if a_working else 0.0) + (
            spouse_b_salary if b_working else 0.0
        )
        # HSA contributions are pre-tax (Box 1 wage reducer alongside
        # traditional 401(k) deferrals).
        wages_box1 = wages - a_pretax_contrib - b_pretax_contrib - hsa_contrib

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

        # The "extra" non-portfolio income items (savings interest, a
        # one-off cap gain from a side hustle, an employer dividend on
        # ESPP shares, etc.) live on `inputs.income.*` and only apply
        # while at least one spouse is still working. Once both retire,
        # those streams stop -- but the taxable brokerage keeps
        # producing dividends and interest forever. That portfolio
        # yield is computed below from the year-start balance.
        extra_interest = inputs.income.interest if a_working else 0.0
        extra_ltcg = inputs.income.capital_gains if a_working else 0.0
        extra_qdiv = inputs.income.dividends if a_working else 0.0

        # Portfolio yield from the taxable brokerage. Applies in every
        # year (working OR retired), proportional to year-start balance
        # and split via asset_location. This is what makes provisional
        # income / NIIT / IRMAA actually track post-retirement reality.
        taxable_eq_balance = max(0.0, state.taxable) * asset_loc.taxable.equity_pct
        taxable_bd_balance = max(0.0, state.taxable) * asset_loc.taxable.bond_pct
        portfolio_qdiv = taxable_eq_balance * cfg.taxable_equity_div_yield
        portfolio_interest = taxable_bd_balance * cfg.taxable_bond_interest_yield

        interest_inc = extra_interest + portfolio_interest
        qdiv_inc = extra_qdiv + portfolio_qdiv
        ltcg_inc = extra_ltcg

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

        # ---- HSA tax-free pay-down of qualified medical expense ----
        # The LTC shock is the only explicit medical-expense line in
        # the model; HSA dollars are spent against it first (the whole
        # point of the triple-tax-advantaged shelter). Net_need drops
        # by the HSA draw, taxes on it stay zero (no AGI hit).
        ltc_today = 0.0
        if spending.ltc_shock and years_until_horizon < spending.ltc_shock.years:
            ltc_today = (
                spending.ltc_shock.annual_cost_today
                * (1 + spending.inflation) ** year_offset
            )
        hsa_withdrawal = min(state.hsa, ltc_today) if ltc_today > 0 else 0.0
        if hsa_withdrawal > 0:
            state.hsa = max(0.0, state.hsa - hsa_withdrawal)
            net_need = max(0.0, net_need - hsa_withdrawal)

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

        # Cash actually flowing into the household this year (extra
        # interest / div / cap-gains on `inputs.income.*` are real cash
        # in working years; portfolio dividends/interest above are
        # *notional* -- already reinvested in the taxable balance via
        # market returns, so they are *not* added to cash_inflow even
        # though they hit the tax line).
        if a_working or b_working:
            cash_inflow = (
                wages_box1
                + extra_interest + extra_qdiv + extra_ltcg
                + gross_cash_in
            )
        else:
            cash_inflow = pension_income + ssn_income + gross_cash_in

        delta = cash_inflow - federal - irmaa_cost - net_need
        unfunded = 0.0

        if delta > 0:
            # Surplus: lands in taxable as new basis.
            state.taxable += delta
            state.cumulative_basis += delta
        elif delta < 0:
            # Deficit (lump-event shortfall, conversion-tax shortfall,
            # gap-year tax overrun, ...). Cascade through tax-efficient
            # buckets and gross up properly. Tracks any genuinely
            # unfunded gap so monte-carlo can see it.
            extra, unfunded = cover_deficit(
                deficit=-delta,
                state=state,
                base_kwargs=base_kwargs,
                basis_frac=current_basis_frac,
                pretax_a_already=withdraws["pretax_a"],
                pretax_b_already=withdraws["pretax_b"],
                taxable_already=withdraws["taxable"],
                roth_already=withdraws["roth"],
                regime=regime,
                filing_status=filing_status,
            )
            if extra["taxable"] > 0:
                state.taxable = max(0.0, state.taxable - extra["taxable"])
                state.cumulative_basis = max(
                    state.cumulative_basis - extra["taxable"] * current_basis_frac, 0.0
                )
                final_kwargs["ltcg"] = (
                    final_kwargs.get("ltcg", 0.0)
                    + extra["taxable"] * (1 - current_basis_frac)
                )
                withdraws["taxable"] += extra["taxable"]
            if extra["roth"] > 0:
                state.roth = max(0.0, state.roth - extra["roth"])
                withdraws["roth"] += extra["roth"]
            if extra["pretax"] > 0:
                state.spouse_a_pretax = max(0.0, state.spouse_a_pretax - extra["pretax_a"])
                state.spouse_b_pretax = max(0.0, state.spouse_b_pretax - extra["pretax_b"])
                final_kwargs["pretax_withdrawal"] = (
                    final_kwargs.get("pretax_withdrawal", 0.0) + extra["pretax"]
                )
                withdraws["pretax_a"] += extra["pretax_a"]
                withdraws["pretax_b"] += extra["pretax_b"]
                withdraws["pretax"] = withdraws["pretax_a"] + withdraws["pretax_b"]

            # Recompute federal / IRMAA against the post-cascade kwargs
            # so the recorded numbers reflect the additional draws.
            if extra["pretax"] > 0 or extra["taxable"] > 0:
                tax_result = federal_tax(
                    regime=regime, filing_status=filing_status, **final_kwargs
                )
                federal = tax_result["tax"]
                irmaa = irmaa_annual_surcharge(
                    tax_result["agi"], n_medicare,
                    regime=regime, filing_status=filing_status,
                )
                irmaa_cost = irmaa["total"]
                irmaa_tier = irmaa["tier"]

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
                "unfunded": unfunded,
                "hsa_contrib": hsa_contrib,
                "hsa_withdrawal": hsa_withdrawal,
                "elective_deferral_a": a_total_contrib,
                "elective_deferral_b": b_total_contrib,
                "employer_match_a": a_employer_match,
                "employer_match_b": b_employer_match,
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
