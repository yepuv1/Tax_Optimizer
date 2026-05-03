"""Retirement Tax Optimizer - Spouse A & Spouse B (standalone script).

Goal: maximize after-tax terminal net worth at the planning horizon, subject
to never running out of money, by jointly choosing:

1. Pre-retirement allocation between Traditional 401(k) and Roth 401(k).
2. In-retirement withdrawal sequencing across taxable / pretax / Roth buckets.
3. Roth-conversion size during the retirement-to-RMD gap years.

This is a script port of ``tax_optimizer_standalone.ipynb``. Edit the
``starting`` / ``income`` / ``current_contrib`` blocks (or the ``Config``
defaults) to model your own scenario, then run::

    python tax_optimizer.py                # text output only
    python tax_optimizer.py --plots        # also display matplotlib figures
    python tax_optimizer.py --save-plots out/  # write figures as PNGs

Glossary: FRA, ERD, NRD, LTCG, NIIT, IRMAA, MAGI, MFJ, PI.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
plt.rcParams["figure.figsize"] = (11, 5)
plt.rcParams["axes.grid"] = True


# =============================================================================
# 1. Hardcoded inputs.
# Generic illustrative defaults for a married-filing-jointly couple in their
# early 50s. Edit any value here to model your own scenario. All amounts USD.
# =============================================================================

starting = {
    "spouse_a_pretax_401k": 500_000.0,
    "spouse_b_pretax_401k": 300_000.0,
    "spouse_a_roth_ira": 50_000.0,
    "spouse_b_pretax_ira": 50_000.0,
    "pension_balance": 0.0,
    "hsa": 20_000.0,
    "taxable_brokerage": 200_000.0,
    "total_nw_excl_real_estate": 1_120_000.0,
}

income = {
    "spouse_a_gross": 150_000.0,
    "spouse_b_gross": 100_000.0,
    "spouse_a_bonus": 0.0,
    "interest": 1_000.0,
    "capital_gains": 5_000.0,
    "dividends": 5_000.0,
}

current_contrib = {
    "spouse_a_pct": 0.10,
    "spouse_b_pct": 0.10,
    "spouse_a_roth_pct": 0.0,
    "spouse_b_roth_pct": 0.0,
    "hsa_family": 8_550.0,
    "std_deduction": 32_200.0,
    "baseline_tax": 0.0,
}

annual_expenses = 80_000.0

pension_balance_today = 0.0
monthly_pension_at_nrd = 0.0
annual_pension_at_nrd = monthly_pension_at_nrd * 12

ssn_monthly_spouse_a = 3_000.0
ssn_monthly_spouse_b = 2_500.0


# =============================================================================
# 2. Configuration
# =============================================================================


@dataclass
class Config:
    start_year: int = 2026
    spouse_a_age_start: int = 50
    spouse_b_age_start: int = 50
    spouse_a_retire_age: int = 65
    spouse_b_retire_age: int = 65
    horizon_age: int = 80

    nominal_growth_rate: float = 0.05
    inflation: float = 0.025
    wage_growth: float = 0.025
    taxable_drag: float = 0.005

    spouse_a_total_contrib_pct: float = (
        current_contrib["spouse_a_pct"] + current_contrib["spouse_a_roth_pct"]
    )
    spouse_b_total_contrib_pct: float = (
        current_contrib["spouse_b_pct"] + current_contrib["spouse_b_roth_pct"]
    )
    spouse_a_roth_401k_pct: float = (
        current_contrib["spouse_a_roth_pct"]
        / (current_contrib["spouse_a_pct"] + current_contrib["spouse_a_roth_pct"])
        if (current_contrib["spouse_a_pct"] + current_contrib["spouse_a_roth_pct"]) > 0
        else 0.0
    )
    spouse_b_roth_401k_pct: float = (
        current_contrib["spouse_b_roth_pct"]
        / (current_contrib["spouse_b_pct"] + current_contrib["spouse_b_roth_pct"])
        if (current_contrib["spouse_b_pct"] + current_contrib["spouse_b_roth_pct"]) > 0
        else 0.0
    )

    withdrawal_strategy: str = "conventional"
    bracket_fill_target: float = 0.22
    roth_conversion_target_bracket: float = 0.0
    roth_conversion_amount: float = 0.0

    ss_start_age: int = 70
    pension_start_age: int = 65
    rmd_start_age: int = 75

    cap_gains_basis_fraction: float = 0.5
    annual_expenses_today: float = annual_expenses
    irmaa_threshold_mfj: float = 212_000.0


# =============================================================================
# 3. Tax engine (2026 MFJ)
# =============================================================================

ORD_BRACKETS_MFJ = [
    (0,        23_850,   0.10),
    (23_850,   96_950,   0.12),
    (96_950,  206_700,   0.22),
    (206_700, 394_600,   0.24),
    (394_600, 501_050,   0.32),
    (501_050, 751_600,   0.35),
    (751_600, math.inf,  0.37),
]
LTCG_BRACKETS_MFJ = [
    (0,        96_700,    0.00),
    (96_700,   600_050,   0.15),
    (600_050,  math.inf,  0.20),
]
STD_DEDUCTION_MFJ = 32_200.0
NIIT_THRESHOLD_MFJ = 250_000.0
NIIT_RATE = 0.038


def _bracket_tax(amount: float, brackets) -> float:
    tax = 0.0
    for lo, hi, rate in brackets:
        if amount <= lo:
            break
        tax += (min(amount, hi) - lo) * rate
    return tax


def _marginal_rate(amount: float, brackets) -> float:
    rate = brackets[0][2]
    for lo, _hi, r in brackets:
        if amount > lo:
            rate = r
    return rate


def social_security_taxable(provisional: float, ss_benefits: float) -> float:
    if ss_benefits <= 0:
        return 0.0
    base1, base2 = 32_000.0, 44_000.0
    if provisional <= base1:
        return 0.0
    if provisional <= base2:
        return min(0.5 * (provisional - base1), 0.5 * ss_benefits)
    tier1 = min(0.5 * (base2 - base1), 0.5 * ss_benefits)
    tier2 = 0.85 * (provisional - base2)
    return min(tier1 + tier2, 0.85 * ss_benefits)


def federal_tax(
    *,
    wages: float = 0.0,
    interest: float = 0.0,
    ordinary_div: float = 0.0,
    qualified_div: float = 0.0,
    ltcg: float = 0.0,
    pension: float = 0.0,
    pretax_withdrawal: float = 0.0,
    roth_conversion: float = 0.0,
    social_security: float = 0.0,
    deduction: float = STD_DEDUCTION_MFJ,
) -> dict:
    other = (
        wages + interest + ordinary_div + qualified_div + ltcg
        + pension + pretax_withdrawal + roth_conversion
    )
    provisional = other + 0.5 * social_security
    ss_taxable = social_security_taxable(provisional, social_security)

    ordinary_income = (
        wages + interest + ordinary_div + pension
        + pretax_withdrawal + roth_conversion + ss_taxable
    )
    preferential = qualified_div + ltcg
    agi = ordinary_income + preferential

    taxable_total = max(0.0, agi - deduction)
    taxable_ordinary = max(0.0, taxable_total - preferential)
    taxable_pref = max(0.0, taxable_total - taxable_ordinary)

    ord_tax = _bracket_tax(taxable_ordinary, ORD_BRACKETS_MFJ)

    ltcg_tax = 0.0
    remaining = taxable_pref
    cursor = taxable_ordinary
    for lo, hi, rate in LTCG_BRACKETS_MFJ:
        if remaining <= 0:
            break
        room = max(0.0, hi - max(lo, cursor))
        slab = min(remaining, room)
        if slab > 0:
            ltcg_tax += slab * rate
            remaining -= slab
            cursor += slab

    investment_income = interest + ordinary_div + qualified_div + ltcg
    niit = NIIT_RATE * max(0.0, min(investment_income, agi - NIIT_THRESHOLD_MFJ))
    niit = max(0.0, niit)

    total = ord_tax + ltcg_tax + niit
    marginal = _marginal_rate(taxable_ordinary, ORD_BRACKETS_MFJ)

    return {
        "tax": total,
        "ordinary_tax": ord_tax,
        "ltcg_tax": ltcg_tax,
        "niit": niit,
        "agi": agi,
        "taxable_income": taxable_total,
        "ss_taxable": ss_taxable,
        "marginal": marginal,
    }


def amount_to_fill_bracket(
    base_taxable: float, target_top: float, brackets=ORD_BRACKETS_MFJ
) -> float:
    for _lo, hi, rate in brackets:
        if math.isclose(rate, target_top, abs_tol=1e-6):
            return max(0.0, hi - base_taxable)
    return 0.0


# IRMAA - Income-Related Monthly Adjustment Amount (Medicare Part B + Part D).
# Lookback: real IRMAA uses MAGI from 2 years prior; we approximate with current AGI.
IRMAA_TIERS_MFJ = [
    (212_000.0,   0.00,   0.00),
    (266_000.0,  74.00,  13.70),
    (334_000.0, 185.00,  35.30),
    (400_000.0, 295.00,  57.00),
    (750_000.0, 406.00,  78.60),
    (math.inf,  443.90,  85.80),
]
MEDICARE_ELIGIBLE_AGE = 65


def irmaa_annual_surcharge(magi: float, n_enrolled: int) -> dict:
    if n_enrolled <= 0 or magi <= IRMAA_TIERS_MFJ[0][0]:
        return {"partB": 0.0, "partD": 0.0, "total": 0.0, "tier": 0}
    for tier_idx, (cap, partB, partD) in enumerate(IRMAA_TIERS_MFJ):
        if magi <= cap:
            annual = (partB + partD) * 12 * n_enrolled
            return {
                "partB": partB * 12 * n_enrolled,
                "partD": partD * 12 * n_enrolled,
                "total": annual,
                "tier": tier_idx,
            }
    _cap, partB, partD = IRMAA_TIERS_MFJ[-1]
    annual = (partB + partD) * 12 * n_enrolled
    return {
        "partB": partB * 12 * n_enrolled,
        "partD": partD * 12 * n_enrolled,
        "total": annual,
        "tier": len(IRMAA_TIERS_MFJ) - 1,
    }


# =============================================================================
# 4. RMD engine (per spouse, per account)
# =============================================================================

UNIFORM_LIFETIME = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4,
    88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9,
    96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2,
    104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1, 108: 3.9, 109: 3.7, 110: 3.5,
}


def rmd_amount(balance: float, age: int, rmd_start_age: int = 75) -> float:
    if age < rmd_start_age or balance <= 0:
        return 0.0
    keys = [k for k in UNIFORM_LIFETIME if k <= age]
    divisor = UNIFORM_LIFETIME.get(age, UNIFORM_LIFETIME[max(keys)] if keys else 1.0)
    return balance / divisor


# =============================================================================
# 5. Pension projector (cash-balance plan)
# =============================================================================

PENSION_INTEREST = 0.048
PENSION_QTR_SSWB = 184_500.0 / 4
PENSION_LOW_RATE = 0.06
PENSION_HIGH_RATE = 0.11


def pension_annual_credit(annual_eligible_earnings: float) -> float:
    monthly_e = annual_eligible_earnings / 12.0
    low_credit = min(monthly_e, PENSION_QTR_SSWB) * PENSION_LOW_RATE
    high_credit = max(0.0, monthly_e - PENSION_QTR_SSWB) * PENSION_HIGH_RATE
    return (low_credit + high_credit) * 12


def project_pension_balance(start_balance, start_earnings, years_to_retire, wage_growth=0.025):
    bal, earn = start_balance, start_earnings
    for _ in range(years_to_retire):
        bal = bal * (1 + PENSION_INTEREST) + pension_annual_credit(earn)
        earn *= 1 + wage_growth
    return bal


# Reference projected pension balance at NRD; used to scale annuity if balance grows faster.
_test_balance = project_pension_balance(pension_balance_today, income["spouse_a_gross"], 11)


# =============================================================================
# 6. Account state
# =============================================================================


@dataclass
class State:
    year: int
    spouse_a_age: int
    spouse_b_age: int
    spouse_a_pretax: float
    spouse_b_pretax: float
    roth: float
    taxable: float
    hsa: float
    pension_balance: float
    pension_annuity: float = 0.0
    cumulative_basis: float = 0.0


def initial_state(cfg: Config) -> State:
    pretax_v = starting["spouse_a_pretax_401k"]
    pretax_b = starting["spouse_b_pretax_401k"] + starting["spouse_b_pretax_ira"]
    roth = starting["spouse_a_roth_ira"]
    taxable = starting["taxable_brokerage"]
    return State(
        year=cfg.start_year,
        spouse_a_age=cfg.spouse_a_age_start,
        spouse_b_age=cfg.spouse_b_age_start,
        spouse_a_pretax=pretax_v,
        spouse_b_pretax=pretax_b,
        roth=roth,
        taxable=taxable,
        hsa=starting["hsa"],
        pension_balance=pension_balance_today,
        pension_annuity=0.0,
        cumulative_basis=taxable * cfg.cap_gains_basis_fraction,
    )


# =============================================================================
# 7. Withdrawal strategies (pure, no mutation)
# =============================================================================


def _solve_pretax_for_net(net_target: float, base_kwargs: dict) -> float:
    if net_target <= 0:
        return 0.0
    base_tax = federal_tax(**base_kwargs)["tax"]
    lo, hi = 0.0, net_target * 2.0 + 50_000.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        kw = dict(base_kwargs)
        kw["pretax_withdrawal"] = kw.get("pretax_withdrawal", 0.0) + mid
        net = mid - (federal_tax(**kw)["tax"] - base_tax)
        if net < net_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0:
            break
    return 0.5 * (lo + hi)


def _solve_taxable_for_net(net_target: float, base_kwargs: dict, basis_frac: float) -> float:
    if net_target <= 0:
        return 0.0
    base_tax = federal_tax(**base_kwargs)["tax"]
    lo, hi = 0.0, net_target * 2.0 + 50_000.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        gain = mid * (1 - basis_frac)
        kw = dict(base_kwargs)
        kw["ltcg"] = kw.get("ltcg", 0.0) + gain
        net = mid - (federal_tax(**kw)["tax"] - base_tax)
        if net < net_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0:
            break
    return 0.5 * (lo + hi)


def withdraw_for_need(net_need, state, cfg, base_kwargs, a_rmd, b_rmd, strategy, basis_frac):
    """Return gross withdrawals split per spouse for pretax.

    Each spouse's RMD must be taken from their own account (IRS rule for
    workplace 401(k)/IRA aggregation). Above-RMD pretax withdrawals are
    allocated pro-rata to remaining pretax balances since both spouses' pretax
    dollars hit the same federal tax line on a joint return.

    `basis_frac` is the live cost-basis / current-balance ratio for the
    taxable account at the start of the year (computed by the caller from
    `state.cumulative_basis / state.taxable`). Using a static `cfg`
    fraction would systematically underestimate LTCG once reinvested gains
    accumulate over a long horizon.
    """
    a_rmd = min(a_rmd, max(state.spouse_a_pretax, 0.0))
    b_rmd = min(b_rmd, max(state.spouse_b_pretax, 0.0))
    rmd_total = a_rmd + b_rmd

    ctx = dict(base_kwargs)
    ctx["pretax_withdrawal"] = ctx.get("pretax_withdrawal", 0.0) + rmd_total

    base_tax = federal_tax(**base_kwargs)["tax"]
    rmd_tax = federal_tax(**ctx)["tax"] - base_tax
    rmd_net = rmd_total - max(0.0, rmd_tax)
    remaining = max(0.0, net_need - rmd_net)

    pretax_a_w = a_rmd
    pretax_b_w = b_rmd
    roth_w = 0.0
    taxable_w = 0.0
    pretax_a_room = max(0.0, state.spouse_a_pretax - pretax_a_w)
    pretax_b_room = max(0.0, state.spouse_b_pretax - pretax_b_w)
    pretax_room = pretax_a_room + pretax_b_room

    def split_extra(extra):
        if extra <= 0 or pretax_room <= 0:
            return 0.0, 0.0
        a_share = pretax_a_room / pretax_room
        ax = min(extra * a_share, pretax_a_room)
        bx = min(extra - ax, pretax_b_room)
        return ax, bx

    if strategy == "conventional":
        if remaining > 0 and state.taxable > 0:
            tw = min(
                _solve_taxable_for_net(min(remaining, state.taxable), ctx, basis_frac),
                state.taxable,
            )
            ctx_after = dict(ctx)
            ctx_after["ltcg"] = ctx_after.get("ltcg", 0.0) + tw * (1 - basis_frac)
            net_from_t = tw - (federal_tax(**ctx_after)["tax"] - federal_tax(**ctx)["tax"])
            taxable_w += tw
            ctx = ctx_after
            remaining = max(0.0, remaining - max(0.0, net_from_t))
        if remaining > 0 and pretax_room > 0:
            extra = min(_solve_pretax_for_net(remaining, ctx), pretax_room)
            ax, bx = split_extra(extra)
            pretax_a_w += ax
            pretax_b_w += bx
            ctx["pretax_withdrawal"] += extra
            remaining = 0.0
        if remaining > 0 and state.roth > 0:
            roth_w += min(remaining, state.roth)

    elif strategy == "proportional":
        avail = max(state.taxable, 0) + max(pretax_room, 0) + max(state.roth, 0)
        if avail > 0 and remaining > 0:
            w_t = remaining * (state.taxable / avail)
            w_p = remaining * (max(pretax_room, 0) / avail)
            w_r = remaining * (state.roth / avail)
            tw = min(
                _solve_taxable_for_net(w_t, ctx, basis_frac),
                state.taxable,
            )
            ctx["ltcg"] = ctx.get("ltcg", 0.0) + tw * (1 - basis_frac)
            taxable_w += tw
            pw = min(_solve_pretax_for_net(w_p, ctx), pretax_room)
            ax, bx = split_extra(pw)
            pretax_a_w += ax
            pretax_b_w += bx
            ctx["pretax_withdrawal"] += pw
            roth_w += min(w_r, state.roth)

    elif strategy == "bracket_fill":
        ti_now = federal_tax(**ctx)["taxable_income"]
        headroom = amount_to_fill_bracket(ti_now, cfg.bracket_fill_target)
        if remaining > 0 and headroom > 0 and pretax_room > 0:
            cap = min(headroom, pretax_room)
            extra = min(_solve_pretax_for_net(min(remaining, cap * 0.85), ctx), cap)
            ax, bx = split_extra(extra)
            pretax_a_w += ax
            pretax_b_w += bx
            ctx["pretax_withdrawal"] += extra
            remaining = max(0.0, remaining - extra * 0.78)
        if remaining > 0 and state.taxable > 0:
            tw = min(
                _solve_taxable_for_net(min(remaining, state.taxable), ctx, basis_frac),
                state.taxable,
            )
            ctx["ltcg"] = ctx.get("ltcg", 0.0) + tw * (1 - basis_frac)
            taxable_w += tw
            remaining = max(0.0, remaining - tw * 0.97)
        if remaining > 0 and state.roth > 0:
            roth_w += min(remaining, state.roth)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    return {
        "pretax_a": pretax_a_w,
        "pretax_b": pretax_b_w,
        "pretax": pretax_a_w + pretax_b_w,
        "roth": roth_w,
        "taxable": taxable_w,
    }


# =============================================================================
# 8. Roth conversion module
# =============================================================================


def planned_roth_conversion(cfg: Config, state: State, base_kwargs: dict):
    """Return (conv_a, conv_b): Roth conversion amounts per spouse this year.

    A conversion may only come from the converting spouse's own pretax
    account. A spouse is eligible when individually retired and pre-RMD age.
    When both are eligible the target is split pro-rata across pretax balances.
    """
    a_in_gap = (state.spouse_a_age >= cfg.spouse_a_retire_age) and (
        state.spouse_a_age < cfg.rmd_start_age
    )
    b_in_gap = (state.spouse_b_age >= cfg.spouse_b_retire_age) and (
        state.spouse_b_age < cfg.rmd_start_age
    )
    a_avail = state.spouse_a_pretax if a_in_gap else 0.0
    b_avail = state.spouse_b_pretax if b_in_gap else 0.0
    cap = a_avail + b_avail
    if cap <= 0:
        return 0.0, 0.0

    if cfg.roth_conversion_amount > 0:
        total = min(cfg.roth_conversion_amount, cap)
    elif cfg.roth_conversion_target_bracket > 0:
        ti_now = federal_tax(**base_kwargs)["taxable_income"]
        headroom = amount_to_fill_bracket(ti_now, cfg.roth_conversion_target_bracket)
        total = min(headroom, cap)
    else:
        return 0.0, 0.0

    if total <= 0:
        return 0.0, 0.0
    a_share = a_avail / cap
    conv_a = total * a_share
    conv_b = total - conv_a
    return conv_a, conv_b


# =============================================================================
# 9. Year-by-year simulator
# =============================================================================


def simulate(cfg: Config) -> pd.DataFrame:
    state = initial_state(cfg)
    rows = []

    spouse_a_salary = income["spouse_a_gross"]
    spouse_b_salary = income["spouse_b_gross"]
    spouse_a_bonus = income["spouse_a_bonus"]

    for year_offset in range(cfg.horizon_age - cfg.spouse_a_age_start + 1):
        year = cfg.start_year + year_offset
        a_age = cfg.spouse_a_age_start + year_offset
        b_age = cfg.spouse_b_age_start + year_offset
        # Keep state ages in sync so age-gated logic (Roth conv) fires.
        state.spouse_a_age = a_age
        state.spouse_b_age = b_age
        state.year = year

        a_working = a_age < cfg.spouse_a_retire_age
        b_working = b_age < cfg.spouse_b_retire_age

        a_total_contrib = spouse_a_salary * cfg.spouse_a_total_contrib_pct if a_working else 0.0
        b_total_contrib = spouse_b_salary * cfg.spouse_b_total_contrib_pct if b_working else 0.0
        a_pretax_contrib = a_total_contrib * (1 - cfg.spouse_a_roth_401k_pct)
        a_roth_contrib = a_total_contrib * cfg.spouse_a_roth_401k_pct
        b_pretax_contrib = b_total_contrib * (1 - cfg.spouse_b_roth_401k_pct)
        b_roth_contrib = b_total_contrib * cfg.spouse_b_roth_401k_pct

        state.spouse_a_pretax += a_pretax_contrib
        state.spouse_b_pretax += b_pretax_contrib
        state.roth += a_roth_contrib + b_roth_contrib

        wages = (spouse_a_salary + spouse_a_bonus if a_working else 0.0) + (
            spouse_b_salary if b_working else 0.0
        )
        wages_box1 = wages - a_pretax_contrib - b_pretax_contrib

        if a_age == cfg.pension_start_age and state.pension_annuity == 0.0:
            scale = state.pension_balance / max(_test_balance, 1.0)
            state.pension_annuity = annual_pension_at_nrd * scale
        pension_income = state.pension_annuity if a_age >= cfg.pension_start_age else 0.0

        ssn_income = 0.0
        if a_age >= cfg.ss_start_age:
            ssn_income += ssn_monthly_spouse_a * 12
        if b_age >= cfg.ss_start_age:
            ssn_income += ssn_monthly_spouse_b * 12

        interest_inc = income["interest"] if a_working else 0.0
        ltcg_inc = income["capital_gains"] if a_working else 0.0
        qdiv_inc = income["dividends"] if a_working else 0.0

        base_kwargs = dict(
            wages=wages_box1,
            interest=interest_inc,
            qualified_div=qdiv_inc,
            ltcg=ltcg_inc,
            pension=pension_income,
            social_security=ssn_income,
        )

        conv_a, conv_b = planned_roth_conversion(cfg, state, base_kwargs)
        conv = conv_a + conv_b
        if conv > 0:
            base_kwargs["roth_conversion"] = conv
            state.spouse_a_pretax = max(0.0, state.spouse_a_pretax - conv_a)
            state.spouse_b_pretax = max(0.0, state.spouse_b_pretax - conv_b)
            state.roth += conv

        a_rmd = rmd_amount(state.spouse_a_pretax, a_age, cfg.rmd_start_age)
        b_rmd = rmd_amount(state.spouse_b_pretax, b_age, cfg.rmd_start_age)
        rmd_total = a_rmd + b_rmd

        net_need = cfg.annual_expenses_today * (1 + cfg.inflation) ** year_offset

        # Live cost-basis fraction for the taxable account. After years of
        # reinvested growth this drifts well below the static config value,
        # so use the tracked basis to compute LTCG accurately.
        current_basis_frac = (
            min(1.0, max(0.0, state.cumulative_basis / state.taxable))
            if state.taxable > 0
            else 1.0
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
                net_need,
                state,
                cfg,
                base_kwargs,
                a_rmd,
                b_rmd,
                cfg.withdrawal_strategy,
                current_basis_frac,
            )

        final_kwargs = dict(base_kwargs)
        final_kwargs["pretax_withdrawal"] = (
            final_kwargs.get("pretax_withdrawal", 0.0) + withdraws["pretax"]
        )
        final_kwargs["ltcg"] = (
            final_kwargs.get("ltcg", 0.0)
            + withdraws["taxable"] * (1 - current_basis_frac)
        )

        tax_result = federal_tax(**final_kwargs)
        federal = tax_result["tax"]

        n_medicare = int(a_age >= MEDICARE_ELIGIBLE_AGE) + int(b_age >= MEDICARE_ELIGIBLE_AGE)
        irmaa = irmaa_annual_surcharge(tax_result["agi"], n_medicare)
        irmaa_cost = irmaa["total"]
        irmaa_tier = irmaa["tier"]

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

        g = 1 + cfg.nominal_growth_rate
        state.spouse_a_pretax *= g
        state.spouse_b_pretax *= g
        state.roth *= g
        state.taxable *= g - cfg.taxable_drag
        state.hsa *= g
        if a_age < cfg.pension_start_age:
            state.pension_balance *= 1 + PENSION_INTEREST
            state.pension_balance += pension_annual_credit(spouse_a_salary if a_working else 0.0)

        spouse_a_salary *= 1 + cfg.wage_growth
        spouse_b_salary *= 1 + cfg.wage_growth
        spouse_a_bonus *= 1 + cfg.wage_growth

        rows.append(
            {
                "year": year,
                "spouse_a_age": a_age,
                "spouse_b_age": b_age,
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


# =============================================================================
# 10. Strategy summarizers
# =============================================================================


def terminal_after_tax_nw(df: pd.DataFrame, *, marginal_rate: float = 0.22) -> float:
    last = df.iloc[-1]
    pretax_after_tax = last["pretax_balance"] * (1 - marginal_rate)
    return pretax_after_tax + last["roth_balance"] + last["taxable_balance"] + last["hsa_balance"]


def lifetime_tax_npv(df: pd.DataFrame, discount: float = 0.025) -> float:
    years = np.arange(len(df))
    return float((df["federal_tax"].values / (1 + discount) ** years).sum())


def summarize(df: pd.DataFrame) -> dict:
    years = np.arange(len(df))
    irmaa_npv = float((df["irmaa"].values / (1 + 0.025) ** years).sum())
    return {
        "lifetime_tax_npv": lifetime_tax_npv(df),
        "lifetime_irmaa_npv": irmaa_npv,
        "terminal_after_tax": terminal_after_tax_nw(df),
        "peak_marginal": float(df["marginal"].max()),
        "years_irmaa": int((df["irmaa"] > 0).sum()),
        "peak_irmaa_tier": int(df["irmaa_tier"].max()),
        "min_balance": float(
            (df["pretax_balance"] + df["roth_balance"] + df["taxable_balance"]).min()
        ),
    }


# =============================================================================
# 11. Optimizer (S3) - SLSQP + grid warm-start
# =============================================================================

BRACKET_CHOICES = [0.0, 0.12, 0.22, 0.24, 0.32]


def x_to_cfg(x: np.ndarray, base_cfg: Config) -> Config:
    a_roth = float(np.clip(x[0], 0.0, 1.0))
    b_roth = float(np.clip(x[1], 0.0, 1.0))
    idx = int(np.clip(round(x[2]), 0, len(BRACKET_CHOICES) - 1))
    return replace(
        base_cfg,
        spouse_a_roth_401k_pct=a_roth,
        spouse_b_roth_401k_pct=b_roth,
        roth_conversion_target_bracket=BRACKET_CHOICES[idx],
    )


def make_objective(base_cfg: Config):
    def objective(x: np.ndarray) -> float:
        cfg = x_to_cfg(x, base_cfg)
        df = simulate(cfg)
        liquid = df["pretax_balance"] + df["roth_balance"] + df["taxable_balance"]
        floor = df["spending_need"]
        deficit = float((floor - liquid).clip(lower=0).sum())
        irmaa_total = float(df["irmaa"].sum())
        terminal = terminal_after_tax_nw(df)
        return -(terminal - 1e3 * deficit - 0.5 * irmaa_total)

    return objective


def optimize_s3(base_cfg: Config, *, seed: int = 0) -> tuple[Config, np.ndarray]:
    """Optimize S3 with `scipy.optimize.differential_evolution`.

    Why DE instead of SLSQP: the objective contains hard step-function
    discontinuities (IRMAA cliffs every $54k of MAGI in retirement, plus
    an internal `round()` on the bracket index). Gradient-based methods
    like SLSQP see a flat gradient between cliffs and are blind at them,
    so they tend to stop at whichever side of a cliff the warm-start
    happens to be on. DE is derivative-free and population-based: it
    samples across the whole bounded box and routinely jumps cliffs.

    The coarse grid is still useful as a sanity check on the DE result.
    """
    objective = make_objective(base_cfg)

    grid_results = []
    for v in np.linspace(0, 1, 4):
        for b in np.linspace(0, 1, 4):
            for c in range(len(BRACKET_CHOICES)):
                x0 = np.array([v, b, c], dtype=float)
                grid_results.append((objective(x0), x0))
    grid_results.sort(key=lambda t: t[0])
    best_grid = grid_results[0]
    print(f"Best grid: terminal=${-best_grid[0]:,.0f}, x={best_grid[1]}")

    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, len(BRACKET_CHOICES) - 1)]
    res = differential_evolution(
        objective,
        bounds=bounds,
        # `polish=False` because the gradient-based polishing step DE
        # would otherwise run is exactly the SLSQP-on-cliffs failure mode
        # we are trying to avoid.
        polish=False,
        maxiter=40,
        popsize=15,
        tol=1e-3,
        seed=seed,
        init="sobol",
        workers=1,
    )
    x_opt = res.x if -res.fun > -best_grid[0] else best_grid[1]
    print(
        f"differential_evolution: success={res.success}, "
        f"terminal=${-res.fun:,.0f}, x={res.x}"
    )
    return x_to_cfg(x_opt, base_cfg), x_opt


# =============================================================================
# 12. Tornado sensitivity + recommended actions
# =============================================================================


def _ranked_strategies(results: dict) -> list[tuple[str, tuple]]:
    return sorted(
        results.items(),
        key=lambda kv: kv[1][2]["terminal_after_tax"],
        reverse=True,
    )


def _winning_cfg(results: dict, default_cfg: Config) -> tuple[str, Config]:
    if not results:
        return "S0_baseline", default_cfg
    name, (cfg, _df, _summary) = _ranked_strategies(results)[0]
    return name, cfg


def tornado_sensitivity(base_cfg: Config) -> tuple[pd.DataFrame, float]:
    base_terminal = terminal_after_tax_nw(simulate(base_cfg))

    def _terminal(**overrides) -> float:
        return terminal_after_tax_nw(simulate(replace(base_cfg, **overrides)))

    def int_clamp(v, lo, hi):
        return int(max(lo, min(hi, v)))

    perturbations = [
        ("spouse_a_roth_401k_pct", 0.0, 1.0),
        ("spouse_b_roth_401k_pct", 0.0, 1.0),
        ("roth_conversion_target_bracket", 0.0, 0.32),
        (
            "spouse_a_total_contrib_pct",
            max(0.0, base_cfg.spouse_a_total_contrib_pct - 0.05),
            min(1.0, base_cfg.spouse_a_total_contrib_pct + 0.05),
        ),
        (
            "spouse_b_total_contrib_pct",
            max(0.0, base_cfg.spouse_b_total_contrib_pct - 0.05),
            min(1.0, base_cfg.spouse_b_total_contrib_pct + 0.05),
        ),
        (
            "spouse_a_retire_age",
            int_clamp(base_cfg.spouse_a_retire_age - 2, 50, 75),
            int_clamp(base_cfg.spouse_a_retire_age + 2, 50, 75),
        ),
        (
            "spouse_b_retire_age",
            int_clamp(base_cfg.spouse_b_retire_age - 2, 50, 75),
            int_clamp(base_cfg.spouse_b_retire_age + 2, 50, 75),
        ),
        (
            "ss_start_age",
            int_clamp(base_cfg.ss_start_age - 3, 62, 70),
            int_clamp(base_cfg.ss_start_age + 3, 62, 70),
        ),
        (
            "nominal_growth_rate",
            max(0.0, base_cfg.nominal_growth_rate - 0.01),
            base_cfg.nominal_growth_rate + 0.01,
        ),
        (
            "inflation",
            max(0.0, base_cfg.inflation - 0.01),
            base_cfg.inflation + 0.01,
        ),
        (
            "annual_expenses_today",
            base_cfg.annual_expenses_today * 0.90,
            base_cfg.annual_expenses_today * 1.10,
        ),
    ]

    rows: list[dict] = []
    for param, lo, hi in perturbations:
        if lo == hi:
            continue
        delta_lo = _terminal(**{param: lo}) - base_terminal
        delta_hi = _terminal(**{param: hi}) - base_terminal
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


def _action_for_param(param: str, direction: str, lo: float, hi: float, base_cfg: Config) -> str:
    base_val = getattr(base_cfg, param, None)
    if param.endswith("_roth_401k_pct"):
        spouse = "Spouse A" if param.startswith("spouse_a") else "Spouse B"
        target = hi if direction == "higher" else lo
        return f"Move {spouse}'s 401(k) deferrals to {target:.0%} Roth (currently {base_val:.0%})."
    if param.endswith("_total_contrib_pct"):
        spouse = "Spouse A" if param.startswith("spouse_a") else "Spouse B"
        target = hi if direction == "higher" else lo
        return (
            f"Adjust {spouse}'s total 401(k) contribution rate to {target:.0%} "
            f"(currently {base_val:.0%})."
        )
    if param == "roth_conversion_target_bracket":
        target = hi if direction == "higher" else lo
        return (
            f"Set Roth-conversion bracket target to {target:.0%} (currently {base_val:.0%})."
        )
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
        return (
            f"{verb.capitalize()} annual spending toward ${target:,.0f} "
            f"(currently ${base_val:,.0f})."
        )
    if param == "nominal_growth_rate":
        return "Market assumption (not an action)."
    if param == "inflation":
        return "Macro assumption only; useful as a stress-test."
    return f"Move toward {direction} end of the tested range."


def render_actions(
    results: dict, sens_df: pd.DataFrame, base_cfg: Config, base_terminal: float
) -> str:
    winner_name, winner_cfg = _winning_cfg(results, base_cfg)
    baseline_sum = results.get("S0_baseline", (None, None, {}))[2]
    winner_sum = results[winner_name][2]

    lines: list[str] = ["### Recommended actions", ""]

    if winner_name == "S0_baseline":
        lines.append(
            "1. Stay the course. Baseline (S0_baseline) already maximizes terminal "
            "after-tax NW under your inputs - no contribution-mix change needed."
        )
    elif winner_name == "S1_all_roth_401k":
        a_pct = winner_cfg.spouse_a_total_contrib_pct
        b_pct = winner_cfg.spouse_b_total_contrib_pct
        lines.append(
            f"1. Switch contributions to Roth 401(k). Direct 100% of Spouse A's "
            f"{a_pct:.0%} and Spouse B's {b_pct:.0%} salary deferrals into the Roth "
            "bucket of their 401(k). No retirement-year conversions needed."
        )
    elif winner_name == "S2_bracket_fill_22":
        lines.append(
            "1. Plan gap-year Roth conversions. Keep current pretax contributions "
            "while working, then in each retirement year before RMDs start, convert "
            f"pretax -> Roth up to the top of the {winner_cfg.roth_conversion_target_bracket:.0%} "
            "federal bracket."
        )
    elif winner_name == "S3_optimized":
        a_roth = winner_cfg.spouse_a_roth_401k_pct
        b_roth = winner_cfg.spouse_b_roth_401k_pct
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
        lines.append(f"1. Adopt `{winner_name}` allocation (best terminal NW in this run).")

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
        action = _action_for_param(param, better_dir, lo, hi, base_cfg)
        lines.append(
            f"   - {param} - swing ${row['swing']:,.0f}; pushing it {better_dir} "
            f"adds ~${better_delta:,.0f}. {action}"
        )

    lines.append("")
    lines.append("3. Always-good hygiene (independent of which strategy wins):")
    lines.append("   - Max out the HSA family contribution every year - triple-tax-advantaged.")
    lines.append(
        "   - Hold an emergency / IRMAA-buffer of 1-2 years of expenses in taxable so you "
        "don't have to realize gains during a bad year."
    )
    lines.append(
        "   - Re-run this script annually with updated balances and salaries; the optimal "
        "mix shifts as you approach retirement."
    )
    return "\n".join(lines)


def render_takeaways(results: dict, cfg: Config) -> str:
    if not results:
        return "(No strategies have been simulated yet.)"

    ranked = sorted(
        results.items(),
        key=lambda kv: kv[1][2]["terminal_after_tax"],
        reverse=True,
    )
    winner_name, (_w_cfg, _winner_df, winner_sum) = ranked[0]
    baseline_name = "S0_baseline" if "S0_baseline" in results else ranked[-1][0]
    baseline_sum = results[baseline_name][2]

    def fmt_money(v: float) -> str:
        return f"${v:,.0f}"

    def delta_phrase(winner_v: float, baseline_v: float, *, lower_is_better: bool) -> str:
        delta = winner_v - baseline_v
        if abs(delta) < 1.0:
            return "matches baseline"
        improved = (delta < 0) if lower_is_better else (delta > 0)
        verb = "saves" if improved else "costs an extra"
        return f"{verb} {fmt_money(abs(delta))} vs `{baseline_name}`"

    lines: list[str] = []

    if winner_name == baseline_name:
        lines.append(
            f"- Winning strategy: {baseline_name} is already optimal - "
            f"terminal after-tax NW {fmt_money(winner_sum['terminal_after_tax'])}."
        )
    else:
        lines.append(
            f"- Winning strategy: {winner_name} - terminal after-tax NW "
            f"{fmt_money(winner_sum['terminal_after_tax'])} "
            f"(+{fmt_money(winner_sum['terminal_after_tax'] - baseline_sum['terminal_after_tax'])} "
            f"vs {baseline_name})."
        )
        lines.append(
            f"- Lifetime federal-tax NPV: {winner_name} pays "
            f"{fmt_money(winner_sum['lifetime_tax_npv'])} "
            f"({delta_phrase(winner_sum['lifetime_tax_npv'], baseline_sum['lifetime_tax_npv'], lower_is_better=True)})."
        )
        lines.append(
            f"- Lifetime IRMAA NPV: {winner_name} pays "
            f"{fmt_money(winner_sum['lifetime_irmaa_npv'])} "
            f"({delta_phrase(winner_sum['lifetime_irmaa_npv'], baseline_sum['lifetime_irmaa_npv'], lower_is_better=True)})."
        )

    ruined = [n for n, (_c, _df, s) in results.items() if s["min_balance"] <= 0]
    tight = [
        n
        for n, (_c, df, s) in results.items()
        if s["min_balance"] > 0 and s["min_balance"] < df["spending_need"].max()
    ]
    if ruined:
        lines.append(
            f"- Solvency: {', '.join(ruined)} run out of liquid assets - "
            "review spending or contribution mix."
        )
    elif tight:
        lines.append(
            f"- Solvency: {', '.join(tight)} dip below one year of spending at some point - thin margin."
        )
    else:
        lines.append(
            "- Solvency: every strategy keeps at least one year of spending in liquid assets "
            "across all years (no-ruin constraint not binding)."
        )

    lines.append(
        f"- IRMAA exposure ({winner_name}): {winner_sum['years_irmaa']} year(s) paying surcharge, "
        f"peak tier {winner_sum['peak_irmaa_tier']} (0 = below threshold)."
    )
    lines.append(
        f"- Peak federal marginal rate ({winner_name}): {winner_sum['peak_marginal']:.0%}."
    )

    conv_totals = []
    for name, (_c, df, _s) in results.items():
        if "roth_conversion" in df.columns:
            total = float(df["roth_conversion"].sum())
            if total > 0:
                conv_totals.append((name, total))
    if conv_totals:
        bullet = "; ".join(f"{n} {fmt_money(t)}" for n, t in conv_totals)
        lines.append(f"- Total Roth conversions over horizon: {bullet}.")
    else:
        lines.append(
            "- Roth conversions: no strategy converted any pretax balances - "
            "set roth_conversion_target_bracket (e.g. 0.22) on a strategy to enable."
        )

    horizon_years = cfg.horizon_age - cfg.spouse_a_age_start + 1
    header = (
        f"### Run summary ({horizon_years}-year horizon, ages "
        f"{cfg.spouse_a_age_start}-{cfg.horizon_age})\n"
    )
    return header + "\n".join(lines)


# =============================================================================
# 13. Plot helpers
# =============================================================================


def _maybe_show_or_save(fig, save_dir: str | None, name: str, show: bool):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_federal_tax_per_year(results: dict, save_dir: str | None, show: bool):
    fig, ax = plt.subplots()
    for name, (_, df, _) in results.items():
        ax.plot(df["year"], df["federal_tax"], label=name)
    ax.set_title("Federal tax paid per year by strategy")
    ax.set_xlabel("Year")
    ax.set_ylabel("Federal tax ($)")
    ax.legend()
    _maybe_show_or_save(fig, save_dir, "federal_tax_per_year", show)


def plot_balances_over_time(results: dict, save_dir: str | None, show: bool):
    n = len(results)
    cols = 2
    rows_n = (n + 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(14, 4.5 * rows_n), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, (name, (_, df, _)) in zip(axes_flat, results.items()):
        ax.stackplot(
            df["year"],
            df["pretax_balance"],
            df["roth_balance"],
            df["taxable_balance"],
            df["hsa_balance"],
            labels=["Pretax", "Roth", "Taxable", "HSA"],
            alpha=0.85,
        )
        ax.set_title(name)
        ax.set_ylabel("Balance ($)")
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes_flat[len(results):]:
        ax.set_visible(False)
    fig.suptitle("Account balances over time", fontsize=14)
    fig.tight_layout()
    _maybe_show_or_save(fig, save_dir, "balances_over_time", show)


def plot_strategy_bars(results: dict, cfg: Config, save_dir: str | None, show: bool):
    names = list(results.keys())
    terminal = [results[n][2]["terminal_after_tax"] for n in names]
    lifetime = [results[n][2]["lifetime_tax_npv"] for n in names]
    irmaa_npv = [results[n][2]["lifetime_irmaa_npv"] for n in names]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))
    ax1.bar(names, terminal)
    for i, v in enumerate(terminal):
        ax1.text(i, v, f"${v / 1e6:.2f}M", ha="center", va="bottom")
    ax1.set_ylabel("Terminal after-tax NW ($)")
    ax1.set_title(f"Terminal after-tax NW @ age {cfg.horizon_age}")
    ax1.tick_params(axis="x", rotation=20)
    ax2.bar(names, lifetime, color="tab:orange")
    for i, v in enumerate(lifetime):
        ax2.text(i, v, f"${v / 1e6:.2f}M", ha="center", va="bottom")
    ax2.set_ylabel("Lifetime federal tax NPV ($)")
    ax2.set_title("Lifetime federal tax (NPV @ 2.5%)")
    ax2.tick_params(axis="x", rotation=20)
    ax3.bar(names, irmaa_npv, color="tab:red")
    for i, v in enumerate(irmaa_npv):
        ax3.text(i, v, f"${v / 1e3:.0f}k", ha="center", va="bottom")
    ax3.set_ylabel("Lifetime IRMAA NPV ($)")
    ax3.set_title("Lifetime IRMAA surcharges (NPV @ 2.5%)")
    ax3.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _maybe_show_or_save(fig, save_dir, "strategy_bars", show)


def plot_roth_pct_heatmap(base_cfg: Config, save_dir: str | None, show: bool):
    vs = np.linspace(0, 1, 6)
    bs = np.linspace(0, 1, 6)
    Z = np.zeros((len(bs), len(vs)))
    for i, b in enumerate(bs):
        for j, v in enumerate(vs):
            cfg = replace(
                base_cfg,
                spouse_a_roth_401k_pct=float(v),
                spouse_b_roth_401k_pct=float(b),
                roth_conversion_target_bracket=0.0,
            )
            Z[i, j] = terminal_after_tax_nw(simulate(cfg))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(Z / 1e6, origin="lower", extent=(0, 1, 0, 1), aspect="auto")
    ax.set_xlabel("Spouse A Roth-401k pct")
    ax.set_ylabel("Spouse B Roth-401k pct")
    ax.set_title("Terminal after-tax NW ($M) - no Roth conversions")
    fig.colorbar(im, ax=ax, label="Terminal NW ($M)")
    best_i, best_j = np.unravel_index(np.argmax(Z), Z.shape)
    ax.scatter([vs[best_j]], [bs[best_i]], color="red", marker="*", s=200, label="Optimum")
    ax.legend()
    _maybe_show_or_save(fig, save_dir, "roth_pct_heatmap", show)
    print(
        f"Best Roth-pct mix: spouse_a={vs[best_j]:.2f}, spouse_b={bs[best_i]:.2f}, "
        f"terminal=${Z.max():,.0f}"
    )


def plot_growth_inflation_sensitivity(s3_cfg: Config, save_dir: str | None, show: bool):
    grates = [0.03, 0.05, 0.07]
    infs = [0.02, 0.03, 0.04]
    M = np.zeros((len(infs), len(grates)))
    for i, inf in enumerate(infs):
        for j, g in enumerate(grates):
            cfg = replace(s3_cfg, nominal_growth_rate=g, inflation=inf)
            M[i, j] = terminal_after_tax_nw(simulate(cfg))
    fig, ax = plt.subplots()
    im = ax.imshow(
        M / 1e6,
        origin="lower",
        aspect="auto",
        extent=(grates[0] * 100, grates[-1] * 100, infs[0] * 100, infs[-1] * 100),
    )
    for i in range(len(infs)):
        for j in range(len(grates)):
            ax.text(
                grates[j] * 100,
                infs[i] * 100,
                f"${M[i, j] / 1e6:.1f}M",
                ha="center",
                va="center",
                color="white",
            )
    ax.set_xlabel("Growth rate (%)")
    ax.set_ylabel("Inflation (%)")
    ax.set_title("S3 terminal after-tax NW sensitivity")
    fig.colorbar(im, ax=ax, label="Terminal NW ($M)")
    _maybe_show_or_save(fig, save_dir, "growth_inflation_sensitivity", show)


def plot_tornado(sens_df: pd.DataFrame, base_terminal: float, save_dir: str | None, show: bool):
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(sens_df) + 1.5)))
    y = np.arange(len(sens_df))[::-1]
    ax.barh(y, sens_df["delta_low"], color="#d62728", alpha=0.7, label="low value")
    ax.barh(y, sens_df["delta_high"], color="#2ca02c", alpha=0.7, label="high value")
    ax.set_yticks(y)
    ax.set_yticklabels(sens_df["param"])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("delta terminal after-tax NW vs base ($)")
    ax.set_title(f"Tornado: sensitivity around base terminal NW ${base_terminal:,.0f}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _maybe_show_or_save(fig, save_dir, "tornado_sensitivity", show)


# =============================================================================
# Main entrypoint
# =============================================================================


def print_inputs() -> None:
    print("--- Starting balances ---")
    for k, v in starting.items():
        print(f"  {k:35s} ${v:>14,.0f}")
    print("\n--- Current income ---")
    for k, v in income.items():
        print(f"  {k:35s} ${v:>14,.0f}")
    print(f"\nAnnual expenses (excl. taxes): ${annual_expenses:,.0f}")
    print(f"Pension @ NRD (annual):        ${annual_pension_at_nrd:,.0f}")
    print(
        f"SSN @ FRA (annual, combined):  "
        f"${(ssn_monthly_spouse_a + ssn_monthly_spouse_b) * 12:,.0f}"
    )


def run_sanity_checks() -> None:
    baseline_year_one_kwargs = dict(
        wages=(income["spouse_a_gross"] - income["spouse_a_gross"] * current_contrib["spouse_a_pct"])
        + (income["spouse_b_gross"] - income["spouse_b_gross"] * current_contrib["spouse_b_pct"])
        + income["spouse_a_bonus"],
        interest=income["interest"],
        qualified_div=income["dividends"],
        ltcg=income["capital_gains"],
    )
    baseline_tax = federal_tax(**baseline_year_one_kwargs)
    target_tax = current_contrib["baseline_tax"]
    print(f"Engine year-1 tax:  ${baseline_tax['tax']:,.0f}")
    if target_tax > 0:
        print(f"Target year-1 tax:  ${target_tax:,.0f}")
        rel_err = abs(baseline_tax["tax"] - target_tax) / target_tax
        print(f"Relative error:     {rel_err:.2%}")
    else:
        print("(set current_contrib['baseline_tax'] to your prior-year federal tax to enable a sanity check)")

    nw_excl = (
        starting["spouse_a_pretax_401k"]
        + starting["spouse_b_pretax_401k"]
        + starting["spouse_a_roth_ira"]
        + starting["spouse_b_pretax_ira"]
        + starting["pension_balance"]
        + starting["hsa"]
        + starting["taxable_brokerage"]
    )
    print(f"Computed starting NW (model buckets):              ${nw_excl:,.0f}")
    print(f"Reported total NW (informational):                 ${starting['total_nw_excl_real_estate']:,.0f}")
    print(
        f"Pension @ NRD model: ${_test_balance:,.0f} balance, "
        f"${annual_pension_at_nrd:,.0f}/yr annuity"
    )


def run_strategies(cfg: Config) -> dict:
    s0 = replace(cfg)
    s1 = replace(cfg, spouse_a_roth_401k_pct=1.0, spouse_b_roth_401k_pct=1.0)
    s2 = replace(cfg, roth_conversion_target_bracket=0.22)

    results: dict = {}
    for name, scenario_cfg in [
        ("S0_baseline", s0),
        ("S1_all_roth_401k", s1),
        ("S2_bracket_fill_22", s2),
    ]:
        df = simulate(scenario_cfg)
        results[name] = (scenario_cfg, df, summarize(df))
    return results


def print_summary(results: dict) -> None:
    summary_df = pd.DataFrame({k: v[2] for k, v in results.items()}).T
    print(summary_df.to_string(float_format=lambda x: f"{x:,.2f}"))


def print_year_by_year(df: pd.DataFrame, label: str) -> None:
    show_cols = [
        "year",
        "spouse_a_age",
        "spouse_b_age",
        "wages",
        "pension",
        "ssn",
        "rmd_a",
        "rmd_b",
        "roth_conversion",
        "pretax_withdrawal",
        "taxable_withdrawal",
        "agi",
        "federal_tax",
        "irmaa",
        "irmaa_tier",
        "marginal",
        "pretax_a_balance",
        "pretax_b_balance",
        "roth_balance",
        "taxable_balance",
    ]
    print(f"\n=== Year-by-year detail ({label}) ===")
    print(df[show_cols].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--plots", action="store_true", help="display matplotlib figures interactively")
    p.add_argument(
        "--save-plots",
        metavar="DIR",
        help="save plots as PNG files into DIR (also enables figure rendering)",
    )
    p.add_argument(
        "--no-detail",
        action="store_true",
        help="skip the year-by-year table for the winning strategy",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    show = bool(args.plots)
    save_dir = args.save_plots

    cfg = Config()

    print("=" * 79)
    print("Retirement Tax Optimizer (standalone script)")
    print("=" * 79)
    print_inputs()

    print("\n--- Config ---")
    print(cfg)

    print("\n--- Sanity checks ---")
    run_sanity_checks()

    print("\n--- Running strategies S0/S1/S2 ---")
    results = run_strategies(cfg)
    print_summary(results)

    print("\n--- Optimizing S3 ---")
    s3_cfg, _x_opt = optimize_s3(cfg)
    df3 = simulate(s3_cfg)
    results["S3_optimized"] = (s3_cfg, df3, summarize(df3))

    print("\n=== Summary across strategies ===")
    print_summary(results)
    print(
        f"\nS3 chosen: spouse_a_roth_401k_pct={s3_cfg.spouse_a_roth_401k_pct:.2f}, "
        f"spouse_b_roth_401k_pct={s3_cfg.spouse_b_roth_401k_pct:.2f}, "
        f"roth_conv_bracket={s3_cfg.roth_conversion_target_bracket}"
    )

    print("\n--- Tornado sensitivity around the winning strategy ---")
    winner_name, winner_cfg = _winning_cfg(results, cfg)
    print(f"Running tornado sensitivity around `{winner_name}` configuration ...")
    sens_df, base_terminal = tornado_sensitivity(winner_cfg)
    print(f"Base terminal NW: ${base_terminal:,.0f}")
    print(
        sens_df.to_string(
            index=False,
            formatters={
                "low_value": lambda v: f"{v:,.4g}",
                "high_value": lambda v: f"{v:,.4g}",
                "delta_low": lambda v: f"${v:+,.0f}",
                "delta_high": lambda v: f"${v:+,.0f}",
                "swing": lambda v: f"${v:,.0f}",
            },
        )
    )

    print("\n" + render_actions(results, sens_df, winner_cfg, base_terminal))
    print("\n" + render_takeaways(results, cfg))

    if not args.no_detail:
        print_year_by_year(df3, "S3_optimized")

    if show or save_dir:
        print("\n--- Generating plots ---")
        plot_federal_tax_per_year(results, save_dir, show)
        plot_balances_over_time(results, save_dir, show)
        plot_strategy_bars(results, cfg, save_dir, show)
        plot_roth_pct_heatmap(cfg, save_dir, show)
        plot_growth_inflation_sensitivity(s3_cfg, save_dir, show)
        plot_tornado(sens_df, base_terminal, save_dir, show)


if __name__ == "__main__":
    main()
