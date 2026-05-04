"""Federal income tax engine, regime + filing-status aware."""

from __future__ import annotations

import math
from typing import Sequence

from .regimes import Bracket, TaxRegime


def _bracket_tax(amount: float, brackets: Sequence[Bracket]) -> float:
    tax = 0.0
    for lo, hi, rate in brackets:
        if amount <= lo:
            break
        tax += (min(amount, hi) - lo) * rate
    return tax


def _marginal_rate(amount: float, brackets: Sequence[Bracket]) -> float:
    rate = brackets[0][2]
    for lo, _hi, r in brackets:
        if amount > lo:
            rate = r
    return rate


def social_security_taxable(
    provisional: float,
    ss_benefits: float,
    *,
    base1: float,
    base2: float,
) -> float:
    """Taxable portion of SS benefits per IRC §86. `base1` / `base2` are
    the filing-status-specific provisional-income thresholds."""
    if ss_benefits <= 0:
        return 0.0
    if provisional <= base1:
        return 0.0
    if provisional <= base2:
        return min(0.5 * (provisional - base1), 0.5 * ss_benefits)
    tier1 = min(0.5 * (base2 - base1), 0.5 * ss_benefits)
    tier2 = 0.85 * (provisional - base2)
    return min(tier1 + tier2, 0.85 * ss_benefits)


def federal_tax(
    *,
    regime: TaxRegime,
    filing_status: str = "mfj",
    wages: float = 0.0,
    interest: float = 0.0,
    ordinary_div: float = 0.0,
    qualified_div: float = 0.0,
    ltcg: float = 0.0,
    pension: float = 0.0,
    pretax_withdrawal: float = 0.0,
    roth_conversion: float = 0.0,
    social_security: float = 0.0,
    deduction: float | None = None,
) -> dict:
    """Compute federal tax liability for one year.

    Pass `regime` (a `TaxRegime`) plus `filing_status` ('mfj' or 'single').
    All income items are in nominal dollars. Returns a dict containing
    `tax`, `ordinary_tax`, `ltcg_tax`, `niit`, `agi`, `taxable_income`,
    `ss_taxable`, and `marginal`.
    """
    ord_brackets = regime.ord_brackets(filing_status)
    ltcg_brackets = regime.ltcg_brackets(filing_status)
    if deduction is None:
        deduction = regime.std_deduction(filing_status)
    niit_threshold = regime.niit_threshold(filing_status)
    base1, base2 = regime.ss_provisional(filing_status)

    other = (
        wages + interest + ordinary_div + qualified_div + ltcg
        + pension + pretax_withdrawal + roth_conversion
    )
    provisional = other + 0.5 * social_security
    ss_taxable = social_security_taxable(provisional, social_security, base1=base1, base2=base2)

    ordinary_income = (
        wages + interest + ordinary_div + pension
        + pretax_withdrawal + roth_conversion + ss_taxable
    )
    preferential = qualified_div + ltcg
    agi = ordinary_income + preferential

    taxable_total = max(0.0, agi - deduction)
    taxable_ordinary = max(0.0, taxable_total - preferential)
    taxable_pref = max(0.0, taxable_total - taxable_ordinary)

    ord_tax = _bracket_tax(taxable_ordinary, ord_brackets)

    ltcg_tax = 0.0
    remaining = taxable_pref
    cursor = taxable_ordinary
    for lo, hi, rate in ltcg_brackets:
        if remaining <= 0:
            break
        room = max(0.0, hi - max(lo, cursor))
        slab = min(remaining, room)
        if slab > 0:
            ltcg_tax += slab * rate
            remaining -= slab
            cursor += slab

    investment_income = interest + ordinary_div + qualified_div + ltcg
    niit = regime.niit_rate * max(
        0.0, min(investment_income, agi - niit_threshold)
    )
    niit = max(0.0, niit)

    total = ord_tax + ltcg_tax + niit
    marginal = _marginal_rate(taxable_ordinary, ord_brackets)

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
    base_taxable: float,
    target_top: float,
    brackets: Sequence[Bracket],
) -> float:
    """Headroom (dollars of taxable income) between `base_taxable` and the
    top of the bracket whose rate is `target_top`."""
    for _lo, hi, rate in brackets:
        if math.isclose(rate, target_top, abs_tol=1e-6):
            return max(0.0, hi - base_taxable)
    return 0.0
