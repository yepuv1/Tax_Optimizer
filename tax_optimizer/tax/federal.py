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
    """Marginal rate the *next dollar* of `amount` would face.

    Use ``>=`` (not ``>``) on the lower bound so that, at exact bracket
    boundaries (e.g. taxable income lands at $23,850), the function
    reports the higher rate that the next dollar would actually pay.
    With ``>`` the function returned the prior bracket's rate at exact
    boundaries — usually invisible because real taxable income rarely
    lands on the cent, but it could bias `bracket_fill` conversion
    sizing right at the boundary.
    """
    rate = brackets[0][2]
    for lo, _hi, r in brackets:
        if amount >= lo:
            rate = r
    return rate


def _compute_amt(
    *,
    regime: TaxRegime,
    filing_status: str,
    taxable_ordinary: float,
    taxable_pref: float,
    deduction: float,
    ord_tax_regular: float,
    ltcg_tax_regular: float,
) -> dict:
    """Compute AMT (Alternative Minimum Tax, IRC §55) liability.

    Returns ``{amt, amti, tmt, amt_exemption_eff}``. ``amt`` is the
    *additional* tax (>= 0) due on top of regular tax; ``tmt`` is the
    tentative minimum tax for comparison vs regular tax (excluding NIIT).

    AMT mechanics modeled:

      * AMTI = taxable_total + std-deduction add-back (pre-TCJA only).
        SALT add-back, personal-exemption add-back, and ISO bargain-
        element preference are NOT modeled (out of scope: the
        retirement model doesn't track these).
      * Exemption phases out at 25% above ``amt_phaseout_start``.
      * Ordinary AMT base taxed at 26% up to ``amt_28pct_threshold``,
        28% above. The preferential (LTCG/QDIV) portion retains the
        regular LTCG rates — re-used directly from ``ltcg_tax_regular``.
      * AMT due = max(0, TMT − regular_tax_excl_NIIT).

    For regimes that don't apply AMT (default ``amt_exemption=inf``),
    short-circuits and returns zero.
    """
    exemption = regime.amt_exemption(filing_status)
    if not math.isfinite(exemption) or exemption <= 0:
        return {"amt": 0.0, "amti": 0.0, "tmt": 0.0, "amt_exemption_eff": 0.0}

    phaseout_start = regime.amt_phaseout_start(filing_status)
    phaseout_rate = regime.amt_phaseout_rate
    threshold_28 = regime.amt_28pct_threshold(filing_status)
    rate_low = regime.amt_rate_low
    rate_high = regime.amt_rate_high

    amti = taxable_ordinary + taxable_pref
    if regime.amt_std_deduction_addback:
        amti += deduction

    excess = max(0.0, amti - phaseout_start)
    exemption_eff = max(0.0, exemption - excess * phaseout_rate)

    # Ordinary AMT base = AMTI − preferential − exemption.
    amt_ord_base = max(0.0, amti - taxable_pref - exemption_eff)
    if amt_ord_base <= threshold_28:
        amt_ord_tax = amt_ord_base * rate_low
    else:
        amt_ord_tax = (
            threshold_28 * rate_low
            + (amt_ord_base - threshold_28) * rate_high
        )

    tmt = amt_ord_tax + ltcg_tax_regular
    regular_tax_excl_niit = ord_tax_regular + ltcg_tax_regular
    amt_due = max(0.0, tmt - regular_tax_excl_niit)

    return {
        "amt": amt_due,
        "amti": amti,
        "tmt": tmt,
        "amt_exemption_eff": exemption_eff,
    }


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
    annuity_taxable: float = 0.0,
    pretax_withdrawal: float = 0.0,
    roth_conversion: float = 0.0,
    social_security: float = 0.0,
    tax_exempt_interest: float = 0.0,
    deduction: float | None = None,
    early_distribution_taxable: float = 0.0,
) -> dict:
    """Compute federal tax liability for one year.

    Pass `regime` (a `TaxRegime`) plus `filing_status` ('mfj' or 'single').
    All income items are in nominal dollars. Returns a dict containing
    `tax`, `ordinary_tax`, `ltcg_tax`, `niit`, `agi`, `taxable_income`,
    `ss_taxable`, and `marginal`.

    ``annuity_taxable`` carries the taxable portion of annuity
    contract distributions (qualified: full payment; non-qualified:
    payment minus the IRC §72(b) excluded portion). Routed into the
    same ordinary-income stack as wages / pension / pretax draws.
    Kept as its own kwarg (rather than rolled into ``pension``) so
    the simulator output can split the two streams in reporting.

    ``early_distribution_taxable`` is the portion of a current-year
    distribution that triggers the IRC §72(t) / §72(q) 10%
    additional tax (e.g. a pension or annuity lump-sum at
    ``a_age < 59.5``). The 10% surtax is computed on this argument
    and added into the returned ``total_tax`` via the new
    ``early_distribution_penalty`` line. The principal itself is
    NOT added by this kwarg — the simulator passes the same dollars
    through ``pension`` / ``pretax_withdrawal`` / ``annuity_taxable``
    on top of this so the income side and the penalty side are both
    accounted for.

    ``tax_exempt_interest`` is municipal-bond / private-activity-bond
    interest that's federally tax-exempt. IRC §86(b)(2)(B) requires
    adding this back when computing the SS provisional-income line
    (i.e. tax-exempt-interest *does* count toward the threshold that
    decides whether SS gets taxed at 0% / 50% / 85%). The amount is
    NOT added to AGI / taxable income here — that would be a federal
    tax on municipal bonds, which is the opposite of the intended
    treatment. Routed *only* into the §86 provisional add-back.
    """
    ord_brackets = regime.ord_brackets(filing_status)
    ltcg_brackets = regime.ltcg_brackets(filing_status)
    if deduction is None:
        deduction = regime.std_deduction(filing_status)
    niit_threshold = regime.niit_threshold(filing_status)
    base1, base2 = regime.ss_provisional(filing_status)

    other = (
        wages + interest + ordinary_div + qualified_div + ltcg
        + pension + annuity_taxable + pretax_withdrawal + roth_conversion
    )
    # IRC §86(b)(2)(B): tax-exempt interest is added back into the
    # provisional-income calculation (but NOT into AGI or taxable
    # income). The downstream `social_security_taxable` call uses
    # `provisional` to decide the 0% / 50% / 85% tier, so a household
    # holding muni bonds correctly sees a higher SS-taxable share
    # without losing the federal exemption on the muni interest itself.
    provisional = other + tax_exempt_interest + 0.5 * social_security
    ss_taxable = social_security_taxable(provisional, social_security, base1=base1, base2=base2)

    ordinary_income = (
        wages + interest + ordinary_div + pension + annuity_taxable
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

    # AMT — parallel tax system. Computed against regular ord+ltcg tax
    # (excluding NIIT, which is its own surtax not in the AMT compare).
    # Returns 0 for regimes where AMT is disabled (amt_exemption=inf).
    amt_result = _compute_amt(
        regime=regime,
        filing_status=filing_status,
        taxable_ordinary=taxable_ordinary,
        taxable_pref=taxable_pref,
        deduction=deduction,
        ord_tax_regular=ord_tax,
        ltcg_tax_regular=ltcg_tax,
    )
    amt = amt_result["amt"]

    # IRC §72(t) / §72(q) 10% additional tax on early distributions
    # from qualified plans / annuity contracts. The simulator passes
    # the principal through the usual income kwargs (``pension``,
    # ``annuity_taxable``, ``pretax_withdrawal``) so it shows up in
    # AGI; the 10% surtax is the *extra* component on top, isolated
    # here for reporting clarity.
    early_distribution_penalty = 0.10 * max(0.0, early_distribution_taxable)

    total = ord_tax + ltcg_tax + niit + amt + early_distribution_penalty
    marginal = _marginal_rate(taxable_ordinary, ord_brackets)

    return {
        "tax": total,
        "ordinary_tax": ord_tax,
        "ltcg_tax": ltcg_tax,
        "niit": niit,
        "amt": amt,
        "amti": amt_result["amti"],
        "tmt": amt_result["tmt"],
        "early_distribution_penalty": early_distribution_penalty,
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
