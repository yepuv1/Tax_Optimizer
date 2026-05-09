"""Roth-conversion sizing.

Two modes are supported via `Config`:

  * Fixed-amount  (``roth_conversion_amount > 0``): convert exactly that
    many dollars per year, capped at the available pretax balance.
    Restricted to the classic "gap" window — between retirement and
    RMD eligibility — so a fixed-dollar plan doesn't accidentally
    detonate working-year tax bills.

  * Bracket-fill  (``roth_conversion_target_bracket > 0``): convert
    just enough to fill the target ordinary bracket, accounting for
    income already in the year (wages, pensions, SS, dividends, AND
    any RMD that's mandatory this year). Self-regulates to zero in
    high-income years, so the gap-only gate isn't needed — we run it
    in any year with a positive pretax balance.

The RMD-first ordering is critical: at age ≥ ``rmd_start_age`` the IRS
requires the RMD before any conversion. The simulator computes RMD
ahead of this call and passes ``rmd_total`` so we size the conversion
on the *post-RMD* taxable-income line.
"""

from __future__ import annotations

from .config import Config
from .inputs import Inputs
from .state import State
from .tax.federal import amount_to_fill_bracket, federal_tax
from .tax.regimes import TaxRegime


def planned_roth_conversion(
    cfg: Config,
    inputs: Inputs,
    state: State,
    base_kwargs: dict,
    *,
    regime: TaxRegime,
    filing_status: str,
    rmd_total: float = 0.0,
) -> tuple[float, float]:
    """Return (conv_a, conv_b): per-spouse pretax→Roth conversion.

    `rmd_total` is the year's total RMD across both spouses (already
    computed by the caller, not yet folded into `base_kwargs`). The
    conversion is sized so that *post-RMD* taxable income just reaches
    the target bracket cap — i.e. RMD eats bracket headroom before any
    optional conversion does.

    Pro-rata split: if both spouses have eligible pretax balances, the
    total is split in proportion to each spouse's balance.
    """
    fixed_mode = cfg.roth_conversion_amount > 0

    if fixed_mode:
        # Backward-compat gap-only gate. A spouse with a fixed annual
        # conversion target probably doesn't want it firing during
        # working years (they'd just be paying ordinary rate on top of
        # already-high wages).
        a_eligible = (
            state.spouse_a_age >= inputs.spouse_a_retire_age
            and state.spouse_a_age < cfg.rmd_start_age
        )
        b_eligible = (
            state.spouse_b_age >= inputs.spouse_b_retire_age
            and state.spouse_b_age < cfg.rmd_start_age
        )
    else:
        # Bracket-fill mode self-regulates: in working years, AGI
        # already pushes past the target bracket → headroom is zero
        # → no conversion. So we open the window to any year with a
        # positive pretax balance, which also unlocks legal post-RMD
        # conversions (just less efficient since you've already taken
        # the RMD; the bracket-fill math accounts for that).
        a_eligible = state.spouse_a_pretax > 0
        b_eligible = state.spouse_b_pretax > 0

    a_avail = state.spouse_a_pretax if a_eligible else 0.0
    b_avail = state.spouse_b_pretax if b_eligible else 0.0
    cap = a_avail + b_avail
    if cap <= 0:
        return 0.0, 0.0

    if fixed_mode:
        total = min(cfg.roth_conversion_amount, cap)
    elif cfg.roth_conversion_target_bracket > 0:
        # Fold the mandatory RMD into the income line BEFORE measuring
        # bracket headroom. Without this, the simulator would convert
        # into bracket space that the RMD is about to consume, then pay
        # higher marginal rates on the RMD itself.
        kwargs_with_rmd = dict(base_kwargs)
        if rmd_total > 0:
            kwargs_with_rmd["pretax_withdrawal"] = (
                kwargs_with_rmd.get("pretax_withdrawal", 0.0) + rmd_total
            )
        ti_now = federal_tax(
            regime=regime, filing_status=filing_status, **kwargs_with_rmd
        )["taxable_income"]
        headroom = amount_to_fill_bracket(
            ti_now,
            cfg.roth_conversion_target_bracket,
            regime.ord_brackets(filing_status),
        )
        total = min(headroom, cap)
    else:
        return 0.0, 0.0

    if total <= 0:
        return 0.0, 0.0
    a_share = a_avail / cap
    conv_a = total * a_share
    conv_b = total - conv_a
    return conv_a, conv_b
