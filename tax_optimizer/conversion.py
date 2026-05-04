"""Roth-conversion sizing during the retirement-to-RMD gap years."""

from __future__ import annotations

from .config import Config
from .state import State
from .tax.federal import amount_to_fill_bracket, federal_tax
from .tax.regimes import TaxRegime


def planned_roth_conversion(
    cfg: Config,
    state: State,
    base_kwargs: dict,
    *,
    regime: TaxRegime,
    filing_status: str,
) -> tuple[float, float]:
    """Return (conv_a, conv_b): per-spouse pretax→Roth conversion amounts.

    A spouse is eligible to convert in any year where they're individually
    retired AND under `cfg.rmd_start_age`. When both are eligible the
    target is split pro-rata across their available pretax balances.
    """
    a_in_gap = (
        state.spouse_a_age >= cfg.spouse_a_retire_age
        and state.spouse_a_age < cfg.rmd_start_age
    )
    b_in_gap = (
        state.spouse_b_age >= cfg.spouse_b_retire_age
        and state.spouse_b_age < cfg.rmd_start_age
    )
    a_avail = state.spouse_a_pretax if a_in_gap else 0.0
    b_avail = state.spouse_b_pretax if b_in_gap else 0.0
    cap = a_avail + b_avail
    if cap <= 0:
        return 0.0, 0.0

    if cfg.roth_conversion_amount > 0:
        total = min(cfg.roth_conversion_amount, cap)
    elif cfg.roth_conversion_target_bracket > 0:
        ti_now = federal_tax(regime=regime, filing_status=filing_status, **base_kwargs)[
            "taxable_income"
        ]
        headroom = amount_to_fill_bracket(
            ti_now, cfg.roth_conversion_target_bracket, regime.ord_brackets(filing_status)
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
