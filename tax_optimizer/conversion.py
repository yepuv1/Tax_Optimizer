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

Liquidity guard (v6.5)
----------------------
When `cfg.cap_conversion_by_liquidity=True` (default), the caller passes
a `tax_paying_capacity` estimate — household cash available to pay the
conversion's marginal federal + state tax, derived from earned-income
surplus + pension + SS + RMD net of FICA/SDI/spending/healthcare/
contributions, plus a configurable slice of the taxable brokerage.
The sizer then bisects the conversion down so its marginal tax fits
inside capacity. Without this guard, an aggressive
`roth_conversion_target_bracket` (or any fixed amount > liquid cash on
hand) silently triggered the deficit cascade, which withdrew tax cash
from the *just-funded* Roth — defeating the strategy and (under IRS
rules) potentially incurring the 10% conversion-principal penalty when
the 5-year clock hadn't matured.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

from .config import Config
from .inputs import Inputs
from .state import State
from .tax.federal import amount_to_fill_bracket, federal_tax
from .tax.regimes import TaxRegime


StateTaxFn = Callable[[dict, float], float]
"""Signature: `state_tax_fn(kwargs, ss_taxable_federal) -> state_tax_dollars`.

Same shape the cascade solvers use (see `withdrawals.StateTaxFn`).
Captures the year's state regime / filing status / ages so the
liquidity bisection can compute the marginal *state* tax delta
incrementally on top of the federal delta.
"""


class ConversionPlan(NamedTuple):
    """Result of `planned_roth_conversion`.

    Attributes:
        conv_a:                Spouse-A pretax → Roth conversion, $.
        conv_b:                Spouse-B pretax → Roth conversion, $.
        capped_by_liquidity:   True iff the liquidity guard bisected
                               the conversion below the bracket /
                               fixed-amount target. Surfaced in the
                               year-row so reports can flag it.
        bracket_target_total:  The conversion total the bracket-fill
                               (or fixed-amount) mode *would* have
                               picked absent the liquidity guard.
                               Useful for diagnostics.
    """

    conv_a: float
    conv_b: float
    capped_by_liquidity: bool = False
    bracket_target_total: float = 0.0


def planned_roth_conversion(
    cfg: Config,
    inputs: Inputs,
    state: State,
    base_kwargs: dict,
    *,
    regime: TaxRegime,
    filing_status: str,
    rmd_total: float = 0.0,
    rmd_a: float = 0.0,
    rmd_b: float = 0.0,
    tax_paying_capacity: Optional[float] = None,
    state_tax_fn: Optional[StateTaxFn] = None,
) -> ConversionPlan:
    """Plan this year's pretax → Roth conversion.

    `rmd_total` is the year's total RMD across both spouses (already
    computed by the caller, not yet folded into `base_kwargs`). The
    conversion is sized so that *post-RMD* taxable income just reaches
    the target bracket cap — i.e. RMD eats bracket headroom before any
    optional conversion does.

    `rmd_a` / `rmd_b` are the per-spouse RMDs. Each spouse's
    conversion is capped at `pretax_balance - rmd`, so a fixed-dollar
    conversion can never consume balance the RMD requires (TC-7 fix).
    Pre-Tier-C, a fixed `roth_conversion_amount` could pull the
    pretax bucket low enough that `withdraw_for_need` silently
    truncated the RMD — internal inconsistency vs IRS sequencing.

    Pro-rata split: if both spouses have eligible pretax balances, the
    total is split in proportion to each spouse's balance.

    If `tax_paying_capacity` is not None AND
    `cfg.cap_conversion_by_liquidity` is True, the conversion is
    bisected down so that the marginal federal + state tax on the
    conversion fits inside capacity. `state_tax_fn` (optional) lets
    the bisection account for state tax; without it the cap is
    federal-only (slightly under-tightens for high-tax states).
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

    # Reserve each spouse's RMD against their own conversion ceiling.
    a_avail = max(0.0, state.spouse_a_pretax - rmd_a) if a_eligible else 0.0
    b_avail = max(0.0, state.spouse_b_pretax - rmd_b) if b_eligible else 0.0
    cap = a_avail + b_avail
    if cap <= 0:
        return ConversionPlan(0.0, 0.0, capped_by_liquidity=False, bracket_target_total=0.0)

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
        return ConversionPlan(0.0, 0.0, capped_by_liquidity=False, bracket_target_total=0.0)

    if total <= 0:
        return ConversionPlan(0.0, 0.0, capped_by_liquidity=False, bracket_target_total=0.0)

    bracket_target_total = total
    capped = False

    # ---- Liquidity guard (v6.5) ----
    # Bisect `total` down so the marginal federal + state tax delta on
    # adding the conversion to (base_kwargs + RMD) stays at or under
    # `tax_paying_capacity`. Anything bigger would have to be funded by
    # the deficit cascade — which (when `cfg.protect_roth_in_conversion_years`
    # is on) refuses to touch the Roth bucket, so it surfaces as
    # `unfunded` instead of silently raiding the just-converted dollars.
    #
    # Negative capacity means the household has *no* liquidity for
    # conversion-marginal tax (committed obligations exceed cash on
    # hand). Pre-fix the guard short-circuited on `< 0` and let the
    # full bracket-fill / fixed-amount conversion through, which is
    # the opposite of safe. Clamp to zero so the bisection forces
    # `total = 0` for the year.
    if cfg.cap_conversion_by_liquidity and tax_paying_capacity is not None:
        tax_paying_capacity = max(0.0, tax_paying_capacity)
        kwargs_with_rmd_no_conv = dict(base_kwargs)
        if rmd_total > 0:
            kwargs_with_rmd_no_conv["pretax_withdrawal"] = (
                kwargs_with_rmd_no_conv.get("pretax_withdrawal", 0.0) + rmd_total
            )
        base_fed = federal_tax(
            regime=regime, filing_status=filing_status, **kwargs_with_rmd_no_conv
        )
        base_fed_tax = base_fed["tax"]
        base_state_tax = (
            state_tax_fn(kwargs_with_rmd_no_conv, base_fed["ss_taxable"])
            if state_tax_fn is not None else 0.0
        )

        def _marginal_tax_on(conv_amount: float) -> float:
            kw = dict(kwargs_with_rmd_no_conv)
            kw["roth_conversion"] = kw.get("roth_conversion", 0.0) + conv_amount
            fed = federal_tax(regime=regime, filing_status=filing_status, **kw)
            fed_delta = max(0.0, fed["tax"] - base_fed_tax)
            state_delta = 0.0
            if state_tax_fn is not None:
                state_delta = max(
                    0.0, state_tax_fn(kw, fed["ss_taxable"]) - base_state_tax
                )
            return fed_delta + state_delta

        marginal_at_total = _marginal_tax_on(total)
        if marginal_at_total > tax_paying_capacity:
            # Bisect on the conversion amount. Monotone in `total`.
            lo, hi = 0.0, total
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if _marginal_tax_on(mid) <= tax_paying_capacity:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 100.0:
                    break
            total = lo
            capped = True

    if total <= 0:
        return ConversionPlan(
            0.0, 0.0,
            capped_by_liquidity=capped,
            bracket_target_total=bracket_target_total,
        )
    a_share = a_avail / cap
    conv_a = total * a_share
    conv_b = total - conv_a
    return ConversionPlan(
        conv_a, conv_b,
        capped_by_liquidity=capped,
        bracket_target_total=bracket_target_total,
    )
