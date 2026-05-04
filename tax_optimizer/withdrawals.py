"""Withdrawal-strategy solvers.

These functions return *gross* withdrawals per bucket given a net-spending
target. They're pure (no state mutation); the simulator applies the
balance changes after computing federal tax + IRMAA on the final
combined income line.
"""

from __future__ import annotations

from .config import Config
from .state import State
from .tax.federal import amount_to_fill_bracket, federal_tax
from .tax.regimes import TaxRegime


def _solve_pretax_for_net(
    net_target: float,
    base_kwargs: dict,
    *,
    regime: TaxRegime,
    filing_status: str,
) -> float:
    if net_target <= 0:
        return 0.0
    base_tax = federal_tax(regime=regime, filing_status=filing_status, **base_kwargs)["tax"]
    lo, hi = 0.0, net_target * 2.0 + 50_000.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        kw = dict(base_kwargs)
        kw["pretax_withdrawal"] = kw.get("pretax_withdrawal", 0.0) + mid
        net = mid - (federal_tax(regime=regime, filing_status=filing_status, **kw)["tax"] - base_tax)
        if net < net_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0:
            break
    return 0.5 * (lo + hi)


def _solve_taxable_for_net(
    net_target: float,
    base_kwargs: dict,
    basis_frac: float,
    *,
    regime: TaxRegime,
    filing_status: str,
) -> float:
    if net_target <= 0:
        return 0.0
    base_tax = federal_tax(regime=regime, filing_status=filing_status, **base_kwargs)["tax"]
    lo, hi = 0.0, net_target * 2.0 + 50_000.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        gain = mid * (1 - basis_frac)
        kw = dict(base_kwargs)
        kw["ltcg"] = kw.get("ltcg", 0.0) + gain
        net = mid - (federal_tax(regime=regime, filing_status=filing_status, **kw)["tax"] - base_tax)
        if net < net_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0:
            break
    return 0.5 * (lo + hi)


def withdraw_for_need(
    net_need: float,
    state: State,
    cfg: Config,
    base_kwargs: dict,
    a_rmd: float,
    b_rmd: float,
    strategy: str,
    basis_frac: float,
    *,
    regime: TaxRegime,
    filing_status: str,
) -> dict:
    """Return gross withdrawals split per spouse for pretax.

    `basis_frac` is the live cost-basis / current-balance ratio for the
    taxable account at the start of the year (the caller computes it
    from `state.cumulative_basis / state.taxable`).
    """
    a_rmd = min(a_rmd, max(state.spouse_a_pretax, 0.0))
    b_rmd = min(b_rmd, max(state.spouse_b_pretax, 0.0))
    rmd_total = a_rmd + b_rmd

    ctx = dict(base_kwargs)
    ctx["pretax_withdrawal"] = ctx.get("pretax_withdrawal", 0.0) + rmd_total

    base_tax = federal_tax(regime=regime, filing_status=filing_status, **base_kwargs)["tax"]
    rmd_tax = federal_tax(regime=regime, filing_status=filing_status, **ctx)["tax"] - base_tax
    rmd_net = rmd_total - max(0.0, rmd_tax)
    remaining = max(0.0, net_need - rmd_net)

    pretax_a_w = a_rmd
    pretax_b_w = b_rmd
    roth_w = 0.0
    taxable_w = 0.0
    pretax_a_room = max(0.0, state.spouse_a_pretax - pretax_a_w)
    pretax_b_room = max(0.0, state.spouse_b_pretax - pretax_b_w)
    pretax_room = pretax_a_room + pretax_b_room

    def split_extra(extra: float) -> tuple[float, float]:
        if extra <= 0 or pretax_room <= 0:
            return 0.0, 0.0
        a_share = pretax_a_room / pretax_room
        ax = min(extra * a_share, pretax_a_room)
        bx = min(extra - ax, pretax_b_room)
        return ax, bx

    if strategy == "conventional":
        if remaining > 0 and state.taxable > 0:
            tw = min(
                _solve_taxable_for_net(
                    min(remaining, state.taxable), ctx, basis_frac,
                    regime=regime, filing_status=filing_status,
                ),
                state.taxable,
            )
            ctx_after = dict(ctx)
            ctx_after["ltcg"] = ctx_after.get("ltcg", 0.0) + tw * (1 - basis_frac)
            net_from_t = tw - (
                federal_tax(regime=regime, filing_status=filing_status, **ctx_after)["tax"]
                - federal_tax(regime=regime, filing_status=filing_status, **ctx)["tax"]
            )
            taxable_w += tw
            ctx = ctx_after
            remaining = max(0.0, remaining - max(0.0, net_from_t))
        if remaining > 0 and pretax_room > 0:
            extra = min(
                _solve_pretax_for_net(remaining, ctx, regime=regime, filing_status=filing_status),
                pretax_room,
            )
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
                _solve_taxable_for_net(w_t, ctx, basis_frac, regime=regime, filing_status=filing_status),
                state.taxable,
            )
            ctx["ltcg"] = ctx.get("ltcg", 0.0) + tw * (1 - basis_frac)
            taxable_w += tw
            pw = min(
                _solve_pretax_for_net(w_p, ctx, regime=regime, filing_status=filing_status),
                pretax_room,
            )
            ax, bx = split_extra(pw)
            pretax_a_w += ax
            pretax_b_w += bx
            ctx["pretax_withdrawal"] += pw
            roth_w += min(w_r, state.roth)

    elif strategy == "bracket_fill":
        ti_now = federal_tax(regime=regime, filing_status=filing_status, **ctx)["taxable_income"]
        headroom = amount_to_fill_bracket(
            ti_now, cfg.bracket_fill_target, regime.ord_brackets(filing_status)
        )
        if remaining > 0 and headroom > 0 and pretax_room > 0:
            cap = min(headroom, pretax_room)
            extra = min(
                _solve_pretax_for_net(min(remaining, cap * 0.85), ctx, regime=regime, filing_status=filing_status),
                cap,
            )
            ax, bx = split_extra(extra)
            pretax_a_w += ax
            pretax_b_w += bx
            ctx["pretax_withdrawal"] += extra
            remaining = max(0.0, remaining - extra * 0.78)
        if remaining > 0 and state.taxable > 0:
            tw = min(
                _solve_taxable_for_net(
                    min(remaining, state.taxable), ctx, basis_frac,
                    regime=regime, filing_status=filing_status,
                ),
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
