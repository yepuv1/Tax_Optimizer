"""Withdrawal-strategy solvers.

These functions return *gross* withdrawals per bucket given a net-spending
target. They're pure (no state mutation); the simulator applies the
balance changes after computing federal tax + IRMAA on the final
combined income line.

Solvers can optionally take a `state_tax_fn` callable so the gross-up
accounts for state income tax on the incremental withdrawal. Without
it (the default, backward compat), the gross-up is federal-only —
which silently underdrew by ~9-12% for CA/MA/OR users in cascade
years until v6.2.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .config import Config
from .state import State
from .tax.federal import amount_to_fill_bracket, federal_tax
from .tax.regimes import TaxRegime


StateTaxFn = Callable[[dict, float], float]
"""Signature: `state_tax_fn(kwargs, ss_taxable_federal) -> state_tax_dollars`.

Takes the same kwargs dict the solver feeds to `federal_tax`, plus the
`ss_taxable` value `federal_tax` returned on that same call (so the
state tax function doesn't have to recompute it). Returns total state
income tax in dollars.
"""


def _solve_pretax_for_net(
    net_target: float,
    base_kwargs: dict,
    *,
    regime: TaxRegime,
    filing_status: str,
    state_tax_fn: Optional[StateTaxFn] = None,
) -> float:
    if net_target <= 0:
        return 0.0
    base_fed = federal_tax(regime=regime, filing_status=filing_status, **base_kwargs)
    base_tax = base_fed["tax"]
    base_state = state_tax_fn(base_kwargs, base_fed["ss_taxable"]) if state_tax_fn else 0.0
    lo, hi = 0.0, net_target * 2.0 + 50_000.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        kw = dict(base_kwargs)
        kw["pretax_withdrawal"] = kw.get("pretax_withdrawal", 0.0) + mid
        new_fed = federal_tax(regime=regime, filing_status=filing_status, **kw)
        fed_delta = new_fed["tax"] - base_tax
        state_delta = (
            state_tax_fn(kw, new_fed["ss_taxable"]) - base_state
            if state_tax_fn else 0.0
        )
        net = mid - fed_delta - state_delta
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
    state_tax_fn: Optional[StateTaxFn] = None,
) -> float:
    if net_target <= 0:
        return 0.0
    # Clamp basis_frac into [0, 1] (F5). A `basis_frac` > 1.0 would
    # imply the taxable account is sitting on an unrealized loss
    # (basis > FMV); a < 0 value would imply negative basis. Neither
    # should produce a negative `gain` (effectively a phantom AGI
    # reduction). The simulator already clamps when it builds the live
    # ratio, but the public solver entry point should be defensive.
    basis_frac = min(1.0, max(0.0, basis_frac))
    base_fed = federal_tax(regime=regime, filing_status=filing_status, **base_kwargs)
    base_tax = base_fed["tax"]
    base_state = state_tax_fn(base_kwargs, base_fed["ss_taxable"]) if state_tax_fn else 0.0
    lo, hi = 0.0, net_target * 2.0 + 50_000.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        gain = mid * (1 - basis_frac)
        kw = dict(base_kwargs)
        kw["ltcg"] = kw.get("ltcg", 0.0) + gain
        new_fed = federal_tax(regime=regime, filing_status=filing_status, **kw)
        fed_delta = new_fed["tax"] - base_tax
        state_delta = (
            state_tax_fn(kw, new_fed["ss_taxable"]) - base_state
            if state_tax_fn else 0.0
        )
        net = mid - fed_delta - state_delta
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
    state_tax_fn: Optional[StateTaxFn] = None,
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
                    state_tax_fn=state_tax_fn,
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
                _solve_pretax_for_net(
                    remaining, ctx,
                    regime=regime, filing_status=filing_status,
                    state_tax_fn=state_tax_fn,
                ),
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
                _solve_taxable_for_net(
                    w_t, ctx, basis_frac,
                    regime=regime, filing_status=filing_status,
                    state_tax_fn=state_tax_fn,
                ),
                state.taxable,
            )
            ctx["ltcg"] = ctx.get("ltcg", 0.0) + tw * (1 - basis_frac)
            taxable_w += tw
            pw = min(
                _solve_pretax_for_net(
                    w_p, ctx,
                    regime=regime, filing_status=filing_status,
                    state_tax_fn=state_tax_fn,
                ),
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
                _solve_pretax_for_net(
                    min(remaining, cap * 0.85), ctx,
                    regime=regime, filing_status=filing_status,
                    state_tax_fn=state_tax_fn,
                ),
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
                    state_tax_fn=state_tax_fn,
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


def cover_deficit(
    deficit: float,
    state: State,
    base_kwargs: dict[str, Any],
    basis_frac: float,
    *,
    pretax_a_already: float = 0.0,
    pretax_b_already: float = 0.0,
    taxable_already: float = 0.0,
    roth_already: float = 0.0,
    hsa_already: float = 0.0,
    hsa_unlocked: bool = False,
    regime: TaxRegime,
    filing_status: str,
    state_tax_fn: Optional[StateTaxFn] = None,
) -> tuple[dict[str, float], float]:
    """Pull `deficit` net dollars from accounts in tax-efficient order.

    Cascades through taxable -> roth -> hsa-after-65 -> pretax.
    Each step grosses up for the appropriate tax (LTCG on the
    taxable gain portion, ordinary on pretax / HSA-non-medical).
    Roth is treated as fully tax-free (we deliberately don't model
    the 5-year clock here -- that's a documented blind spot).

    `hsa_unlocked=True` ⇒ at least one spouse is 65+. After 65 HSA
    non-medical withdrawals incur ordinary tax but no 20% penalty,
    so the HSA acts like a Traditional IRA without RMDs. We cascade
    HSA *before* pretax because the HSA has no RMD obligation, no
    state-tax friction in some regimes, and no widow's-penalty
    risk on a pretax balance the survivor inherits.

    `*_already` is the gross already withdrawn from each bucket this
    year (so room calculations stay correct). `base_kwargs` already
    reflects those prior withdrawals' AGI contributions.

    Returns ``(extra_withdrawals, unfunded)`` where ``extra_withdrawals``
    has the same shape as ``withdraw_for_need``'s output (with an
    extra ``hsa`` key) and ``unfunded`` is dollars still un-met after
    exhausting every bucket.
    """
    out = {
        "pretax_a": 0.0,
        "pretax_b": 0.0,
        "pretax": 0.0,
        "roth": 0.0,
        "taxable": 0.0,
        "hsa": 0.0,
    }
    if deficit <= 0:
        return out, 0.0

    kw = dict(base_kwargs)

    def _net_after_marginal_tax(prev_kw: dict, new_kw: dict, gross: float) -> float:
        """Net produced from `gross` after both federal and state marginal tax.

        Computes the delta in federal + state tax between `prev_kw` and
        `new_kw` (which differ by the additional cascade draw the caller
        just added to `new_kw`), and subtracts from gross.
        """
        prev_fed = federal_tax(regime=regime, filing_status=filing_status, **prev_kw)
        new_fed = federal_tax(regime=regime, filing_status=filing_status, **new_kw)
        fed_delta = max(0.0, new_fed["tax"] - prev_fed["tax"])
        state_delta = 0.0
        if state_tax_fn is not None:
            prev_state = state_tax_fn(prev_kw, prev_fed["ss_taxable"])
            new_state = state_tax_fn(new_kw, new_fed["ss_taxable"])
            state_delta = max(0.0, new_state - prev_state)
        return gross - fed_delta - state_delta

    # 1. Taxable: realize gains at LTCG rate. Solver returns the gross
    # withdrawal needed to net the (capped) target.
    taxable_room = max(0.0, state.taxable - taxable_already)
    if deficit > 0 and taxable_room > 0:
        target = min(deficit, taxable_room)
        tw = min(
            _solve_taxable_for_net(
                target, kw, basis_frac,
                regime=regime, filing_status=filing_status,
                state_tax_fn=state_tax_fn,
            ),
            taxable_room,
        )
        if tw > 0:
            prev_kw = dict(kw)
            kw["ltcg"] = kw.get("ltcg", 0.0) + tw * (1 - basis_frac)
            net_produced = _net_after_marginal_tax(prev_kw, kw, tw)
            out["taxable"] += tw
            deficit = max(0.0, deficit - max(0.0, net_produced))

    # 2. Roth: tax-free (blind spot: 5-year rule on conversions ignored).
    roth_room = max(0.0, state.roth - roth_already)
    if deficit > 0 and roth_room > 0:
        rw = min(deficit, roth_room)
        if rw > 0:
            out["roth"] += rw
            deficit = max(0.0, deficit - rw)

    # 2.5 HSA-after-65: ordinary income, no penalty. Treated as pretax
    # for tax purposes (federal_tax sees `pretax_withdrawal` and grosses
    # up identically). No RMD on HSA, so it's strictly preferred over
    # the pretax buckets when both are available — that's the "stealth
    # IRA" property of an HSA.
    hsa_room = max(0.0, state.hsa - hsa_already) if hsa_unlocked else 0.0
    if deficit > 0 and hsa_room > 0:
        gross = min(
            _solve_pretax_for_net(
                deficit, kw,
                regime=regime, filing_status=filing_status,
                state_tax_fn=state_tax_fn,
            ),
            hsa_room,
        )
        if gross > 0:
            out["hsa"] += gross
            prev_kw = dict(kw)
            kw["pretax_withdrawal"] = kw.get("pretax_withdrawal", 0.0) + gross
            net_produced = _net_after_marginal_tax(prev_kw, kw, gross)
            deficit = max(0.0, deficit - max(0.0, net_produced))

    # 3. Pretax: gross up for marginal ordinary rate; split pro-rata.
    pretax_a_room = max(0.0, state.spouse_a_pretax - pretax_a_already)
    pretax_b_room = max(0.0, state.spouse_b_pretax - pretax_b_already)
    pretax_room = pretax_a_room + pretax_b_room
    if deficit > 0 and pretax_room > 0:
        gross = min(
            _solve_pretax_for_net(
                deficit, kw,
                regime=regime, filing_status=filing_status,
                state_tax_fn=state_tax_fn,
            ),
            pretax_room,
        )
        if gross > 0:
            a_share = pretax_a_room / pretax_room if pretax_room > 0 else 0.0
            ax = min(gross * a_share, pretax_a_room)
            bx = min(gross - ax, pretax_b_room)
            out["pretax_a"] += ax
            out["pretax_b"] += bx
            out["pretax"] = out["pretax_a"] + out["pretax_b"]
            prev_kw = dict(kw)
            kw["pretax_withdrawal"] = kw.get("pretax_withdrawal", 0.0) + gross
            net_produced = _net_after_marginal_tax(prev_kw, kw, gross)
            deficit = max(0.0, deficit - max(0.0, net_produced))

    return out, max(0.0, deficit)
