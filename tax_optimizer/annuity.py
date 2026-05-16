"""Helpers for the optional annuity account type.

The simulator's annuity logic lives inline in
``tax_optimizer.simulator.simulate``; this module just hosts the
small, side-effect-free bits that are easier to read (and test) on
their own. Right now that's a single helper, ``exclusion_ratio``,
implementing the IRC §72(b) split for non-qualified annuity
payments.

§72 simplification
==================

The genuine §72 exclusion ratio is

    exclusion_ratio = investment_in_contract / expected_return

where ``expected_return`` is computed from IRS-published life-
expectancy tables (Treas. Reg. §1.72-9, Table V) for an immediate
straight-life annuity, or a multi-life table for joint-and-
survivor contracts. We do NOT consult those tables here. Instead
we approximate the denominator with

    expected_return ≈ annual_payment * expected_payout_years

where ``expected_payout_years`` is supplied by the user
(``inputs.annuity.expected_payout_years``, default 20). This keeps
the math transparent and tweakable: a shorter horizon front-loads
basis recovery (faster tax-free starting payments, abrupt switch
to 100% taxable when the basis is exhausted); a longer horizon
spreads the exclusion over more years.

The simulator separately tracks ``state.annuity_basis_remaining``
and switches each payment to 100% taxable once the cumulative
excluded amount has matched the cost basis — independent of any
discrepancy between ``expected_payout_years`` and the actual
simulation horizon.
"""

from __future__ import annotations


def exclusion_ratio(
    cost_basis: float,
    annual_payment: float,
    expected_payout_years: int,
) -> float:
    """Return the §72(b) exclusion ratio for a non-qualified annuity.

    Each annual payment is split:

        tax_free_part = annual_payment * exclusion_ratio
        ordinary_part = annual_payment * (1 - exclusion_ratio)

    until ``state.annuity_basis_remaining`` is exhausted, after
    which the entire payment is taxable as ordinary income.

    Edge cases
    ----------
    - ``cost_basis <= 0`` (qualified contract or fully recovered
      basis) → returns 0.0 (every dollar is taxable).
    - ``annual_payment <= 0`` → returns 0.0 (nothing to split, and
      we don't want a divide-by-zero).
    - ``expected_payout_years <= 0`` → would zero the denominator;
      validated upstream in ``Inputs.__post_init__`` so this is a
      defensive guard only. Returns 1.0 (every dollar is treated
      as basis recovery — not realistic, but a graceful fallback
      that's still bounded so downstream simulator code can clamp
      against ``annuity_basis_remaining``).
    - ``cost_basis >= annual_payment * expected_payout_years``
      (over-funded basis) → returns 1.0, capped. The simulator's
      ``annuity_basis_remaining`` decrement still ensures the
      cumulative tax-free amount can't exceed the actual basis.
    """
    if cost_basis <= 0 or annual_payment <= 0:
        return 0.0
    if expected_payout_years <= 0:
        return 1.0
    expected_return = annual_payment * expected_payout_years
    raw = cost_basis / expected_return
    # Clamp into [0, 1] so the simulator doesn't have to defend
    # against pathological inputs separately. Over-funded contracts
    # (basis > expected total payments) hit the 1.0 ceiling here;
    # the per-year ``annuity_basis_remaining`` bookkeeping in the
    # simulator caps the tax-free amount at the actual basis.
    return max(0.0, min(1.0, raw))
