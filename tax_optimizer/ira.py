"""Per-spouse IRA contribution helpers.

Three contribution paths are modeled per spouse, all capped together
under one annual `IRA_CONTRIBUTION_LIMIT` (+ catch-up):

  1. **Deductible Traditional IRA** — reduces ordinary income (Box 1)
     by the contribution amount; balance lands in pretax. We treat
     it as universally deductible for simplicity. The IRS deductibility
     phase-out (when covered by a workplace plan) is NOT modeled —
     households with AGI past the phase-out should prefer the
     **backdoor** path below for the same dollar effect.

  2. **Direct Roth IRA** — after-tax contribution → Roth balance.
     Subject to the IRS MAGI phase-out (zeroed out above the cap).

  3. **Backdoor Roth** — non-deductible Traditional IRA contribution
     (no Box 1 reduction) immediately converted to Roth. Income-
     uncapped. The pro-rata rule applies: if the spouse already has
     a non-zero pretax IRA / 401(k) balance, a fraction of the
     conversion is taxable as ordinary income.

Cap allocation is done in priority order: Traditional first, then
Roth direct (after phase-out), then backdoor on whatever room is
left. So a user setting `traditional_ira_contrib=7000 AND
backdoor_roth=True` gets the Traditional contribution and the
backdoor is silently skipped (cap exhausted).

Eligibility is gated by the simulator: the spouse must be alive and
EITHER spouse must have W-2 earned income (allows a non-working
spouse to use a "spousal IRA" against the working spouse's wages).
"""

from __future__ import annotations

from dataclasses import dataclass

from .limits import ira_contribution_cap, roth_ira_phaseout_factor


@dataclass
class IRAAllocation:
    """How much was actually contributed (after caps & phase-outs).

    `backdoor_taxable_conversion` is the portion of the backdoor's
    Roth conversion that's taxable as ordinary income under the
    pro-rata rule. The simulator adds this to `roth_conversion` in
    the federal-tax kwargs so the tax bill reflects it.
    """

    traditional: float
    roth_direct: float
    backdoor: float
    backdoor_taxable_conversion: float

    @property
    def total_cash_outflow(self) -> float:
        """Total dollars leaving the household's after-tax cash flow.

        Traditional IRA still draws from after-tax cash (the user
        gets the deduction back via reduced federal tax), so all
        three paths contribute to cash outflow.
        """
        return self.traditional + self.roth_direct + self.backdoor


def allocate_ira_contributions(
    *,
    age: int,
    eligible: bool,
    pretax_existing: float,
    traditional_target: float,
    roth_direct_target: float,
    backdoor_enabled: bool,
    magi_estimate: float,
    filing_status: str,
) -> IRAAllocation:
    """Compute the actual IRA contribution split for one spouse.

    `pretax_existing` is the spouse's **IRA-only** pretax balance
    BEFORE this year's contributions — used for the backdoor
    pro-rata rule (IRC §408(d)(2)). 401(k) balances are NOT
    aggregated for pro-rata, so the simulator passes
    `state.spouse_*_pretax_ira` here, NOT the combined pretax bucket.

    `magi_estimate` is the household's MAGI estimate (federal AGI
    is a close-enough approximation) used for the direct-Roth
    phase-out check.
    """
    if not eligible:
        return IRAAllocation(0.0, 0.0, 0.0, 0.0)

    cap = ira_contribution_cap(age)
    remaining = cap

    traditional = min(max(0.0, traditional_target), remaining)
    remaining -= traditional

    phase_factor = roth_ira_phaseout_factor(magi_estimate, filing_status)
    roth_direct_eligible = max(0.0, roth_direct_target) * phase_factor
    roth_direct = min(roth_direct_eligible, remaining)
    remaining -= roth_direct

    backdoor = remaining if backdoor_enabled else 0.0

    # Pro-rata rule (IRC §408(d)(2)): the converted backdoor amount
    # is taxable in proportion to how much of the spouse's pretax IRA
    # basis is already "all earnings, no basis" (i.e. existing
    # pretax). With zero existing pretax (the clean-backdoor case)
    # the conversion is fully tax-free.
    #
    # Modeling scope (F14): we treat *all* of `pretax_existing` as
    # zero-basis (pre-tax) and *all* of the freshly-contributed
    # nondeductible amount as basis. In real life a spouse may have
    # pre-existing **after-tax basis** in their Traditional IRA
    # (from prior nondeductible contributions filed on Form 8606),
    # which would *reduce* the taxable conversion fraction. We don't
    # track that historical basis. For typical users the pretax IRA
    # is dominated by pre-tax 401(k) rollovers, so this simplification
    # leans conservative (slightly overstates the backdoor tax cost
    # when historical 8606 basis exists).
    backdoor_taxable = 0.0
    if backdoor > 0 and pretax_existing > 0:
        taxable_fraction = pretax_existing / (pretax_existing + backdoor)
        backdoor_taxable = backdoor * taxable_fraction

    return IRAAllocation(
        traditional=traditional,
        roth_direct=roth_direct,
        backdoor=backdoor,
        backdoor_taxable_conversion=backdoor_taxable,
    )
