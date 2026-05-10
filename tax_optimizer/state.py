"""Mutable per-year account state."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .inputs import Inputs


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
    # IRA-only sub-balance of `spouse_*_pretax`. Tracked separately for
    # the **backdoor Roth pro-rata rule** (IRC §408(d)(2)) which
    # aggregates only Traditional/SEP/SIMPLE IRA balances, NOT 401(k).
    # Pre-Tier-C, the simulator passed total pretax (IRA + 401k) to
    # `allocate_ira_contributions` and overstated taxable conversion
    # for any spouse with rolled-over 401(k) money — TC-6 fix.
    spouse_a_pretax_ira: float = 0.0
    spouse_b_pretax_ira: float = 0.0
    # AGI lag chain. Read at the start of each simulator iteration,
    # rolled forward at the end. Naming convention:
    #
    #   prior_agi  := AGI from year T-1 (used as IRA MAGI estimate, TC-5)
    #   agi_lag_2  := AGI from year T-2 (used as IRMAA lookback,    TC-11)
    #
    # End-of-loop bookkeeping:
    #   agi_lag_2  <- prior_agi      (was T-1 → becomes "T-2" for year T+1)
    #   prior_agi  <- tax_result.agi (current-year AGI → "T-1" next year)
    prior_agi: float = 0.0
    agi_lag_2: float = 0.0


def initial_state(cfg: Config, inputs: Inputs) -> State:
    s = inputs.starting
    pretax_a = s.spouse_a_pretax_401k + s.spouse_a_pretax_ira
    pretax_b = s.spouse_b_pretax_401k + s.spouse_b_pretax_ira
    roth = s.spouse_a_roth_ira + s.spouse_b_roth_ira
    taxable = s.taxable_brokerage
    return State(
        year=cfg.start_year,
        spouse_a_age=inputs.spouse_a_age_start,
        spouse_b_age=inputs.spouse_b_age_start,
        spouse_a_pretax=pretax_a,
        spouse_b_pretax=pretax_b,
        spouse_a_pretax_ira=s.spouse_a_pretax_ira,
        spouse_b_pretax_ira=s.spouse_b_pretax_ira,
        roth=roth,
        taxable=taxable,
        hsa=s.hsa,
        pension_balance=s.pension_balance,
        pension_annuity=0.0,
        cumulative_basis=taxable * cfg.cap_gains_basis_fraction,
    )
