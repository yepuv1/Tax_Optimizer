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


def initial_state(cfg: Config, inputs: Inputs) -> State:
    s = inputs.starting
    pretax_a = s.spouse_a_pretax_401k + s.spouse_a_pretax_ira
    pretax_b = s.spouse_b_pretax_401k + s.spouse_b_pretax_ira
    roth = s.spouse_a_roth_ira + s.spouse_b_roth_ira
    taxable = s.taxable_brokerage
    return State(
        year=cfg.start_year,
        spouse_a_age=cfg.spouse_a_age_start,
        spouse_b_age=cfg.spouse_b_age_start,
        spouse_a_pretax=pretax_a,
        spouse_b_pretax=pretax_b,
        roth=roth,
        taxable=taxable,
        hsa=s.hsa,
        pension_balance=s.pension_balance,
        pension_annuity=0.0,
        cumulative_basis=taxable * cfg.cap_gains_basis_fraction,
    )
