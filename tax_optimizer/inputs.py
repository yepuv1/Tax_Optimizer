"""Default scenario inputs.

These are the user-editable starting balances, income, and contribution
settings. They're separated from `Config` so users can edit just the
"about me" data without touching the simulation knobs.

Defaults reflect a typical dual-income married-filing-jointly couple turning
50 in 2026:
  - Combined household income ~$170k (mid-career professional + a lower-
    earning spouse). ~70th percentile of MFJ households per Census 2023 ACS.
  - Combined 401(k) balance $375k. Mean 401(k) balance for 50-somethings
    is $220-250k per Vanguard/Fidelity 2024 reporting; this assumes
    somewhat-better-than-mean savers, typical of tax-optimizer users.
  - Modest IRA, HSA, and taxable brokerage balances.
  - No private-sector pension (rare today; set if you have one).
  - SS estimates are SSA Quick-Calculator PIA figures at FRA for those
    career earnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StartingBalances:
    spouse_a_pretax_401k: float = 225_000.0
    spouse_b_pretax_401k: float = 150_000.0
    spouse_a_roth_ira: float = 40_000.0
    spouse_b_pretax_ira: float = 35_000.0
    pension_balance: float = 0.0
    hsa: float = 18_000.0
    taxable_brokerage: float = 80_000.0

    @property
    def total_excl_real_estate(self) -> float:
        return (
            self.spouse_a_pretax_401k
            + self.spouse_b_pretax_401k
            + self.spouse_a_roth_ira
            + self.spouse_b_pretax_ira
            + self.pension_balance
            + self.hsa
            + self.taxable_brokerage
        )


@dataclass
class CurrentIncome:
    spouse_a_gross: float = 95_000.0
    spouse_b_gross: float = 70_000.0
    spouse_a_bonus: float = 5_000.0
    interest: float = 500.0
    capital_gains: float = 1_000.0
    dividends: float = 2_000.0


@dataclass
class CurrentContrib:
    spouse_a_pct: float = 0.08
    spouse_b_pct: float = 0.06
    spouse_a_roth_pct: float = 0.0
    spouse_b_roth_pct: float = 0.0
    hsa_family: float = 8_550.0
    std_deduction: float = 32_200.0
    baseline_tax: float = 0.0


@dataclass
class PensionInputs:
    balance_today: float = 0.0
    monthly_at_nrd: float = 0.0

    @property
    def annual_at_nrd(self) -> float:
        return self.monthly_at_nrd * 12


@dataclass
class SocialSecurity:
    monthly_spouse_a: float = 2_700.0
    monthly_spouse_b: float = 2_200.0


@dataclass
class Inputs:
    """Container for all user-editable scenario data."""

    starting: StartingBalances = field(default_factory=StartingBalances)
    income: CurrentIncome = field(default_factory=CurrentIncome)
    contrib: CurrentContrib = field(default_factory=CurrentContrib)
    pension: PensionInputs = field(default_factory=PensionInputs)
    ss: SocialSecurity = field(default_factory=SocialSecurity)
    annual_expenses: float = 85_000.0
