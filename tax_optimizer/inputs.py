"""Default scenario inputs.

These are the user-editable household-specific values: spouse ages,
retire ages, salaries, contribution rates, Roth-401(k) splits, starting
balances, Social Security amounts, etc. They're separated from `Config`
so users can edit just the "about me" data without touching the
simulation knobs (and so the same `Inputs` can drive different `Config`
strategies side-by-side).

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
    spouse_b_roth_ira: float = 0.0
    spouse_a_pretax_ira: float = 0.0
    spouse_b_pretax_ira: float = 35_000.0
    pension_balance: float = 0.0
    hsa: float = 18_000.0
    taxable_brokerage: float = 80_000.0

    # NOTE: the simulator pools all Roth dollars into a single bucket
    # (state.roth) regardless of which spouse owns them. Pre-tax IRAs are
    # pooled per-spouse with that spouse's pre-tax 401(k) (so RMDs stay
    # tied to the right age). The per-account split is preserved here for
    # documentation and reporting only.

    @property
    def total_excl_real_estate(self) -> float:
        return (
            self.spouse_a_pretax_401k
            + self.spouse_b_pretax_401k
            + self.spouse_a_roth_ira
            + self.spouse_b_roth_ira
            + self.spouse_a_pretax_ira
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
    start_age: int = 65  # NRD: age Spouse A's pension annuity begins.

    @property
    def annual_at_nrd(self) -> float:
        return self.monthly_at_nrd * 12


@dataclass
class SocialSecurity:
    monthly_spouse_a: float = 2_700.0
    monthly_spouse_b: float = 2_200.0
    start_age: int = 70  # Claim age applied to both spouses (single knob).


@dataclass
class Inputs:
    """Container for all user-editable household scenario data.

    The first block (ages, retire ages, contribution rates, Roth-401(k)
    splits) lives directly on `Inputs` rather than nested under
    `contrib` because they're used independently throughout the
    simulator (timing, salary deferral math, optimizer decision
    variables) and grouping them top-level keeps the JSON scenario
    layout readable.
    """

    spouse_a_age_start: int = 50
    spouse_b_age_start: int = 50
    spouse_a_retire_age: int = 65
    spouse_b_retire_age: int = 65

    spouse_a_total_contrib_pct: float = 0.08
    spouse_b_total_contrib_pct: float = 0.06
    spouse_a_roth_401k_pct: float = 0.0
    spouse_b_roth_401k_pct: float = 0.0

    starting: StartingBalances = field(default_factory=StartingBalances)
    income: CurrentIncome = field(default_factory=CurrentIncome)
    contrib: CurrentContrib = field(default_factory=CurrentContrib)
    pension: PensionInputs = field(default_factory=PensionInputs)
    ss: SocialSecurity = field(default_factory=SocialSecurity)
    annual_expenses: float = 85_000.0
