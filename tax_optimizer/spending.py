"""Realistic spending models.

The default behavior of a flat `base_spending` inflated forward by 2.5%/yr
is a rough approximation. Empirically, retirement spending follows a
"smile" curve documented by Blanchett (2014) and Bernicke (2005):

    * High in the early "go-go" years (travel, hobbies, kids' weddings).
    * Lower in the middle "slow-go" years.
    * Rising again at the end as long-term-care, medical, and
      residence costs accelerate.

`SpendingProfile` lets you express that shape via age-banded multipliers.
`LumpEvent` adds one-off cash outflows (new car, helping a child with a
down payment, a known LTC episode). `LongTermCareShock` is a
convenience for "the last N years of life are X% more expensive" which
is the most common LTC-modeling pattern in retirement planning software.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class SpendingPhase:
    """Multiplier applied to `base_spending` while either spouse is in the
    inclusive age range `[age_lo, age_hi]`. The first matching phase wins.
    """

    age_lo: int
    age_hi: int
    multiplier: float
    label: str = ""


@dataclass
class LumpEvent:
    """A one-off, inflation-adjusted cash outflow during the simulation.

    `year_offset` is the simulation year offset (0 = first year).
    `amount_today` is the amount in today's dollars; the simulator
    inflation-adjusts it forward to the year of payment using the
    profile's `inflation` rate.

    Lump events always add to that year's spending need; the active
    `withdrawal_strategy` decides which bucket(s) actually pay for them.
    `preferred_source` is a hint — currently informational only; future
    strategies may consult it, but the simulator does not enforce it
    today (avoids double-counting the cash in the surplus accounting).
    """

    year_offset: int
    amount_today: float
    label: str = ""
    preferred_source: Literal["taxable", "pretax", "roth", "any"] = "any"


@dataclass
class LongTermCareShock:
    """A late-life cost ramp that fires for the last `years` years of life.

    For couples, the standard planning assumption is at least one episode
    of care (~3 years for women, ~2 for men per Genworth's Cost of Care
    Survey) at $80-150k/yr depending on geography and care setting.
    """

    years: int = 3
    annual_cost_today: float = 80_000.0


@dataclass
class SpendingProfile:
    """Variable-by-age spending plus optional lump events and an LTC shock.

    Backward-compatible default mimics the old flat behavior: a single
    `SpendingPhase(0, 200, 1.0)` over all ages, no events, no LTC shock.

    The "retirement smile" preset (`retirement_smile()`) is the typical
    Blanchett shape calibrated to a 50-year-old couple retiring at 65.
    """

    base_spending: float
    inflation: float = 0.025
    phases: list[SpendingPhase] = field(
        default_factory=lambda: [SpendingPhase(0, 200, 1.0, "flat")]
    )
    lump_events: list[LumpEvent] = field(default_factory=list)
    ltc_shock: LongTermCareShock | None = None

    @classmethod
    def flat(cls, base_spending: float, inflation: float = 0.025) -> "SpendingProfile":
        return cls(base_spending=base_spending, inflation=inflation)

    @classmethod
    def retirement_smile(
        cls,
        base_spending: float,
        inflation: float = 0.025,
        ltc_years: int = 3,
        ltc_annual_today: float = 80_000.0,
    ) -> "SpendingProfile":
        """Blanchett-style retirement smile around `base_spending`.

        Working years run flat at base. Retirement starts with a 15%
        bump (go-go), drops 10% below base in the slow-go years, then
        the LTC shock layers on top in the final years.
        """
        return cls(
            base_spending=base_spending,
            inflation=inflation,
            phases=[
                SpendingPhase(0, 64, 1.00, "working"),
                SpendingPhase(65, 74, 1.15, "go-go"),
                SpendingPhase(75, 84, 0.90, "slow-go"),
                SpendingPhase(85, 200, 1.00, "no-go"),
            ],
            ltc_shock=LongTermCareShock(years=ltc_years, annual_cost_today=ltc_annual_today),
        )

    def _phase_multiplier(self, age: int) -> float:
        for phase in self.phases:
            if phase.age_lo <= age <= phase.age_hi:
                return phase.multiplier
        return 1.0

    def amount_for(
        self,
        year_offset: int,
        age_a: int,
        *,
        years_until_horizon: int,
        years_until_death: Optional[int] = None,
    ) -> tuple[float, list[LumpEvent]]:
        """Return (recurring_spending_for_year, list_of_lump_events_firing_this_year).

        LTC shock timing:

          * If `years_until_death` is provided, it anchors the shock
            to the household's actual end-of-life year — this is the
            economically meaningful definition. The shock fires in
            the last `ltc_shock.years` years before all spouses are
            dead.
          * If `years_until_death` is `None` (legacy callers), we fall
            back to `years_until_horizon`. That works only when the
            horizon and the life expectancy coincide; otherwise the
            shock incorrectly fires at the *simulation* end instead of
            at the *life* end.

        The simulator now passes both arguments. The pre-v6.2 default
        attached the shock to `horizon - years` which fired the shock
        on still-alive spouses' last years when `horizon_age > life
        expectancy` and skipped it entirely when `horizon_age < life
        expectancy`.
        """
        mult = self._phase_multiplier(age_a)
        recurring = self.base_spending * mult * (1 + self.inflation) ** year_offset
        ltc_anchor = years_until_death if years_until_death is not None else years_until_horizon
        if self.ltc_shock and ltc_anchor >= 0 and ltc_anchor < self.ltc_shock.years:
            recurring += (
                self.ltc_shock.annual_cost_today * (1 + self.inflation) ** year_offset
            )
        events_this_year = [e for e in self.lump_events if e.year_offset == year_offset]
        return recurring, events_this_year
