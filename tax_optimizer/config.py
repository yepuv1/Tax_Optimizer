"""Aggregated simulation configuration.

`Config` bundles the simulation knobs (macro assumptions, policy
choices, modular blocks). The household-specific "about me" data --
spouse ages, retire ages, contribution rates, Roth-401(k) splits,
starting balances, salaries, Social Security amounts, etc. -- lives
on `Inputs` instead so the two concerns don't bleed into each other.
A clean rule of thumb:

  * If a number describes the *household*, it goes on `Inputs`.
  * If a number describes the *simulation / strategy / world*, it
    goes on `Config`.

Modular blocks on `Config`:

  * `tax_regime`           : which `TaxRegime` is active
  * `regime_change_*`      : optional mid-simulation regime swap
  * `mortality`            : per-spouse death year + survivor election
  * `market`               : `DeterministicModel` / `LognormalModel` /
                              `BootstrapModel` for stochastic returns
  * `asset_location`       : per-account equity/bond split
  * `spending`             : `SpendingProfile` (smile + lump events + LTC)

When `market` is a `DeterministicModel(equity=cfg.nominal_growth_rate,
bond=cfg.nominal_growth_rate)` AND `asset_location` is uniform AND
`spending` is the flat default, the simulator reproduces the v1
deterministic numbers byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .market import AssetLocation, DeterministicModel, MarketModel
from .mortality import Mortality
from .spending import SpendingProfile
from .tax.regimes import TCJA_EXTENDED, TaxRegime


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    start_year: int = 2026
    horizon_age: int = 90

    # ------------------------------------------------------------------
    # Macro assumptions (used by `DeterministicModel` if `market` is
    # left as default; also used as the `inflation` driver for the
    # default flat `SpendingProfile`).
    # ------------------------------------------------------------------
    nominal_growth_rate: float = 0.06  # equity & bond when deterministic
    inflation: float = 0.025
    wage_growth: float = 0.030
    taxable_drag: float = 0.005

    # ------------------------------------------------------------------
    # Withdrawal & conversion strategy
    # ------------------------------------------------------------------
    withdrawal_strategy: str = "conventional"
    bracket_fill_target: float = 0.22
    roth_conversion_target_bracket: float = 0.0
    roth_conversion_amount: float = 0.0

    # ------------------------------------------------------------------
    # Age-gated income events
    # ------------------------------------------------------------------
    ss_start_age: int = 70
    pension_start_age: int = 65
    rmd_start_age: int = 75

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    cap_gains_basis_fraction: float = 0.5
    annual_expenses_today: float = 85_000.0  # overridden by `spending` if set

    # ------------------------------------------------------------------
    # New v2 modular types
    # ------------------------------------------------------------------
    tax_regime: TaxRegime = TCJA_EXTENDED
    regime_change_year_offset: int | None = None
    regime_change_target: TaxRegime | None = None

    mortality: Mortality = field(default_factory=Mortality)

    market: MarketModel | None = None  # None => DeterministicModel from scalars
    asset_location: AssetLocation = field(
        default_factory=lambda: AssetLocation.uniform(equity_pct=1.0)
    )
    # Note: `AssetLocation.uniform(1.0)` makes every account behave like
    # the v1 single-rate growth (no bond drag). Set `asset_location =
    # AssetLocation()` for the textbook bonds-in-pretax / equities-in-Roth
    # split.

    spending: SpendingProfile | None = None  # None => flat profile from scalars

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def regime_for_year(self, year_offset: int) -> TaxRegime:
        if (
            self.regime_change_year_offset is not None
            and self.regime_change_target is not None
            and year_offset >= self.regime_change_year_offset
        ):
            return self.regime_change_target
        return self.tax_regime

    def resolved_market(self) -> MarketModel:
        if self.market is not None:
            return self.market
        # Backward compat: scalar growth → uniform deterministic returns.
        return DeterministicModel(
            equity=self.nominal_growth_rate, bond=self.nominal_growth_rate
        )

    def resolved_spending(self) -> SpendingProfile:
        if self.spending is not None:
            return self.spending
        return SpendingProfile.flat(
            base_spending=self.annual_expenses_today, inflation=self.inflation
        )
