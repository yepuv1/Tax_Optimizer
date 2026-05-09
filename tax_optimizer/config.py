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
from .tax.state import STATELESS, StateTaxRegime


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

    # Annual SS benefit COLA. None ⇒ follow `cfg.inflation`. SSA grants a
    # CPI-W-based COLA each year; this approximates with a constant rate.
    # Setting this to 0.0 reproduces the v1 (broken) behavior of flat-
    # nominal SS benefits across the entire horizon.
    ss_cola_rate: float | None = None

    # Annual rate at which ordinary tax brackets, LTCG brackets, the
    # standard deduction, and IRMAA tier thresholds + surcharge dollars
    # index. None ⇒ follow `cfg.inflation`. IRS uses chained CPI for
    # bracket indexing since TCJA, slightly slower than headline CPI;
    # close enough for projections. NIIT threshold and SS-provisional
    # thresholds are statutory and don't index in real life — they're
    # left flat-nominal inside `TaxRegime.inflated()`.
    #
    # Setting this to 0.0 freezes the regime at its quoted-year nominals
    # across the full horizon (the v1 behavior — produces stealth
    # bracket creep as nominal income grows). 0.0 is also useful as a
    # stress test for "what if Congress freezes brackets".
    bracket_indexing_rate: float | None = None

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
    # SS claim age and pension NRD now live on the nested Inputs blocks
    # (`inputs.ss.start_age`, `inputs.pension.start_age`) so all "about
    # the household" timing knobs cluster with the dollar amounts they
    # gate.
    rmd_start_age: int = 75

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    cap_gains_basis_fraction: float = 0.5
    annual_expenses_today: float = 85_000.0  # overridden by `spending` if set

    # ------------------------------------------------------------------
    # Bequest tax (terminal-NW objective)
    # ------------------------------------------------------------------
    # Marginal rate the simulator assumes heirs will pay on inherited
    # pretax (and HSA) balances. Roth and taxable (stepped-up basis)
    # bequests are tax-free to heirs; pretax / HSA must be drained at
    # ordinary rates. The SECURE Act 10-year rule's exact intra-period
    # timing isn't modeled — this is a flat haircut on the terminal
    # pretax + HSA balance.
    #
    # Default 0.22 is the typical adult-child marginal rate under TCJA
    # brackets. Set higher (28-32%) if heirs are themselves high earners,
    # or zero to recover the v1 behavior of $1-pretax = $1-Roth at
    # horizon (which silently over-rewards "leave it pretax" plans).
    heir_marginal_rate: float = 0.22

    # ------------------------------------------------------------------
    # Taxable-account yield model
    # ------------------------------------------------------------------
    # Dividends and interest produced by the taxable brokerage every
    # year (in addition to realized gains on withdrawals). These flow
    # into AGI / NIIT / IRMAA / SS-provisional income whether or not a
    # spouse is still working, so post-retirement portfolios still feed
    # the tax line. Yields are applied to the year-start `state.taxable`
    # balance, allocated to equity (qualified-dividend) vs bond
    # (ordinary-interest) using `asset_location.taxable`.
    #
    # Defaults reflect a roughly-2024 broad-market portfolio: ~1.6%
    # qualified dividend yield on equities, ~3.5% interest on bonds.
    # Setting either to 0.0 reverts to the v1 behavior of "the taxable
    # account is invisible to AGI between withdrawals" (and any
    # `taxable_drag` you've left in place will fully model friction).
    taxable_equity_div_yield: float = 0.016
    taxable_bond_interest_yield: float = 0.035

    # ------------------------------------------------------------------
    # New v2 modular types
    # ------------------------------------------------------------------
    tax_regime: TaxRegime = TCJA_EXTENDED
    regime_change_year_offset: int | None = None
    regime_change_target: TaxRegime | None = None

    # State income tax regime. Default is `STATELESS` (no state
    # income tax, correct for FL / TX / WA / NV / TN / NH / AK /
    # SD / WY). Bundled non-zero presets: CA, NY, IL, MA. Pass a
    # custom `StateTaxRegime` for other states. State brackets index
    # annually using `bracket_indexing_rate` (same as federal).
    #
    # State income tax can flip optimal Roth-conversion timing:
    # converting in a high-tax state vs. in retirement after moving
    # to a no-tax state is one of the largest decisions a household
    # in CA / NY can make, and is invisible without this.
    state_regime: StateTaxRegime = STATELESS
    state_regime_change_year_offset: int | None = None
    state_regime_change_target: StateTaxRegime | None = None

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
        """Pick the regime SWAP that's active in `year_offset`.

        Doesn't apply bracket indexing — call `effective_regime` for
        the inflation-indexed copy actually used by the simulator.
        """
        if (
            self.regime_change_year_offset is not None
            and self.regime_change_target is not None
            and year_offset >= self.regime_change_year_offset
        ):
            return self.regime_change_target
        return self.tax_regime

    def effective_regime(self, year_offset: int) -> TaxRegime:
        """Return the active regime for `year_offset` with brackets, std
        deduction, and IRMAA tier thresholds + surcharges scaled forward
        by `(1 + bracket_indexing_rate) ** year_offset` (rate falls back
        to `cfg.inflation`).
        """
        raw = self.regime_for_year(year_offset)
        rate = (
            self.bracket_indexing_rate
            if self.bracket_indexing_rate is not None
            else self.inflation
        )
        return raw.inflated((1.0 + rate) ** year_offset)

    def state_regime_for_year(self, year_offset: int) -> StateTaxRegime:
        """Pick the state regime active in `year_offset`. Mirrors
        `regime_for_year`. Lets users model a move (e.g. CA → STATELESS
        after retiring to FL).
        """
        if (
            self.state_regime_change_year_offset is not None
            and self.state_regime_change_target is not None
            and year_offset >= self.state_regime_change_year_offset
        ):
            return self.state_regime_change_target
        return self.state_regime

    def effective_state_regime(self, year_offset: int) -> StateTaxRegime:
        """State-regime equivalent of `effective_regime`."""
        raw = self.state_regime_for_year(year_offset)
        rate = (
            self.bracket_indexing_rate
            if self.bracket_indexing_rate is not None
            else self.inflation
        )
        return raw.inflated((1.0 + rate) ** year_offset)

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
