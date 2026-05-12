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
    # Roth-conversion liquidity guards (v6.5)
    # ------------------------------------------------------------------
    # When True, size each year's Roth conversion so its marginal
    # federal + state tax fits within the household's *non-Roth*
    # tax-paying capacity (earned cash + pension + SS + RMD net of
    # FICA/SDI/spending/healthcare/contributions, plus an allowed
    # slice of the taxable brokerage). Prevents the simulator from
    # converting more than the household could realistically pay tax
    # on, which previously would silently trigger the deficit
    # cascade and raid the just-funded Roth bucket. Defaults True.
    # Set False to recover pre-v6.5 behavior (sizes to bracket
    # headroom regardless of cash on hand) — useful for sensitivity
    # analysis on the "what if I could find tax cash from elsewhere"
    # boundary.
    cap_conversion_by_liquidity: bool = True

    # Fraction of the taxable-brokerage balance the conversion sizer
    # is willing to earmark for paying the conversion's marginal tax.
    # Combined with the earned/retirement-income surplus to form
    # `tax_paying_capacity`. Default 0.5 leaves half the taxable
    # account as runway for spending shocks. Set 1.0 to allow the
    # full taxable balance, 0.0 to require the conversion tax come
    # entirely from earned-income surplus / pension / SS / RMD.
    conversion_taxable_use_ratio: float = 0.5

    # When True, the deficit cascade excludes the Roth bucket in any
    # year a Roth conversion fires. Otherwise an under-sized
    # liquidity check (or a fixed-dollar conversion that overshoots
    # capacity) silently withdraws from the just-converted Roth to
    # pay the conversion tax — which under IRS rules can trigger
    # a 10% penalty on conversion principal if the holder is < 59½
    # or the 5-year clock hasn't matured (neither tracked by the
    # model). Default True. Set False to allow Roth in the cascade
    # (recovers pre-v6.5 behavior).
    protect_roth_in_conversion_years: bool = True

    # ------------------------------------------------------------------
    # Age-gated income events
    # ------------------------------------------------------------------
    # SS claim age and pension NRD now live on the nested Inputs blocks
    # (`inputs.ss.start_age`, `inputs.pension.start_age`) so all "about
    # the household" timing knobs cluster with the dollar amounts they
    # gate.
    rmd_start_age: int = 75

    # ------------------------------------------------------------------
    # Healthcare cost knobs (Tier C-B)
    # ------------------------------------------------------------------
    # Combined Medicare Part B + Part D base premium per enrolled
    # spouse, in today's dollars. Inflated forward using
    # `cfg.inflation`. The 2026 published Part B base is ~$2,084/yr;
    # adding a typical Part D premium (~$650/yr, varies widely by
    # plan) lands in the $2,500–$2,800 range. Default $2,500 is a
    # mid-of-range, conservative-low setting; users with rich Part D
    # plans should set this higher. This is **separate from IRMAA**
    # (income-related surcharge), which still flows through
    # `irmaa_annual_surcharge`.
    medicare_base_b_d_premium: float = 2_500.0

    # Pre-Medicare household healthcare cost in today's dollars,
    # charged each year either spouse is alive AND at least one
    # spouse is below `MEDICARE_ELIGIBLE_AGE` (65). Per-spouse share
    # is the household total times (n_pre_medicare / 2). Set to 0.0
    # if you've baked healthcare into `spending.base_spending`
    # already.
    health_pre65_today: float = 0.0

    # IRMAA MAGI lookback. SSA computes IRMAA in year T from MAGI in
    # year T-2 (with a one-year fallback for filing-year edge cases).
    # The simulator keeps a running lag chain in `state.agi_lag_*`;
    # set to 0 to revert to the pre-Tier-C behavior (use current-year
    # AGI), which over-states IRMAA exposure in any year with a
    # large transient income spike (Roth conversion, retirement-year
    # severance, etc.).
    irmaa_lookback_years: int = 2

    # ------------------------------------------------------------------
    # ACA premium tax credit (Tier C-B)
    # ------------------------------------------------------------------
    # Enable ACA enhanced subsidies (post-IRA-2022 8.5%-of-MAGI cap on
    # benchmark premium). Models the post-Inflation Reduction Act of
    # 2022 rules: no income cliff (the pre-2021 "400% FPL cliff" is
    # gone), premium contribution capped at 8.5% of MAGI for any
    # household above ~150% FPL. Below 150% FPL, contribution drops
    # toward zero (we use a single `aca_max_contrib_pct` knob; for a
    # full Form 8962 you'd want the seven-step income×age curve).
    aca_enabled: bool = False
    # Benchmark premium per enrolled adult (second-lowest-cost silver
    # plan, age-adjusted). Today's dollars. Inflates by `cfg.inflation`
    # (a stand-in for healthcare CPI; if you have a separate healthcare
    # inflation series, fold it into a custom regime). Default $14k is
    # a rough national average for a 60-year-old in 2026.
    aca_benchmark_premium_per_adult: float = 14_000.0
    # Maximum applicable percentage of MAGI a household pays toward
    # premiums. Post-IRA-2022 = 8.5% for all incomes >= 150% FPL.
    aca_max_contrib_pct: float = 0.085

    # ------------------------------------------------------------------
    # Step-up in basis on first spouse's death (Tier C-B)
    # ------------------------------------------------------------------
    # If True, the surviving spouse's cost basis on the taxable account
    # is reset to fair-market-value when the first spouse dies. Models
    # the community-property full step-up (CA, WA, ID, etc.). Default
    # False = no step-up (pre-Tier-C behavior; conservative).
    stepup_at_first_death: bool = False

    # ------------------------------------------------------------------
    # Optimizer scope (Tier C-C)
    # ------------------------------------------------------------------
    # If True, the optimizer adds per-spouse SS claim age (62/65/67/70
    # grid) to the decision vector. Defaults False to keep the
    # baseline 3-axis search (Roth %, conv bracket) backward-compatible.
    optimize_ss_claim_age: bool = False

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

    # Fraction of taxable-equity dividends that are "qualified" (LTCG-
    # rate-eligible). The remainder is non-qualified — REIT dividends,
    # certain foreign-issuer payors, money-market sweeps inside an
    # equity fund — and is taxed at ordinary rates. For a broad
    # US-equity index, ~0.85–0.90 is typical; mixed funds with REIT
    # exposure run lower. Set to 1.0 to recover the pre-v6.3 behavior
    # of treating all dividends as qualified (slightly optimistic).
    taxable_equity_qualified_fraction: float = 0.85

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

    def __post_init__(self) -> None:
        # The IRS Uniform Lifetime Table starts at age 72 (SECURE 2.0
        # currently sets the RMD start age to 73, going to 75 in 2033;
        # 75 is our default). Anything below 72 has no published
        # divisor — silently mishandling that previously caused the
        # entire pretax balance to be returned as the RMD. Reject it
        # at construction so misconfiguration fails loudly.
        if self.rmd_start_age < 72:
            raise ValueError(
                f"rmd_start_age must be >= 72 (the youngest age in the IRS "
                f"Uniform Lifetime Table); got {self.rmd_start_age}."
            )

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
