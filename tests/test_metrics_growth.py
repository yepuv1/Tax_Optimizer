"""Pure-function tests for the new growth-rate helpers in
``tax_optimizer.metrics``.

Each helper is a leaf calculation that the Overview-tab KPI
tiles + multi-strategy growth chart depend on. Pinning their
behavior here means the dashboard layer can rely on a stable
contract without re-deriving CAGRs from raw config / inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tax_optimizer import Config, Inputs
from tax_optimizer.inputs import StartingBalances
from tax_optimizer.metrics import (
    effective_cagr,
    nw_after_tax_series,
    real_cagr,
    stage_cagr,
    starting_after_tax_nw,
    summarize,
    terminal_after_tax_nw,
    total_growth_multiplier,
)
from tax_optimizer.simulator import simulate


# ---------------------------------------------------------------------
# starting_after_tax_nw — bequest-tax-aware starting balance
# ---------------------------------------------------------------------


class TestStartingAfterTaxNW:
    def test_matches_terminal_formula_symmetrically(self) -> None:
        """The whole point of having ``starting_after_tax_nw`` and
        ``terminal_after_tax_nw`` as a pair is that they apply the
        same lens — pretax + HSA discounted by the heir's marginal
        rate, Roth + taxable at face value. Without that symmetry
        the growth ratio bundles a methodology change in with the
        actual growth.
        """
        sb = StartingBalances(
            spouse_a_pretax_401k=200_000.0,
            spouse_b_pretax_401k=100_000.0,
            spouse_a_pretax_ira=0.0,
            spouse_b_pretax_ira=0.0,
            spouse_a_roth_ira=50_000.0,
            spouse_b_roth_ira=10_000.0,
            taxable_brokerage=80_000.0,
            hsa=20_000.0,
            pension_balance=999_999.0,  # excluded by design
        )
        rate = 0.22
        expected = (
            (200_000 + 100_000) * (1 - rate)
            + (50_000 + 10_000)
            + 80_000
            + 20_000 * (1 - rate)
        )
        assert math.isclose(
            starting_after_tax_nw(sb, heir_marginal_rate=rate), expected
        )

    def test_pension_reserve_excluded(self) -> None:
        """Pension reserve is an annuity backing, not heir-accessible
        liquidity. Mirrors ``terminal_after_tax_nw`` which only
        consumes the four liquid-balance columns from the simulator
        DataFrame.
        """
        a = StartingBalances(pension_balance=0.0)
        b = StartingBalances(pension_balance=500_000.0)
        assert math.isclose(
            starting_after_tax_nw(a, heir_marginal_rate=0.22),
            starting_after_tax_nw(b, heir_marginal_rate=0.22),
        )

    def test_zero_buckets(self) -> None:
        sb = StartingBalances(
            spouse_a_pretax_401k=0.0,
            spouse_b_pretax_401k=0.0,
            spouse_a_pretax_ira=0.0,
            spouse_b_pretax_ira=0.0,
            spouse_a_roth_ira=0.0,
            spouse_b_roth_ira=0.0,
            taxable_brokerage=0.0,
            hsa=0.0,
            pension_balance=0.0,
        )
        assert starting_after_tax_nw(sb, heir_marginal_rate=0.22) == 0.0

    def test_aggregates_both_spouses_across_pretax_buckets(self) -> None:
        """Pretax aggregation must combine both spouses' 401(k) AND
        IRA balances — same pooling the simulator does in
        ``state.initial_state``. A bug here would understate the
        bequest-tax discount.
        """
        sb = StartingBalances(
            spouse_a_pretax_401k=100.0,
            spouse_a_pretax_ira=200.0,
            spouse_b_pretax_401k=400.0,
            spouse_b_pretax_ira=800.0,
            spouse_a_roth_ira=0.0,
            spouse_b_roth_ira=0.0,
            taxable_brokerage=0.0,
            hsa=0.0,
            pension_balance=0.0,
        )
        rate = 0.30
        # 1500 * (1 - 0.30) = 1050
        assert math.isclose(
            starting_after_tax_nw(sb, heir_marginal_rate=rate), 1050.0
        )


# ---------------------------------------------------------------------
# total_growth_multiplier
# ---------------------------------------------------------------------


class TestTotalGrowthMultiplier:
    def test_known_example(self) -> None:
        assert math.isclose(
            total_growth_multiplier(1_000_000, 3_150_000), 3.15
        )

    def test_zero_starting_returns_nan(self) -> None:
        assert math.isnan(total_growth_multiplier(0.0, 1_000_000))

    def test_negative_starting_returns_nan(self) -> None:
        # Defensive: a negative starting net worth is meaningless
        # for a growth multiplier.
        assert math.isnan(total_growth_multiplier(-1.0, 1_000_000))

    def test_terminal_lower_than_starting(self) -> None:
        # Decumulation: 1M -> 800K -> 0.8x
        assert math.isclose(
            total_growth_multiplier(1_000_000, 800_000), 0.8
        )


# ---------------------------------------------------------------------
# effective_cagr — nominal CAGR
# ---------------------------------------------------------------------


class TestEffectiveCAGR:
    def test_known_example_30y(self) -> None:
        # 1M -> 3.15M over 30 years  ->  ~3.9%/yr
        cagr = effective_cagr(1_000_000, 3_150_000, 30)
        assert math.isclose(cagr, 0.0390, abs_tol=0.001)

    def test_known_example_doubles_in_one_year(self) -> None:
        assert math.isclose(effective_cagr(100, 200, 1), 1.0)

    def test_zero_years_returns_zero(self) -> None:
        assert effective_cagr(1_000_000, 1_500_000, 0) == 0.0

    def test_negative_years_returns_zero(self) -> None:
        # Defensive: a negative-year horizon is degenerate, not
        # an "implied negative growth" case.
        assert effective_cagr(1_000_000, 1_500_000, -5) == 0.0

    def test_zero_starting_returns_nan(self) -> None:
        assert math.isnan(effective_cagr(0, 1_000_000, 30))

    def test_negative_starting_returns_nan(self) -> None:
        assert math.isnan(effective_cagr(-1.0, 1_000_000, 30))

    def test_loss_returns_negative_cagr(self) -> None:
        """A plan that loses money over the horizon must produce a
        negative CAGR, not NaN. The signed-power root preserves the
        sign of the ratio so the rate stays interpretable.
        """
        cagr = effective_cagr(1_000_000, 800_000, 30)
        assert cagr < 0.0
        assert math.isclose(cagr, (0.8 ** (1 / 30)) - 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------
# real_cagr — inflation-adjusted CAGR
# ---------------------------------------------------------------------


class TestRealCAGR:
    def test_zero_inflation_matches_nominal(self) -> None:
        nominal = effective_cagr(1_000_000, 3_150_000, 30)
        real = real_cagr(1_000_000, 3_150_000, 30, 0.0)
        assert math.isclose(nominal, real, abs_tol=1e-12)

    def test_fisher_approximation_small_rates(self) -> None:
        """For small rates: real ~= nominal - inflation. The Fisher
        equation is exact: (1 + nominal) = (1 + real)(1 + inflation).
        At 3.9 % nominal / 2.5 % inflation the approximation should
        hold to within 0.05 %.
        """
        starting = 1_000_000
        terminal = 3_150_000
        years = 30
        nominal = effective_cagr(starting, terminal, years)
        real = real_cagr(starting, terminal, years, 0.025)
        # Fisher: (1+nominal) = (1+real)(1+infl)  =>  real = (1+nominal)/(1+infl) - 1
        fisher_real = (1 + nominal) / 1.025 - 1
        assert math.isclose(real, fisher_real, abs_tol=1e-9)
        # Linear approximation
        assert abs((nominal - 0.025) - real) < 0.001

    def test_zero_starting_returns_nan(self) -> None:
        assert math.isnan(real_cagr(0, 1_000_000, 30, 0.025))

    def test_zero_years_returns_zero(self) -> None:
        assert real_cagr(1_000_000, 1_500_000, 0, 0.025) == 0.0


# ---------------------------------------------------------------------
# stage_cagr — accumulation / decumulation split
# ---------------------------------------------------------------------


def _synth_df(
    *,
    spouse_a_ages: list[int],
    pretax: list[float],
    roth: list[float] | None = None,
    taxable: list[float] | None = None,
    hsa: list[float] | None = None,
) -> pd.DataFrame:
    n = len(spouse_a_ages)
    return pd.DataFrame(
        {
            "spouse_a_age": spouse_a_ages,
            "pretax_balance": pretax,
            "roth_balance": roth or [0.0] * n,
            "taxable_balance": taxable or [0.0] * n,
            "hsa_balance": hsa or [0.0] * n,
        }
    )


class TestStageCAGR:
    def test_accumulation_positive_decumulation_negative(self) -> None:
        """Synthetic: balances ramp up to the retirement boundary,
        then taper off. Accumulation CAGR must be > 0, decumulation
        < 0.
        """
        ages = list(range(60, 70))  # 10 years
        # Boundary at age 65: ages [60..64] = accumulation (5 yrs)
        # ages [65..69] = decumulation (5 yrs)
        # Accumulation: balance grows 1.0M -> 1.5M
        # Decumulation: balance falls 1.5M -> 1.0M
        pretax = [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000,
                  1_500_000, 1_400_000, 1_300_000, 1_200_000, 1_000_000]
        df = _synth_df(spouse_a_ages=ages, pretax=pretax)
        starting_nw = 1_000_000 * (1 - 0.22)
        accum, decum = stage_cagr(
            df, starting_nw, retire_age=65, heir_marginal_rate=0.22
        )
        assert accum is not None and accum > 0
        assert decum is not None and decum < 0

    def test_returns_none_for_empty_df(self) -> None:
        df = pd.DataFrame(
            columns=["spouse_a_age", "pretax_balance", "roth_balance",
                     "taxable_balance", "hsa_balance"]
        )
        accum, decum = stage_cagr(df, 1.0, retire_age=65)
        assert accum is None and decum is None

    def test_no_accumulation_phase(self) -> None:
        """Already-retired household: every row has age >= retire_age,
        so accumulation is None and decumulation runs from starting
        NW directly.
        """
        ages = list(range(70, 75))
        pretax = [1_000_000, 950_000, 900_000, 850_000, 800_000]
        df = _synth_df(spouse_a_ages=ages, pretax=pretax)
        starting_nw = 1_000_000 * (1 - 0.22)
        accum, decum = stage_cagr(df, starting_nw, retire_age=65)
        assert accum is None
        assert decum is not None and decum < 0

    def test_no_decumulation_phase(self) -> None:
        """Horizon ends before retirement (e.g. early-retirement
        sandbox): decumulation None, accumulation populated.
        """
        ages = list(range(50, 55))
        pretax = [500_000, 600_000, 700_000, 800_000, 900_000]
        df = _synth_df(spouse_a_ages=ages, pretax=pretax)
        starting_nw = 500_000 * (1 - 0.22)
        accum, decum = stage_cagr(df, starting_nw, retire_age=65)
        assert accum is not None and accum > 0
        assert decum is None

    def test_missing_spouse_a_age_returns_none(self) -> None:
        df = pd.DataFrame(
            {
                "pretax_balance": [100, 200],
                "roth_balance": [0, 0],
                "taxable_balance": [0, 0],
                "hsa_balance": [0, 0],
            }
        )
        accum, decum = stage_cagr(df, 100, retire_age=65)
        assert accum is None and decum is None


# ---------------------------------------------------------------------
# nw_after_tax_series — vectorized terminal_after_tax_nw
# ---------------------------------------------------------------------


class TestNWAfterTaxSeries:
    def test_last_value_matches_terminal(self) -> None:
        """The last value of the series MUST equal the scalar
        ``terminal_after_tax_nw`` — the Overview chart's final
        point is the same number as the Terminal-NW tile.
        """
        cfg = Config(horizon_age=58)
        inputs = Inputs()
        df = simulate(cfg, inputs)
        ser = nw_after_tax_series(
            df, heir_marginal_rate=cfg.heir_marginal_rate
        )
        scalar = terminal_after_tax_nw(
            df, heir_marginal_rate=cfg.heir_marginal_rate
        )
        assert math.isclose(ser.iloc[-1], scalar)

    def test_per_row_matches_scalar_on_subset(self) -> None:
        """Every intermediate row of the series must equal what
        ``terminal_after_tax_nw`` would compute on a one-row slice.
        Vectorization shouldn't change the formula.
        """
        cfg = Config(horizon_age=58)
        df = simulate(cfg, Inputs())
        ser = nw_after_tax_series(
            df, heir_marginal_rate=cfg.heir_marginal_rate
        )
        for i in [0, len(df) // 2, len(df) - 1]:
            scalar = terminal_after_tax_nw(
                df.iloc[: i + 1], heir_marginal_rate=cfg.heir_marginal_rate
            )
            assert math.isclose(ser.iloc[i], scalar)

    def test_series_index_matches_input(self) -> None:
        df = pd.DataFrame(
            {
                "pretax_balance": [100, 200, 300],
                "roth_balance": [10, 20, 30],
                "taxable_balance": [5, 15, 25],
                "hsa_balance": [1, 2, 3],
            },
            index=[2025, 2026, 2027],
        )
        ser = nw_after_tax_series(df, heir_marginal_rate=0.22)
        assert list(ser.index) == [2025, 2026, 2027]


# ---------------------------------------------------------------------
# summarize — new keys appear on the summary dict
# ---------------------------------------------------------------------


class TestSummarizeGrowthKeys:
    def test_new_keys_present_with_full_kwargs(self) -> None:
        cfg = Config(horizon_age=58)
        inputs = Inputs()
        df = simulate(cfg, inputs)
        summary = summarize(
            df,
            heir_marginal_rate=cfg.heir_marginal_rate,
            starting_balances=inputs.starting,
            inflation=cfg.inflation,
            retire_age=inputs.spouse_a_retire_age,
        )
        # All six growth keys must be present.
        for key in (
            "starting_after_tax",
            "total_growth_mult",
            "effective_cagr",
            "real_cagr",
            "accumulation_cagr",
            "decumulation_cagr",
        ):
            assert key in summary, f"missing growth key {key!r}"
        # Sanity: starting matches the standalone helper exactly.
        assert math.isclose(
            summary["starting_after_tax"],
            starting_after_tax_nw(
                inputs.starting, heir_marginal_rate=cfg.heir_marginal_rate
            ),
        )

    def test_legacy_call_yields_none_growth_keys(self) -> None:
        """Existing tests (and any custom callers) that call
        ``summarize(df)`` without the new kwargs must still work —
        the growth keys come back as None instead of raising.
        """
        df = simulate(Config(horizon_age=58), Inputs())
        summary = summarize(df)
        for key in (
            "starting_after_tax",
            "total_growth_mult",
            "effective_cagr",
            "real_cagr",
            "accumulation_cagr",
            "decumulation_cagr",
        ):
            assert summary[key] is None

    def test_partial_kwargs_yield_partial_metrics(self) -> None:
        """``starting_balances`` alone is enough for starting-NW,
        multiplier, and nominal CAGR; ``inflation`` and
        ``retire_age`` are independent gates for real-CAGR and
        stage-CAGR respectively.
        """
        cfg = Config(horizon_age=58)
        inputs = Inputs()
        df = simulate(cfg, inputs)
        summary = summarize(
            df,
            heir_marginal_rate=cfg.heir_marginal_rate,
            starting_balances=inputs.starting,
        )
        assert summary["starting_after_tax"] is not None
        assert summary["total_growth_mult"] is not None
        assert summary["effective_cagr"] is not None
        assert summary["real_cagr"] is None  # no inflation supplied
        assert summary["accumulation_cagr"] is None  # no retire_age
        assert summary["decumulation_cagr"] is None
