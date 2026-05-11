"""End-to-end smoke tests for the deterministic simulator.

These run the full year-by-year engine against the example scenario file
and verify shape, key invariants, and (where deterministic) the
relationships across columns. They're slower than the unit tests above
but still complete in well under a second.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tax_optimizer import (
    Config,
    Inputs,
    Mortality,
    StrategyResult,
    apply_scenario,
    load_scenario_file,
    simulate,
    summarize,
)
from tax_optimizer.market import DeterministicModel, LognormalModel
from tax_optimizer.metrics import lifetime_irmaa_npv, lifetime_tax_npv, terminal_after_tax_nw
from tax_optimizer.state import State, initial_state


# ---------------------------------------------------------------------------
# initial_state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_pools_pretax_per_spouse(self, cfg: Config, inputs: Inputs) -> None:
        st = initial_state(cfg, inputs)
        assert isinstance(st, State)
        # Spouse A: 401k + IRA pooled together (RMDs key off A's age).
        assert st.spouse_a_pretax == pytest.approx(
            inputs.starting.spouse_a_pretax_401k + inputs.starting.spouse_a_pretax_ira
        )
        # Spouse B: same pooling.
        assert st.spouse_b_pretax == pytest.approx(
            inputs.starting.spouse_b_pretax_401k + inputs.starting.spouse_b_pretax_ira
        )
        # Roth pooled across spouses (couple-level account).
        assert st.roth == pytest.approx(
            inputs.starting.spouse_a_roth_ira + inputs.starting.spouse_b_roth_ira
        )
        assert st.taxable == inputs.starting.taxable_brokerage
        assert st.hsa == inputs.starting.hsa
        assert st.pension_balance == inputs.starting.pension_balance

    def test_initial_basis_uses_cap_gains_fraction(self, cfg: Config, inputs: Inputs) -> None:
        st = initial_state(cfg, inputs)
        assert st.cumulative_basis == pytest.approx(
            inputs.starting.taxable_brokerage * cfg.cap_gains_basis_fraction
        )


# ---------------------------------------------------------------------------
# Single-path simulator
# ---------------------------------------------------------------------------


class TestSimulateDefaults:
    def test_returns_dataframe_of_expected_length(
        self, cfg: Config, inputs: Inputs
    ) -> None:
        df = simulate(cfg, inputs)
        # Years simulated = horizon_age - age_start + 1
        expected_len = cfg.horizon_age - inputs.spouse_a_age_start + 1
        assert isinstance(df, pd.DataFrame)
        assert len(df) == expected_len

    def test_required_columns_present(self, cfg: Config, inputs: Inputs) -> None:
        df = simulate(cfg, inputs)
        required = {
            "year", "spouse_a_age", "spouse_b_age",
            "alive_a", "alive_b", "filing_status", "regime",
            "wages", "pension", "ssn",
            "rmd", "rmd_a", "rmd_b",
            "roth_conversion", "roth_conversion_a", "roth_conversion_b",
            "pretax_withdrawal", "pretax_withdrawal_a", "pretax_withdrawal_b",
            "roth_withdrawal", "taxable_withdrawal",
            "agi", "taxable_income", "federal_tax", "marginal",
            "irmaa", "irmaa_tier",
            "spending_need",
            "equity_return", "bond_return",
            "pretax_balance", "pretax_a_balance", "pretax_b_balance",
            "roth_balance", "taxable_balance", "hsa_balance",
            "pension_balance",
        }
        missing = required - set(df.columns)
        assert not missing, f"missing columns: {missing}"

    def test_ages_are_monotonic(self, cfg: Config, inputs: Inputs) -> None:
        df = simulate(cfg, inputs)
        assert (df["spouse_a_age"].diff().dropna() == 1).all()
        assert (df["spouse_b_age"].diff().dropna() == 1).all()
        assert (df["year"].diff().dropna() == 1).all()

    def test_balances_non_negative(self, cfg: Config, inputs: Inputs) -> None:
        df = simulate(cfg, inputs)
        for col in (
            "pretax_balance", "pretax_a_balance", "pretax_b_balance",
            "roth_balance", "hsa_balance", "pension_balance",
        ):
            # tiny negatives can creep in via float arithmetic; allow epsilon
            assert (df[col] >= -1e-6).all(), f"{col} went negative"

    def test_no_wages_after_retirement(self, cfg: Config, inputs: Inputs) -> None:
        df = simulate(cfg, inputs)
        post = df[df["spouse_a_age"] >= inputs.spouse_a_retire_age]
        post = post[post["spouse_b_age"] >= inputs.spouse_b_retire_age]
        assert (post["wages"] == 0.0).all()

    def test_ss_starts_at_configured_age(self, cfg: Config, inputs: Inputs) -> None:
        df = simulate(cfg, inputs)
        # SS = 0 strictly before the claim age, > 0 from claim onward.
        a_claim = inputs.ss.effective_start_age_a
        before = df[df["spouse_a_age"] < a_claim]
        after = df[df["spouse_a_age"] >= a_claim]
        assert (before["ssn"] == 0.0).all()
        # SS now scales by (a) actuarial claim-age factor on FRA-PIA, and
        # (b) annual COLA. The PIA scaling uses claim_age_factor; COLA
        # follows cfg.ss_cola_rate (None ⇒ cfg.inflation). With default
        # inputs (ss.start_age=70, ss.fra_a=ss.fra_b=67), the factor is
        # 1.24 (3 years × 8% delayed-retirement credit).
        if not after.empty:
            both_alive = after[after["alive_a"] & after["alive_b"]]
            if not both_alive.empty:
                from tax_optimizer.inputs import claim_age_factor
                a_fac = claim_age_factor(a_claim, inputs.ss.fra_a)
                b_fac = claim_age_factor(
                    inputs.ss.effective_start_age_b, inputs.ss.fra_b
                )
                ss_cola = (
                    cfg.ss_cola_rate if cfg.ss_cola_rate is not None else cfg.inflation
                )
                offsets = (
                    both_alive["spouse_a_age"].values - inputs.spouse_a_age_start
                )
                expected = (
                    (
                        inputs.ss.monthly_spouse_a * 12 * a_fac
                        + inputs.ss.monthly_spouse_b * 12 * b_fac
                    )
                    * (1 + ss_cola) ** offsets
                )
                assert np.allclose(both_alive["ssn"].values, expected)

    def test_rmd_starts_at_configured_age(self, cfg: Config, inputs: Inputs) -> None:
        df = simulate(cfg, inputs)
        pre_rmd = df[df["spouse_a_age"] < cfg.rmd_start_age]
        # No RMD before start age.
        assert (pre_rmd["rmd"] == 0.0).all()
        # Once both spouses are past start age and have a positive balance,
        # RMD must be > 0.
        post = df[
            (df["spouse_a_age"] >= cfg.rmd_start_age)
            & (df["spouse_b_age"] >= cfg.rmd_start_age)
            & (df["pretax_balance"] > 0)
        ]
        assert (post["rmd"] > 0).all()

    def test_filing_status_default_mfj(self, cfg: Config, inputs: Inputs) -> None:
        df = simulate(cfg, inputs)
        assert (df["filing_status"] == "mfj").all()

    def test_simulate_accepts_default_inputs(self, cfg: Config) -> None:
        # Calling without inputs uses Inputs() defaults; just shouldn't blow up.
        df = simulate(cfg)
        assert len(df) > 0


class TestSimulateWidowsPenalty:
    def test_filing_status_flips_at_widow_year(self) -> None:
        cfg = Config(mortality=Mortality(year_of_death_a=20))
        inputs = Inputs()
        df = simulate(cfg, inputs)
        # IRS year-of-death rule: years 0..19 are MFJ both alive, year
        # 20 is the year of A's death (still MFJ per IRS), year 21+ is
        # single (qualifying-surviving-spouse not modeled).
        assert (df.iloc[:20]["filing_status"] == "mfj").all()
        assert df.iloc[20]["filing_status"] == "mfj"
        assert (df.iloc[21:]["filing_status"] == "single").all()

    def test_alive_a_flag_flips(self) -> None:
        cfg = Config(mortality=Mortality(year_of_death_a=20))
        df = simulate(cfg, Inputs())
        assert df.iloc[19]["alive_a"]
        assert not df.iloc[20]["alive_a"]


class TestSimulateRegimeChange:
    def test_regime_label_changes_at_offset(self) -> None:
        cfg = Config(
            tax_regime=__import__(
                "tax_optimizer.tax.regimes", fromlist=["TCJA_EXTENDED"]
            ).TCJA_EXTENDED,
            regime_change_year_offset=10,
            regime_change_target=__import__(
                "tax_optimizer.tax.regimes", fromlist=["SUNSET_2026"]
            ).SUNSET_2026,
        )
        df = simulate(cfg, Inputs())
        # First 10 rows under starting regime, rest under target.
        assert (df.iloc[:10]["regime"] == "TCJA_extended_2026").all()
        assert (df.iloc[10:]["regime"] == "tcja_sunset_2026").all()


class TestSimulateExampleScenario:
    """Smoke run against the shipped example scenarios (real-world wiring)."""

    @pytest.mark.parametrize(
        "scenario_path",
        [
            Path("scenarios/example01.json"),
            Path("scenarios/example02.json"),
        ],
    )
    def test_example_scenario_simulates(self, scenario_path: Path) -> None:
        # The example scenarios are part of the repo and must exist; a
        # silent skip here previously hid the fact that the path was
        # broken in the docs (see CHANGELOG entry for v6.1).
        assert scenario_path.exists(), f"missing {scenario_path}"
        scenario = load_scenario_file(scenario_path)
        cfg, inputs = apply_scenario(Config(), Inputs(), scenario)
        df = simulate(cfg, inputs)
        # Years simulated = horizon_age - spouse_a_age_start + 1
        expected = cfg.horizon_age - inputs.spouse_a_age_start + 1
        assert len(df) == expected
        # AGI is non-negative every year (spending may net to zero but AGI
        # is gross of withdrawals and never strictly negative).
        assert (df["agi"] >= 0).all()


# ---------------------------------------------------------------------------
# Metrics + StrategyResult
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_summarize_keys_and_types(self, cfg: Config, inputs: Inputs) -> None:
        df = simulate(cfg, inputs)
        s = summarize(df)
        for k in (
            "lifetime_tax_npv", "lifetime_irmaa_npv", "terminal_after_tax",
            "peak_marginal", "years_irmaa", "peak_irmaa_tier", "min_balance",
        ):
            assert k in s
        # Marginal is a fractional rate, so peak_marginal should sit in [0,1].
        assert 0 <= s["peak_marginal"] <= 1

    def test_lifetime_tax_npv_matches_summarize(
        self, cfg: Config, inputs: Inputs
    ) -> None:
        df = simulate(cfg, inputs)
        s = summarize(df)
        assert s["lifetime_tax_npv"] == pytest.approx(lifetime_tax_npv(df))
        assert s["lifetime_irmaa_npv"] == pytest.approx(lifetime_irmaa_npv(df))

    def test_terminal_after_tax_nw_signs(
        self, cfg: Config, inputs: Inputs
    ) -> None:
        df = simulate(cfg, inputs)
        # Terminal NW should be positive for the default scenario (it's an
        # accumulator, no death assumed) but the function should also work
        # when balances are zero.
        out = terminal_after_tax_nw(df)
        assert isinstance(out, float)


class TestStrategyResult:
    def test_dataclass_round_trip(self, cfg: Config, inputs: Inputs) -> None:
        df = simulate(cfg, inputs)
        s = summarize(df)
        sr = StrategyResult(cfg=cfg, inputs=inputs, df=df, summary=s)
        # The 4-tuple unpacking the simulator/optimizer used pre-refactor:
        assert sr.cfg is cfg
        assert sr.inputs is inputs
        assert sr.df is df
        assert sr.summary is s
